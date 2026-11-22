// Fused "glue + matvec" kernels of the Inkling 2 decode graph (phase 3).
//
// One persistent launch of 128-thread blocks: a few blocks run the latency-bound glue (the previous MoE epilogue, the
// attention-input mix + norm -> Q8_1 activation) while the other blocks preload the first rows of the weight matrix into
// registers, wait for a device-side flag, and then stream the rest of the matrix. The glue is hidden behind the weight
// preload (the memory system is busy the whole time), and no kernel boundary separates the dependent steps.
//
// The grid never exceeds the number of co-resident blocks (occupancy), so the spinning blocks cannot starve the glue
// blocks. The flags/counters live in the persistent zeroed counter area and are reset by the last block to finish, so
// the kernel is replayable from a CUDA graph.

#include "inkling.cuh"
#include "inkling-common.cuh"
#include "vecdotq.cuh"

// Each output row belongs to one of the three input groups. The exclusive
// attention consumer reads only that column of the ordinary matmul result.
template <bool q6>
static __global__ void ik_segmented_projection(const void * weights, const block_q8_1 * x, float * out,
        int rows, int q_rows, int k_rows) {
    constexpr int qk = q6 ? QK_K : QK4_NL;
    constexpr int blocks = 2304/qk;
    constexpr int vdr = q6 ? VDR_Q6_K_Q8_1_MMVQ : VDR_IQ4_NL_Q8_1_MMVQ;
    constexpr int lanes = (q6 ? QI6_K : QI4_NL)/vdr;
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= rows) {
        return;
    }
    const int col = row < q_rows ? 0 : row < q_rows + k_rows ? 1 : 2;
    // One warp computes a row. Keep the four partial sums of Ampere's ordinary
    // three-column MMVQ in registers so the final additions follow the same order.
    constexpr int stride = WARP_SIZE/lanes;
    float partial[4] = {};
#pragma unroll
    for (int base = 0; base < blocks; base += 4*stride) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int k = base + w*stride + threadIdx.x/lanes;
            if (k < blocks) {
                const block_q8_1 * y = x + col*(2304/QK8_1) + k*(qk/QK8_1);
                const int iqs = (threadIdx.x % lanes)*vdr;
                if constexpr (q6) {
                    partial[w] += vec_dot_q6_K_q8_1(weights, y, row*blocks + k, iqs);
                } else {
                    partial[w] += vec_dot_iq4_nl_q8_1(weights, y, row*blocks + k, iqs);
                }
            }
        }
    }
    const float sum = warp_reduce_sum(((partial[0] + partial[1]) + partial[2]) + partial[3]);
    if (threadIdx.x == (row % 2)) {
        out[col*rows + row] = sum;
    }
}

void ggml_cuda_inkling_segmented_projection(ggml_backend_cuda_context & ctx, ggml_tensor * projection, const ggml_tensor * attn) {
    const int rows = projection->ne[0];
    const int q_rows = ggml_get_op_params_i32(attn, 0) *
            (2*ggml_get_op_params_i32(attn, 2) + ggml_get_op_params_i32(attn, 3) + 1);
    const dim3 grid((rows + 3)/4), threads(WARP_SIZE, 4);
    const void * w = projection->src[0]->data;
    const block_q8_1 * x = (const block_q8_1 *) projection->src[1]->data;
    float * out = (float *) projection->data;
    if (projection->src[0]->type == GGML_TYPE_Q6_K) {
        ik_segmented_projection<true><<<grid, threads, 0, ctx.stream()>>>(w, x, out, rows, q_rows, (rows - q_rows)/2);
    } else {
        ik_segmented_projection<false><<<grid, threads, 0, ctx.stream()>>>(w, x, out, rows, q_rows, (rows - q_rows)/2);
    }
}

// The rows of a window: the projection of each token's column of the row's group, per (row, token) the arithmetic of
// ik_segmented_projection. x: [2304, n_tok, 3] Q8_1 (the token columns of a group are consecutive), out: [rows, 3*n_tok].
template <bool q6>
static __global__ void ik_segmented_projection_rows(const void * weights, const block_q8_1 * x, float * out,
        int rows, int q_rows, int k_rows, int n_tok) {
    constexpr int qk = q6 ? QK_K : QK4_NL;
    constexpr int blocks = 2304/qk;
    constexpr int vdr = q6 ? VDR_Q6_K_Q8_1_MMVQ : VDR_IQ4_NL_Q8_1_MMVQ;
    constexpr int lanes = (q6 ? QI6_K : QI4_NL)/vdr;
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= rows) {
        return;
    }
    const int col = row < q_rows ? 0 : row < q_rows + k_rows ? 1 : 2;
    constexpr int stride = WARP_SIZE/lanes;
    float partial[8][4] = {};
#pragma unroll
    for (int base = 0; base < blocks; base += 4*stride) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int k = base + w*stride + threadIdx.x/lanes;
            if (k < blocks) {
                const int iqs = (threadIdx.x % lanes)*vdr;
                if constexpr (!q6) {
                    // the weights are looked up once for all the tokens
                    const ik_iq4_nl_w wd = ik_iq4_nl_decode(weights, row*blocks + k, iqs);
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        if (t < n_tok) {
                            partial[t][w] += ik_iq4_nl_dot(wd, x + (col*n_tok + t)*(2304/QK8_1) + k*(qk/QK8_1), iqs);
                        }
                    }
                    continue;
                }
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (t < n_tok) {
                        const block_q8_1 * y = x + (col*n_tok + t)*(2304/QK8_1) + k*(qk/QK8_1);
                        if constexpr (q6) {
                            partial[t][w] += vec_dot_q6_K_q8_1(weights, y, row*blocks + k, iqs);
                        } else {
                            partial[t][w] += vec_dot_iq4_nl_q8_1(weights, y, row*blocks + k, iqs);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t < n_tok) {
            const float sum = warp_reduce_sum(((partial[t][0] + partial[t][1]) + partial[t][2]) + partial[t][3]);
            if (threadIdx.x == (row % 2)) {
                out[(col*n_tok + t)*rows + row] = sum;
            }
        }
    }
}

bool ggml_cuda_inkling_segmented_projection_rows(ggml_backend_cuda_context & ctx, ggml_tensor * projection, int q_rows, int n_tok) {
    const ggml_tensor * w = projection->src[0];
    const ggml_tensor * x = projection->src[1];
    if ((w->type != GGML_TYPE_Q6_K && w->type != GGML_TYPE_IQ4_NL) || x->type != GGML_TYPE_Q8_1 || w->ne[0] != 2304 || n_tok < 1 || n_tok > 8 ||
            projection->ne[1] != 3*n_tok || !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(projection)) {
        return false;
    }
    const int rows = projection->ne[0];
    const dim3 grid((rows + 3)/4), threads(WARP_SIZE, 4);
    if (w->type == GGML_TYPE_Q6_K) {
        ik_segmented_projection_rows<true><<<grid, threads, 0, ctx.stream()>>>(w->data, (const block_q8_1 *) x->data, (float *) projection->data, rows, q_rows, (rows - q_rows)/2, n_tok);
    } else {
        ik_segmented_projection_rows<false><<<grid, threads, 0, ctx.stream()>>>(w->data, (const block_q8_1 *) x->data, (float *) projection->data, rows, q_rows, (rows - q_rows)/2, n_tok);
    }
    return true;
}

#define IK_FMV_BLOCK      128
#define IK_FMV_WARPS      (IK_FMV_BLOCK/WARP_SIZE)
#ifndef IK_FMV_MIN_BLOCKS
#define IK_FMV_MIN_BLOCKS 8
#endif
#ifndef IK_FMV_PIPE
#define IK_FMV_PIPE       1
#endif
#define IK_FMV_MAX_STATES 6      // states of the attention-residual mix
#define IK_FMV_GRAM       (IK_FMV_MAX_STATES*(IK_FMV_MAX_STATES + 1)/2)   // the Gram matrix of the states (upper triangle)
#define IK_FMV_PART       (IK_FMV_GRAM + IK_FMV_MAX_STATES*IK_FMV_MAX_COLS) // per glue block: gram, then dot[col][state]
#define IK_FMV_MAX_EMBD   3072   // the activation columns kept in shared memory
#define IK_FMV_MAX_COLS   3
#define IK_FMV_MAX_RP     16     // row pairs per block between two result flushes

struct ik_fmv_sync {
    int epi_flag, epi_arrivals;   // the barrier of the glue blocks
    int flag, glue_arrivals;      // the glue blocks -> the matvec blocks
    int mv_arrivals;              // the matvec blocks -> reset
    int work;                     // the next row group to process (dynamic distribution: the active rows stay contiguous)
    int pad[2];
};

static __device__ __forceinline__ void ik_wait_flag(const int * flag) {
    if (threadIdx.x == 0) {
        while (ik_ld_acquire(flag) == 0) {
            __nanosleep(64);
        }
    }
    __syncthreads();
}

// all threads of the block have written (the barrier orders them before thread 0's release, which is cumulative):
// count the block's arrival, the last one releases the flags: one word per waiting block (hundreds of blocks polling a
// single word saturate its L2 slice), set with one fence and relaxed stores (a release store is a fence each)
static __device__ __forceinline__ void ik_arrive_release(int * arrivals, int * flags, const int n_flags, const int n_blocks, int * s_last) {
    __syncthreads();
    if (threadIdx.x == 0) {
        *s_last = ik_atomic_add_acq_rel(arrivals, 1) == n_blocks - 1;
    }
    __syncthreads();
    if (*s_last) {
        __threadfence();
        for (int i = threadIdx.x; i < n_flags; i += blockDim.x) {
            ik_st_relaxed(flags + i, 1);
        }
    }
}

// all blocks of a group arrive, every block waits for the whole group (a grid barrier of the group)
static __device__ __forceinline__ void ik_group_barrier(int * arrivals, const int n_blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        ik_atomic_add_acq_rel(arrivals, 1);
        while (ik_ld_acquire(arrivals) < n_blocks) {
            __nanosleep(64);
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------------------------------
// the quantized weights of R rows at one (block, int offset) position of a 128-thread block: `L` lanes share a quant
// block, `BPI` blocks are covered per iteration; the dot products with the Q8_1 activation blocks of a column

template <ggml_type T, int R> struct ik_qrows;

template <int R> struct ik_qrows<GGML_TYPE_Q8_0, R> {
    static constexpr int qk  = QK8_0;
    static constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    static constexpr int L   = QI8_0/vdr;
    static constexpr int BPI = IK_FMV_BLOCK/L;
    int  v[R][vdr];
    half d[R];
    __device__ __forceinline__ int & first() { return v[R - 1][vdr - 1]; } // the last-issued word (pacing)

    __device__ __forceinline__ void load(const char * const * rows, const int kbx, const int kqs) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const block_q8_0 * b = (const block_q8_0 *) rows[r] + kbx;
#pragma unroll
            for (int i = 0; i < vdr; ++i) {
                v[r][i] = get_int_b2(b->qs, kqs + i);
            }
            d[r] = b->d;
        }
    }

    __device__ __forceinline__ void dot(const block_q8_1 * const * y, const int kbx, const int kqs, float * acc) const {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            int u[vdr];
#pragma unroll
            for (int i = 0; i < vdr; ++i) {
                u[i] = get_int_b4(y[r][kbx].qs, kqs + i);
            }
            acc[r] += vec_dot_q8_0_q8_1_impl<float, vdr>(v[r], u, __half2float(d[r]), __low2float(y[r][kbx].ds));
        }
    }
};

template <int R> struct ik_qrows<GGML_TYPE_IQ4_NL, R> {
    static constexpr int qk  = QK4_NL;
    static constexpr int vdr = VDR_IQ4_NL_Q8_1_MMVQ;
    static constexpr int L   = QI4_NL/vdr;
    static constexpr int BPI = IK_FMV_BLOCK/L;
    int  q[R][vdr];
    half d[R];
    __device__ __forceinline__ int & first() { return q[R - 1][vdr - 1]; } // the last-issued word (pacing)

    __device__ __forceinline__ void load(const char * const * rows, const int kbx, const int kqs) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const block_iq4_nl * b = (const block_iq4_nl *) rows[r] + kbx;
#pragma unroll
            for (int l = 0; l < vdr; ++l) {
                q[r][l] = get_int_b2(b->qs, kqs + l);
            }
            d[r] = b->d;
        }
    }

    __device__ __forceinline__ void dot(const block_q8_1 * const * y, const int kbx, const int kqs, float * acc) const {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int * q8 = (const int *) y[r][kbx].qs + kqs;
            int sumi = 0;
#pragma unroll
            for (int l = 0; l < vdr; ++l) {
                const int2 v = get_int_from_table_16(q[r][l], kvalues_iq4nl);
                sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
                sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
            }
            acc[r] += __half2float(d[r])*__low2float(y[r][kbx].ds)*sumi;
        }
    }
};

template <int R> struct ik_qrows<GGML_TYPE_Q6_K, R> {
    static constexpr int qk  = QK_K;
    static constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    static constexpr int L   = QI6_K/vdr;
    static constexpr int BPI = IK_FMV_BLOCK/L;
    int  vl[R][vdr];
    int  vh[R][vdr];
    int  sc[R];       // the two scales, packed
    half d[R];
    __device__ __forceinline__ int & first() { return vh[R - 1][vdr - 1]; } // the last-issued word (pacing)

    __device__ __forceinline__ void load(const char * const * rows, const int kbx, const int iqs) {
        const int vh_shift     = 2*((iqs % (QI6_K/2))/(QI6_K/4));
        const int scale_offset = (QI6_K/4)*(iqs/(QI6_K/2)) + (iqs % (QI6_K/2))/(QI6_K/8);
        const int vh_offset    = (QI6_K/4)*(iqs/(QI6_K/2)) + iqs % (QI6_K/4);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const block_q6_K * b = (const block_q6_K *) rows[r] + kbx;
#pragma unroll
            for (int j = 0; j < vdr; ++j) {
                vl[r][j] = get_int_b2(b->ql, iqs + j);
                vh[r][j] = get_int_b2(b->qh, vh_offset + j) >> vh_shift;
            }
            sc[r] = (int) (uint8_t) b->scales[scale_offset] | ((int) (uint8_t) b->scales[scale_offset + 4] << 8);
            d[r]  = b->d;
        }
    }

    __device__ __forceinline__ void dot(const block_q8_1 * const * y, const int kbx, const int iqs, float * acc) const {
        const int bq8_offset = 2*QR6_K*(iqs/(QI6_K/2)) + (iqs % (QI6_K/2))/(QI6_K/4);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const block_q8_1 * yb = y[r] + kbx*(QK_K/QK8_1) + bq8_offset;
            int   u[vdr*QR6_K];
            float d8[QR6_K];
#pragma unroll
            for (int i = 0; i < QR6_K; ++i) {
#pragma unroll
                for (int j = 0; j < vdr; ++j) {
                    u[vdr*i + j] = get_int_b4(yb[2*i].qs, iqs % QI8_1 + j);
                }
                d8[i] = __low2float(yb[2*i].ds);
            }
            const int8_t scs[5] = { (int8_t) (sc[r] & 0xff), 0, 0, 0, (int8_t) ((sc[r] >> 8) & 0xff) };
            acc[r] += vec_dot_q6_K_q8_1_impl_mmvq(vl[r], vh[r], u, scs, __half2float(d[r]), d8);
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// inkling_proj: [MoE epilogue of the previous layer ->] attention-input mix + norm -> Q8_1 -> attention projection matvec

struct ik_fmv_args {
    // the matvec: dst[c(r)*dst_s1 + r] = w[r] . y[c(r)], c(r) = 0 (one column) or the row group (3 columns)
    const char   * w; int64_t w_rb; int n_rows; int n_cols; int n_q, n_k;
    float        * dst; int64_t dst_s1;
    block_q8_1   * y;             // scratch: the activation columns, written by the mix blocks
    float        * part;          // scratch: the pass-1 partial sums of the glue blocks [n_epi][IK_FMV_PART]
    int          * flags;         // one flag word per matvec block (persistent counter area, reset by the last matvec block)
    int            n_embd;
    ik_fmv_sync  * sync;
    int            n_epi;         // epilogue blocks (0: no epilogue)
    // the mix + norm: states p[0 .. n_states), the last one is cur (== layer_out when the epilogue runs)
    const float   * query; int64_t q_nb1;
    ik_state_ptrs   st; int n_states;
    const float   * norm_w; int64_t nw_nb1;
    float eps, scale, norm_eps;
    // the MoE epilogue: layer_out = residual + conv(sum_e w[e]*d[:, e]) with the recurrent state of the FFN channels
    const float   * moe_d; int64_t moe_d_s1; const float * moe_w; int moe_w_s1; int n_used;
    const float   * conv_w; float * state; int64_t state_s1; const int32_t * rs_ids; int write_cell, chan_offset, n_conv; float conv_scale;
    const float   * residual; float * layer_out;
    int dbg; // timing experiments (IK_FMV_DBG): 1 glue blocks only release the flags, 2 no preload, 4 unpaced preload
    unsigned long long * times; // timing experiments (IK_FMV_TIMING): globaltimer at the phase boundaries of glue block 0 / matvec block 0
};

static __device__ __forceinline__ unsigned ik_smid() {
    unsigned id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

static __device__ __forceinline__ unsigned long long ik_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#define IK_FMV_T(k) do { if (a.times && threadIdx.x == 0) { a.times[k] = ik_globaltimer(); } } while (0)

// the glue blocks (128 channels each): the MoE epilogue of the previous layer into layer_out (the weighted expert sum,
// the FFN short convolution, the residual: the math of inkling_moe_sum_conv), then the attention-residual mix: pass 1
// (the Gram matrix of the states and their dots with the query columns) over the block's channels, a barrier of the
// glue blocks, the softmax weights and the RMS factor of the mix (|y|^2 from the Gram matrix: no second reduction),
// pass 2 (the normalized mix scaled by the norm weight) stored as the Q8_1 activation columns.
static __device__ __forceinline__ void ik_fmv_glue(const ik_fmv_args & a, const int e, float * s_red) {
    // s_red: [IK_FMV_WARPS*IK_FMV_PART] reductions / totals, then s_rms, s_p, s_ss
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int c    = e*IK_FMV_BLOCK + tid;
    const bool ok  = c < a.n_embd;
    const int ns   = a.n_states;
    float * s_rms = s_red + IK_FMV_WARPS*IK_FMV_PART;        // [IK_FMV_MAX_COLS][IK_FMV_MAX_STATES]
    float * s_p   = s_rms + IK_FMV_MAX_COLS*IK_FMV_MAX_STATES;
    float * s_ss  = s_p + IK_FMV_MAX_COLS*IK_FMV_MAX_STATES;  // [IK_FMV_MAX_COLS]: the RMS factors

    const bool t0 = e == 0;
    if (t0) { IK_FMV_T(0); }
    // ===== the epilogue: cur =====
    float cur = 0.0f;
    if (a.moe_d) {
        if (ok) {
            const int64_t ch   = a.chan_offset + c;
            const int     nc   = a.n_conv;
            const float * past = a.state + (int64_t) a.rs_ids[0]*a.state_s1 + ch*(nc - 1);
            const float * cw   = a.conv_w + (int64_t) c*nc;
            float av[8], xs[8], wv[8], we[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                av[k] = k < a.n_used ? a.moe_d[c + k*a.moe_d_s1] : 0.0f;
                we[k] = k < a.n_used ? a.moe_w[k*a.moe_w_s1] : 0.0f;
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                xs[j] = j < nc - 1 ? past[j] : 0.0f;
                wv[j] = j < nc ? cw[j] : 0.0f;
            }
            const float res = a.residual[c];
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                sum += av[k]*we[k];
            }
            xs[nc - 1] = sum;
            float out = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < nc) {
                    out += xs[j]*wv[j];
                }
            }
            float * st = a.state + (int64_t) a.write_cell*a.state_s1 + ch*(nc - 1);
#pragma unroll
            for (int j = 0; j < 7; ++j) {
                if (j < nc - 1) {
                    st[j] = xs[j + 1];
                }
            }
            cur = res + (sum + out)*a.conv_scale;
            a.layer_out[c] = cur;
        }
    } else if (ok) {
        cur = a.st.p[ns - 1][c];
    }

    if (t0) { IK_FMV_T(1); }
    // ===== pass 1 of the mix over this block's channels: the Gram matrix of the states (its diagonal: the sums of
    //       squares) and their dot products with the query columns =====
    float x[IK_FMV_MAX_STATES], q[IK_FMV_MAX_COLS], nw[IK_FMV_MAX_COLS];
#pragma unroll
    for (int i = 0; i < IK_FMV_MAX_STATES; ++i) {
        x[i] = ok && i < ns - 1 ? a.st.p[i][c] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < IK_FMV_MAX_STATES; ++i) {
        if (i == ns - 1) {
            x[i] = cur;
        }
    }
#pragma unroll
    for (int k = 0; k < IK_FMV_MAX_COLS; ++k) {
        q[k]  = ok && k < a.n_cols && a.query ? ((const float *) ((const char *) a.query + k*a.q_nb1))[c] : 0.0f;
        nw[k] = ok && k < a.n_cols ? ((const float *) ((const char *) a.norm_w + k*a.nw_nb1))[c] : 0.0f;
    }
    {
        float acc[IK_FMV_PART];
        int n = 0;
#pragma unroll
        for (int i = 0; i < IK_FMV_MAX_STATES; ++i) {
#pragma unroll
            for (int j = i; j < IK_FMV_MAX_STATES; ++j) {
                acc[n++] = warp_reduce_sum(x[i]*x[j]);
            }
        }
#pragma unroll
        for (int k = 0; k < IK_FMV_MAX_COLS; ++k) {
#pragma unroll
            for (int i = 0; i < IK_FMV_MAX_STATES; ++i) {
                acc[IK_FMV_GRAM + IK_FMV_MAX_STATES*k + i] = warp_reduce_sum(x[i]*q[k]);
            }
        }
        // the warp's reduced values (valid in every lane) to shared memory, up to 2 per lane
#pragma unroll
        for (int base = 0; base < IK_FMV_PART; base += WARP_SIZE) {
            if (base + lane < IK_FMV_PART) {
                float v = 0.0f;
#pragma unroll
                for (int i = 0; i < IK_FMV_PART; ++i) {
                    if (i == base + lane) {
                        v = acc[i];
                    }
                }
                s_red[warp*IK_FMV_PART + base + lane] = v;
            }
        }
        __syncthreads();
        if (tid < IK_FMV_PART) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < IK_FMV_WARPS; ++w) {
                v += s_red[w*IK_FMV_PART + tid];
            }
            a.part[e*IK_FMV_PART + tid] = v;
        }
    }

    if (t0) { IK_FMV_T(2); }
    // ===== all glue blocks: the totals; warp k: the mix weights of column k and the RMS factor of the mix =====
    ik_group_barrier(&a.sync->epi_arrivals, a.n_epi);
    if (t0) { IK_FMV_T(3); }
    float * s_tot = s_red; // [IK_FMV_PART] totals
    for (int i = tid; i < IK_FMV_PART; i += IK_FMV_BLOCK) {
        float v = 0.0f;
        for (int blk = 0; blk < a.n_epi; ++blk) {
            v += a.part[blk*IK_FMV_PART + i];
        }
        s_tot[i] = v;
    }
    __syncthreads();
    if (warp < a.n_cols) {
        // lane i: state i
        auto gram = [&](const int i, const int j) -> float {
            const int lo = min(i, j), hi = max(i, j);
            return s_tot[lo*IK_FMV_MAX_STATES - lo*(lo - 1)/2 + (hi - lo)];
        };
        float wgt = 0.0f, r = 1.0f;
        if (a.query) {
            const float ss = lane < ns ? gram(lane, lane) : 0.0f;
            const float dt = lane < ns ? s_tot[IK_FMV_GRAM + IK_FMV_MAX_STATES*warp + lane] : 0.0f;
            r = rsqrtf(ss/a.n_embd + a.eps);
            float       sc = lane < ns ? dt*r*a.scale : -INFINITY;
            const float mx = warp_reduce_max(sc);
            sc = lane < ns ? expf(sc - mx) : 0.0f;
            const float sum = warp_reduce_sum(sc);
            wgt = lane < ns ? r*(sc/sum) : 0.0f;
        } else {
            wgt = lane == 0 ? 1.0f : 0.0f;
        }
        // |y|^2 = sum_ij w_i w_j G_ij
        float ssy = 0.0f;
        for (int j = 0; j < ns; ++j) {
            const float wj = __shfl_sync(0xffffffff, wgt, j); // all lanes take part
            if (lane < ns) {
                ssy += wgt*wj*gram(lane, j);
            }
        }
        ssy = warp_reduce_sum(ssy);
        const float factor = rsqrtf(ssy/a.n_embd + a.norm_eps);
        if (lane < ns) {
            s_rms[warp*IK_FMV_MAX_STATES + lane] = r;              // the mix weight is split as in res_mix_norm: (x*rms)*p
            s_p[warp*IK_FMV_MAX_STATES + lane]   = wgt/(r == 0.0f ? 1.0f : r);
        }
        if (lane == 0) {
            s_ss[warp] = factor;
        }
    }
    __syncthreads();

    if (t0) { IK_FMV_T(4); }
    // ===== pass 2: the mix of each column, normalized and scaled by the norm weight (the operation order of
    //       res_mix_norm), as Q8_1: one block of 32 channels per warp =====
    const int nyb = a.n_embd/QK8_1;
#pragma unroll
    for (int k = 0; k < IK_FMV_MAX_COLS; ++k) {
        if (k < a.n_cols) {
            float y = 0.0f;
#pragma unroll
            for (int i = 0; i < IK_FMV_MAX_STATES; ++i) {
                if (i < ns) {
                    y += (x[i]*s_rms[k*IK_FMV_MAX_STATES + i])*s_p[k*IK_FMV_MAX_STATES + i];
                }
            }
            const float v = s_ss[k]*y*nw[k];
            ik_warp_store_q8_1(v, a.y + (int64_t) k*nyb + c/QK8_1, lane, ok);
        }
    }
    if (t0) { IK_FMV_T(5); }
}

template <ggml_type T, int R, int NIT, int P, int MINB = IK_FMV_MIN_BLOCKS>
static __global__ void __launch_bounds__(IK_FMV_BLOCK, MINB) ik_fmv_proj_kernel(const ik_fmv_args a) {
    typedef ik_qrows<T, R> Q;
    __shared__ float s_red[IK_FMV_WARPS][R];
    __shared__ float s_glue[IK_FMV_WARPS*IK_FMV_PART + 2*IK_FMV_MAX_COLS*IK_FMV_MAX_STATES + IK_FMV_MAX_COLS + 8];
    __shared__ int   s_last;

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int n_glue = a.n_epi;

    if ((int) blockIdx.x < n_glue) {
        // ===== the glue: the previous MoE epilogue, the mix + norm -> Q8_1 activation columns =====
        if (!(a.dbg & 1)) {
            ik_fmv_glue(a, blockIdx.x, s_glue);
        }
        ik_arrive_release(&a.sync->glue_arrivals, a.flags, gridDim.x - n_glue, n_glue, &s_last);
        if (blockIdx.x == 0) { IK_FMV_T(6); if (a.times && tid == 0) { a.times[7] = ik_smid(); } }
        return;
    }
    if (blockIdx.x == (unsigned) n_glue) { IK_FMV_T(8); if (a.times && tid == 0) { a.times[13] = ik_smid(); } }
    // the matvec blocks sharing an SM with a glue block do not preload: their outstanding loads would sit in front of
    // the glue's dependent loads in the SM's memory pipeline (the glue blocks are 0..n_glue-1, dispatched one per SM)
    const bool skip_pre = (a.dbg & 8) && ik_smid() < (unsigned) n_glue;

    // ===== the matvec blocks: row groups of R rows at a static stride; the first P groups are preloaded into registers
    //       before the wait for the activation (the memory system streams weights while the glue runs) =====
    const int b    = blockIdx.x - n_glue;
    const int nb   = gridDim.x - n_glue;
    const int n_rg = (a.n_rows + R - 1)/R;
    const int nbpr = a.n_embd/Q::qk;             // weight quant blocks per row
    const int nyb  = a.n_embd/QK8_1;             // activation blocks per column
    const int kb0  = tid / Q::L;
    const int kqs  = Q::vdr*(tid % Q::L);

    auto col_of = [&](const int r) -> int {
        return a.n_cols == 1 ? 0 : (r < a.n_q ? 0 : (r < a.n_q + a.n_k ? 1 : 2));
    };
    auto group_rows = [&](const int rg, const char ** rows, const block_q8_1 ** y) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = min(R*rg + r, a.n_rows - 1);
            rows[r] = a.w + (int64_t) row*a.w_rb;
            y[r]    = a.y + col_of(row)*nyb;
        }
    };
    // the block's result of row group rg (the warps' partial sums), written to its column of dst
    auto reduce_write = [&](float * acc, const int rg) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = warp_reduce_sum(acc[r]);
            if (lane == 0) {
                s_red[warp][r] = acc[r];
            }
        }
        __syncthreads();
        if (tid < R) {
            const int row = R*rg + tid;
            if (row < a.n_rows) {
                float v = 0.0f;
#pragma unroll
                for (int w = 0; w < IK_FMV_WARPS; ++w) {
                    v += s_red[w][tid];
                }
                const int c = col_of(row);
                for (int cc = 0; cc < a.n_cols; ++cc) {
                    a.dst[cc*a.dst_s1 + row] = cc == c ? v : 0.0f;
                }
            }
        }
        __syncthreads();
    };

    Q pre[P > 0 ? P : 1][NIT];
    int rg = b;
    if (a.dbg & 2) {
        ik_wait_flag(a.flags + b);
    }
    // paced: one iteration's loads per block in flight at a time (about 1.4 MB over the GPU, enough to stream at full
    // rate), waiting for them before issuing the next; a burst of everything would queue microseconds of latency in front
    // of the glue blocks' dependent loads and slow the glue down more than the preload saves
    const bool pace = !(a.dbg & 4);
    if (skip_pre) {
        ik_wait_flag(a.flags + b);
    }
#pragma unroll
    for (int p = 0; p < P; ++p) {
        if (rg + p*nb < n_rg) {
            const char * rows[R];
            const block_q8_1 * y[R];
            group_rows(rg + p*nb, rows, y);
#pragma unroll
            for (int it = 0; it < NIT; ++it) {
                const int kbx = it*Q::BPI + kb0;
                if (kbx < nbpr) {
                    pre[p][it].load(rows, kbx, kqs);
                    if (pace && !skip_pre) {
                        asm volatile("" : "+r"(pre[p][it].first()) :: "memory"); // the next loads issue after this one returned
                    }
                }
            }
        }
    }

    if (b == 0) { IK_FMV_T(9); }
    if (!(a.dbg & 2) && !skip_pre) {
        ik_wait_flag(a.flags + b);
    }
    if (b == 0) { IK_FMV_T(10); }

    // the preloaded groups
#pragma unroll
    for (int p = 0; p < P; ++p) {
        if (rg < n_rg) {
            const char * rows[R];
            const block_q8_1 * y[R];
            group_rows(rg, rows, y);
            float acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r] = 0.0f;
            }
#pragma unroll
            for (int it = 0; it < NIT; ++it) {
                const int kbx = it*Q::BPI + kb0;
                if (kbx < nbpr) {
                    pre[p][it].dot(y, kbx, kqs, acc);
                }
            }
            reduce_write(acc, rg);
            rg += nb;
            if (b == 0 && p == 0) { IK_FMV_T(11); }
        }
    }
    // the rest: load + dot per iteration (a burst of all the group's loads per thread was measured slower), the block's
    // warps kept together by the barrier of the write (they stream adjacent chunks of the same rows)
#pragma unroll 1
    while (rg < n_rg) {
        const char * rows[R];
        const block_q8_1 * y[R];
        group_rows(rg, rows, y);
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] = 0.0f;
        }
#pragma unroll
        for (int it = 0; it < NIT; ++it) {
            const int kbx = it*Q::BPI + kb0;
            if (kbx < nbpr) {
                Q q;
                q.load(rows, kbx, kqs);
                q.dot(y, kbx, kqs, acc);
            }
        }
        reduce_write(acc, rg);
        rg += nb;
    }

    if (b == 0) { IK_FMV_T(12); }
    // the last matvec block resets the synchronization words for the next launch
    if (tid == 0) {
        s_last = ik_atomic_add_acq_rel(&a.sync->mv_arrivals, 1) == nb - 1;
    }
    __syncthreads();
    if (s_last) {
        for (int i = tid; i < nb; i += IK_FMV_BLOCK) {
            a.flags[i] = 0;
        }
        if (tid == 0) {
            a.sync->epi_flag = 0; a.sync->epi_arrivals = 0;
            a.sync->flag = 0;     a.sync->glue_arrivals = 0;
            a.sync->mv_arrivals = 0; a.sync->work = 0;
        }
    }
}

// the co-resident block count of a kernel instantiation (cached)
template <typename K>
static int ik_fmv_max_blocks(K kernel, const int device) {
    static int cached[GGML_CUDA_MAX_DEVICES] = { 0 };
    if (cached[device] == 0) {
        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, IK_FMV_BLOCK, 0));
        GGML_ASSERT(per_sm > 0);
        cached[device] = per_sm*ggml_cuda_info().devices[device].nsm;
    }
    return cached[device];
}

template <ggml_type T, int R, int NIT>
static void ik_fmv_proj_launch(ggml_backend_cuda_context & ctx, ik_fmv_args & a) {
    constexpr int P = 2; // preloaded row groups per block
    const int n_glue  = a.n_epi;
    const int max_blk = ik_fmv_max_blocks(ik_fmv_proj_kernel<T, R, NIT, P>, ctx.device);
    GGML_ASSERT(max_blk > n_glue + 32);
    const int n_rg = (a.n_rows + R - 1)/R;
    static const int blk_override = getenv("IK_FMV_BLOCKS") ? atoi(getenv("IK_FMV_BLOCKS")) : 0;
    const int n_mv = std::min(std::min(blk_override > 0 ? std::min(blk_override, max_blk - n_glue) : max_blk - n_glue, n_rg), IK_N_COUNTERS - 256);
    // tuning experiments: IK_FMV_P (preloaded groups), IK_FMV_MINB (register budget: 65536/(128*MINB))
    static const int p_env = getenv("IK_FMV_P") ? atoi(getenv("IK_FMV_P")) : -1;
    static const int m_env = getenv("IK_FMV_MINB") ? atoi(getenv("IK_FMV_MINB")) : -1;
    if constexpr (T == GGML_TYPE_Q8_0 && R == 2 && NIT == 3) {
        if (p_env >= 0 || m_env >= 0) {
            const int pp = p_env >= 0 ? p_env : P;
            const int mm = m_env >= 0 ? m_env : IK_FMV_MIN_BLOCKS;
#define IK_FMV_LAUNCH(PP, MM) if (pp == PP && mm == MM) { ik_fmv_proj_kernel<T, R, NIT, PP, MM><<<n_glue + n_mv, IK_FMV_BLOCK, 0, ctx.stream()>>>(a); return; }
            IK_FMV_LAUNCH(0, 8) IK_FMV_LAUNCH(1, 8) IK_FMV_LAUNCH(2, 8)
            IK_FMV_LAUNCH(3, 6) IK_FMV_LAUNCH(4, 6) IK_FMV_LAUNCH(3, 5) IK_FMV_LAUNCH(4, 5) IK_FMV_LAUNCH(4, 4) IK_FMV_LAUNCH(6, 4)
#undef IK_FMV_LAUNCH
            GGML_ABORT("inkling_proj: unsupported tuning combination");
        }
    }
    ik_fmv_proj_kernel<T, R, NIT, P><<<n_glue + n_mv, IK_FMV_BLOCK, 0, ctx.stream()>>>(a);
}

template <ggml_type T, int R>
static void ik_fmv_proj_switch(ggml_backend_cuda_context & ctx, ik_fmv_args & a) {
    typedef ik_qrows<T, R> Q;
    const int nit = (a.n_embd/Q::qk + Q::BPI - 1)/Q::BPI;
    switch (nit) {
        case 1: ik_fmv_proj_launch<T, R, 1>(ctx, a); break;
        case 2: ik_fmv_proj_launch<T, R, 2>(ctx, a); break;
        case 3: ik_fmv_proj_launch<T, R, 3>(ctx, a); break;
        case 4: ik_fmv_proj_launch<T, R, 4>(ctx, a); break;
        default: GGML_ABORT("inkling_proj: unsupported n_embd %d for type %d", a.n_embd, (int) T);
    }
}

void ggml_cuda_op_inkling_proj(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int n_states = ggml_get_op_params_i32(dst, 3);
    const ggml_tensor * query  = dst->src[0];
    const ggml_tensor * norm_w = dst->src[1 + n_states];
    const ggml_tensor * w      = dst->src[2 + n_states];
    const ggml_tensor * moe_d  = dst->src[3 + n_states];

    ik_fmv_args a;
    memset(&a, 0, sizeof(a));
    a.eps      = ggml_get_op_params_f32(dst, 0);
    a.scale    = ggml_get_op_params_f32(dst, 1);
    a.norm_eps = ggml_get_op_params_f32(dst, 2);
    a.n_states = n_states;
    a.n_cols   = ggml_get_op_params_i32(dst, 4);
    a.n_q      = ggml_get_op_params_i32(dst, 5);
    a.n_k      = ggml_get_op_params_i32(dst, 6);

    a.n_embd = w->ne[0];
    a.n_rows = w->ne[1];
    a.w      = (const char *) w->data;
    a.w_rb   = w->nb[1];
    a.dst    = (float *) dst->data;
    a.dst_s1 = dst->nb[1]/sizeof(float);
    GGML_ASSERT(dst->ne[0] == a.n_rows && dst->ne[1] == a.n_cols);
    GGML_ASSERT(a.n_embd % QK8_1 == 0 && a.n_embd <= IK_FMV_MAX_EMBD && a.n_cols >= 1 && a.n_cols <= IK_FMV_MAX_COLS);
    GGML_ASSERT(n_states >= 1 && n_states <= IK_FMV_MAX_STATES);

    a.query = query ? (const float *) query->data : nullptr;
    a.q_nb1 = query ? query->nb[1] : 0;
    for (int i = 0; i < n_states; ++i) {
        a.st.p[i]   = (const float *) dst->src[1 + i]->data;
        a.st.nb1[i] = 0;
        GGML_ASSERT(((uintptr_t) a.st.p[i]) % 16 == 0);
    }
    a.norm_w = (const float *) norm_w->data;
    a.nw_nb1 = norm_w->ne[1] == 1 ? 0 : norm_w->nb[1];
    GGML_ASSERT(((uintptr_t) a.query) % 16 == 0 && ((uintptr_t) a.norm_w) % 16 == 0 && a.nw_nb1 % 16 == 0 && a.q_nb1 % 16 == 0);

    if (moe_d) {
        const ggml_tensor * moe_w    = dst->src[4 + n_states];
        const ggml_tensor * conv_w   = dst->src[5 + n_states];
        ggml_tensor       * state    = dst->src[6 + n_states];
        const ggml_tensor * rs_ids   = dst->src[7 + n_states];
        const ggml_tensor * residual = dst->src[8 + n_states];
        ggml_tensor       * layer_out = dst->src[9 + n_states];
        a.moe_d       = (const float *) moe_d->data;
        a.moe_d_s1    = moe_d->nb[1]/sizeof(float);
        a.moe_w       = (const float *) moe_w->data;
        a.moe_w_s1    = moe_w->nb[1]/sizeof(float);
        a.n_used      = moe_d->ne[1];
        a.conv_w      = (const float *) conv_w->data;
        a.state       = (float *) state->data;
        a.state_s1    = state->nb[1]/sizeof(float);
        a.rs_ids      = (const int32_t *) rs_ids->data;
        a.write_cell  = ggml_get_op_params_i32(dst, 7);
        a.chan_offset = ggml_get_op_params_i32(dst, 8);
        a.n_conv      = conv_w->ne[0];
        a.conv_scale  = ggml_get_op_params_f32(dst, 9);
        a.residual    = (const float *) residual->data;
        a.layer_out   = (float *) layer_out->data;
        GGML_ASSERT(a.n_used <= 8 && a.n_conv <= 8 && a.layer_out == a.st.p[n_states - 1]);
    }
    a.n_epi = (a.n_embd + IK_FMV_BLOCK - 1)/IK_FMV_BLOCK; // the epilogue blocks also run the first pass of the mix
    GGML_ASSERT(n_states <= IK_FMV_MAX_STATES);

    static const int dbg = getenv("IK_FMV_DBG") ? atoi(getenv("IK_FMV_DBG")) : 0;
    a.dbg = dbg;
    static unsigned long long * times = [] () -> unsigned long long * {
        if (!getenv("IK_FMV_TIMING")) {
            return nullptr;
        }
        unsigned long long * t = nullptr;
        CUDA_CHECK(cudaMallocManaged(&t, 16*sizeof(unsigned long long)));
        memset(t, 0, 16*sizeof(unsigned long long));
        static unsigned long long * t_static = t;
        atexit([] {
            cudaDeviceSynchronize();
            const unsigned long long * t = t_static;
            const double t0 = (double) t[0];
            fprintf(stderr, "inkling_proj glue block 0 (us from its start): cur %.2f, partials %.2f, barrier %.2f, weights %.2f, pass2 %.2f, released %.2f\n",
                (t[1]-t0)/1e3, (t[2]-t0)/1e3, (t[3]-t0)/1e3, (t[4]-t0)/1e3, (t[5]-t0)/1e3, (t[6]-t0)/1e3);
            fprintf(stderr, "inkling_proj matvec block 0 (us from glue start): start %.2f, preload issued %.2f, flag %.2f, first group %.2f, end %.2f; smid glue0 %llu, mv0 %llu\n",
                ((double) t[8]-t0)/1e3, ((double) t[9]-t0)/1e3, ((double) t[10]-t0)/1e3, ((double) t[11]-t0)/1e3, ((double) t[12]-t0)/1e3, t[7], t[13]);
        });
        return t;
    }();
    a.times = times;
    ggml_cuda_pool_alloc<block_q8_1> y(ctx.pool(), (size_t) a.n_cols*(a.n_embd/QK8_1));
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) a.n_epi*IK_FMV_PART);
    a.y    = y.get();
    a.part = part.get();
    a.sync  = (ik_fmv_sync *) ik_counters(ctx, 16, sizeof(ik_fmv_sync)/sizeof(int));
    a.flags = ik_counters(ctx, 256, IK_N_COUNTERS - 256); // one word per matvec block
    GGML_ASSERT(((uintptr_t) a.y) % 16 == 0);

    switch (w->type) {
        case GGML_TYPE_Q8_0:   ik_fmv_proj_switch<GGML_TYPE_Q8_0, 2>(ctx, a);   break;
        case GGML_TYPE_Q6_K:   ik_fmv_proj_switch<GGML_TYPE_Q6_K, 1>(ctx, a);   break;
        case GGML_TYPE_IQ4_NL: ik_fmv_proj_switch<GGML_TYPE_IQ4_NL, 2>(ctx, a); break;
        default: GGML_ABORT("inkling_proj: unsupported weight type %s", ggml_type_name(w->type));
    }
}
