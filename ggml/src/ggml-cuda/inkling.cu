#include "inkling.cuh"

#include <cmath>

// sum reduction over a block of up to 1024 threads, result valid in every thread
template <int block_size>
static __device__ __forceinline__ float ik_block_sum(float v, float * smem) {
    v = warp_reduce_sum(v);
    if constexpr (block_size > WARP_SIZE) {
        const int lane = threadIdx.x % WARP_SIZE;
        const int wid  = threadIdx.x / WARP_SIZE;
        __syncthreads();
        if (lane == 0) {
            smem[wid] = v;
        }
        __syncthreads();
        v = lane < block_size/WARP_SIZE ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
    }
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------
// l2_clip: y = x*b * min(1, threshold/(|x*b| + eps)), one warp per row

static __global__ void l2_clip_f32(const float * __restrict__ x, const float * __restrict__ b, float * __restrict__ y,
        const int ne00, const int ne01, const int ne02, const int ne03,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int bne0, const int bne1, const int bne2, const int bne3,
        const int64_t bnb1, const int64_t bnb2, const int64_t bnb3,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const float threshold, const float eps) {
    const int64_t row = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int     lane = threadIdx.x % WARP_SIZE;

    const int64_t nrows = (int64_t) ne01*ne02*ne03;
    if (row >= nrows) {
        return;
    }

    const int i3 = row/((int64_t) ne02*ne01);
    const int i2 = (row - (int64_t) i3*ne02*ne01)/ne01;
    const int i1 = row - (int64_t) i3*ne02*ne01 - (int64_t) i2*ne01;

    const float * xr = (const float *) ((const char *) x + i1*nb01 + i2*nb02 + i3*nb03);
    float       * yr = (float *)       ((char *)       y + i1*nb1  + i2*nb2  + i3*nb3);

    const float * br = nullptr;
    if (b) {
        br = (const float *) ((const char *) b + (i1 % bne1)*bnb1 + (i2 % bne2)*bnb2 + (i3 % bne3)*bnb3);
    }

    float ss = 0.0f;
    for (int i0 = lane; i0 < ne00; i0 += WARP_SIZE) {
        const float v = br ? xr[i0]*br[i0 % bne0] : xr[i0];
        ss += v*v;
    }
    ss = warp_reduce_sum(ss);

    const float norm = sqrtf(ss)*1.0f + eps;
    const float factor = fminf(fmaxf(threshold/norm, 0.0f), 1.0f);

    for (int i0 = lane; i0 < ne00; i0 += WARP_SIZE) {
        const float v = br ? xr[i0]*br[i0 % bne0] : xr[i0];
        yr[i0] = v*factor;
    }
}

void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float));

    float threshold, eps;
    memcpy(&threshold, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&eps,       (const float *) dst->op_params + 1, sizeof(float));

    const int64_t nrows = ggml_nrows(src0);
    const int block_size = 256;
    const int64_t nblocks = (nrows*WARP_SIZE + block_size - 1)/block_size;

    l2_clip_f32<<<nblocks, block_size, 0, ctx.stream()>>>(
            (const float *) src0->data, src1 ? (const float *) src1->data : nullptr, (float *) dst->data,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[1], src0->nb[2], src0->nb[3],
            src1 ? (int) src1->ne[0] : 1, src1 ? (int) src1->ne[1] : 1, src1 ? (int) src1->ne[2] : 1, src1 ? (int) src1->ne[3] : 1,
            src1 ? src1->nb[1] : 0, src1 ? src1->nb[2] : 0, src1 ? src1->nb[3] : 0,
            dst->nb[1], dst->nb[2], dst->nb[3], threshold, eps);
}

// ---------------------------------------------------------------------------------------------------------------------
// res_mix: softmax-weighted mix of RMS-normalized states, one block per (token, query)

#define IK_MAX_STATES (GGML_MAX_SRC - 1)

struct ik_state_ptrs {
    const float * p[IK_MAX_STATES];
    int64_t       nb1[IK_MAX_STATES]; // bytes
};

template <int block_size>
static __global__ void res_mix_f32(const float * __restrict__ query, const ik_state_ptrs st, float * __restrict__ dst,
        const int n_embd, const int n_tok, const int n_states, const int64_t q_nb1, const float eps, const float scale) {
    constexpr int n_warps = block_size/WARP_SIZE;

    const int t  = blockIdx.x;
    const int iq = blockIdx.y;

    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;

    __shared__ float red[n_warps][2*IK_MAX_STATES];
    __shared__ float s_rms[IK_MAX_STATES];
    __shared__ float s_p[IK_MAX_STATES];

    const float * q = (const float *) ((const char *) query + iq*q_nb1);

    // partial sums of squares and of the dot products with the query, over the elements of this thread
    float acc[2*IK_MAX_STATES];
#pragma unroll
    for (int i = 0; i < 2*IK_MAX_STATES; ++i) {
        acc[i] = 0.0f;
    }

    for (int k = threadIdx.x; k < n_embd; k += block_size) {
        const float qk = q[k];
#pragma unroll
        for (int i = 0; i < IK_MAX_STATES; ++i) {
            if (i < n_states) {
                const float x = ((const float *) ((const char *) st.p[i] + t*st.nb1[i]))[k];
                acc[2*i + 0] += x*x;
                acc[2*i + 1] += x*qk;
            }
        }
    }

    // one reduction pass for all the partial sums
#pragma unroll
    for (int i = 0; i < 2*IK_MAX_STATES; ++i) {
        if (i < 2*n_states) {
            acc[i] = warp_reduce_sum(acc[i]);
        }
    }
    if (lane < 2*n_states) {
        float v = 0.0f;
#pragma unroll
        for (int i = 0; i < 2*IK_MAX_STATES; ++i) {
            if (i == lane) {
                v = acc[i];
            }
        }
        red[wid][lane] = v;
    }
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        // the scores and the softmax, in the first warp
        float ss = 0.0f, dt = 0.0f;
        if (lane < n_states) {
#pragma unroll
            for (int w = 0; w < n_warps; ++w) {
                ss += red[w][2*lane + 0];
                dt += red[w][2*lane + 1];
            }
        }
        const float r  = rsqrtf(ss/n_embd + eps);
        float       sc = lane < n_states ? dt*r*scale : -INFINITY;
        const float mx = warp_reduce_max(sc);
        sc = lane < n_states ? expf(sc - mx) : 0.0f;
        const float sum = warp_reduce_sum(sc);
        if (lane < n_states) {
            s_rms[lane] = r;
            s_p[lane]   = sc/sum;
        }
    }
    __syncthreads();

    float * y = dst + ((int64_t) iq*n_tok + t)*n_embd;
    for (int k = threadIdx.x; k < n_embd; k += block_size) {
        float acc_y = 0.0f;
#pragma unroll
        for (int i = 0; i < IK_MAX_STATES; ++i) {
            if (i < n_states) {
                const float x = ((const float *) ((const char *) st.p[i] + t*st.nb1[i]))[k];
                acc_y += (x*s_rms[i])*s_p[i];
            }
        }
        y[k] = acc_y;
    }
}

void ggml_cuda_op_res_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * query = dst->src[0];

    int n_states = 0;
    ik_state_ptrs st;
    while (n_states < IK_MAX_STATES && dst->src[1 + n_states]) {
        st.p[n_states]   = (const float *) dst->src[1 + n_states]->data;
        st.nb1[n_states] = dst->src[1 + n_states]->nb[1];
        n_states++;
    }
    GGML_ASSERT(n_states >= 1);

    float eps, scale;
    memcpy(&eps,   (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&scale, (const float *) dst->op_params + 1, sizeof(float));

    const int n_embd = dst->ne[0];
    const int n_tok  = dst->ne[1];
    const int n_q    = dst->ne[2];

    const dim3 grid(n_tok, n_q, 1);
    if (n_embd >= 2048) {
        res_mix_f32<1024><<<grid, 1024, 0, ctx.stream()>>>((const float *) query->data, st, (float *) dst->data,
                n_embd, n_tok, n_states, query->nb[1], eps, scale);
    } else {
        res_mix_f32<256><<<grid, 256, 0, ctx.stream()>>>((const float *) query->data, st, (float *) dst->data,
                n_embd, n_tok, n_states, query->nb[1], eps, scale);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// shortconv_res: y = res + (cur + conv(sx, w) + dynamic taps)*scale, one thread per (channel, token)

template <int d_conv, bool has_taps>
static __global__ void shortconv_res_f32(const float * __restrict__ sx, const float * __restrict__ w,
        const float * __restrict__ cur, const float * __restrict__ taps, const float * __restrict__ res, float * __restrict__ dst,
        const int n_chan, const int n_t, const int n_s,
        const int64_t sx_nb1, const int64_t sx_nb2, const int64_t w_nb1,
        const int64_t cur_nb1, const int64_t taps_nb1, const int64_t taps_nb2, const int64_t res_nb1, const int64_t dst_nb1,
        const int hd, const float scale) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t n_tok = (int64_t) n_t*n_s;
    if (idx >= n_tok*n_chan) {
        return;
    }
    const int c   = idx % n_chan;
    const int tok = idx / n_chan;
    const int s   = tok / n_t;
    const int t   = tok - s*n_t;

    const float * xs = (const float *) ((const char *) sx + c*sx_nb1 + s*sx_nb2) + t;
    const float * wc = (const float *) ((const char *) w + c*w_nb1);

    float x[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        x[j] = xs[j];
    }

    float out = 0.0f;
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        out += x[j]*wc[j];
    }
    if constexpr (has_taps) {
        const float * tp = (const float *) ((const char *) taps + (c/hd)*taps_nb1 + tok*taps_nb2);
#pragma unroll
        for (int lag = 0; lag < d_conv; ++lag) {
            out += x[d_conv - 1 - lag]*tp[lag];
        }
    }

    const float xc = *(const float *) ((const char *) cur + tok*cur_nb1 + c*sizeof(float));
    float y = (xc + out)*scale;
    if (res) {
        y += *(const float *) ((const char *) res + tok*res_nb1 + c*sizeof(float));
    }
    *(float *) ((char *) dst + tok*dst_nb1 + c*sizeof(float)) = y;
}

void ggml_cuda_op_shortconv_res(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sx   = dst->src[0];
    const ggml_tensor * w    = dst->src[1];
    const ggml_tensor * cur  = dst->src[2];
    const ggml_tensor * taps = dst->src[3];
    const ggml_tensor * res  = dst->src[4];

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int d_conv = w->ne[0];
    const int n_chan = w->ne[1];
    const int n_t    = sx->ne[0] - d_conv + 1;
    const int n_s    = sx->ne[2];

    GGML_ASSERT(sx->nb[0] == sizeof(float) && w->nb[0] == sizeof(float));

    const int hd = taps ? n_chan/taps->ne[1] : n_chan;

    const int64_t n = (int64_t) n_chan*n_t*n_s;
    const int block_size = 256;
    const int64_t nblocks = (n + block_size - 1)/block_size;

    auto launch = [&](auto DC) {
        constexpr int kDC = decltype(DC)::value;
        if (taps) {
            shortconv_res_f32<kDC, true><<<nblocks, block_size, 0, ctx.stream()>>>(
                    (const float *) sx->data, (const float *) w->data, (const float *) cur->data, (const float *) taps->data,
                    res ? (const float *) res->data : nullptr, (float *) dst->data, n_chan, n_t, n_s,
                    sx->nb[1], sx->nb[2], w->nb[1], cur->nb[1], taps->nb[1], taps->nb[2], res ? res->nb[1] : 0, dst->nb[1], hd, scale);
        } else {
            shortconv_res_f32<kDC, false><<<nblocks, block_size, 0, ctx.stream()>>>(
                    (const float *) sx->data, (const float *) w->data, (const float *) cur->data, nullptr,
                    res ? (const float *) res->data : nullptr, (float *) dst->data, n_chan, n_t, n_s,
                    sx->nb[1], sx->nb[2], w->nb[1], cur->nb[1], 0, 0, res ? res->nb[1] : 0, dst->nb[1], hd, scale);
        }
    };

    switch (d_conv) {
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 7: launch(std::integral_constant<int, 7>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        default: GGML_ABORT("shortconv_res: unsupported d_conv %d", d_conv);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// rel_bias: out[kv, t, h] = sum_r table[r, pos[kv, t]]*r[r, h, t] - sinks[h, t] + mask[kv, t], one block per (head, token)

#define IK_REL_BIAS_BLOCK 1024
#define IK_REL_MAX_SMEM_EXTENT 12288

template <typename dst_t, typename mask_t, bool use_smem>
static __global__ void rel_bias_kernel(const float * __restrict__ r, const float * __restrict__ table, const float * __restrict__ sinks,
        const int32_t * __restrict__ pos, const mask_t * __restrict__ mask, dst_t * __restrict__ dst,
        const int n_rel, const int n_head, const int n_tok, const int n_kv, const int n_extent, const int64_t mask_ne0) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    extern __shared__ float tbl[];
    __shared__ float rv[64];

    const float * rr = r + ((int64_t) t*n_head + h)*n_rel;
    for (int i = threadIdx.x; i < n_rel; i += IK_REL_BIAS_BLOCK) {
        rv[i] = rr[i];
    }
    __syncthreads();

    if constexpr (use_smem) {
        for (int p = threadIdx.x; p < n_extent; p += IK_REL_BIAS_BLOCK) {
            const float * tb = table + (int64_t) p*n_rel;
            float bias = 0.0f;
            for (int ir = 0; ir < n_rel; ++ir) {
                bias += tb[ir]*rv[ir];
            }
            tbl[p] = bias;
        }
        __syncthreads();
    }

    const float sk = sinks ? sinks[(int64_t) t*n_head + h] : 0.0f;
    const int32_t * pv = pos + (int64_t) t*n_kv;
    const mask_t  * mv = mask ? mask + (int64_t) t*mask_ne0 : nullptr;
    dst_t * out = dst + ((int64_t) h*n_tok + t)*n_kv;

    for (int kv = threadIdx.x; kv < n_kv; kv += IK_REL_BIAS_BLOCK) {
        const int32_t p = pv[kv];
        float bias = 0.0f;
        if (p >= 0 && p < n_extent) {
            if constexpr (use_smem) {
                bias = tbl[p];
            } else {
                const float * tb = table + (int64_t) p*n_rel;
                for (int ir = 0; ir < n_rel; ++ir) {
                    bias += tb[ir]*rv[ir];
                }
            }
        }
        float v = bias - sk;
        if (mv) {
            v += (float) mv[kv];
        }
        out[kv] = (dst_t) v;
    }
}

template <typename dst_t, typename mask_t>
static void rel_bias_launch(const float * r, const float * table, const float * sinks, const int32_t * pos, const mask_t * mask,
        dst_t * dst, const int n_rel, const int n_head, const int n_tok, const int n_kv, const int n_extent, const int64_t mask_ne0,
        cudaStream_t stream) {
    const dim3 grid(n_head, n_tok, 1);
    if (n_extent <= IK_REL_MAX_SMEM_EXTENT) {
        rel_bias_kernel<dst_t, mask_t, true><<<grid, IK_REL_BIAS_BLOCK, n_extent*sizeof(float), stream>>>(
                r, table, sinks, pos, mask, dst, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0);
    } else {
        rel_bias_kernel<dst_t, mask_t, false><<<grid, IK_REL_BIAS_BLOCK, 0, stream>>>(
                r, table, sinks, pos, mask, dst, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0);
    }
}

void ggml_cuda_op_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * r     = dst->src[0];
    const ggml_tensor * table = dst->src[1];
    const ggml_tensor * sinks = dst->src[2];
    const ggml_tensor * pos   = dst->src[3];
    const ggml_tensor * mask  = dst->src[4];

    const int n_rel    = r->ne[0];
    const int n_head   = r->ne[1];
    const int n_tok    = r->ne[2];
    const int n_kv     = pos->ne[0];
    const int n_extent = table->ne[1];

    GGML_ASSERT(n_rel <= 64);
    GGML_ASSERT(ggml_is_contiguous(dst));

    const float   * r_d     = (const float *) r->data;
    const float   * table_d = (const float *) table->data;
    const float   * sinks_d = sinks ? (const float *) sinks->data : nullptr;
    const int32_t * pos_d   = (const int32_t *) pos->data;
    const int64_t   mask_ne0 = mask ? mask->ne[0] : 0;

    cudaStream_t stream = ctx.stream();

    if (dst->type == GGML_TYPE_F16) {
        if (mask && mask->type == GGML_TYPE_F16) {
            rel_bias_launch<half, half>(r_d, table_d, sinks_d, pos_d, (const half *) mask->data, (half *) dst->data, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0, stream);
        } else {
            rel_bias_launch<half, float>(r_d, table_d, sinks_d, pos_d, mask ? (const float *) mask->data : nullptr, (half *) dst->data, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0, stream);
        }
    } else {
        if (mask && mask->type == GGML_TYPE_F16) {
            rel_bias_launch<float, half>(r_d, table_d, sinks_d, pos_d, (const half *) mask->data, (float *) dst->data, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0, stream);
        } else {
            rel_bias_launch<float, float>(r_d, table_d, sinks_d, pos_d, mask ? (const float *) mask->data : nullptr, (float *) dst->data, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0, stream);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// loop_ctrl_down: d[k, t] = sum_i w[i, k]*rmsnorm(a)[i, t] + w[n + i, k]*rmsnorm(b)[i, t], one block per (k, t)

template <int block_size>
static __global__ void loop_ctrl_down_f32(const float * __restrict__ w, const float * __restrict__ a, const float * __restrict__ b,
        float * __restrict__ dst, const int n, const int rank, const int64_t a_nb1, const int64_t b_nb1, const float eps) {
    constexpr int n_warps = block_size/WARP_SIZE;

    const int k = blockIdx.x;
    const int t = blockIdx.y;

    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;

    __shared__ float red[n_warps][4];

    const float * av = (const float *) ((const char *) a + t*a_nb1);
    const float * bv = (const float *) ((const char *) b + t*b_nb1);
    const float * wk = w + (int64_t) k*2*n;

    // the sums of squares and the raw dot products in one pass: w^T rmsnorm(a) = rsqrt(mean(a^2) + eps) * w^T a
    float sa = 0.0f, sb = 0.0f, da = 0.0f, db = 0.0f;
    for (int i = threadIdx.x; i < n; i += block_size) {
        const float x = av[i];
        const float y = bv[i];
        sa += x*x;
        sb += y*y;
        da += wk[i]*x;
        db += wk[n + i]*y;
    }
    sa = warp_reduce_sum(sa);
    sb = warp_reduce_sum(sb);
    da = warp_reduce_sum(da);
    db = warp_reduce_sum(db);
    if (lane == 0) {
        red[wid][0] = sa;
        red[wid][1] = sb;
        red[wid][2] = da;
        red[wid][3] = db;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        sa = sb = da = db = 0.0f;
#pragma unroll
        for (int i = 0; i < n_warps; ++i) {
            sa += red[i][0];
            sb += red[i][1];
            da += red[i][2];
            db += red[i][3];
        }
        const float ra = rsqrtf(sa/n + eps);
        const float rb = rsqrtf(sb/n + eps);
        dst[(int64_t) t*rank + k] = da*ra + db*rb;
    }
}

void ggml_cuda_op_loop_ctrl_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * a = dst->src[1];
    const ggml_tensor * b = dst->src[2];

    float eps;
    memcpy(&eps, (const float *) dst->op_params + 0, sizeof(float));

    const int n     = a->ne[0];
    const int n_tok = a->ne[1];
    const int rank  = w->ne[1];

    const dim3 grid(rank, n_tok, 1);
    loop_ctrl_down_f32<256><<<grid, 256, 0, ctx.stream()>>>((const float *) w->data, (const float *) a->data, (const float *) b->data,
            (float *) dst->data, n, rank, a->nb[1], b->nb[1], eps);
}

// ---------------------------------------------------------------------------------------------------------------------
// loop_ctrl_apply: y = cur - tanh(wf^T d)*cur + tanh(wg^T d)*msg, one warp per output element

static __global__ void loop_ctrl_apply_f32(const float * __restrict__ cur, const float * __restrict__ msg, const float * __restrict__ d,
        const float * __restrict__ wg, const float * __restrict__ wf, float * __restrict__ dst,
        const int n, const int n_tok, const int rank, const int64_t cur_nb1, const int64_t msg_nb1) {
    const int64_t gw   = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int     lane = threadIdx.x % WARP_SIZE;
    if (gw >= (int64_t) n*n_tok) {
        return;
    }
    const int t = gw / n;
    const int i = gw - (int64_t) t*n;

    const float * dv = d + (int64_t) t*rank;
    const float * g  = wg + (int64_t) i*rank;
    const float * f  = wf + (int64_t) i*rank;

    float sg = 0.0f, sf = 0.0f;
    for (int k = lane; k < rank; k += WARP_SIZE) {
        const float dk = dv[k];
        sg += g[k]*dk;
        sf += f[k]*dk;
    }
    sg = warp_reduce_sum(sg);
    sf = warp_reduce_sum(sf);

    if (lane == 0) {
        const float gain   = tanhf(sg);
        const float forget = tanhf(sf);
        const float c = *(const float *) ((const char *) cur + t*cur_nb1 + i*sizeof(float));
        const float m = *(const float *) ((const char *) msg + t*msg_nb1 + i*sizeof(float));
        dst[(int64_t) t*n + i] = (c - forget*c) + gain*m;
    }
}

void ggml_cuda_op_loop_ctrl_apply(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur = dst->src[0];
    const ggml_tensor * msg = dst->src[1];
    const ggml_tensor * d   = dst->src[2];
    const ggml_tensor * wg  = dst->src[3];
    const ggml_tensor * wf  = dst->src[4];

    const int n     = cur->ne[0];
    const int n_tok = cur->ne[1];
    const int rank  = d->ne[0];

    const int block_size = 256;
    const int64_t nblocks = ((int64_t) n*n_tok*WARP_SIZE + block_size - 1)/block_size;

    loop_ctrl_apply_f32<<<nblocks, block_size, 0, ctx.stream()>>>((const float *) cur->data, (const float *) msg->data,
            (const float *) d->data, (const float *) wg->data, (const float *) wf->data, (float *) dst->data,
            n, n_tok, rank, cur->nb[1], msg->nb[1]);
}
