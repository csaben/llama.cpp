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

template <int n_per_lane>
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

    // the row in registers when it fits, else two passes
    float v[n_per_lane > 0 ? n_per_lane : 1];
    float ss = 0.0f;
    if constexpr (n_per_lane > 0) {
#pragma unroll
        for (int j = 0; j < n_per_lane; ++j) {
            const int i0 = lane + j*WARP_SIZE;
            v[j] = i0 < ne00 ? (br ? xr[i0]*br[i0 % bne0] : xr[i0]) : 0.0f;
            ss += v[j]*v[j];
        }
    } else {
        for (int i0 = lane; i0 < ne00; i0 += WARP_SIZE) {
            const float t = br ? xr[i0]*br[i0 % bne0] : xr[i0];
            ss += t*t;
        }
    }
    ss = warp_reduce_sum(ss);

    const float norm = sqrtf(ss)*1.0f + eps;
    const float factor = fminf(fmaxf(threshold/norm, 0.0f), 1.0f);

    if constexpr (n_per_lane > 0) {
#pragma unroll
        for (int j = 0; j < n_per_lane; ++j) {
            const int i0 = lane + j*WARP_SIZE;
            if (i0 < ne00) {
                yr[i0] = v[j]*factor;
            }
        }
    } else {
        for (int i0 = lane; i0 < ne00; i0 += WARP_SIZE) {
            const float t = br ? xr[i0]*br[i0 % bne0] : xr[i0];
            yr[i0] = t*factor;
        }
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

    auto launch = [&](auto NPL) {
        constexpr int kNPL = decltype(NPL)::value;
        l2_clip_f32<kNPL><<<nblocks, block_size, 0, ctx.stream()>>>(
                (const float *) src0->data, src1 ? (const float *) src1->data : nullptr, (float *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[1], src0->nb[2], src0->nb[3],
                src1 ? (int) src1->ne[0] : 1, src1 ? (int) src1->ne[1] : 1, src1 ? (int) src1->ne[2] : 1, src1 ? (int) src1->ne[3] : 1,
                src1 ? src1->nb[1] : 0, src1 ? src1->nb[2] : 0, src1 ? src1->nb[3] : 0,
                dst->nb[1], dst->nb[2], dst->nb[3], threshold, eps);
    };
    const int64_t ne00 = src0->ne[0];
    if (ne00 <= 32) {
        launch(std::integral_constant<int, 1>{});
    } else if (ne00 <= 64) {
        launch(std::integral_constant<int, 2>{});
    } else if (ne00 <= 128) {
        launch(std::integral_constant<int, 4>{});
    } else if (ne00 <= 256) {
        launch(std::integral_constant<int, 8>{});
    } else {
        launch(std::integral_constant<int, 0>{});
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Q8_1 stores of a row held by the lanes of a warp: one value per lane (32 consecutive values = one block), or one float4
// per lane (8 consecutive lanes = one block). All lanes of the warp take part (v = 0 past the row), only `store` lanes write.

static __device__ __forceinline__ void ik_warp_store_q8_1(const float v, block_q8_1 * __restrict__ y, const int lane, const bool store) {
    const float amax = warp_reduce_max(fabsf(v));
    const float d    = amax/127.0f;
    const int   q    = amax == 0.0f ? 0 : (int) roundf(v/d);
    const int   sum  = warp_reduce_sum(q);
    if (store) {
        y->qs[lane] = (int8_t) q;
        if (lane == 0) {
            y->ds = make_half2(d, d*sum);
        }
    }
}

static __device__ __forceinline__ void ik_store_q8_1_vec4(const float4 v, block_q8_1 * __restrict__ y, const int k, const bool store) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
    const float d = amax/127.0f;
    int q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    if (amax != 0.0f) {
        q0 = (int) roundf(v.x/d);
        q1 = (int) roundf(v.y/d);
        q2 = (int) roundf(v.z/d);
        q3 = (int) roundf(v.w/d);
    }
    int sum = q0 + q1 + q2 + q3;
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    if (store) {
        block_q8_1 * b = y + k/8;
        char4 qs;
        qs.x = (int8_t) q0; qs.y = (int8_t) q1; qs.z = (int8_t) q2; qs.w = (int8_t) q3;
        *(char4 *) (b->qs + (k % 8)*4) = qs;
        if (k % 8 == 0) {
            b->ds = make_half2(d, d*sum);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// res_mix: softmax-weighted mix of RMS-normalized states, one block per (token, query); optional RMS norm of the result
// (scaled by nw) and F32 or Q8_1 output. query == nullptr: no mix, the (normalized) first state.

#define IK_MAX_STATES (GGML_MAX_SRC - 1)

struct ik_state_ptrs {
    const float * p[IK_MAX_STATES];
    int64_t       nb1[IK_MAX_STATES]; // bytes
};

struct ik_res_mix_norm {
    const float * nw;     // nullptr: no norm
    int64_t       nw_nb1; // bytes per query (0: shared)
    float         eps;
};

template <int block_size, bool q8>
static __global__ void res_mix_f32(const float * __restrict__ query, const ik_state_ptrs st, void * __restrict__ dst_v,
        const int n_embd, const int n_tok, const int n_states, const int64_t q_nb1, const float eps, const float scale,
        const ik_res_mix_norm norm) {
    constexpr int n_warps = block_size/WARP_SIZE;

    const int t  = blockIdx.x;
    const int iq = blockIdx.y;

    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;

    __shared__ float red[n_warps][2*IK_MAX_STATES];
    __shared__ float s_rms[IK_MAX_STATES];
    __shared__ float s_p[IK_MAX_STATES];
    __shared__ float s_red[n_warps];

    if (query) {
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
    } else if (threadIdx.x == 0) {
        s_rms[0] = 1.0f;
        s_p[0]   = 1.0f;
    }
    __syncthreads();

    auto mixed = [&](const int k) {
        float acc_y = 0.0f;
#pragma unroll
        for (int i = 0; i < IK_MAX_STATES; ++i) {
            if (i < n_states) {
                const float x = ((const float *) ((const char *) st.p[i] + t*st.nb1[i]))[k];
                acc_y += (x*s_rms[i])*s_p[i];
            }
        }
        return acc_y;
    };

    float factor = 1.0f;
    const float * nw = nullptr;
    if (norm.nw) {
        float ss = 0.0f;
        for (int k = threadIdx.x; k < n_embd; k += block_size) {
            const float y = mixed(k);
            ss += y*y;
        }
        ss = ik_block_sum<block_size>(ss, s_red);
        factor = rsqrtf(ss/n_embd + norm.eps);
        nw = (const float *) ((const char *) norm.nw + iq*norm.nw_nb1);
    }

    const int64_t row = (int64_t) iq*n_tok + t;
    if constexpr (q8) {
        block_q8_1 * y = (block_q8_1 *) dst_v + row*(n_embd/QK8_1);
        for (int k0 = wid*WARP_SIZE; k0 < n_embd; k0 += block_size) {
            const int k = k0 + lane;
            float v = 0.0f;
            if (k < n_embd) {
                v = mixed(k)*factor;
                if (nw) {
                    v *= nw[k];
                }
            }
            ik_warp_store_q8_1(v, y + k0/QK8_1, lane, k < n_embd);
        }
    } else {
        float * y = (float *) dst_v + row*n_embd;
        for (int k = threadIdx.x; k < n_embd; k += block_size) {
            float v = mixed(k)*factor;
            if (nw) {
                v *= nw[k];
            }
            y[k] = v;
        }
    }
}

// n_embd % 4 == 0 and n_embd/4 <= block_size: one float4 of every state per thread, a single pass of loads
template <int block_size, bool q8>
static __global__ void res_mix_f32_vec4(const float4 * __restrict__ query, const ik_state_ptrs st, void * __restrict__ dst_v,
        const int n_embd4, const int n_tok, const int n_states, const int64_t q_nb1, const float eps, const float scale,
        const ik_res_mix_norm norm) {
    constexpr int n_warps = block_size/WARP_SIZE;

    const int t  = blockIdx.x;
    const int iq = blockIdx.y;

    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;
    const int k    = threadIdx.x;

    __shared__ float red[n_warps][2*IK_MAX_STATES];
    __shared__ float s_rms[IK_MAX_STATES];
    __shared__ float s_p[IK_MAX_STATES];
    __shared__ float s_red[n_warps];

    float4 x[IK_MAX_STATES];
#pragma unroll
    for (int i = 0; i < IK_MAX_STATES; ++i) {
        x[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (k < n_embd4) {
#pragma unroll
        for (int i = 0; i < IK_MAX_STATES; ++i) {
            if (i < n_states) {
                x[i] = ((const float4 *) ((const char *) st.p[i] + t*st.nb1[i]))[k];
            }
        }
    }

    if (query) {
        const float4 * q = (const float4 *) ((const char *) query + iq*q_nb1);

        float acc[2*IK_MAX_STATES];
#pragma unroll
        for (int i = 0; i < 2*IK_MAX_STATES; ++i) {
            acc[i] = 0.0f;
        }

        if (k < n_embd4) {
            const float4 qk = q[k];
#pragma unroll
            for (int i = 0; i < IK_MAX_STATES; ++i) {
                if (i < n_states) {
                    acc[2*i + 0] = x[i].x*x[i].x + x[i].y*x[i].y + x[i].z*x[i].z + x[i].w*x[i].w;
                    acc[2*i + 1] = x[i].x*qk.x + x[i].y*qk.y + x[i].z*qk.z + x[i].w*qk.w;
                }
            }
        }

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
            float ss = 0.0f, dt = 0.0f;
            if (lane < n_states) {
#pragma unroll
                for (int w = 0; w < n_warps; ++w) {
                    ss += red[w][2*lane + 0];
                    dt += red[w][2*lane + 1];
                }
            }
            const float r  = rsqrtf(ss/(4*n_embd4) + eps);
            float       sc = lane < n_states ? dt*r*scale : -INFINITY;
            const float mx = warp_reduce_max(sc);
            sc = lane < n_states ? expf(sc - mx) : 0.0f;
            const float sum = warp_reduce_sum(sc);
            if (lane < n_states) {
                s_rms[lane] = r;
                s_p[lane]   = sc/sum;
            }
        }
    } else if (threadIdx.x == 0) {
        s_rms[0] = 1.0f;
        s_p[0]   = 1.0f;
    }
    __syncthreads();

    float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (k < n_embd4) {
#pragma unroll
        for (int i = 0; i < IK_MAX_STATES; ++i) {
            if (i < n_states) {
                const float f = s_rms[i]*s_p[i];
                y.x += (x[i].x*s_rms[i])*s_p[i];
                y.y += (x[i].y*s_rms[i])*s_p[i];
                y.z += (x[i].z*s_rms[i])*s_p[i];
                y.w += (x[i].w*s_rms[i])*s_p[i];
                GGML_UNUSED(f);
            }
        }
    }

    if (norm.nw) {
        float ss = y.x*y.x + y.y*y.y + y.z*y.z + y.w*y.w;
        ss = ik_block_sum<block_size>(ss, s_red);
        const float factor = rsqrtf(ss/(4*n_embd4) + norm.eps);
        if (k < n_embd4) {
            const float4 w = ((const float4 *) ((const char *) norm.nw + iq*norm.nw_nb1))[k];
            // the operation order of the rms_norm + mul kernel this replaces
            y.x = factor*y.x*w.x;
            y.y = factor*y.y*w.y;
            y.z = factor*y.z*w.z;
            y.w = factor*y.w*w.w;
        }
    }

    const int64_t row = (int64_t) iq*n_tok + t;
    if constexpr (q8) {
        ik_store_q8_1_vec4(y, (block_q8_1 *) dst_v + row*(4*n_embd4/QK8_1), k, k < n_embd4);
    } else if (k < n_embd4) {
        ((float4 *) dst_v)[row*n_embd4 + k] = y;
    }
}

void ggml_cuda_op_res_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * query = dst->src[0];

    const bool has_norm = ggml_get_op_params_i32(dst, 3) != 0;

    int n_states = 0;
    ik_state_ptrs st;
    while (n_states < IK_MAX_STATES && dst->src[1 + n_states]) {
        st.p[n_states]   = (const float *) dst->src[1 + n_states]->data;
        st.nb1[n_states] = dst->src[1 + n_states]->nb[1];
        n_states++;
    }
    if (has_norm) {
        n_states--;
    }
    GGML_ASSERT(n_states >= 1);
    const ggml_tensor * norm_w = has_norm ? dst->src[1 + n_states] : nullptr;

    float eps, scale, norm_eps;
    memcpy(&eps,      (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&scale,    (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&norm_eps, (const float *) dst->op_params + 2, sizeof(float));

    const int n_embd = dst->ne[0];
    const int n_tok  = dst->ne[1];
    const int n_q    = dst->ne[2];

    const bool q8 = dst->type == GGML_TYPE_Q8_1;
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || q8);
    GGML_ASSERT(!q8 || (n_embd % QK8_1 == 0 && norm_w != nullptr));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ik_res_mix_norm norm;
    norm.nw     = norm_w ? (const float *) norm_w->data : nullptr;
    norm.nw_nb1 = norm_w && ggml_nrows(norm_w) > 1 ? norm_w->nb[1] : 0;
    norm.eps    = norm_eps;

    const dim3 grid(n_tok, n_q, 1);

    const float * q_d    = query ? (const float *) query->data : nullptr;
    const int64_t q_nb1  = query ? query->nb[1] : 0;
    cudaStream_t  stream = ctx.stream();

    bool aligned = n_embd % 4 == 0 && q_nb1 % 16 == 0 && norm.nw_nb1 % 16 == 0 && ((uintptr_t) norm.nw) % 16 == 0;
    for (int i = 0; i < n_states; ++i) {
        aligned = aligned && st.nb1[i] % 16 == 0 && ((uintptr_t) st.p[i]) % 16 == 0;
    }

    auto launch = [&](auto Q8) {
        constexpr bool kQ8 = decltype(Q8)::value;
        if (aligned && n_embd/4 <= 1024) {
            const int n_embd4 = n_embd/4;
            if (n_embd4 <= 256) {
                res_mix_f32_vec4<256, kQ8><<<grid, 256, 0, stream>>>((const float4 *) q_d, st, dst->data, n_embd4, n_tok, n_states, q_nb1, eps, scale, norm);
            } else if (n_embd4 <= 512) {
                res_mix_f32_vec4<512, kQ8><<<grid, 512, 0, stream>>>((const float4 *) q_d, st, dst->data, n_embd4, n_tok, n_states, q_nb1, eps, scale, norm);
            } else if (n_embd4 <= 640) {
                res_mix_f32_vec4<640, kQ8><<<grid, 640, 0, stream>>>((const float4 *) q_d, st, dst->data, n_embd4, n_tok, n_states, q_nb1, eps, scale, norm);
            } else {
                res_mix_f32_vec4<1024, kQ8><<<grid, 1024, 0, stream>>>((const float4 *) q_d, st, dst->data, n_embd4, n_tok, n_states, q_nb1, eps, scale, norm);
            }
        } else if (n_embd >= 2048) {
            res_mix_f32<1024, kQ8><<<grid, 1024, 0, stream>>>(q_d, st, dst->data, n_embd, n_tok, n_states, q_nb1, eps, scale, norm);
        } else {
            res_mix_f32<256, kQ8><<<grid, 256, 0, stream>>>(q_d, st, dst->data, n_embd, n_tok, n_states, q_nb1, eps, scale, norm);
        }
    };
    if (q8) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// shortconv_res: y = res + (cur + conv([past, cur], w) + dynamic taps)*scale, one thread per (channel, token);
// the thread of token n_t - 1 - k also stores the inputs it saw into state slot k

struct ik_shortconv_args {
    const float * past; const float * w; const float * cur; const float * taps; const float * res;
    const int32_t * ids;
    float * dst; float * state;
    int n_chan, n_t, n_s, n_slots;
    int64_t past_nb1, past_nb2, w_nb1, cur_nb1, taps_nb1, taps_nb2, res_nb1, dst_nb1, st_nb1, st_nb2, st_nb3;
    int hd; float scale;
    // post-normalization per head of hd_post channels
    int post; int hd_post; float post_eps; float post_threshold;
};

// the convolution of one (channel, token), the state store included; returns the unnormalized result
template <int d_conv, bool has_taps>
static __device__ __forceinline__ float shortconv_res_one(const ik_shortconv_args & args, const int c, const int tok) {
    const int s = tok / args.n_t;
    const int t = tok - s*args.n_t;
    const int s_src = args.ids ? args.ids[s] : s;

    const float * ps = (const float *) ((const char *) args.past + c*args.past_nb1 + s_src*args.past_nb2);
    const float * wc = (const float *) ((const char *) args.w + c*args.w_nb1);

    // the window [t, t + d_conv) of the inputs [past, cur] of this channel
    float x[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        const int i = t + j;
        x[j] = i < d_conv - 1 ? ps[i] : *(const float *) ((const char *) args.cur + (int64_t) (s*args.n_t + i - (d_conv - 1))*args.cur_nb1 + c*sizeof(float));
    }

    float out = 0.0f;
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        out += x[j]*wc[j];
    }
    if constexpr (has_taps) {
        const float * tp = (const float *) ((const char *) args.taps + (c/args.hd)*args.taps_nb1 + tok*args.taps_nb2);
#pragma unroll
        for (int lag = 0; lag < d_conv; ++lag) {
            out += x[d_conv - 1 - lag]*tp[lag];
        }
    }

    float y = (x[d_conv - 1] + out)*args.scale;
    if (args.res) {
        y += *(const float *) ((const char *) args.res + tok*args.res_nb1 + c*sizeof(float));
    }

    const int k = args.n_t - 1 - t;
    if (k < args.n_slots) {
        float * st = (float *) ((char *) args.state + k*args.st_nb1 + c*args.st_nb2 + s*args.st_nb3);
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            st[j] = x[j + 1];
        }
    }
    return y;
}

// one thread per (channel, token)
template <int d_conv, bool has_taps>
static __global__ void shortconv_res_f32(const ik_shortconv_args args) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t n_tok = (int64_t) args.n_t*args.n_s;
    if (idx >= n_tok*args.n_chan) {
        return;
    }
    const int c   = idx % args.n_chan;
    const int tok = idx / args.n_chan;

    const float y = shortconv_res_one<d_conv, has_taps>(args, c, tok);
    *(float *) ((char *) args.dst + tok*args.dst_nb1 + c*sizeof(float)) = y;
}

// one block per (head, token), the block normalizes the head
template <int d_conv, bool has_taps, int block_size>
static __global__ void shortconv_res_post_f32(const ik_shortconv_args args) {
    const int head = blockIdx.x;
    const int tok  = blockIdx.y;

    __shared__ float red[block_size/WARP_SIZE];

    const int c = head*args.hd_post + threadIdx.x;
    float y = 0.0f;
    if (threadIdx.x < args.hd_post) {
        y = shortconv_res_one<d_conv, has_taps>(args, c, tok);
    }
    const float ss = ik_block_sum<block_size>(y*y, red);

    float factor;
    if (args.post == GGML_SHORTCONV_POST_RMS_NORM) {
        factor = rsqrtf(ss/args.hd_post + args.post_eps);
    } else {
        const float norm = sqrtf(ss)*1.0f + args.post_eps;
        factor = fminf(fmaxf(args.post_threshold/norm, 0.0f), 1.0f);
    }
    if (threadIdx.x < args.hd_post) {
        *(float *) ((char *) args.dst + tok*args.dst_nb1 + c*sizeof(float)) = y*factor;
    }
}

void ggml_cuda_op_shortconv_res(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * past  = dst->src[0];
    const ggml_tensor * w     = dst->src[1];
    const ggml_tensor * cur   = dst->src[2];
    const ggml_tensor * taps  = dst->src[3];
    const ggml_tensor * res   = dst->src[4];
    ggml_tensor       * state = dst->src[5];
    const ggml_tensor * ids   = dst->src[6];

    ik_shortconv_args args;
    memcpy(&args.scale,          (const float *)   dst->op_params + 0, sizeof(float));
    memcpy(&args.post,           (const int32_t *) dst->op_params + 1, sizeof(int32_t));
    int32_t n_heads_post = 1;
    memcpy(&n_heads_post,        (const int32_t *) dst->op_params + 2, sizeof(int32_t));
    memcpy(&args.post_eps,       (const float *)   dst->op_params + 3, sizeof(float));
    memcpy(&args.post_threshold, (const float *)   dst->op_params + 4, sizeof(float));

    const int d_conv = w->ne[0];
    args.n_chan  = w->ne[1];
    args.n_s     = ids ? ids->ne[0] : past->ne[2];
    args.n_t     = cur->ne[1]/args.n_s;
    args.ids     = ids ? (const int32_t *) ids->data : nullptr;
    args.n_slots = state ? state->ne[1] : 0;
    args.hd      = taps ? args.n_chan/taps->ne[1] : args.n_chan;
    args.hd_post = args.post ? args.n_chan/n_heads_post : args.n_chan;

    GGML_ASSERT(past->nb[0] == sizeof(float) && w->nb[0] == sizeof(float));
    GGML_ASSERT(!state || state->nb[0] == sizeof(float));

    args.past = (const float *) past->data; args.past_nb1 = past->nb[1]; args.past_nb2 = past->nb[2];
    args.w    = (const float *) w->data;    args.w_nb1    = w->nb[1];
    args.cur  = (const float *) cur->data;  args.cur_nb1  = cur->nb[1];
    args.taps = taps ? (const float *) taps->data : nullptr; args.taps_nb1 = taps ? taps->nb[1] : 0; args.taps_nb2 = taps ? taps->nb[2] : 0;
    args.res  = res ? (const float *) res->data : nullptr;   args.res_nb1  = res ? res->nb[1] : 0;
    args.dst  = (float *) dst->data; args.dst_nb1 = dst->nb[1];
    args.state = state ? (float *) state->data : nullptr;
    args.st_nb1 = state ? state->nb[1] : 0; args.st_nb2 = state ? state->nb[2] : 0; args.st_nb3 = state ? state->nb[3] : 0;

    const int64_t n_tok = (int64_t) args.n_t*args.n_s;
    cudaStream_t stream = ctx.stream();

    auto launch = [&](auto DC, auto TAPS) {
        constexpr int  kDC   = decltype(DC)::value;
        constexpr bool kTaps = decltype(TAPS)::value;
        if (args.post) {
            const dim3 grid(args.n_chan/args.hd_post, n_tok, 1);
            if (args.hd_post <= 32) {
                shortconv_res_post_f32<kDC, kTaps, 32><<<grid, 32, 0, stream>>>(args);
            } else if (args.hd_post <= 64) {
                shortconv_res_post_f32<kDC, kTaps, 64><<<grid, 64, 0, stream>>>(args);
            } else if (args.hd_post <= 128) {
                shortconv_res_post_f32<kDC, kTaps, 128><<<grid, 128, 0, stream>>>(args);
            } else if (args.hd_post <= 256) {
                shortconv_res_post_f32<kDC, kTaps, 256><<<grid, 256, 0, stream>>>(args);
            } else if (args.hd_post <= 512) {
                shortconv_res_post_f32<kDC, kTaps, 512><<<grid, 512, 0, stream>>>(args);
            } else {
                GGML_ASSERT(args.hd_post <= 1024);
                shortconv_res_post_f32<kDC, kTaps, 1024><<<grid, 1024, 0, stream>>>(args);
            }
        } else {
            const int block_size = 256;
            const int64_t nblocks = (n_tok*args.n_chan + block_size - 1)/block_size;
            shortconv_res_f32<kDC, kTaps><<<nblocks, block_size, 0, stream>>>(args);
        }
    };

    auto launch_dc = [&](auto DC) {
        if (taps) {
            launch(DC, std::true_type{});
        } else {
            launch(DC, std::false_type{});
        }
    };

    switch (d_conv) {
        case 2: launch_dc(std::integral_constant<int, 2>{}); break;
        case 3: launch_dc(std::integral_constant<int, 3>{}); break;
        case 4: launch_dc(std::integral_constant<int, 4>{}); break;
        case 5: launch_dc(std::integral_constant<int, 5>{}); break;
        case 6: launch_dc(std::integral_constant<int, 6>{}); break;
        case 7: launch_dc(std::integral_constant<int, 7>{}); break;
        case 8: launch_dc(std::integral_constant<int, 8>{}); break;
        default: GGML_ABORT("shortconv_res: unsupported d_conv %d", d_conv);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// rel_bias: out[kv, t, h] = sum_r table[r, pos[kv, t]]*r[r, h, t] - sinks[h, t] + mask[kv, t], one block per (head, token)

#define IK_REL_BIAS_BLOCK 256
#define IK_REL_MAX_SMEM_EXTENT 12288

template <typename dst_t, typename mask_t, bool use_smem>
static __global__ void rel_bias_kernel(const float * __restrict__ r, const float * __restrict__ table, const float * __restrict__ sinks,
        const int32_t * __restrict__ pos, const mask_t * __restrict__ mask, dst_t * __restrict__ dst,
        const int n_rel, const int n_head, const int n_tok, const int n_kv, const int n_extent, const int64_t mask_ne0,
        const int64_t sinks_nb1) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    extern __shared__ float tbl[];

    // r of this head in registers (n_rel <= 16 on the fast path, else shared memory)
    const float * rr = r + ((int64_t) t*n_head + h)*n_rel;
    float rv[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        rv[i] = i < n_rel ? rr[i] : 0.0f;
    }

    if constexpr (use_smem) {
        if (n_rel == 16 && ((uintptr_t) table) % 16 == 0) {
            // a table entry is 64 bytes: four float4 loads per thread
            for (int p = threadIdx.x; p < n_extent; p += IK_REL_BIAS_BLOCK) {
                const float4 * tb = (const float4 *) (table + (int64_t) p*16);
                const float4 t0 = tb[0], t1 = tb[1], t2 = tb[2], t3 = tb[3];
                float bias = 0.0f;
                bias += t0.x*rv[0]  + t0.y*rv[1]  + t0.z*rv[2]  + t0.w*rv[3];
                bias += t1.x*rv[4]  + t1.y*rv[5]  + t1.z*rv[6]  + t1.w*rv[7];
                bias += t2.x*rv[8]  + t2.y*rv[9]  + t2.z*rv[10] + t2.w*rv[11];
                bias += t3.x*rv[12] + t3.y*rv[13] + t3.z*rv[14] + t3.w*rv[15];
                tbl[p] = bias;
            }
        } else {
            for (int p = threadIdx.x; p < n_extent; p += IK_REL_BIAS_BLOCK) {
                const float * tb = table + (int64_t) p*n_rel;
                float bias = 0.0f;
#pragma unroll
                for (int ir = 0; ir < 16; ++ir) {
                    if (ir < n_rel) {
                        bias += tb[ir]*rv[ir];
                    }
                }
                tbl[p] = bias;
            }
        }
        __syncthreads();
    }

    const float sk = sinks ? *(const float *) ((const char *) sinks + t*sinks_nb1 + h*sizeof(float)) : 0.0f;
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
#pragma unroll
                for (int ir = 0; ir < 16; ++ir) {
                    if (ir < n_rel) {
                        bias += tb[ir]*rv[ir];
                    }
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
        const int64_t sinks_nb1, cudaStream_t stream) {
    const dim3 grid(n_head, n_tok, 1);
    if (n_extent <= IK_REL_MAX_SMEM_EXTENT) {
        rel_bias_kernel<dst_t, mask_t, true><<<grid, IK_REL_BIAS_BLOCK, n_extent*sizeof(float), stream>>>(
                r, table, sinks, pos, mask, dst, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0, sinks_nb1);
    } else {
        rel_bias_kernel<dst_t, mask_t, false><<<grid, IK_REL_BIAS_BLOCK, 0, stream>>>(
                r, table, sinks, pos, mask, dst, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0, sinks_nb1);
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

    GGML_ASSERT(n_rel <= 16);
    GGML_ASSERT(ggml_is_contiguous(dst));

    const float   * r_d     = (const float *) r->data;
    const float   * table_d = (const float *) table->data;
    const float   * sinks_d = sinks ? (const float *) sinks->data : nullptr;
    const int32_t * pos_d   = (const int32_t *) pos->data;
    const int64_t   mask_ne0 = mask ? mask->ne[0] : 0;
    const int64_t   sinks_nb1 = sinks ? sinks->nb[1] : 0;

    cudaStream_t stream = ctx.stream();

    if (dst->type == GGML_TYPE_F16) {
        if (mask && mask->type == GGML_TYPE_F16) {
            rel_bias_launch<half, half>(r_d, table_d, sinks_d, pos_d, (const half *) mask->data, (half *) dst->data, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0, sinks_nb1, stream);
        } else {
            rel_bias_launch<half, float>(r_d, table_d, sinks_d, pos_d, mask ? (const float *) mask->data : nullptr, (half *) dst->data, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0, sinks_nb1, stream);
        }
    } else {
        if (mask && mask->type == GGML_TYPE_F16) {
            rel_bias_launch<float, half>(r_d, table_d, sinks_d, pos_d, (const half *) mask->data, (float *) dst->data, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0, sinks_nb1, stream);
        } else {
            rel_bias_launch<float, float>(r_d, table_d, sinks_d, pos_d, mask ? (const float *) mask->data : nullptr, (float *) dst->data, n_rel, n_head, n_tok, n_kv, n_extent, mask_ne0, sinks_nb1, stream);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// loop_ctrl_down: d[k, t, b] = sum_i w[i, k, b]*rmsnorm(a)[i, t, b] + w[n + i, k, b]*rmsnorm(msg)[i, t, b], one block per (k, t, b)

template <int block_size>
static __global__ void loop_ctrl_down_f32(const float * __restrict__ w, const float * __restrict__ a, const float * __restrict__ b,
        float * __restrict__ dst, const int n, const int rank, const int64_t a_nb1, const int64_t a_nb2, const int64_t b_nb1, const int64_t b_nb2,
        const float eps) {
    constexpr int n_warps = block_size/WARP_SIZE;

    const int k  = blockIdx.x;
    const int t  = blockIdx.y;
    const int ib = blockIdx.z;

    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;

    __shared__ float red[n_warps][4];

    const float * av = (const float *) ((const char *) a + t*a_nb1 + ib*a_nb2);
    const float * bv = (const float *) ((const char *) b + t*b_nb1 + ib*b_nb2);
    const float * wk = w + ((int64_t) ib*rank + k)*2*n;

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
        dst[((int64_t) ib*gridDim.y + t)*rank + k] = da*ra + db*rb;
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
    const int B     = w->ne[2];

    const dim3 grid(rank, n_tok, B);
    loop_ctrl_down_f32<256><<<grid, 256, 0, ctx.stream()>>>((const float *) w->data, (const float *) a->data, (const float *) b->data,
            (float *) dst->data, n, rank, a->nb[1], a->ne[2] == 1 ? 0 : a->nb[2], b->nb[1], b->nb[2], eps);
}

// ---------------------------------------------------------------------------------------------------------------------
// loop_ctrl_apply: y = cur - tanh(wf^T d)*cur + tanh(wg^T d)*msg, one warp per output element

static __global__ void loop_ctrl_apply_f32(const float * __restrict__ cur, const float * __restrict__ msg, const float * __restrict__ d,
        const float * __restrict__ wg, const float * __restrict__ wf, float * __restrict__ dst,
        const int n, const int n_tok, const int rank, const int64_t cur_nb1, const int64_t cur_nb2, const int64_t msg_nb1, const int64_t msg_nb2) {
    const int64_t gw   = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int     lane = threadIdx.x % WARP_SIZE;
    const int64_t n_out = (int64_t) n*n_tok;
    if (gw >= n_out*gridDim.y) {
        return;
    }
    const int ib = blockIdx.y;
    const int t  = gw / n;
    const int i  = gw - (int64_t) t*n;
    if (t >= n_tok) {
        return;
    }

    const float * dv = d + ((int64_t) ib*n_tok + t)*rank;
    const float * g  = wg + ((int64_t) ib*n + i)*rank;
    const float * f  = wf + ((int64_t) ib*n + i)*rank;

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
        const float c = *(const float *) ((const char *) cur + t*cur_nb1 + ib*cur_nb2 + i*sizeof(float));
        const float m = *(const float *) ((const char *) msg + t*msg_nb1 + ib*msg_nb2 + i*sizeof(float));
        dst[((int64_t) ib*n_tok + t)*n + i] = (c - forget*c) + gain*m;
    }
}

// with the norm: a second launch RMS-normalizes the rows and scales them by the norm weight, in place (F32) or into
// the Q8_1 output (from the F32 scratch)
template <int block_size, bool q8>
static __global__ void rms_norm_mul_inplace_f32(float * __restrict__ y, const float * __restrict__ nw, void * __restrict__ dst_v,
        const int n, const int n_tok, const int64_t nw_nb1, const float eps) {
    const int t  = blockIdx.x;
    const int ib = blockIdx.y;
    __shared__ float red[block_size/WARP_SIZE];

    const int64_t row = (int64_t) ib*n_tok + t;
    float       * yt  = y + row*n;
    const float * nwb = (const float *) ((const char *) nw + ib*nw_nb1);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += block_size) {
        ss += yt[i]*yt[i];
    }
    ss = ik_block_sum<block_size>(ss, red);
    const float scale = rsqrtf(ss/n + eps);
    if constexpr (q8) {
        const int lane = threadIdx.x % WARP_SIZE;
        const int wid  = threadIdx.x / WARP_SIZE;
        block_q8_1 * out = (block_q8_1 *) dst_v + row*(n/QK8_1);
        for (int i0 = wid*WARP_SIZE; i0 < n; i0 += block_size) {
            const int i = i0 + lane;
            const float v = i < n ? yt[i]*scale*nwb[i] : 0.0f;
            ik_warp_store_q8_1(v, out + i0/QK8_1, lane, i < n);
        }
    } else {
        for (int i = threadIdx.x; i < n; i += block_size) {
            yt[i] = yt[i]*scale*nwb[i];
        }
    }
}

void ggml_cuda_op_loop_ctrl_apply(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur = dst->src[0];
    const ggml_tensor * msg = dst->src[1];
    const ggml_tensor * d   = dst->src[2];
    const ggml_tensor * wg  = dst->src[3];
    const ggml_tensor * wf  = dst->src[4];
    const ggml_tensor * nw  = dst->src[5];

    const int n     = cur->ne[0];
    const int n_tok = cur->ne[1];
    const int rank  = d->ne[0];
    const int B     = msg->ne[2];

    const bool q8 = dst->type == GGML_TYPE_Q8_1;
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || q8);
    GGML_ASSERT(!q8 || (nw != nullptr && n % QK8_1 == 0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    // the F32 result: the output itself, or a scratch when the output is quantized
    ggml_cuda_pool_alloc<float> scratch(ctx.pool());
    float * y = (float *) dst->data;
    if (q8) {
        scratch.alloc((size_t) n*n_tok*B);
        y = scratch.get();
    }

    const int block_size = 256;
    const int64_t nblocks = ((int64_t) n*n_tok*WARP_SIZE + block_size - 1)/block_size;
    const dim3 grid(nblocks, B, 1);

    loop_ctrl_apply_f32<<<grid, block_size, 0, ctx.stream()>>>((const float *) cur->data, (const float *) msg->data,
            (const float *) d->data, (const float *) wg->data, (const float *) wf->data, y,
            n, n_tok, rank, cur->nb[1], cur->ne[2] == 1 ? 0 : cur->nb[2], msg->nb[1], msg->nb[2]);

    if (nw) {
        float eps;
        memcpy(&eps, (const float *) dst->op_params + 0, sizeof(float));
        const dim3 grid_norm(n_tok, B, 1);
        const int64_t nw_nb1 = ggml_nrows(nw) == 1 ? 0 : nw->nb[1];
        if (q8) {
            rms_norm_mul_inplace_f32<1024, true><<<grid_norm, 1024, 0, ctx.stream()>>>(y, (const float *) nw->data, dst->data, n, n_tok, nw_nb1, eps);
        } else {
            rms_norm_mul_inplace_f32<1024, false><<<grid_norm, 1024, 0, ctx.stream()>>>(y, (const float *) nw->data, dst->data, n, n_tok, nw_nb1, eps);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// moe_sum: y[i, t] = sum_k a[i, k, t]*w[0, k, t], one thread per output

static __global__ void moe_sum_f32(const float * __restrict__ a, const float * __restrict__ w, float * __restrict__ dst,
        const int n0, const int n1, const int n2, const int64_t a_nb1, const int64_t a_nb2, const int64_t w_nb1, const int64_t w_nb2) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= (int64_t) n0*n2) {
        return;
    }
    const int i = idx % n0;
    const int t = idx / n0;

    float sum = 0.0f;
    for (int k = 0; k < n1; ++k) {
        const float x  = *(const float *) ((const char *) a + i*sizeof(float) + k*a_nb1 + t*a_nb2);
        const float wk = *(const float *) ((const char *) w + k*w_nb1 + t*w_nb2);
        sum += x*wk;
    }
    dst[(int64_t) t*n0 + i] = sum;
}

void ggml_cuda_op_moe_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * w = dst->src[1];

    const int n0 = a->ne[0];
    const int n1 = a->ne[1];
    const int n2 = a->ne[2];

    const int block_size = 256;
    const int64_t nblocks = ((int64_t) n0*n2 + block_size - 1)/block_size;
    moe_sum_f32<<<nblocks, block_size, 0, ctx.stream()>>>((const float *) a->data, (const float *) w->data, (float *) dst->data,
            n0, n1, n2, a->nb[1], a->nb[2], w->nb[1], w->nb[2]);
}

// ---------------------------------------------------------------------------------------------------------------------
// inkling_attn: the single-token attention block, one block of 256 threads per (kv head, kv split): the K/V convolutions,
// normalizations and cache stores, the Q/R clips, the relative bias table of the head group, the attention with a zero
// sink logit over the cells of the split, the combination of the splits (last block) and the sigmoid gate

#define IK_ATTN_HD         128
#define IK_ATTN_GQA_MAX    4
#define IK_ATTN_MAX_EXTENT 1024
#define IK_ATTN_BLOCK      256
#define IK_ATTN_WARPS      (IK_ATTN_BLOCK/WARP_SIZE)

struct ik_attn_args {
    const float   * proj;    int64_t proj_s1; int proj_ncols; // proj_s1: floats per column
    const float   * consts;
    float         * state;   int64_t state_s1;                // floats per cell
    const int32_t * rs_ids;  int write_cell;
    char          * k_cache; int64_t kc_nb1, kc_nb2;
    char          * v_cache; int64_t vc_nb1, vc_nb2;
    const int64_t * kv_idxs;
    const int32_t * rel_pos;
    const void    * mask;    int mask_f16;
    const int32_t * pos;
    void          * dst;
    int n_head, n_head_kv, n_rel, n_conv, has_taps, n_kv, n_extent, n_split, kv_per_split;
    float qk_clip, rel_clip, norm_eps, tau_alpha, tau_floor, kq_scale, clip_eps;
    int64_t off_rel, off_sink, off_gate, off_k, off_ktaps, off_v, off_vtaps;
    int64_t c_relclip, c_kconv, c_vconv, c_table;
    float * part;  // [n_head_kv, n_split, gqa, HD + 2]: the partial sums, max and denominator of each split
    int   * counters;
};

template <typename cache_t>
static __device__ __forceinline__ void ik_attn_load4(const char * __restrict__ row, const int lane, float * v) {
    if constexpr (std::is_same<cache_t, half>::value) {
        const half2 * hp = (const half2 *) row + 2*lane;
        const float2 a = __half22float2(hp[0]);
        const float2 b = __half22float2(hp[1]);
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const block_q8_0 * b = (const block_q8_0 *) row + lane/8;
        int8_t qs[4];
        ggml_cuda_memcpy_1<4, 2>(qs, b->qs + 4*(lane % 8));
        const float d = __half2float(b->d);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            v[e] = d*qs[e];
        }
    }
}

template <typename cache_t, bool q8_out>
static __global__ void __launch_bounds__(IK_ATTN_BLOCK) inkling_attn_kernel(const ik_attn_args args) {
    constexpr int HD = IK_ATTN_HD;

    const int g     = blockIdx.x;
    const int split = blockIdx.y;
    const int tid   = threadIdx.x;
    const int lane  = tid % WARP_SIZE;
    const int warp  = tid / WARP_SIZE;
    const int gqa   = args.n_head/args.n_head_kv;

    __shared__ float s_q[IK_ATTN_GQA_MAX][HD];
    __shared__ float s_k[HD];
    __shared__ float s_v[HD];
    __shared__ float s_r[IK_ATTN_GQA_MAX][32];
    __shared__ float s_sink[IK_ATTN_GQA_MAX];
    __shared__ float s_tbl[IK_ATTN_GQA_MAX*IK_ATTN_MAX_EXTENT];
    __shared__ float s_red[IK_ATTN_WARPS];
    __shared__ float s_m[IK_ATTN_GQA_MAX][IK_ATTN_WARPS];
    __shared__ float s_l[IK_ATTN_GQA_MAX][IK_ATTN_WARPS];
    __shared__ float s_acc[IK_ATTN_GQA_MAX][IK_ATTN_WARPS][HD];
    __shared__ int   s_last;

    const int     read_cell = args.rs_ids[0];
    const int64_t kv_idx    = args.kv_idxs[0];

    float tau = 1.0f;
    if (args.pos) {
        const float t = args.pos[0] + 1.0f;
        tau = 1.0f + args.tau_alpha*logf(fmaxf(1.0f, t/args.tau_floor));
    }

    const float * pr0 = args.proj; // column 0: Q, R, sink, gate

    // --- the K (threads 0..127) and V (threads 128..255) channels of kv head g: convolution, state, post-normalization
    {
        const int     which = tid/HD;
        const int     i     = tid % HD;
        const int64_t ch    = (int64_t) which*HD*args.n_head_kv + (int64_t) g*HD + i; // state channel
        const int64_t off   = which ? args.off_v : args.off_k;
        const int64_t offt  = which ? args.off_vtaps : args.off_ktaps;
        const int     col   = args.proj_ncols == 3 ? (which ? 2 : 1) : 0;
        const float * pr    = args.proj + col*args.proj_s1;
        const float * w     = args.consts + (which ? args.c_vconv : args.c_kconv) + ch*args.n_conv;
        const int     n_conv = args.n_conv;
        const float   scale  = 1.0f/sqrtf((args.has_taps ? 2 : 1)*n_conv + 1);

        float x[8];
        const float * past = args.state + (int64_t) read_cell*args.state_s1 + ch*(n_conv - 1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < n_conv - 1) {
                x[j] = past[j];
            } else if (j == n_conv - 1) {
                x[j] = pr[off + (int64_t) g*HD + i];
            } else {
                x[j] = 0.0f;
            }
        }
        float out = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < n_conv) {
                out += x[j]*w[j];
            }
        }
        if (args.has_taps) {
#pragma unroll
            for (int lag = 0; lag < 8; ++lag) {
                if (lag < n_conv) {
                    out += x[n_conv - 1 - lag]*pr[offt + (int64_t) g*n_conv + lag];
                }
            }
        }
        float y = (x[n_conv - 1] + out)*scale;

        if (split == 0) {
            float * st = args.state + (int64_t) args.write_cell*args.state_s1 + ch*(n_conv - 1);
#pragma unroll
            for (int j = 0; j < 7; ++j) {
                if (j < n_conv - 1) {
                    st[j] = x[j + 1];
                }
            }
        }

        // the sums of squares of the K half (warps 0-3) and the V half (warps 4-7)
        float ss = warp_reduce_sum(y*y);
        if (lane == 0) {
            s_red[warp] = ss;
        }
        __syncthreads();
        ss = s_red[4*which + 0] + s_red[4*which + 1] + s_red[4*which + 2] + s_red[4*which + 3];
        float factor;
        if (which == 0) {
            const float norm = sqrtf(ss)*1.0f + args.clip_eps;
            factor = fminf(fmaxf(args.qk_clip/norm, 0.0f), 1.0f);
        } else {
            factor = rsqrtf(ss/HD + args.norm_eps);
        }
        y *= factor;

        // rounded through the cache type; split 0 stores the row
        char * row = (which ? args.v_cache : args.k_cache) + kv_idx*(which ? args.vc_nb2 : args.kc_nb2) + g*(which ? args.vc_nb1 : args.kc_nb1);
        float yq;
        if constexpr (std::is_same<cache_t, half>::value) {
            const half h = __float2half(y);
            if (split == 0) {
                ((half *) row)[i] = h;
            }
            yq = __half2float(h);
        } else {
            // the 32 channels of the warp form one block
            const float amax = warp_reduce_max(fabsf(y));
            const float d    = amax/127.0f;
            const float id   = d != 0.0f ? 1.0f/d : 0.0f;
            const int   q    = (int) roundf(y*id);
            const half  dh   = __float2half(d);
            if (split == 0) {
                block_q8_0 * b = (block_q8_0 *) row + (warp % 4);
                b->qs[lane] = (int8_t) q;
                if (lane == 0) {
                    b->d = dh;
                }
            }
            yq = __half2float(dh)*q;
        }
        (which ? s_v : s_k)[i] = yq;
    }
    __syncthreads();

    // --- the clipped Q and R of the heads of the group, the sinks
    for (int hh = 0; hh < gqa; ++hh) {
        const int h = g*gqa + hh;
        float qv = 0.0f, rv = 0.0f, ss = 0.0f;
        if (tid < HD) {
            qv = pr0[(int64_t) h*HD + tid]*args.consts[(int64_t) h*HD + tid];
            ss = qv*qv;
        } else if (warp == 4 && lane < args.n_rel) {
            rv = pr0[args.off_rel + (int64_t) h*args.n_rel + lane]*args.consts[args.c_relclip + (int64_t) h*args.n_rel + lane];
            ss = rv*rv;
        }
        ss = warp_reduce_sum(ss);
        if (lane == 0) {
            s_red[warp] = ss;
        }
        __syncthreads();
        if (tid < HD) {
            const float ssq    = s_red[0] + s_red[1] + s_red[2] + s_red[3];
            const float norm   = sqrtf(ssq)*1.0f + args.clip_eps;
            const float factor = fminf(fmaxf(args.qk_clip/norm, 0.0f), 1.0f);
            s_q[hh][tid] = qv*factor*tau;
        } else if (warp == 4) {
            const float norm   = sqrtf(s_red[4])*1.0f + args.clip_eps;
            const float factor = fminf(fmaxf(args.rel_clip/norm, 0.0f), 1.0f);
            if (lane < 32) {
                s_r[hh][lane] = lane < args.n_rel ? rv*factor*tau : 0.0f;
            }
            if (lane == 0) {
                s_sink[hh] = pr0[args.off_sink + h];
            }
        }
        __syncthreads();
    }

    // --- the relative bias of every distance, per head
    for (int idx = tid; idx < gqa*args.n_extent; idx += IK_ATTN_BLOCK) {
        const int hh = idx/args.n_extent;
        const int d  = idx - hh*args.n_extent;
        const float * tb = args.consts + args.c_table + (int64_t) d*args.n_rel;
        float b = 0.0f;
        for (int j = 0; j < args.n_rel; ++j) {
            b += tb[j]*s_r[hh][j];
        }
        s_tbl[hh*args.n_extent + d] = b;
    }
    __syncthreads();

    // --- the attention over the cells of the split: warp w takes the cells w, w + 8, ...; a lane holds 4 dims
    float qreg[IK_ATTN_GQA_MAX][4];
    float acc[IK_ATTN_GQA_MAX][4];
    float m[IK_ATTN_GQA_MAX], l[IK_ATTN_GQA_MAX];
#pragma unroll
    for (int hh = 0; hh < IK_ATTN_GQA_MAX; ++hh) {
        m[hh] = -INFINITY;
        l[hh] = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            qreg[hh][e] = hh < gqa ? s_q[hh][4*lane + e] : 0.0f;
            acc[hh][e]  = 0.0f;
        }
    }

    const int kv0 = split*args.kv_per_split;
    const int kv1 = min(args.n_kv, kv0 + args.kv_per_split);
    for (int kv = kv0 + warp; kv < kv1; kv += IK_ATTN_WARPS) {
        const float mv = args.mask_f16 ? __half2float(((const half *) args.mask)[kv]) : ((const float *) args.mask)[kv];
        if (mv == -INFINITY) {
            continue;
        }
        float kvals[4], vvals[4];
        if (kv == kv_idx) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                kvals[e] = s_k[4*lane + e];
                vvals[e] = s_v[4*lane + e];
            }
        } else {
            ik_attn_load4<cache_t>(args.k_cache + kv*args.kc_nb2 + g*args.kc_nb1, lane, kvals);
            ik_attn_load4<cache_t>(args.v_cache + kv*args.vc_nb2 + g*args.vc_nb1, lane, vvals);
        }
        const int d = args.rel_pos[kv];
#pragma unroll
        for (int hh = 0; hh < IK_ATTN_GQA_MAX; ++hh) {
            if (hh < gqa) {
                float dot = qreg[hh][0]*kvals[0] + qreg[hh][1]*kvals[1] + qreg[hh][2]*kvals[2] + qreg[hh][3]*kvals[3];
                dot = warp_reduce_sum(dot);
                const float bias = d >= 0 && d < args.n_extent ? s_tbl[hh*args.n_extent + d] : 0.0f;
                const float s    = args.kq_scale*dot + bias - s_sink[hh] + mv;
                const float m_new = fmaxf(m[hh], s);
                const float corr  = expf(m[hh] - m_new);
                const float pk    = expf(s - m_new);
                l[hh] = l[hh]*corr + pk;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[hh][e] = acc[hh][e]*corr + pk*vvals[e];
                }
                m[hh] = m_new;
            }
        }
    }

    // --- combine the warps
#pragma unroll
    for (int hh = 0; hh < IK_ATTN_GQA_MAX; ++hh) {
        if (hh < gqa) {
            if (lane == 0) {
                s_m[hh][warp] = m[hh];
                s_l[hh][warp] = l[hh];
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                s_acc[hh][warp][4*lane + e] = acc[hh][e];
            }
        }
    }
    __syncthreads();

    // threads 0..127: dim i of every head; M, L of the split (M = -inf: no cell)
    float M[IK_ATTN_GQA_MAX], L[IK_ATTN_GQA_MAX], A[IK_ATTN_GQA_MAX];
    if (tid < HD) {
#pragma unroll
        for (int hh = 0; hh < IK_ATTN_GQA_MAX; ++hh) {
            M[hh] = -INFINITY; L[hh] = 0.0f; A[hh] = 0.0f;
            if (hh < gqa) {
#pragma unroll
                for (int w = 0; w < IK_ATTN_WARPS; ++w) {
                    M[hh] = fmaxf(M[hh], s_m[hh][w]);
                }
#pragma unroll
                for (int w = 0; w < IK_ATTN_WARPS; ++w) {
                    if (s_m[hh][w] != -INFINITY) {
                        const float c = expf(s_m[hh][w] - M[hh]);
                        L[hh] += s_l[hh][w]*c;
                        A[hh] += s_acc[hh][w][tid]*c;
                    }
                }
            }
        }
    }

    if (args.n_split > 1) {
        // publish the partial result, the last block to arrive combines
        if (tid < HD) {
#pragma unroll
            for (int hh = 0; hh < IK_ATTN_GQA_MAX; ++hh) {
                if (hh < gqa) {
                    float * part = args.part + (((int64_t) g*args.n_split + split)*gqa + hh)*(HD + 2);
                    part[tid] = A[hh];
                    if (tid == 0) {
                        part[HD]     = M[hh];
                        part[HD + 1] = L[hh];
                    }
                }
            }
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int old = atomicAdd(&args.counters[g], 1);
            s_last = old == args.n_split - 1;
        }
        __syncthreads();
        if (!s_last) {
            return;
        }
        __threadfence();
        if (tid < HD) {
#pragma unroll
            for (int hh = 0; hh < IK_ATTN_GQA_MAX; ++hh) {
                if (hh < gqa) {
                    float Mt = -INFINITY;
                    for (int sp = 0; sp < args.n_split; ++sp) {
                        const float * part = args.part + (((int64_t) g*args.n_split + sp)*gqa + hh)*(HD + 2);
                        Mt = fmaxf(Mt, part[HD]);
                    }
                    float Lt = 0.0f, At = 0.0f;
                    for (int sp = 0; sp < args.n_split; ++sp) {
                        const float * part = args.part + (((int64_t) g*args.n_split + sp)*gqa + hh)*(HD + 2);
                        const float ms = part[HD];
                        if (ms != -INFINITY) {
                            const float c = expf(ms - Mt);
                            Lt += part[HD + 1]*c;
                            At += part[tid]*c;
                        }
                    }
                    M[hh] = Mt; L[hh] = Lt; A[hh] = At;
                }
            }
        }
        if (tid == 0) {
            args.counters[g] = 0;
        }
    }

    // --- the zero sink logit, the normalization and the sigmoid gate; threads 0..127 (warps 0-3) write the heads
    if (tid < HD) {
#pragma unroll
        for (int hh = 0; hh < IK_ATTN_GQA_MAX; ++hh) {
            if (hh < gqa) {
                const int   h     = g*gqa + hh;
                const float m_new = fmaxf(M[hh], 0.0f);
                const float corr  = expf(M[hh] - m_new);
                const float Lf    = L[hh]*corr + expf(0.0f - m_new);
                const float gate  = pr0[args.off_gate + (int64_t) h*HD + tid];
                const float o     = A[hh]*corr/Lf * (1.0f/(1.0f + expf(-gate)));
                if constexpr (q8_out) {
                    ik_warp_store_q8_1(o, (block_q8_1 *) args.dst + ((int64_t) h*HD)/QK8_1 + warp, lane, true);
                } else {
                    ((float *) args.dst)[(int64_t) h*HD + tid] = o;
                }
            }
        }
    }
}

void ggml_cuda_op_inkling_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * proj    = dst->src[0];
    const ggml_tensor * consts  = dst->src[1];
    ggml_tensor       * state   = dst->src[2];
    const ggml_tensor * rs_ids  = dst->src[3];
    ggml_tensor       * k_cache = dst->src[4];
    ggml_tensor       * v_cache = dst->src[5];
    const ggml_tensor * kv_idxs = dst->src[6];
    const ggml_tensor * rel_pos = dst->src[7];
    const ggml_tensor * mask    = dst->src[8];
    const ggml_tensor * pos     = dst->src[9];

    ik_attn_args a;
    a.n_head     = ggml_get_op_params_i32(dst, 0);
    a.n_head_kv  = ggml_get_op_params_i32(dst, 1);
    const int hd = ggml_get_op_params_i32(dst, 2);
    a.n_rel      = ggml_get_op_params_i32(dst, 3);
    a.n_conv     = ggml_get_op_params_i32(dst, 4);
    a.has_taps   = ggml_get_op_params_i32(dst, 5);
    a.write_cell = ggml_get_op_params_i32(dst, 6);
    a.qk_clip    = ggml_get_op_params_f32(dst, 7);
    a.rel_clip   = ggml_get_op_params_f32(dst, 8);
    a.norm_eps   = ggml_get_op_params_f32(dst, 9);
    a.tau_alpha  = ggml_get_op_params_f32(dst, 10);
    a.tau_floor  = ggml_get_op_params_f32(dst, 11);
    a.kq_scale   = ggml_get_op_params_f32(dst, 12);
    a.clip_eps   = ggml_get_op_params_f32(dst, 13);

    GGML_ASSERT(hd == IK_ATTN_HD && a.n_head % a.n_head_kv == 0 && a.n_head/a.n_head_kv <= IK_ATTN_GQA_MAX);
    GGML_ASSERT(a.n_rel <= 32 && a.n_conv <= 8);

    const int64_t n_kv_ch = (int64_t) hd*a.n_head_kv;
    const int64_t n_taps  = a.has_taps ? (int64_t) a.n_conv*a.n_head_kv : 0;
    a.off_rel   = (int64_t) hd*a.n_head;
    a.off_sink  = a.off_rel + (int64_t) a.n_rel*a.n_head;
    a.off_gate  = a.off_sink + a.n_head;
    a.off_k     = a.off_gate + (int64_t) hd*a.n_head;
    a.off_ktaps = a.off_k + n_kv_ch;
    a.off_v     = a.off_ktaps + n_taps;
    a.off_vtaps = a.off_v + n_kv_ch;
    a.c_relclip = (int64_t) hd*a.n_head;
    a.c_kconv   = a.c_relclip + (int64_t) a.n_rel*a.n_head;
    a.c_vconv   = a.c_kconv + (int64_t) a.n_conv*n_kv_ch;
    a.c_table   = a.c_vconv + (int64_t) a.n_conv*n_kv_ch;
    a.n_extent  = (ggml_nelements(consts) - a.c_table)/a.n_rel;
    GGML_ASSERT(a.n_extent <= IK_ATTN_MAX_EXTENT);

    a.proj       = (const float *) proj->data; a.proj_s1 = proj->nb[1]/sizeof(float); a.proj_ncols = proj->ne[1];
    a.consts     = (const float *) consts->data;
    a.state      = (float *) state->data; a.state_s1 = state->nb[1]/sizeof(float);
    a.rs_ids     = (const int32_t *) rs_ids->data;
    a.k_cache    = (char *) k_cache->data; a.kc_nb1 = k_cache->nb[1]; a.kc_nb2 = k_cache->nb[2];
    a.v_cache    = (char *) v_cache->data; a.vc_nb1 = v_cache->nb[1]; a.vc_nb2 = v_cache->nb[2];
    a.kv_idxs    = (const int64_t *) kv_idxs->data;
    a.rel_pos    = (const int32_t *) rel_pos->data;
    a.mask       = mask->data; a.mask_f16 = mask->type == GGML_TYPE_F16;
    a.pos        = pos ? (const int32_t *) pos->data : nullptr;
    a.dst        = dst->data;
    a.n_kv       = k_cache->ne[2];

    // the cells split over blocks: about 256 cells per block, at most 16 splits
    a.n_split      = a.n_kv <= 256 ? 1 : std::min(16, (a.n_kv + 255)/256);
    a.kv_per_split = (a.n_kv + a.n_split - 1)/a.n_split;

    cudaStream_t stream = ctx.stream();
    const int gqa = a.n_head/a.n_head_kv;

    ggml_cuda_pool_alloc<float> part(ctx.pool());
    ggml_cuda_pool_alloc<int>   counters(ctx.pool());
    a.part = nullptr; a.counters = nullptr;
    if (a.n_split > 1) {
        part.alloc((size_t) a.n_head_kv*a.n_split*gqa*(IK_ATTN_HD + 2));
        counters.alloc(a.n_head_kv);
        a.part     = part.get();
        a.counters = counters.get();
        CUDA_CHECK(cudaMemsetAsync(a.counters, 0, a.n_head_kv*sizeof(int), stream));
    }

    const bool q8 = dst->type == GGML_TYPE_Q8_1;
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || q8);
    GGML_ASSERT(k_cache->type == v_cache->type && (k_cache->type == GGML_TYPE_F16 || k_cache->type == GGML_TYPE_Q8_0));

    const dim3 grid(a.n_head_kv, a.n_split, 1);
    if (k_cache->type == GGML_TYPE_F16) {
        if (q8) {
            inkling_attn_kernel<half, true><<<grid, IK_ATTN_BLOCK, 0, stream>>>(a);
        } else {
            inkling_attn_kernel<half, false><<<grid, IK_ATTN_BLOCK, 0, stream>>>(a);
        }
    } else {
        if (q8) {
            inkling_attn_kernel<block_q8_0, true><<<grid, IK_ATTN_BLOCK, 0, stream>>>(a);
        } else {
            inkling_attn_kernel<block_q8_0, false><<<grid, IK_ATTN_BLOCK, 0, stream>>>(a);
        }
    }
}
