#include "inkling.cuh"
#include "inkling-common.cuh"
#include "vecdotq.cuh"
#include "cp-async.cuh"

#include <cmath>

// ---------------------------------------------------------------------------------------------------------------------
// l2_clip: y = x*b * min(1, threshold/(|x*b| + eps)), one warp per row

// IK_MR=1: windows of 2..8 tokens take kernels whose per-row arithmetic is that of the single-token kernels
static bool ik_mr_enabled() {
    static const bool on = getenv("IK_MR") && atoi(getenv("IK_MR")) != 0;
    return on;
}

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
// res_mix: softmax-weighted mix of RMS-normalized states, one block per (token, query); optional RMS norm of the result
// (scaled by nw) and F32 or Q8_1 output. query == nullptr: no mix, the (normalized) first state.

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
template <int block_size, bool q8, int NS = 0>
static __global__ void res_mix_f32_vec4(const float4 * __restrict__ query, const ik_state_ptrs st, void * __restrict__ dst_v,
        const int n_embd4, const int n_tok, const int state_count, const int64_t q_nb1, const float eps, const float scale,
        const ik_res_mix_norm norm) {
    const int n_states = NS ? NS : state_count;
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

// The trailing normalization weight is a source but never a mixed state.
static int ik_res_mix_state_count(const ggml_tensor * node) {
    int count = 0;
    while (count + 1 < GGML_MAX_SRC && node->src[1 + count]) {
        ++count;
    }
    return count - (ggml_get_op_params_i32(node, 3) != 0);
}

bool ggml_cuda_supports_res_mix(const ggml_tensor * node) {
    const int count = ik_res_mix_state_count(node);
    return count >= 1 && count <= IK_MAX_STATES;
}

void ggml_cuda_op_res_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * query = dst->src[0];

    const bool has_norm = ggml_get_op_params_i32(dst, 3) != 0;

    const int n_states = ik_res_mix_state_count(dst);
    GGML_ASSERT(n_states >= 1 && n_states <= IK_MAX_STATES);
    ik_state_ptrs st = {};
    for (int i = 0; i < n_states; ++i) {
        st.p[i]   = (const float *) dst->src[1 + i]->data;
        st.nb1[i] = dst->src[1 + i]->nb[1];
    }
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
                static const bool specialize = getenv("IK_MIX_SPECIALIZE") && atoi(getenv("IK_MIX_SPECIALIZE")) != 0;
                auto launch_mix = [&](auto state_count) {
                    constexpr int ns = decltype(state_count)::value;
                    res_mix_f32_vec4<640, kQ8, ns><<<grid, 640, 0, stream>>>((const float4 *) q_d, st, dst->data,
                            n_embd4, n_tok, n_states, q_nb1, eps, scale, norm);
                };
                switch (specialize && (n_tok == 1 || (ik_mr_enabled() && n_tok <= 8)) ? n_states : 0) {
                    case 1: launch_mix(std::integral_constant<int, 1>{}); break;
                    case 2: launch_mix(std::integral_constant<int, 2>{}); break;
                    case 3: launch_mix(std::integral_constant<int, 3>{}); break;
                    case 4: launch_mix(std::integral_constant<int, 4>{}); break;
                    case 5: launch_mix(std::integral_constant<int, 5>{}); break;
                    default: launch_mix(std::integral_constant<int, 0>{}); break;
                }
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

// the rows of a window whose state is read in place (ids): a thread per channel walks the tokens in order, so the
// state it overwrites was read first; per token the statements of shortconv_res_one
template <int d_conv>
static __global__ void shortconv_res_rows_f32(const ik_shortconv_args args) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= args.n_chan) {
        return;
    }
    const float * ps = (const float *) ((const char *) args.past + c*args.past_nb1 + args.ids[0]*args.past_nb2);
    const float * wc = (const float *) ((const char *) args.w + c*args.w_nb1);
    float x[d_conv], w[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        x[j] = j < d_conv - 1 ? ps[j] : 0.0f;
        w[j] = wc[j];
    }
    for (int t = 0; t < args.n_t; ++t) {
        x[d_conv - 1] = *(const float *) ((const char *) args.cur + (int64_t) t*args.cur_nb1 + c*sizeof(float));
        float out = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            out += x[j]*w[j];
        }
        float y = (x[d_conv - 1] + out)*args.scale;
        if (args.res) {
            y += *(const float *) ((const char *) args.res + t*args.res_nb1 + c*sizeof(float));
        }
        const int k = args.n_t - 1 - t;
        if (k < args.n_slots) {
            float * st = (float *) ((char *) args.state + k*args.st_nb1 + c*args.st_nb2);
#pragma unroll
            for (int j = 0; j < d_conv - 1; ++j) {
                st[j] = x[j + 1];
            }
        }
        *(float *) ((char *) args.dst + t*args.dst_nb1 + c*sizeof(float)) = y;
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            x[j] = x[j + 1];
        }
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

    if (args.ids && args.n_t > 1 && args.n_s == 1 && !taps && !args.post && d_conv == 4 && state != nullptr) {
        // a window over the state in place
        shortconv_res_rows_f32<4><<<(args.n_chan + 255)/256, 256, 0, stream>>>(args);
        return;
    }

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

template <typename T>
static __device__ __forceinline__ float ik_w(const T * w, const int i) {
    if constexpr (std::is_same<T, half>::value) {
        return __half2float(w[i]);
    } else {
        return w[i];
    }
}

template <int block_size, typename wt>
static __global__ void loop_ctrl_down_f32(const wt * __restrict__ w, const float * __restrict__ a, const float * __restrict__ b,
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
    const wt * wk = w + ((int64_t) ib*rank + k)*2*n;

    // the sums of squares and the raw dot products in one pass: w^T rmsnorm(a) = rsqrt(mean(a^2) + eps) * w^T a
    // (the loads of the thread's elements are issued together)
    constexpr int PER = block_size >= 1024 ? 5 : 10; // elements per thread per round (n <= PER*block_size for one round)
    float sa = 0.0f, sb = 0.0f, da = 0.0f, db = 0.0f;
    for (int i0 = 0; i0 < n; i0 += PER*block_size) {
        float x[PER], y[PER], wa[PER], wb[PER];
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int i = i0 + threadIdx.x + j*block_size;
            const bool ok = i < n;
            x[j]  = ok ? av[i] : 0.0f;
            y[j]  = ok ? bv[i] : 0.0f;
            wa[j] = ok ? ik_w(wk, i) : 0.0f;
            wb[j] = ok ? ik_w(wk, n + i) : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            sa += x[j]*x[j];
            sb += y[j]*y[j];
            da += wa[j]*x[j];
            db += wb[j]*y[j];
        }
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

// ---------------------------------------------------------------------------------------------------------------------
// The controllers of a batch (prefill): the per-token kernels above launch a block per (rank, token) and a warp per
// output element, which was 35% of the prefill time (phase 5 profile). For n_tok >= IK_CTRL_GEMM_MIN the same sums are
// two GEMMs: D = W_down^T [rmsnorm(a); rmsnorm(msg)] and [G, F] = [W_gain, W_forget] D, in FP32 (the handle's TF32
// mode is switched off around them; F16 weights are widened to a scratch).

static __global__ void ik_ctrl_stack_norm(const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ x,
        const int n, const int n_tok, const int64_t a_nb1, const int64_t a_nb2, const int64_t b_nb1, const int64_t b_nb2, const float eps) {
    const int t  = blockIdx.x;
    const int ib = blockIdx.y;
    __shared__ float red[2][256/WARP_SIZE];
    const float * av = (const float *) ((const char *) a + t*a_nb1 + ib*a_nb2);
    const float * bv = (const float *) ((const char *) b + t*b_nb1 + ib*b_nb2);
    float sa = 0.0f, sb = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) {
        sa += av[i]*av[i];
        sb += bv[i]*bv[i];
    }
    sa = ik_block_sum<256>(sa, red[0]);
    sb = ik_block_sum<256>(sb, red[1]);
    const float ra = rsqrtf(sa/n + eps);
    const float rb = rsqrtf(sb/n + eps);
    float * xt = x + ((int64_t) ib*n_tok + t)*2*n;
    for (int i = threadIdx.x; i < n; i += 256) {
        xt[i]     = av[i]*ra;
        xt[n + i] = bv[i]*rb;
    }
}

static __global__ void ik_half_to_float(const half * __restrict__ x, float * __restrict__ y, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __half2float(x[i]);
    }
}

// y = cur - tanh(f)*cur + tanh(g)*msg, one thread per element
static __global__ void ik_ctrl_apply_elem(const float * __restrict__ cur, const float * __restrict__ msg, const float * __restrict__ g, const float * __restrict__ f,
        float * __restrict__ y, const int n, const int n_tok, const int64_t cur_nb1, const int64_t cur_nb2, const int64_t msg_nb1, const int64_t msg_nb2) {
    const int64_t e  = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int     ib = blockIdx.y;
    if (e >= (int64_t) n*n_tok) {
        return;
    }
    const int t = e/n;
    const int i = e - (int64_t) t*n;
    const int64_t o = ((int64_t) ib*n_tok + t)*n + i;
    const float c = *(const float *) ((const char *) cur + t*cur_nb1 + ib*cur_nb2 + i*sizeof(float));
    const float m = *(const float *) ((const char *) msg + t*msg_nb1 + ib*msg_nb2 + i*sizeof(float));
    y[o] = (c - tanhf(f[o])*c) + tanhf(g[o])*m;
}

static int ik_ctrl_gemm_min() {
    static const int v = getenv("IK_CTRL_GEMM_MIN") ? atoi(getenv("IK_CTRL_GEMM_MIN")) : 8; // tokens; 0: never
    return v;
}

// the weights as FP32: the tensor itself, or a widened copy in the scratch
static const float * ik_ctrl_weights_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * w, ggml_cuda_pool_alloc<float> & scratch) {
    if (w->type == GGML_TYPE_F32) {
        return (const float *) w->data;
    }
    const int64_t ne = ggml_nelements(w);
    scratch.alloc(ne);
    ik_half_to_float<<<(ne + 255)/256, 256, 0, ctx.stream()>>>((const half *) w->data, scratch.get(), ne);
    return scratch.get();
}

// c[m, n_col] (ldc) = a[k, m]^T (lda) b[k, n_col] (ldb), FP32 math
static void ik_sgemm_tn(ggml_backend_cuda_context & ctx, const int m, const int n_col, const int k, const float * a, const int lda, const float * b, const int ldb,
        float * c, const int ldc) {
    cublasHandle_t h = ctx.cublas_handle();
    const float one = 1.0f, zero = 0.0f;
    CUBLAS_CHECK(cublasSetStream(h, ctx.stream()));
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n_col, k, &one, a, lda, b, ldb, &zero, c, ldc));
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
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

    if (ik_ctrl_gemm_min() > 0 && n_tok >= ik_ctrl_gemm_min() && !(ik_mr_enabled() && n_tok <= 8) && (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16)) {
        ggml_cuda_pool_alloc<float> x(ctx.pool(), (size_t) 2*n*n_tok*B);
        ggml_cuda_pool_alloc<float> wbuf(ctx.pool());
        ik_ctrl_stack_norm<<<dim3(n_tok, B), 256, 0, ctx.stream()>>>((const float *) a->data, (const float *) b->data, x.get(), n, n_tok,
                a->nb[1], a->ne[2] == 1 ? 0 : a->nb[2], b->nb[1], b->nb[2], eps);
        const float * wf32 = ik_ctrl_weights_f32(ctx, w, wbuf);
        for (int ib = 0; ib < B; ++ib) {
            ik_sgemm_tn(ctx, rank, n_tok, 2*n, wf32 + (int64_t) ib*rank*2*n, 2*n, x.get() + (int64_t) ib*n_tok*2*n, 2*n,
                    (float *) dst->data + (int64_t) ib*n_tok*rank, rank);
        }
        return;
    }

    const dim3 grid(rank, n_tok, B);
    // 1024 threads: the whole dot of a block in one round of loads (n <= 10240)
    static const bool big = getenv("IK_CTRL_DOWN_1024") && atoi(getenv("IK_CTRL_DOWN_1024")) != 0; // measured slower (8.7 vs 5.7 us)
    if (w->type == GGML_TYPE_F16) {
        if (big) {
            loop_ctrl_down_f32<1024, half><<<grid, 1024, 0, ctx.stream()>>>((const half *) w->data, (const float *) a->data, (const float *) b->data,
                    (float *) dst->data, n, rank, a->nb[1], a->ne[2] == 1 ? 0 : a->nb[2], b->nb[1], b->nb[2], eps);
        } else {
            loop_ctrl_down_f32<256, half><<<grid, 256, 0, ctx.stream()>>>((const half *) w->data, (const float *) a->data, (const float *) b->data,
                    (float *) dst->data, n, rank, a->nb[1], a->ne[2] == 1 ? 0 : a->nb[2], b->nb[1], b->nb[2], eps);
        }
    } else {
        if (big) {
            loop_ctrl_down_f32<1024, float><<<grid, 1024, 0, ctx.stream()>>>((const float *) w->data, (const float *) a->data, (const float *) b->data,
                    (float *) dst->data, n, rank, a->nb[1], a->ne[2] == 1 ? 0 : a->nb[2], b->nb[1], b->nb[2], eps);
        } else {
            loop_ctrl_down_f32<256, float><<<grid, 256, 0, ctx.stream()>>>((const float *) w->data, (const float *) a->data, (const float *) b->data,
                    (float *) dst->data, n, rank, a->nb[1], a->ne[2] == 1 ? 0 : a->nb[2], b->nb[1], b->nb[2], eps);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// loop_ctrl_apply: y = cur - tanh(wf^T d)*cur + tanh(wg^T d)*msg, one warp per output element

// NORM: n_tok == 1 only; the last block of a batch row to finish re-reads the row, RMS-normalizes it and scales it by the
// norm weight, in place (F32) or into the Q8_1 output (dst_v; the F32 result is the row buffer)
template <typename wt, int NORM, bool q8>
static __global__ void __launch_bounds__(256) loop_ctrl_apply_f32(const float * __restrict__ cur, const float * __restrict__ msg, const float * __restrict__ d,
        const wt * __restrict__ wg, const wt * __restrict__ wf, float * __restrict__ dst,
        const int n, const int n_tok, const int rank, const int64_t cur_nb1, const int64_t cur_nb2, const int64_t msg_nb1, const int64_t msg_nb2,
        const float * __restrict__ nw, const int64_t nw_nb1, void * __restrict__ dst_v, const float eps, int * __restrict__ counters) {
    const int64_t gw   = (int64_t) blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int     lane = threadIdx.x % WARP_SIZE;
    const int64_t n_out = (int64_t) n*n_tok;
    const int ib = blockIdx.y;
    const int t  = gw / n;
    const int i  = gw - (int64_t) t*n;
    if (gw < n_out && t < n_tok) {
        const float * dv = d + ((int64_t) ib*n_tok + t)*rank;
        const wt    * g  = wg + ((int64_t) ib*n + i)*rank;
        const wt    * f  = wf + ((int64_t) ib*n + i)*rank;

        float sg = 0.0f, sf = 0.0f;
        for (int k = lane; k < rank; k += WARP_SIZE) {
            const float dk = dv[k];
            sg += ik_w(g, k)*dk;
            sf += ik_w(f, k)*dk;
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
    if constexpr (NORM) {
        __shared__ float s_red[256/WARP_SIZE];
        __shared__ int   s_last;
        __syncthreads();
        if (threadIdx.x == 0) {
            s_last = ik_atomic_add_acq_rel(counters + ib, 1) == (int) gridDim.x - 1;
        }
        __syncthreads();
        if (!s_last) {
            return;
        }
        float       * yt  = dst + (int64_t) ib*n;
        const float * nwb = (const float *) ((const char *) nw + ib*nw_nb1);
        float ss = 0.0f;
        for (int k = threadIdx.x; k < n; k += 256) {
            ss += yt[k]*yt[k];
        }
        ss = ik_block_sum<256>(ss, s_red);
        const float scale = rsqrtf(ss/n + eps);
        if constexpr (q8) {
            const int wid = threadIdx.x / WARP_SIZE;
            block_q8_1 * out = (block_q8_1 *) dst_v + (int64_t) ib*(n/QK8_1);
            for (int i0 = wid*WARP_SIZE; i0 < n; i0 += 256) {
                const int k = i0 + lane;
                const float v = k < n ? yt[k]*scale*nwb[k] : 0.0f;
                ik_warp_store_q8_1(v, out + i0/QK8_1, lane, k < n);
            }
        } else {
            for (int k = threadIdx.x; k < n; k += 256) {
                yt[k] = yt[k]*scale*nwb[k];
            }
        }
        if (threadIdx.x == 0) {
            counters[ib] = 0;
        }
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

// v2: one thread per output element (its two weight rows of `rank` values as float4 loads, all in flight), d in shared
// memory; with the norm, the last block of a row to finish re-reads the row, RMS-normalizes it and scales it by the norm
// weight, in place (F32) or into the Q8_1 output (the F32 scratch is the row buffer). rank <= 64.
#define IK_CA2_BLOCK 256
#define IK_CA2_RANK  64

template <typename wt, bool q8, bool norm>
static __global__ void __launch_bounds__(IK_CA2_BLOCK) loop_ctrl_apply2_f32(const float * __restrict__ cur, const float * __restrict__ msg, const float * __restrict__ d,
        const wt * __restrict__ wg, const wt * __restrict__ wf, float * __restrict__ y, const float * __restrict__ nw, void * __restrict__ dst_v,
        const int n, const int n_tok, const int rank, const int64_t cur_nb1, const int64_t cur_nb2, const int64_t msg_nb1, const int64_t msg_nb2,
        const int64_t nw_nb1, const float eps, int * __restrict__ counters) {
    __shared__ float s_d[IK_CA2_RANK];
    __shared__ float s_red[IK_CA2_BLOCK/WARP_SIZE];
    __shared__ int   s_last;
    const int tid = threadIdx.x;
    const int t   = blockIdx.y;
    const int ib  = blockIdx.z;
    const int i   = blockIdx.x*IK_CA2_BLOCK + tid;
    const int64_t row = (int64_t) ib*n_tok + t;

    if (tid < rank) {
        s_d[tid] = d[row*rank + tid];
    }
    // the two weight rows of this output (issued before the barrier)
    float sg = 0.0f, sf = 0.0f;
    if (i < n) {
        const wt * g = wg + ((int64_t) ib*n + i)*rank;
        const wt * f = wf + ((int64_t) ib*n + i)*rank;
        if constexpr (std::is_same<wt, half>::value) {
            // 64 halves = 8 x 16 bytes per row
            uint4 gv[IK_CA2_RANK/8], fv[IK_CA2_RANK/8];
#pragma unroll
            for (int k = 0; k < IK_CA2_RANK/8; ++k) {
                gv[k] = k*8 < rank ? ((const uint4 *) g)[k] : make_uint4(0, 0, 0, 0);
                fv[k] = k*8 < rank ? ((const uint4 *) f)[k] : make_uint4(0, 0, 0, 0);
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < IK_CA2_RANK/8; ++k) {
                const half2 * gh = (const half2 *) &gv[k];
                const half2 * fh = (const half2 *) &fv[k];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 g2 = __half22float2(gh[e]);
                    const float2 f2 = __half22float2(fh[e]);
                    const float d0 = s_d[8*k + 2*e], d1 = s_d[8*k + 2*e + 1];
                    sg += g2.x*d0 + g2.y*d1;
                    sf += f2.x*d0 + f2.y*d1;
                }
            }
        } else {
            float4 gv[IK_CA2_RANK/4], fv[IK_CA2_RANK/4];
#pragma unroll
            for (int k = 0; k < IK_CA2_RANK/4; ++k) {
                gv[k] = k*4 < rank ? ((const float4 *) g)[k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                fv[k] = k*4 < rank ? ((const float4 *) f)[k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < IK_CA2_RANK/4; ++k) {
                sg += gv[k].x*s_d[4*k] + gv[k].y*s_d[4*k + 1] + gv[k].z*s_d[4*k + 2] + gv[k].w*s_d[4*k + 3];
                sf += fv[k].x*s_d[4*k] + fv[k].y*s_d[4*k + 1] + fv[k].z*s_d[4*k + 2] + fv[k].w*s_d[4*k + 3];
            }
        }
        const float gain   = tanhf(sg);
        const float forget = tanhf(sf);
        const float c = *(const float *) ((const char *) cur + t*cur_nb1 + ib*cur_nb2 + i*sizeof(float));
        const float m = *(const float *) ((const char *) msg + t*msg_nb1 + ib*msg_nb2 + i*sizeof(float));
        y[row*n + i] = (c - forget*c) + gain*m;
    } else {
        __syncthreads();
    }
    if constexpr (!norm) {
        return;
    }
    // the last block of the row normalizes it
    __syncthreads();
    if (tid == 0) {
        s_last = ik_atomic_add_acq_rel(counters + row, 1) == (int) gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    float * yt = y + row*n;
    const float * nwb = (const float *) ((const char *) nw + ib*nw_nb1);
    float ss = 0.0f;
    for (int k = tid; k < n; k += IK_CA2_BLOCK) {
        ss += yt[k]*yt[k];
    }
    ss = ik_block_sum<IK_CA2_BLOCK>(ss, s_red);
    const float scale = rsqrtf(ss/n + eps);
    if constexpr (q8) {
        const int lane = tid % WARP_SIZE;
        const int wid  = tid / WARP_SIZE;
        block_q8_1 * out = (block_q8_1 *) dst_v + row*(n/QK8_1);
        for (int i0 = wid*WARP_SIZE; i0 < n; i0 += IK_CA2_BLOCK) {
            const int k = i0 + lane;
            const float v = k < n ? yt[k]*scale*nwb[k] : 0.0f;
            ik_warp_store_q8_1(v, out + i0/QK8_1, lane, k < n);
        }
    } else {
        for (int k = tid; k < n; k += IK_CA2_BLOCK) {
            yt[k] = yt[k]*scale*nwb[k];
        }
    }
    if (tid == 0) {
        counters[row] = 0;
    }
}

template <typename wt>
static void loop_ctrl_apply2_launch(ggml_backend_cuda_context & ctx, const float * cur, const float * msg, const float * d, const wt * wg, const wt * wf,
        float * y, const float * nw, void * dst, const int n, const int n_tok, const int rank, const int64_t cur_nb1, const int64_t cur_nb2,
        const int64_t msg_nb1, const int64_t msg_nb2, const int64_t nw_nb1, const float eps, const bool q8, const int B) {
    const dim3 grid((n + IK_CA2_BLOCK - 1)/IK_CA2_BLOCK, n_tok, B);
    int * counters = nw ? ik_counters(ctx, 32, B*n_tok) : nullptr;
    if (nw == nullptr) {
        loop_ctrl_apply2_f32<wt, false, false><<<grid, IK_CA2_BLOCK, 0, ctx.stream()>>>(cur, msg, d, wg, wf, y, nw, dst, n, n_tok, rank, cur_nb1, cur_nb2, msg_nb1, msg_nb2, nw_nb1, eps, counters);
    } else if (q8) {
        loop_ctrl_apply2_f32<wt, true, true><<<grid, IK_CA2_BLOCK, 0, ctx.stream()>>>(cur, msg, d, wg, wf, y, nw, dst, n, n_tok, rank, cur_nb1, cur_nb2, msg_nb1, msg_nb2, nw_nb1, eps, counters);
    } else {
        loop_ctrl_apply2_f32<wt, false, true><<<grid, IK_CA2_BLOCK, 0, ctx.stream()>>>(cur, msg, d, wg, wf, y, nw, dst, n, n_tok, rank, cur_nb1, cur_nb2, msg_nb1, msg_nb2, nw_nb1, eps, counters);
    }
}

// Four rank-64 output rows per warp. Each eight-lane group loads two
// contiguous four-element vectors per lane, keeping the full-warp sum order.
static __global__ void ik_ctrl_apply8(const float * cur, const float * msg, const float * d,
        const half * wg, const half * wf, float * y, int n, size_t cur_stride, size_t msg_stride) {
    const int i = blockIdx.x*32 + threadIdx.x/8;
    const int lane = threadIdx.x % 8;
    const int b = blockIdx.y;
    const int row = b*n + i;
    const float * dv = d + b*64 + lane*4;
    float gv[4], fv[4];
    const half * g = wg + row*64 + lane*4;
    const half * f = wf + row*64 + lane*4;
    float4 d0 = *(const float4 *) dv, d1 = *(const float4 *) (dv+32);
    uint2 g0 = *(const uint2 *) g, g1 = *(const uint2 *) (g+32);
    uint2 f0 = *(const uint2 *) f, f1 = *(const uint2 *) (f+32);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a = ((const float *) &d0)[j], z = ((const float *) &d1)[j];
        gv[j] = warp_reduce_sum<8>(__fmaf_rn(__half2float(((const half *) &g1)[j]), z,
                __fmul_rn(__half2float(((const half *) &g0)[j]), a)));
        fv[j] = warp_reduce_sum<8>(__fmaf_rn(__half2float(((const half *) &f1)[j]), z,
                __fmul_rn(__half2float(((const half *) &f0)[j]), a)));
    }
    if (lane == 0) {
        const float c = cur[b*cur_stride+i], m = msg[b*msg_stride+i];
        const float gain = tanhf((gv[0]+gv[2])+(gv[1]+gv[3]));
        const float forget = tanhf((fv[0]+fv[2])+(fv[1]+fv[3]));
        y[row] = (c-forget*c) + gain*m;
    }
}

// ik_ctrl_apply8 for the rows of a window (grid z: the row); the same arithmetic per element
static __global__ void ik_ctrl_apply8_rows(const float * cur, const float * msg, const float * d,
        const half * wg, const half * wf, float * y, int n, int n_tok, size_t cur_stride, size_t msg_stride, size_t cur_ts, size_t msg_ts) {
    const int i = blockIdx.x*32 + threadIdx.x/8;
    const int lane = threadIdx.x % 8;
    const int b = blockIdx.y;
    const int t = blockIdx.z;
    const int row = b*n + i;
    const float * dv = d + ((size_t) b*n_tok + t)*64 + lane*4;
    float gv[4], fv[4];
    const half * g = wg + row*64 + lane*4;
    const half * f = wf + row*64 + lane*4;
    float4 d0 = *(const float4 *) dv, d1 = *(const float4 *) (dv+32);
    uint2 g0 = *(const uint2 *) g, g1 = *(const uint2 *) (g+32);
    uint2 f0 = *(const uint2 *) f, f1 = *(const uint2 *) (f+32);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a = ((const float *) &d0)[j], z = ((const float *) &d1)[j];
        gv[j] = warp_reduce_sum<8>(__fmaf_rn(__half2float(((const half *) &g1)[j]), z,
                __fmul_rn(__half2float(((const half *) &g0)[j]), a)));
        fv[j] = warp_reduce_sum<8>(__fmaf_rn(__half2float(((const half *) &f1)[j]), z,
                __fmul_rn(__half2float(((const half *) &f0)[j]), a)));
    }
    if (lane == 0) {
        const float c = cur[b*cur_stride + t*cur_ts + i], m = msg[b*msg_stride + t*msg_ts + i];
        const float gain = tanhf((gv[0]+gv[2])+(gv[1]+gv[3]));
        const float forget = tanhf((fv[0]+fv[2])+(fv[1]+fv[3]));
        y[((size_t) b*n_tok + t)*n + i] = (c-forget*c) + gain*m;
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

    static const bool v2 = getenv("IK_CTRL_APPLY_V2") && atoi(getenv("IK_CTRL_APPLY_V2")) != 0; // measured slower (thread per output: 27 blocks)
    if (v2 && rank <= IK_CA2_RANK && rank % 8 == 0 && B*n_tok <= 32 && (nw == nullptr || ggml_nrows(nw) == 1 || nw->nb[1] % 16 == 0)) {
        float eps = 0.0f;
        if (nw) {
            memcpy(&eps, (const float *) dst->op_params + 0, sizeof(float));
        }
        const int64_t nw_nb1 = nw && ggml_nrows(nw) != 1 ? nw->nb[1] : 0;
        if (wg->type == GGML_TYPE_F16) {
            loop_ctrl_apply2_launch<half>(ctx, (const float *) cur->data, (const float *) msg->data, (const float *) d->data, (const half *) wg->data, (const half *) wf->data,
                    y, nw ? (const float *) nw->data : nullptr, dst->data, n, n_tok, rank, cur->nb[1], cur->ne[2] == 1 ? 0 : cur->nb[2], msg->nb[1], msg->nb[2], nw_nb1, eps, q8, B);
        } else {
            loop_ctrl_apply2_launch<float>(ctx, (const float *) cur->data, (const float *) msg->data, (const float *) d->data, (const float *) wg->data, (const float *) wf->data,
                    y, nw ? (const float *) nw->data : nullptr, dst->data, n, n_tok, rank, cur->nb[1], cur->ne[2] == 1 ? 0 : cur->nb[2], msg->nb[1], msg->nb[2], nw_nb1, eps, q8, B);
        }
        return;
    }

    const int block_size = 256;
    const int64_t nblocks = ((int64_t) n*n_tok*WARP_SIZE + block_size - 1)/block_size;
    const dim3 grid(nblocks, B, 1);

    float eps = 0.0f;
    if (nw) {
        memcpy(&eps, (const float *) dst->op_params + 0, sizeof(float));
    }
    const int64_t nw_nb1 = nw && ggml_nrows(nw) != 1 ? nw->nb[1] : 0;
    // the fused norm (last block of 288 arriving on one counter) measured slower than the separate norm kernel: opt-in
    static const bool fuse_norm = getenv("IK_CTRL_NORM_FUSED") && atoi(getenv("IK_CTRL_NORM_FUSED")) != 0;
    const bool fused = nw != nullptr && n_tok == 1 && B <= 32 && fuse_norm;
    int * counters = fused ? ik_counters(ctx, 32, B) : nullptr;
#define IK_CA_LAUNCH(wt_t, NORM, Q8) loop_ctrl_apply_f32<wt_t, NORM, Q8><<<grid, block_size, 0, ctx.stream()>>>((const float *) cur->data, (const float *) msg->data, \
            (const float *) d->data, (const wt_t *) wg->data, (const wt_t *) wf->data, y, n, n_tok, rank, cur->nb[1], cur->ne[2] == 1 ? 0 : cur->nb[2], msg->nb[1], msg->nb[2], \
            nw ? (const float *) nw->data : nullptr, nw_nb1, dst->data, eps, counters)
    static const bool group8 = getenv("IK_CTRL_APPLY8") && atoi(getenv("IK_CTRL_APPLY8"));
    const bool mr_rows = ik_mr_enabled() && n_tok > 1 && n_tok <= 8;
    if (mr_rows && group8 && !fused && wg->type == GGML_TYPE_F16 && rank == 64 && n % 32 == 0 &&
            (uintptr_t) d->data % 16 == 0 && (uintptr_t) wg->data % 8 == 0 && (uintptr_t) wf->data % 8 == 0) {
        ik_ctrl_apply8_rows<<<dim3(n/32, B, n_tok), 256, 0, ctx.stream()>>>((const float *) cur->data, (const float *) msg->data,
                (const float *) d->data, (const half *) wg->data, (const half *) wf->data, y, n, n_tok,
                cur->ne[2] == 1 ? 0 : cur->nb[2]/sizeof(float), msg->nb[2]/sizeof(float), cur->nb[1]/sizeof(float), msg->nb[1]/sizeof(float));
    } else if (ik_ctrl_gemm_min() > 0 && n_tok >= ik_ctrl_gemm_min() && !mr_rows && !fused && (wg->type == GGML_TYPE_F32 || wg->type == GGML_TYPE_F16) && wf->type == wg->type) {
        // the batch: G = W_gain D and F = W_forget D as GEMMs, then the element-wise update
        ggml_cuda_pool_alloc<float> gbuf(ctx.pool(), (size_t) n*n_tok*B), fbuf(ctx.pool(), (size_t) n*n_tok*B);
        ggml_cuda_pool_alloc<float> wgbuf(ctx.pool()), wfbuf(ctx.pool());
        const float * wg32 = ik_ctrl_weights_f32(ctx, wg, wgbuf);
        const float * wf32 = ik_ctrl_weights_f32(ctx, wf, wfbuf);
        for (int ib = 0; ib < B; ++ib) {
            const float * dv = (const float *) d->data + (int64_t) ib*n_tok*rank;
            ik_sgemm_tn(ctx, n, n_tok, rank, wg32 + (int64_t) ib*n*rank, rank, dv, rank, gbuf.get() + (int64_t) ib*n_tok*n, n);
            ik_sgemm_tn(ctx, n, n_tok, rank, wf32 + (int64_t) ib*n*rank, rank, dv, rank, fbuf.get() + (int64_t) ib*n_tok*n, n);
        }
        const int64_t ne = (int64_t) n*n_tok;
        ik_ctrl_apply_elem<<<dim3((ne + 255)/256, B), 256, 0, ctx.stream()>>>((const float *) cur->data, (const float *) msg->data, gbuf.get(), fbuf.get(), y,
                n, n_tok, cur->nb[1], cur->ne[2] == 1 ? 0 : cur->nb[2], msg->nb[1], msg->nb[2]);
    } else if (group8 && !fused && wg->type == GGML_TYPE_F16 && rank == 64 && n_tok == 1 && n % 32 == 0 &&
            (uintptr_t) d->data % 16 == 0 && (uintptr_t) wg->data % 8 == 0 && (uintptr_t) wf->data % 8 == 0) {
        ik_ctrl_apply8<<<dim3(n/32, B), 256, 0, ctx.stream()>>>((const float *) cur->data, (const float *) msg->data,
                (const float *) d->data, (const half *) wg->data, (const half *) wf->data, y, n,
                cur->ne[2] == 1 ? 0 : cur->nb[2]/sizeof(float), msg->nb[2]/sizeof(float));
    } else if (wg->type == GGML_TYPE_F16) {
        if (!fused) { IK_CA_LAUNCH(half, 0, false); } else if (q8) { IK_CA_LAUNCH(half, 1, true); } else { IK_CA_LAUNCH(half, 1, false); }
    } else {
        if (!fused) { IK_CA_LAUNCH(float, 0, false); } else if (q8) { IK_CA_LAUNCH(float, 1, true); } else { IK_CA_LAUNCH(float, 1, false); }
    }
#undef IK_CA_LAUNCH
    if (fused) {
        return;
    }

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
#define IK_ATTN_MAX_EXTENT 1024 // == 4*IK_ATTN_BLOCK (one table step)
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
    int     ext_scan; // the cells were scanned by inkling_attn_scan_kernel (parts 0 .. n_split - 2); this kernel adds the current cell
    int     dbg_skip; // timing experiments only (IK_ATTN_SKIP): 1 bias table, 2 attention loop, 4 Q/R phase
    // multi-row (speculative window) launches: row r of the window is blockIdx.z of the MR kernel
    const int32_t * mr_kv_before;
    int     mr_rows;        // tokens of the window (1: the single-token kernel)
    int     mr_planes;      // rollback planes of the recurrent state written by the window (slot k: the state k tokens back)
    int     mr_plane_cells; // cells per plane
    int64_t mr_dst_rs;      // bytes between the output rows
    int64_t mr_part_rs;     // floats between the partial results of the rows
    int     mr_mma, mr_mma_split, mr_mma_step, mr_cps, mr_medium; // the split rule of the launch, to redo it for a row's own n_kv
    long long * dbg_times; // timing experiments only (IK_ATTN_TIMING): clock64 at the phase boundaries of block (0, 0) / the last block
};

#define IK_DBG_T(k) do { if (args.dbg_times && tid == 0 && blockIdx.x == 0 && blockIdx.y == 0) { args.dbg_times[k] = clock64(); } } while (0)

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
    constexpr int G = IK_ATTN_GQA_MAX;
    constexpr int U = 8; // cells per warp step (pairs of cells share a table-row load: lanes 0-15 / 16-31)
    IK_DBG_T(0);

    // ===== round 1 of loads: the conv inputs, Q, gate, R, sinks, the first tile's mask and distances =====
    const int     which  = tid/HD;
    const int     ci     = tid % HD;
    const int64_t ch     = (int64_t) which*HD*args.n_head_kv + (int64_t) g*HD + ci; // state channel
    const int     n_conv = args.n_conv;
    float cur_in, w_c[8], taps[8];
    {
        const int64_t off  = which ? args.off_v : args.off_k;
        const int64_t offt = which ? args.off_vtaps : args.off_ktaps;
        const int     col  = args.proj_ncols == 3 ? (which ? 2 : 1) : 0;
        const float * pr   = args.proj + col*args.proj_s1;
        const float * w    = args.consts + (which ? args.c_vconv : args.c_kconv) + ((int64_t) g*HD + ci)*n_conv;
        cur_in = pr[off + (int64_t) g*HD + ci];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w_c[j]  = j < n_conv ? w[j] : 0.0f;
            taps[j] = args.has_taps && j < n_conv ? pr[offt + (int64_t) g*n_conv + j] : 0.0f;
        }
    }
    float qv[G], gate_v[G], rv[G], sink_v[G];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        qv[hh] = gate_v[hh] = rv[hh] = sink_v[hh] = 0.0f;
        if (hh < gqa) {
            const int h = g*gqa + hh;
            if (tid < HD) {
                qv[hh]     = pr0[(int64_t) h*HD + tid]*args.consts[(int64_t) h*HD + tid];
                gate_v[hh] = pr0[args.off_gate + (int64_t) h*HD + tid];
            } else if (warp == 4 && lane < args.n_rel) {
                rv[hh] = pr0[args.off_rel + (int64_t) h*args.n_rel + lane]*args.consts[args.c_relclip + (int64_t) h*args.n_rel + lane];
            }
            if (warp == 4) {
                sink_v[hh] = pr0[args.off_sink + h];
            }
        }
    }
    const int kv0 = split*args.kv_per_split;
    const int kv1 = min(args.n_kv, kv0 + args.kv_per_split);
    const int per_warp = (kv1 - kv0 + IK_ATTN_WARPS - 1)/IK_ATTN_WARPS;
    const int w0 = kv0 + warp*per_warp;
    const int w1 = min(kv1, w0 + per_warp);
    const bool has_cells = w0 < w1;
    auto load_tile = [&](const int tile, float & mv_l, int & d_l) {
        const int kvl = min(tile + lane, args.n_kv - 1);
        mv_l = tile + lane < w1 ? (args.mask_f16 ? __half2float(((const half *) args.mask)[kvl]) : ((const float *) args.mask)[kvl]) : -INFINITY;
        d_l  = args.rel_pos[kvl];
    };
    float mv_l = -INFINITY;
    int   d_l  = 0;
    if (has_cells) {
        load_tile(w0, mv_l, d_l);
    }

    // ===== round 2: the past inputs (after read_cell) =====
    float x[8];
    {
        const float * past = args.state + (int64_t) read_cell*args.state_s1 + ch*(n_conv - 1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[j] = j < n_conv - 1 ? past[j] : (j == n_conv - 1 ? cur_in : 0.0f);
        }
    }
    IK_DBG_T(1);

    // ===== the K/V convolution, state, post-normalization, cache store; the clipped Q and R =====
    float x_new[7];
    {
        const float scale = 1.0f/sqrtf((args.has_taps ? 2 : 1)*n_conv + 1);
        float out = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < n_conv) {
                out += x[j]*w_c[j];
            }
        }
        if (args.has_taps) {
#pragma unroll
            for (int lag = 0; lag < 8; ++lag) {
                if (lag < n_conv) {
                    out += x[n_conv - 1 - lag]*taps[lag];
                }
            }
        }
        float y = (x[n_conv - 1] + out)*scale;
#pragma unroll
        for (int j = 0; j < 7; ++j) {
            x_new[j] = j < n_conv - 1 ? x[j + 1] : 0.0f;
        }

        // the sums of squares of the K half (warps 0-3) and the V half (warps 4-7), and of Q and R
        float ss = warp_reduce_sum(y*y);
        __shared__ float s_ssq[G][IK_ATTN_WARPS];
        if (lane == 0) {
            s_red[warp] = ss;
        }
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            if (hh < gqa) {
                const float v = warp_reduce_sum(tid < HD ? qv[hh]*qv[hh] : rv[hh]*rv[hh]);
                if (lane == 0) {
                    s_ssq[hh][warp] = v;
                }
            }
        }
        __syncthreads();
        IK_DBG_T(2);
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
            const half hv = __float2half(y);
            if (split == 0) {
                ((half *) row)[ci] = hv;
            }
            yq = __half2float(hv);
        } else {
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
        (which ? s_v : s_k)[ci] = yq;

#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            if (hh < gqa) {
                if (tid < HD) {
                    const float ssq  = s_ssq[hh][0] + s_ssq[hh][1] + s_ssq[hh][2] + s_ssq[hh][3];
                    const float norm = sqrtf(ssq)*1.0f + args.clip_eps;
                    const float f    = fminf(fmaxf(args.qk_clip/norm, 0.0f), 1.0f);
                    s_q[hh][tid] = qv[hh]*f*tau;
                } else if (warp == 4) {
                    const float norm = sqrtf(s_ssq[hh][4])*1.0f + args.clip_eps;
                    const float f    = fminf(fmaxf(args.rel_clip/norm, 0.0f), 1.0f);
                    s_r[hh][lane] = lane < args.n_rel ? rv[hh]*f*tau : 0.0f;
                    if (lane == 0) {
                        s_sink[hh] = sink_v[hh];
                    }
                }
            }
        }
    }
    __syncthreads();
    IK_DBG_T(3);

    // ===== the attention over the cells of this warp (online softmax per head, a lane holds 4 dims) =====
    // the relative bias of a cell is table[d] . R: lane l holds R[l % 16] (the table rows of two cells are loaded by the
    // two half-warps, 16 floats each, and reduced over the half-warps)
    const float * table = args.consts + args.c_table;
    float qreg[G][4], rreg[G], acc[G][4], m[G], l[G];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        m[hh] = -INFINITY;
        l[hh] = 0.0f;
        rreg[hh] = hh < gqa && (lane % 16) < args.n_rel ? s_r[hh][lane % 16] : 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            qreg[hh][e] = hh < gqa ? s_q[hh][4*lane + e] : 0.0f;
            acc[hh][e]  = 0.0f;
        }
    }
    const float sinks[G] = { s_sink[0], s_sink[1], s_sink[2], s_sink[3] };

    if (has_cells && !(args.dbg_skip & 2)) {
        int n_tile = min(WARP_SIZE, w1 - w0);
        for (int tile = w0, u0 = 0; tile < w1; ) {
            // --- the loads of this step: K/V rows and table rows of U cells (clamped indices, in flight together)
            //     the table row of a cell is loaded by both half-warps (lane l: entry l % 16): its contribution to the
            //     score is folded into the dot-product partial sums (with a factor 1/2), so one reduction gives dot + bias
            float kvals[U][4], vvals[U][4], trow[U];
            int   d[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                d[u] = __shfl_sync(0xffffffff, d_l, min(u0 + u, WARP_SIZE - 1));
            }
            if (!(args.dbg_skip & 8)) {
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int kv = min(tile + u0 + u, args.n_kv - 1);
                    ik_attn_load4<cache_t>(args.k_cache + (int64_t) kv*args.kc_nb2 + g*args.kc_nb1, lane, kvals[u]);
                    ik_attn_load4<cache_t>(args.v_cache + (int64_t) kv*args.vc_nb2 + g*args.vc_nb1, lane, vvals[u]);
                }
            } else {
#pragma unroll
                for (int u = 0; u < U; ++u) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        kvals[u][e] = s_k[4*lane + e];
                        vvals[u][e] = s_v[4*lane + e];
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int dc = min(max(d[u], 0), args.n_extent - 1);
                trow[u] = table[(int64_t) dc*args.n_rel + min(lane % 16, args.n_rel - 1)];
                trow[u] = d[u] >= 0 && d[u] < args.n_extent && !(args.dbg_skip & 1) ? 0.5f*trow[u] : 0.0f;
            }
            // --- the scores: kq_scale*(q . k) + bias, reduced over the warp for the 4 (padded) heads with a transposed
            //     butterfly (6 shuffles), then each head's score broadcast from its lane group (lanes 8*hh..8*hh+7)
            float sc[G][U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int kv = tile + u0 + u;
                if (kv == kv_idx) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        kvals[u][e] = s_k[4*lane + e];
                        vvals[u][e] = s_v[4*lane + e];
                    }
                }
                float v[G];
#pragma unroll
                for (int hh = 0; hh < G; ++hh) {
                    v[hh] = hh < gqa ? args.kq_scale*(qreg[hh][0]*kvals[u][0] + qreg[hh][1]*kvals[u][1] + qreg[hh][2]*kvals[u][2] + qreg[hh][3]*kvals[u][3])
                                     + trow[u]*rreg[hh] : 0.0f;
                }
                const bool hi16 = lane >= 16;
                float a0 = hi16 ? v[2] : v[0], a1 = hi16 ? v[3] : v[1];
                const float b0 = hi16 ? v[0] : v[2], b1 = hi16 ? v[1] : v[3];
                a0 += __shfl_xor_sync(0xffffffff, b0, 16);
                a1 += __shfl_xor_sync(0xffffffff, b1, 16);
                const bool hi8 = (lane & 8) != 0;
                float c = hi8 ? a1 : a0;
                c += __shfl_xor_sync(0xffffffff, hi8 ? a0 : a1, 8);
                c += __shfl_xor_sync(0xffffffff, c, 4);
                c += __shfl_xor_sync(0xffffffff, c, 2);
                c += __shfl_xor_sync(0xffffffff, c, 1);
                // lanes 8*hh..8*hh+7 hold the score of head hh
#pragma unroll
                for (int hh = 0; hh < G; ++hh) {
                    if (hh < gqa) {
                        sc[hh][u] = __shfl_sync(0xffffffff, c, 8*hh);
                    }
                }
            }
            // --- the softmax update of the step: the masks, the step maximum per head, then the U cells at once
            float mv[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                mv[u] = u0 + u < n_tile ? __shfl_sync(0xffffffff, mv_l, u0 + u) : -INFINITY;
            }
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                if (hh < gqa) {
                    float m_new = m[hh];
#pragma unroll
                    for (int u = 0; u < U; ++u) {
                        sc[hh][u] = mv[u] == -INFINITY ? -INFINITY : sc[hh][u] - sinks[hh] + mv[u];
                        m_new = fmaxf(m_new, sc[hh][u]);
                    }
                    if (m_new == -INFINITY) {
                        continue; // no cell of this step
                    }
                    const float corr = expf(m[hh] - m_new);
                    l[hh] *= corr;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        acc[hh][e] *= corr;
                    }
#pragma unroll
                    for (int u = 0; u < U; ++u) {
                        const float pk = sc[hh][u] == -INFINITY ? 0.0f : expf(sc[hh][u] - m_new);
                        l[hh] += pk;
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            acc[hh][e] += pk*vvals[u][e];
                        }
                    }
                    m[hh] = m_new;
                }
            }
            // --- next step / tile
            u0 += U;
            if (u0 >= n_tile) {
                tile += WARP_SIZE;
                u0 = 0;
                if (tile < w1) {
                    load_tile(tile, mv_l, d_l);
                    n_tile = min(WARP_SIZE, w1 - tile);
                }
            }
        }
    }
    IK_DBG_T(4);

    // ===== combine the warps =====
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
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
    IK_DBG_T(5);

    // threads 0..127: dim i of every head; M, L of the split (M = -inf: no cell)
    float M[G], L[G], A[G];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        M[hh] = -INFINITY; L[hh] = 0.0f; A[hh] = 0.0f;
        if (hh < gqa && tid < HD) {
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

    if (args.n_split > 1) {
        // publish the partial result, the last block to arrive combines
        if (tid < HD) {
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
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
        IK_DBG_T(6);
        if (!s_last) {
            return;
        }
        __threadfence();
        if (tid < HD) {
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                if (hh < gqa) {
                    const float * part0 = args.part + ((int64_t) g*args.n_split*gqa + hh)*(HD + 2);
                    const int64_t ps = (int64_t) gqa*(HD + 2); // stride between splits
                    // chunks of 8 splits: one round of loads each, merged online
                    float Mt = -INFINITY, Lt = 0.0f, At = 0.0f;
                    for (int sp0 = 0; sp0 < args.n_split; sp0 += 8) {
                        float ms[8], ls[8], as[8];
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const int sp = min(sp0 + k, args.n_split - 1);
                            ms[k] = sp0 + k < args.n_split ? part0[sp*ps + HD] : -INFINITY;
                            ls[k] = part0[sp*ps + HD + 1];
                            as[k] = part0[sp*ps + tid];
                        }
                        float Mc = Mt;
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            Mc = fmaxf(Mc, ms[k]);
                        }
                        if (Mc != -INFINITY) {
                            const float c0 = Mt != -INFINITY ? expf(Mt - Mc) : 0.0f;
                            Lt *= c0;
                            At *= c0;
#pragma unroll
                            for (int k = 0; k < 8; ++k) {
                                if (ms[k] != -INFINITY) {
                                    const float c = expf(ms[k] - Mc);
                                    Lt += ls[k]*c;
                                    At += as[k]*c;
                                }
                            }
                            Mt = Mc;
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

    // ===== the new recurrent state of the K and V channels (every block has read the old state by now) =====
    {
        float * st = args.state + (int64_t) args.write_cell*args.state_s1 + ch*(n_conv - 1);
#pragma unroll
        for (int j = 0; j < 7; ++j) {
            if (j < n_conv - 1) {
                st[j] = x_new[j];
            }
        }
    }

    // ===== the zero sink logit, the normalization and the sigmoid gate; threads 0..127 (warps 0-3) write the heads =====
    if (tid < HD) {
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            if (hh < gqa) {
                const int   h     = g*gqa + hh;
                const float m_new = fmaxf(M[hh], 0.0f);
                const float corr  = expf(M[hh] - m_new);
                const float Lf    = L[hh]*corr + expf(0.0f - m_new);
                const float o     = A[hh]*corr/Lf * (1.0f/(1.0f + expf(-gate_v[hh])));
                if constexpr (q8_out) {
                    ik_warp_store_q8_1(o, (block_q8_1 *) args.dst + ((int64_t) h*HD)/QK8_1 + warp, lane, true);
                } else {
                    ((float *) args.dst)[(int64_t) h*HD + tid] = o;
                }
            }
        }
    }
    IK_DBG_T(7);
}

// ---------------------------------------------------------------------------------------------------------------------
// inkling_attn v2: one block of 256 threads per (kv head, split of the cells). All independent loads are issued at the
// start (the projection values, the constants, the first tile of the cells); the recurrent-state loads (which depend on
// the cell id) are in flight while the warps attend over their cells; the current token's cell is skipped in the loop
// and added last, from the convolved K/V in shared memory. Raw cache words are kept in registers (2 per cell and side)
// so two blocks fit an SM; the last block of a head to arrive combines the splits (acq_rel hand-off).

#define IK_A2_BLOCK 256
#define IK_A2_WARPS (IK_A2_BLOCK/WARP_SIZE)
#define IK_A2_U     8   // cells per warp step

template <typename cache_t>
struct ik_a2_raw;

template <> struct ik_a2_raw<half> {
    uint32_t w[2]; // 4 halves: dims 4*lane .. 4*lane+3
    __device__ __forceinline__ void load(const char * row, const int lane) {
        const uint2 v = *(const uint2 *) (row + 8*lane);
        w[0] = v.x; w[1] = v.y;
    }
    __device__ __forceinline__ void get(float * v) const {
        const float2 a = __half22float2(*(const half2 *) &w[0]);
        const float2 b = __half22float2(*(const half2 *) &w[1]);
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
};

template <> struct ik_a2_raw<block_q8_0> {
    uint32_t q;  // 4 int8 quants
    half     d;
    __device__ __forceinline__ void load(const char * row, const int lane) {
        const block_q8_0 * b = (const block_q8_0 *) row + lane/8;
        ggml_cuda_memcpy_1<4, 2>(&q, b->qs + 4*(lane % 8));
        d = b->d;
    }
    __device__ __forceinline__ void get(float * v) const {
        const float df = __half2float(d);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            v[e] = df*(float) ((int8_t) ((q >> (8*e)) & 0xff));
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Tensor-core scan of the cells (IK_ATTN_MMA, phase 5). The structure (K as the A operand of mma.sync m16n8k16 straight
// from the cache, P V with two dims of four keys per lane, one online softmax per lane group, the mma16816 wrapper and
// the "integer as an FP16 bit pattern" conversion) is adapted from Project Maya's maya-ik2 engine
// (attn_decode_mma_kernel in src/ik2/ik2_kernels.cu, https://github.com/alesha-pro/maya-inkling-mini):
//   MIT License. Copyright (c) 2026 Peasant Smith and the Project Maya contributors,
//   Copyright (c) 2026 Niko1221 and the Strata contributors. See NOTICE.inkling.
// What differs here: the cache stays ggml Q8_0 (no new KV format). The int8 quants become FP16 exactly in registers,
// the per-block scales are applied in FP32 (K: on the block's partial dot; V: folded into P), and Q and P go in as two
// FP16 limbs (hi + lo) in the spare columns of the 8-column operand, so the result equals the FP32 kernel up to the
// order of the sums.

static __device__ __forceinline__ void ik_mma16816(float * c, const uint32_t * a, const uint32_t b0, const uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// two halves with the bits 0x6400 | n (n < 256) are 1024 + n; minus 1152 they are the int8 values n - 128, exactly
static __device__ __forceinline__ uint32_t ik_u8x2_to_h2(const uint32_t biased) {
    uint32_t r;
    asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(biased), "r"(0x64806480u));
    return r;
}
static __device__ __forceinline__ uint32_t ik_pack_h2(const float a, const float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *(const uint32_t *) &h;
}
// the limb of v that an FP16 column carries: the value (rounded to FP16 by the pack), or what the rounding left
static __device__ __forceinline__ float ik_limb(const float v, const bool lo) {
    const float hi = __half2float(__float2half_rn(v)); // (no branch: the lanes of a warp carry both limbs)
    return v - (lo ? hi : 0.0f);
}
// A (cell, kv head) row of a Q8_0 cache: 4 blocks {half d; int8 qs[32]} = 34 aligned words. The blocks 1 and 3 have
// aligned quants. The blocks 0 and 2 are read as the 32-byte stream qs[2..31], qs[0..1]: seven aligned words and an
// eighth made of the low half of the tail word (qs[30..31] | d of the next block) and the high half of the head word
// (d | qs[0..1]). Stream position p of block b is the dim 32 b + (b odd ? p : (p + 2) % 32).
struct ik_q8row {
    const uint32_t * w;
    __device__ __forceinline__ uint32_t head(const int P) const { return w[17*P]; }
    __device__ __forceinline__ uint32_t tail(const int P) const { return w[17*P + 8]; }
    __device__ __forceinline__ uint32_t quad(const int b, const int c, const uint32_t h) const {
        if (b & 1) {
            return w[17*(b >> 1) + 9 + c];
        }
        const uint32_t x = w[17*(b >> 1) + 1 + c]; // c == 7: the tail word
        return c == 7 ? (x & 0xffffu) | (h & 0xffff0000u) : x;
    }
    static __device__ __forceinline__ float scale(const int b, const uint32_t h, const uint32_t t) {
        return __half2float(__ushort_as_half((unsigned short) ((b & 1) ? t >> 16 : h & 0xffffu)));
    }
    static __device__ __forceinline__ int dim(const int b, const int p) {
        return 32*b + ((b & 1) ? p : ((p + 2) & 31));
    }
};

// The scan state of a lane (gr = lane/4, tq = lane%4): the lane holds the logits of the cells gr and gr + 8 of a step
// for the head tq, and 16 dims of that head's output (o, scaled by IK_MMA_PS).
#define IK_MMA_PS 4096.0f // P is scaled up before it becomes FP16 (the V scales are small)
template <int G>
struct ik_mma_lane {
    uint32_t qf[4][2][2]; // the Q operand: [block][half][register]
    float    o[4][2][4];  // the output: the C fragment of each P V product
    float    mm, ll, sink;
};

template <int G>
static __device__ __forceinline__ void ik_mma_init(ik_mma_lane<G> & st, const float (* s_q)[IK_ATTN_HD], const float * s_sink) {
    const int  lane = threadIdx.x % WARP_SIZE;
    const int  gr = lane >> 2, tq = lane & 3;
    const int  hq = min(gr >> 1, G - 1);
    const bool col_ok = (gr >> 1) < G; // operand column gr: head gr/2, limb gr%2
    const bool lo_limb = gr & 1;
    st.mm   = -INFINITY;
    st.ll   = 0.0f;
    st.sink = s_sink[min(tq, G - 1)];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            st.o[b][hf][0] = st.o[b][hf][1] = st.o[b][hf][2] = st.o[b][hf][3] = 0.0f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int p = 16*hf + 4*tq + 2*j;
                const float x0 = s_q[hq][ik_q8row::dim(b, p)], x1 = s_q[hq][ik_q8row::dim(b, p + 1)];
                st.qf[b][hf][j] = col_ok ? ik_pack_h2(ik_limb(x0, lo_limb), ik_limb(x1, lo_limb)) : 0u;
            }
        }
    }
}

// One step: the 16 cells kt .. kt + 15, of which the cells below w1 other than kv_idx (the current token's, not stored
// yet) count. The Q8_0 rows of the kv head are at kb + row*k_nb2 and vb + row*v_nb2, row = min(cell - row0, row_max):
// the cache itself, or a tile of it in shared memory.
template <int G>
static __device__ __forceinline__ void ik_mma_step(const ik_attn_args & args, ik_mma_lane<G> & st, const float (* s_r)[32],
        const char * kb, const int64_t k_nb2, const char * vb, const int64_t v_nb2, const int row0, const int row_max,
        const int kt, const int w1, const int64_t kv_idx) {
    constexpr float PS = IK_MMA_PS;
    const int  lane = threadIdx.x % WARP_SIZE;
    const int  gr = lane >> 2, tq = lane & 3;
    const int  hq = min(gr >> 1, G - 1), ht = min(tq, G - 1);
    const bool col_ok = (gr >> 1) < G, head_ok = tq < G;
    const bool lo_limb = gr & 1;
    const float * table = args.consts + args.c_table;
    const int     n_rel = args.n_rel;
    const int     last  = args.n_kv - 1;

    // ---- every load of the step ----
    uint32_t kh[2][2], kl[2][2], kq[2][4][2];
    float    mk[2];
    int      dk[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int cell = kt + gr + 8*i;
        const int cc   = min(cell, last);
        const ik_q8row r = { (const uint32_t *) (kb + (int64_t) min(cell - row0, row_max)*k_nb2) };
#pragma unroll
        for (int P = 0; P < 2; ++P) {
            kh[i][P] = r.head(P);
            kl[i][P] = r.tail(P);
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                kq[i][b][hf] = r.quad(b, 4*hf + tq, kh[i][b >> 1]);
            }
        }
        const float mv = args.mask_f16 ? __half2float(((const half *) args.mask)[cc]) : ((const float *) args.mask)[cc];
        mk[i] = head_ok && cell < w1 && cell != kv_idx ? mv : -INFINITY;
        dk[i] = args.rel_pos[cc];
    }
    uint32_t vh[4][2], vl[4][2], vq[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int cell = kt + 2*tq + (i & 1) + 8*(i >> 1);
        const ik_q8row r = { (const uint32_t *) (vb + (int64_t) min(cell - row0, row_max)*v_nb2) };
#pragma unroll
        for (int P = 0; P < 2; ++P) {
            vh[i][P] = r.head(P);
            vl[i][P] = r.tail(P);
        }
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            vq[i][b] = r.quad(b, gr, vh[i][b >> 1]);
        }
    }
    // ---- Q K^T: C[16 cells][8 columns] += A[16 cells][16 dims] B[16 dims][8 columns], block by block ----
    float s[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        float cb[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const uint32_t x0 = kq[0][b][hf] ^ 0x80808080u, x1 = kq[1][b][hf] ^ 0x80808080u;
            const uint32_t a[4] = { ik_u8x2_to_h2(__byte_perm(x0, 0x64u, 0x4140u)), ik_u8x2_to_h2(__byte_perm(x1, 0x64u, 0x4140u)),
                                    ik_u8x2_to_h2(__byte_perm(x0, 0x64u, 0x4342u)), ik_u8x2_to_h2(__byte_perm(x1, 0x64u, 0x4342u)) };
            ik_mma16816(cb, a, st.qf[b][hf][0], st.qf[b][hf][1]);
        }
        s[0] += ik_q8row::scale(b, kh[0][b >> 1], kl[0][b >> 1])*(cb[0] + cb[1]);
        s[1] += ik_q8row::scale(b, kh[1][b >> 1], kl[1][b >> 1])*(cb[2] + cb[3]);
    }
    // ---- the logits: scale, relative bias (within the extent), sink, mask ----
    const bool e0 = dk[0] >= 0 && dk[0] < args.n_extent, e1 = dk[1] >= 0 && dk[1] < args.n_extent;
    float bias[2] = { 0.0f, 0.0f };
    if (__ballot_sync(0xffffffffu, e0 || e1)) {
        const float * t0 = table + (int64_t) (e0 ? dk[0] : 0)*n_rel;
        const float * t1 = table + (int64_t) (e1 ? dk[1] : 0)*n_rel;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < n_rel) {
                const float rj = s_r[ht][j];
                bias[0] += t0[j]*rj;
                bias[1] += t1[j]*rj;
            }
        }
        bias[0] = e0 ? bias[0] : 0.0f;
        bias[1] = e1 ? bias[1] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        s[i] = mk[i] == -INFINITY ? -INFINITY : args.kq_scale*s[i] + bias[i] - st.sink + mk[i];
    }
    float mx = fmaxf(s[0], s[1]); // over the lanes of the same head
#pragma unroll
    for (int sh = 4; sh < 32; sh <<= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, sh));
    }
    if (__ballot_sync(0xffffffffu, mx > st.mm)) { // a new maximum: rescale (rare after the first steps)
        const float mn = fmaxf(st.mm, mx);
        const float c  = st.mm == -INFINITY ? 0.0f : expf(st.mm - mn);
        st.mm  = mn;
        st.ll *= c;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                st.o[b][hf][0] *= c; st.o[b][hf][1] *= c; st.o[b][hf][2] *= c; st.o[b][hf][3] *= c;
            }
        }
    }
    const float u  = st.mm == -INFINITY ? 0.0f : st.mm;
    const float p0 = expf(s[0] - u), p1 = expf(s[1] - u);
    st.ll += p0 + p1;
    // ---- P to the lanes of its operand: head gr/2 of the cells 2 tq, 2 tq + 1, 2 tq + 8, 2 tq + 9 ----
    const int src = 8*tq + hq;
    const float pv[4] = { __shfl_sync(0xffffffffu, p0, src), __shfl_sync(0xffffffffu, p0, src + 4),
                          __shfl_sync(0xffffffffu, p1, src), __shfl_sync(0xffffffffu, p1, src + 4) };
    const float pvz[4] = { col_ok ? pv[0] : 0.0f, col_ok ? pv[1] : 0.0f, col_ok ? pv[2] : 0.0f, col_ok ? pv[3] : 0.0f };
    // ---- P V: C[16 dims][8 columns] += A[16 dims][16 cells] B[16 cells][8 columns]; an mma takes 16 dims of one block
    //      (two per lane group), so the block's scales ride on P. A cell without weight may be past the end: no scale. ----
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        float w[4];
        uint32_t el[4], eh[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float d = ik_q8row::scale(b, vh[i][b >> 1], vl[i][b >> 1]);
            w[i] = ik_limb(pvz[i]*(pvz[i] > 0.0f ? d : 0.0f)*PS, lo_limb);
            const uint32_t x = vq[i][b] ^ 0x80808080u;
            el[i] = __byte_perm(x, 0x64u, 0x4140u);
            eh[i] = __byte_perm(x, 0x64u, 0x4342u);
        }
        const uint32_t B0 = ik_pack_h2(w[0], w[1]), B1 = ik_pack_h2(w[2], w[3]);
        {
            const uint32_t a[4] = { ik_u8x2_to_h2(__byte_perm(el[0], el[1], 0x5410u)), ik_u8x2_to_h2(__byte_perm(el[0], el[1], 0x7632u)),
                                    ik_u8x2_to_h2(__byte_perm(el[2], el[3], 0x5410u)), ik_u8x2_to_h2(__byte_perm(el[2], el[3], 0x7632u)) };
            ik_mma16816(st.o[b][0], a, B0, B1);
        }
        {
            const uint32_t a[4] = { ik_u8x2_to_h2(__byte_perm(eh[0], eh[1], 0x5410u)), ik_u8x2_to_h2(__byte_perm(eh[0], eh[1], 0x7632u)),
                                    ik_u8x2_to_h2(__byte_perm(eh[2], eh[3], 0x5410u)), ik_u8x2_to_h2(__byte_perm(eh[2], eh[3], 0x7632u)) };
            ik_mma16816(st.o[b][1], a, B0, B1);
        }
    }
}

// the head's denominator over the lanes; the lane's 16 dims of the head tq to dst[HD]
template <int G>
static __device__ __forceinline__ void ik_mma_finish(ik_mma_lane<G> & st, float * dst) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int gr = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int sh = 4; sh < 32; sh <<= 1) {
        st.ll += __shfl_xor_sync(0xffffffffu, st.ll, sh);
    }
    if (tq < G) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                dst[ik_q8row::dim(b, 4*gr + 2*hf)]     = (st.o[b][hf][0] + st.o[b][hf][1])*(1.0f/IK_MMA_PS);
                dst[ik_q8row::dim(b, 4*gr + 2*hf + 1)] = (st.o[b][hf][2] + st.o[b][hf][3])*(1.0f/IK_MMA_PS);
            }
        }
    }
}

#ifndef IK_A2_MIN_BLOCKS
#define IK_A2_MIN_BLOCKS 1
#endif
template <typename cache_t, bool q8_out, int G, bool MMA = false, bool MR = false, int HAS_TAPS = -1, bool SHORT_SCAN = false>
static __global__ void __launch_bounds__(IK_A2_BLOCK, IK_A2_MIN_BLOCKS) inkling_attn2_kernel(const ik_attn_args args_in) {
    constexpr int HD = IK_ATTN_HD;
    constexpr int U  = IK_A2_U;
    typedef ik_a2_raw<cache_t> raw_t;

    // MR: the block serves row blockIdx.z of a window. Its K/V are in the caches already (inkling_attn_kv_rows_kernel,
    // with the cells of the earlier rows), so this kernel neither convolves nor stores nor touches the recurrent state.
    ik_attn_args args_m;
    if constexpr (MR) {
        const int row = blockIdx.z;
        args_m = args_in;
        args_m.proj    += (int64_t) row*args_in.proj_s1;
        args_m.kv_idxs += row;
        args_m.rel_pos += (int64_t) row*args_in.n_kv;
        args_m.mask     = (const char *) args_in.mask + (int64_t) row*args_in.n_kv*(args_in.mask_f16 ? sizeof(half) : sizeof(float));
        args_m.pos      = args_in.pos ? args_in.pos + row : nullptr;
        args_m.dst      = (char *) args_in.dst + (int64_t) row*args_in.mr_dst_rs;
        if (args_in.part) {
            args_m.part     += (int64_t) row*args_in.mr_part_rs;
            args_m.counters += row*args_in.n_head_kv;
        }
        // Masked cells still count toward the padded cache size after an SWA wrap.
        int row_kv = args_in.n_kv;
        if (args_in.mr_kv_before) {
            row_kv = args_in.mr_kv_before[args_in.has_taps ? 0 : 1];
            for (int r = 0; r <= row; ++r) {
                row_kv = max(row_kv, ((int) args_in.kv_idxs[r] + 256) & ~255);
            }
        }
        if (row_kv < args_in.n_kv) {
            args_m.n_kv = row_kv;
            if (args_in.mr_mma) {
                args_m.n_split = max(1, min(args_in.mr_mma_split, (row_kv + args_in.mr_mma_step - 1)/args_in.mr_mma_step));
            } else {
                const int cps = args_in.mr_medium && row_kv > 512 && row_kv <= 2304 ? 128 : args_in.mr_cps;
                args_m.n_split = row_kv <= cps ? 1 : min(64, (row_kv + cps - 1)/cps);
            }
            args_m.kv_per_split = (row_kv + args_m.n_split - 1)/args_m.n_split;
        }
        if ((int) blockIdx.y >= args_m.n_split) {
            return;
        }
    }
    if constexpr (HAS_TAPS >= 0) {
        if constexpr (!MR) {
            args_m = args_in;
        }
        args_m.n_conv = 4;
        args_m.has_taps = HAS_TAPS;
        args_m.n_head_kv = 8;
        args_m.n_head = 24;
        args_m.n_extent = 1024;
        args_m.n_rel = 16;
        args_m.kc_nb1 = args_m.vc_nb1 = 4*sizeof(block_q8_0);
        args_m.kc_nb2 = args_m.vc_nb2 = 32*sizeof(block_q8_0);
        args_m.ext_scan = 0;
        args_m.dbg_skip = 0;
        args_m.dbg_times = nullptr;
    }
    const ik_attn_args & args = MR || HAS_TAPS >= 0 ? args_m : args_in;

    const int g     = blockIdx.x;
    const int split = blockIdx.y;
    const int tid   = threadIdx.x;
    const int lane  = tid % WARP_SIZE;
    const int warp  = tid / WARP_SIZE;

    __shared__ float s_q[G][HD];
    __shared__ float s_k[HD];
    __shared__ float s_v[HD];
    __shared__ float s_r[G][32];
    __shared__ float s_sink[G];
    __shared__ float s_ssq[G][IK_A2_WARPS];
    __shared__ float s_red[2][IK_A2_WARPS];
    __shared__ float s_m[G][IK_A2_WARPS];
    __shared__ float s_l[G][IK_A2_WARPS];
    __shared__ float s_acc[G][IK_A2_WARPS][HD];
    __shared__ int   s_last;
    IK_DBG_T(0);

    // ===== round 1: every load that does not depend on the recurrent cell =====
    const int     read_cell = args.rs_ids[0];
    const int64_t kv_idx    = args.kv_idxs[0];
    __shared__ uint32_t visible[IK_A2_WARPS];
    if constexpr (SHORT_SCAN) {
        static_assert(IK_A2_BLOCK == 256, "one mask cell per thread");
        const float mask = args.mask_f16 ? __half2float(((const half *) args.mask)[tid]) : ((const float *) args.mask)[tid];
        const unsigned bits = __ballot_sync(0xffffffffu, mask != -INFINITY && tid != kv_idx);
        if (lane == 0) {
            visible[warp] = bits;
        }
    }
    float tau = 1.0f;
    if (args.pos) {
        const float t = args.pos[0] + 1.0f;
        tau = 1.0f + args.tau_alpha*logf(fmaxf(1.0f, t/args.tau_floor));
    }
    const float * pr0    = args.proj; // column 0: Q, R, sink, gate
    const int     which  = tid/HD;    // 0: a K channel, 1: a V channel
    const int     ci     = tid % HD;
    const int     n_conv = args.n_conv;
    const int64_t ch     = (int64_t) which*HD*args.n_head_kv + (int64_t) g*HD + ci; // state channel
    float cur_in, w_c[8], taps[8];
    if constexpr (!MR) {
        const int64_t off  = which ? args.off_v : args.off_k;
        const int64_t offt = which ? args.off_vtaps : args.off_ktaps;
        const int     col  = args.proj_ncols == 3 ? (which ? 2 : 1) : 0;
        const float * pr   = args.proj + col*args.proj_s1;
        const float * w    = args.consts + (which ? args.c_vconv : args.c_kconv) + ((int64_t) g*HD + ci)*n_conv;
        cur_in = pr[off + (int64_t) g*HD + ci];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w_c[j]  = j < n_conv ? w[j] : 0.0f;
            taps[j] = args.has_taps && j < n_conv ? pr[offt + (int64_t) g*n_conv + j] : 0.0f;
        }
    }
    float qv[G], gate_v[G], rv[G], sink_v[G];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        const int h = g*G + hh;
        qv[hh] = gate_v[hh] = rv[hh] = sink_v[hh] = 0.0f;
        if (tid < HD) {
            qv[hh]     = pr0[(int64_t) h*HD + tid]*args.consts[(int64_t) h*HD + tid];
            gate_v[hh] = pr0[args.off_gate + (int64_t) h*HD + tid];
        } else if (warp == 4) {
            if (lane < args.n_rel) {
                rv[hh] = pr0[args.off_rel + (int64_t) h*args.n_rel + lane]*args.consts[args.c_relclip + (int64_t) h*args.n_rel + lane];
            }
            sink_v[hh] = pr0[args.off_sink + h];
        }
    }
    // the cells of this warp
    const int kv0 = args.ext_scan ? 0 : split*args.kv_per_split;
    const int kv1 = args.ext_scan ? 0 : min(args.n_kv, kv0 + args.kv_per_split);
    const int per_warp = (kv1 - kv0 + IK_A2_WARPS - 1)/IK_A2_WARPS;
    const int w0 = kv0 + warp*per_warp;
    const int w1 = min(kv1, w0 + per_warp);
    const bool has_cells = w0 < w1;
    auto load_tile = [&](const int tile, float & mv_l, int & d_l) {
        const int kvl = min(tile + lane, args.n_kv - 1);
        mv_l = tile + lane < w1 ? (args.mask_f16 ? __half2float(((const half *) args.mask)[kvl]) : ((const float *) args.mask)[kvl]) : -INFINITY;
        d_l  = args.rel_pos[kvl];
    };
    float mv_l = -INFINITY;
    int   d_l  = 0;
    if (!MMA && has_cells) {
        load_tile(w0, mv_l, d_l);
    }
    // the first step's raw K/V and table rows
    const float * table = args.consts + args.c_table;
    raw_t kraw[U], vraw[U];
    float trow[U];
    auto load_step = [&](const int tile, const int u0) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int kv = min(tile + u0 + u, args.n_kv - 1);
            kraw[u].load(args.k_cache + (int64_t) kv*args.kc_nb2 + g*args.kc_nb1, lane);
            vraw[u].load(args.v_cache + (int64_t) kv*args.vc_nb2 + g*args.vc_nb1, lane);
            const int d  = __shfl_sync(0xffffffff, d_l, min(u0 + u, WARP_SIZE - 1));
            const int dc = min(max(d, 0), args.n_extent - 1);
            trow[u] = table[(int64_t) dc*args.n_rel + min(lane % 16, args.n_rel - 1)];
            trow[u] = d >= 0 && d < args.n_extent ? 0.5f*trow[u] : 0.0f;
        }
    };
    IK_DBG_T(3); // round-1 loads issued
    if (!MMA && has_cells) {
        load_step(w0, 0);
    }
    IK_DBG_T(1); // the first tile's distances landed (the table rows need them)

    // ===== round 2: the past inputs of the convolution (after the cell id), consumed after the cell loop =====
    float x[8];
    if constexpr (!MR) {
        const float * past = args.state + (int64_t) read_cell*args.state_s1 + ch*(n_conv - 1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[j] = j < n_conv - 1 ? past[j] : (j == n_conv - 1 ? cur_in : 0.0f);
        }
    }

    // ===== the clipped Q and R of the heads (block reductions), scaled by tau =====
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        const float v = warp_reduce_sum(tid < HD ? qv[hh]*qv[hh] : rv[hh]*rv[hh]);
        if (lane == 0) {
            s_ssq[hh][warp] = v;
        }
    }
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        if (tid < HD) {
            const float ssq  = s_ssq[hh][0] + s_ssq[hh][1] + s_ssq[hh][2] + s_ssq[hh][3];
            const float norm = sqrtf(ssq) + args.clip_eps;
            const float f    = fminf(fmaxf(args.qk_clip/norm, 0.0f), 1.0f);
            s_q[hh][tid] = qv[hh]*f*tau;
        } else if (warp == 4) {
            const float norm = sqrtf(s_ssq[hh][4]) + args.clip_eps;
            const float f    = fminf(fmaxf(args.rel_clip/norm, 0.0f), 1.0f);
            s_r[hh][lane] = lane < args.n_rel ? rv[hh]*f*tau : 0.0f;
            if (lane == 0) {
                s_sink[hh] = sink_v[hh];
            }
        }
    }
    __syncthreads();
    IK_DBG_T(2);

    // ===== the attention over the cells of this warp (online softmax per head, a lane holds 4 dims), the current
    //       token's cell skipped (added last from the convolved K/V) =====
    float qreg[G][4], rreg[G], acc[G][4], m[G], l[G], sinks[G];
    auto init_lane_state = [&]() {
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            m[hh] = -INFINITY;
            l[hh] = 0.0f;
            rreg[hh]  = (lane % 16) < args.n_rel ? s_r[hh][lane % 16] : 0.0f;
            sinks[hh] = s_sink[hh];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                qreg[hh][e] = s_q[hh][4*lane + e];
                acc[hh][e]  = 0.0f;
            }
        }
    };
    if constexpr (!MMA) {
        init_lane_state();
    }
    // one step of U cells with the raw words in registers; the scores and the softmax update
    auto step = [&](const int tile, const int u0, const int n_tile, const bool skip_cur) {
        float sc[G][U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            float kv4[4];
            kraw[u].get(kv4);
            float v[G];
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                v[hh] = args.kq_scale*(qreg[hh][0]*kv4[0] + qreg[hh][1]*kv4[1] + qreg[hh][2]*kv4[2] + qreg[hh][3]*kv4[3]) + trow[u]*rreg[hh];
            }
            // transposed butterfly over 4 (padded) heads: lanes 8*hh..8*hh+7 hold the score of head hh
            const bool hi16 = lane >= 16;
            const float v2 = G > 2 ? v[2] : 0.0f, v3 = G > 3 ? v[3] : 0.0f;
            float a0 = hi16 ? v2 : v[0], a1 = hi16 ? v3 : (G > 1 ? v[1] : 0.0f);
            const float b0 = hi16 ? v[0] : v2, b1 = hi16 ? (G > 1 ? v[1] : 0.0f) : v3;
            a0 += __shfl_xor_sync(0xffffffff, b0, 16);
            a1 += __shfl_xor_sync(0xffffffff, b1, 16);
            const bool hi8 = (lane & 8) != 0;
            float c = hi8 ? a1 : a0;
            c += __shfl_xor_sync(0xffffffff, hi8 ? a0 : a1, 8);
            c += __shfl_xor_sync(0xffffffff, c, 4);
            c += __shfl_xor_sync(0xffffffff, c, 2);
            c += __shfl_xor_sync(0xffffffff, c, 1);
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                sc[hh][u] = __shfl_sync(0xffffffff, c, 8*hh);
            }
        }
        float mv[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int kv = tile + u0 + u;
            mv[u] = u0 + u < n_tile && !(skip_cur && kv == kv_idx) ? __shfl_sync(0xffffffff, mv_l, u0 + u) : -INFINITY;
        }
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            float m_new = m[hh];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                sc[hh][u] = mv[u] == -INFINITY ? -INFINITY : sc[hh][u] - sinks[hh] + mv[u];
                m_new = fmaxf(m_new, sc[hh][u]);
            }
            if (m_new == -INFINITY) {
                continue;
            }
            const float corr = expf(m[hh] - m_new);
            l[hh] *= corr;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[hh][e] *= corr;
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const float pk = sc[hh][u] == -INFINITY ? 0.0f : expf(sc[hh][u] - m_new);
                l[hh] += pk;
                float vv[4];
                vraw[u].get(vv);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[hh][e] += pk*vv[e];
                }
            }
            m[hh] = m_new;
        }
    };
    if (!MMA && has_cells && !(args.dbg_skip & 2)) {
        int n_tile = min(WARP_SIZE, w1 - w0);
        int tile = w0, u0 = 0;
        while (true) {
            // the next step's loads are issued after this step consumed the registers
            const int tile_c = tile, u0_c = u0, n_tile_c = n_tile;
            u0 += U;
            bool more = true;
            if (u0 >= n_tile) {
                tile += WARP_SIZE;
                u0 = 0;
                if (tile < w1) {
                    n_tile = min(WARP_SIZE, w1 - tile);
                } else {
                    more = false;
                }
            }
            step(tile_c, u0_c, n_tile_c, true);
            if (!more) {
                break;
            }
            if (u0 == 0) {
                load_tile(tile, mv_l, d_l);
            }
            load_step(tile, u0);
        }
    }
    if constexpr (MMA) {
        // ===== the tensor-core scan of the warp's cells =====
        const int ht = min(lane & 3, G - 1);
        ik_mma_lane<G> st;
        ik_mma_init<G>(st, s_q, s_sink);
        if (has_cells && !(args.dbg_skip & 2)) {
            const char * kb = args.k_cache + g*args.kc_nb1;
            const char * vb = args.v_cache + g*args.vc_nb1;
            if constexpr (SHORT_SCAN) {
                // Each warp takes two tiles. Skip tiles that only contain masked cells or the current token.
#pragma unroll
                for (int tile = warp; tile < 16; tile += IK_A2_WARPS) {
                    if (visible[tile/2] & (0xffffu << (16*(tile%2)))) {
                        ik_mma_step<G>(args, st, s_r, kb, args.kc_nb2, vb, args.vc_nb2, 0, 255, 16*tile, 16*tile + 16, kv_idx);
                    }
                }
            } else {
                for (int kt = w0; kt < w1; kt += 16) {
                    ik_mma_step<G>(args, st, s_r, kb, args.kc_nb2, vb, args.vc_nb2, 0, args.n_kv - 1, kt, w1, kv_idx);
                }
            }
        }
        // the warp's partial result
        ik_mma_finish<G>(st, s_acc[ht][warp]);
        if (lane < G) {
            s_m[lane][warp] = st.mm;
            s_l[lane][warp] = st.ll;
        }
    }
    IK_DBG_T(4);

    // ===== the K/V convolution of the token, the post-normalization, the cache store (split 0), the new state =====
    float x_new[7];
    if constexpr (MR) {
        const char * row = (which ? args.v_cache : args.k_cache) + kv_idx*(which ? args.vc_nb2 : args.kc_nb2) + g*(which ? args.vc_nb1 : args.kc_nb1);
        float yq;
        if constexpr (std::is_same<cache_t, half>::value) {
            yq = __half2float(((const half *) row)[ci]);
        } else {
            const block_q8_0 * b = (const block_q8_0 *) row + (warp % 4);
            const int q = b->qs[lane];
            yq = __half2float(b->d)*q;
        }
        (which ? s_v : s_k)[ci] = yq;
    } else {
        const float scale = 1.0f/sqrtf((args.has_taps ? 2 : 1)*n_conv + 1);
        float out = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < n_conv) {
                out += x[j]*w_c[j];
            }
        }
        if (args.has_taps) {
#pragma unroll
            for (int lag = 0; lag < 8; ++lag) {
                if (lag < n_conv) {
                    out += x[n_conv - 1 - lag]*taps[lag];
                }
            }
        }
        float y = (x[n_conv - 1] + out)*scale;
#pragma unroll
        for (int j = 0; j < 7; ++j) {
            x_new[j] = j < n_conv - 1 ? x[j + 1] : 0.0f;
        }
        const float ss = warp_reduce_sum(y*y);
        if (lane == 0) {
            s_red[which][warp % 4] = ss;
        }
        __syncthreads();
        const float sst = s_red[which][0] + s_red[which][1] + s_red[which][2] + s_red[which][3];
        float factor;
        if (which == 0) {
            const float norm = sqrtf(sst) + args.clip_eps;
            factor = fminf(fmaxf(args.qk_clip/norm, 0.0f), 1.0f);
        } else {
            factor = rsqrtf(sst/HD + args.norm_eps);
        }
        y *= factor;
        char * row = (which ? args.v_cache : args.k_cache) + kv_idx*(which ? args.vc_nb2 : args.kc_nb2) + g*(which ? args.vc_nb1 : args.kc_nb1);
        float yq;
        if constexpr (std::is_same<cache_t, half>::value) {
            const half hv = __float2half(y);
            if (split == 0) {
                ((half *) row)[ci] = hv;
            }
            yq = __half2float(hv);
        } else {
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
        (which ? s_v : s_k)[ci] = yq;
    }
    __syncthreads();

    // ===== the current token's cell, by the warp whose range holds it =====
    if constexpr (MMA) {
        init_lane_state();
    }
    const bool owns_current = SHORT_SCAN ? (kv_idx/16)%IK_A2_WARPS == warp : args.ext_scan ? warp == 0 : (kv_idx >= w0 && kv_idx < w1);
    if (owns_current && !(args.dbg_skip & 2)) {
        float kv4[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            kv4[e] = s_k[4*lane + e];
        }
        // the table row of distance 0 is folded like in the loop (both half-warps, factor 1/2)
        const float t0 = args.n_extent > 0 ? 0.5f*table[min(lane % 16, args.n_rel - 1)] : 0.0f;
        float v[G];
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            v[hh] = args.kq_scale*(qreg[hh][0]*kv4[0] + qreg[hh][1]*kv4[1] + qreg[hh][2]*kv4[2] + qreg[hh][3]*kv4[3]) + t0*rreg[hh];
        }
        const bool hi16 = lane >= 16;
        const float v2 = G > 2 ? v[2] : 0.0f, v3 = G > 3 ? v[3] : 0.0f;
        float a0 = hi16 ? v2 : v[0], a1 = hi16 ? v3 : (G > 1 ? v[1] : 0.0f);
        const float b0 = hi16 ? v[0] : v2, b1 = hi16 ? (G > 1 ? v[1] : 0.0f) : v3;
        a0 += __shfl_xor_sync(0xffffffff, b0, 16);
        a1 += __shfl_xor_sync(0xffffffff, b1, 16);
        const bool hi8 = (lane & 8) != 0;
        float c = hi8 ? a1 : a0;
        c += __shfl_xor_sync(0xffffffff, hi8 ? a0 : a1, 8);
        c += __shfl_xor_sync(0xffffffff, c, 4);
        c += __shfl_xor_sync(0xffffffff, c, 2);
        c += __shfl_xor_sync(0xffffffff, c, 1);
        const float mvc = args.mask_f16 ? __half2float(((const half *) args.mask)[kv_idx]) : ((const float *) args.mask)[kv_idx];
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            const float s = __shfl_sync(0xffffffff, c, 8*hh) - sinks[hh] + mvc;
            if (mvc != -INFINITY) {
                const float m_new = fmaxf(m[hh], s);
                const float corr  = expf(m[hh] - m_new);
                const float pk    = expf(s - m_new);
                l[hh] = l[hh]*corr + pk;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[hh][e] = acc[hh][e]*corr + pk*s_v[4*lane + e];
                }
                m[hh] = m_new;
            }
        }
    }

    // ===== combine the warps =====
    if constexpr (MMA) {
        // the scan left the warp's partial in shared memory; the warp with the current cell merges it in
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            if (m[hh] != -INFINITY) {
                const float mo = s_m[hh][warp], lo = s_l[hh][warp];
                const float mn = fmaxf(mo, m[hh]);
                const float co = mo == -INFINITY ? 0.0f : expf(mo - mn);
                const float cn = expf(m[hh] - mn);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    s_acc[hh][warp][4*lane + e] = (mo == -INFINITY ? 0.0f : s_acc[hh][warp][4*lane + e]*co) + acc[hh][e]*cn;
                }
                if (lane == 0) {
                    s_m[hh][warp] = mn;
                    s_l[hh][warp] = lo*co + l[hh]*cn;
                }
            }
        }
    } else {
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
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
    IK_DBG_T(5);

    float M[G], L[G], A[G];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        M[hh] = -INFINITY; L[hh] = 0.0f; A[hh] = 0.0f;
        if (tid < HD) {
#pragma unroll
            for (int w = 0; w < IK_A2_WARPS; ++w) {
                M[hh] = fmaxf(M[hh], s_m[hh][w]);
            }
#pragma unroll
            for (int w = 0; w < IK_A2_WARPS; ++w) {
                if (s_m[hh][w] != -INFINITY) {
                    const float c = expf(s_m[hh][w] - M[hh]);
                    L[hh] += s_l[hh][w]*c;
                    A[hh] += s_acc[hh][w][tid]*c;
                }
            }
        }
    }

    if (args.n_split > 1) {
        // publish the partial result; the last block of the head to arrive combines
        if (tid < HD) {
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                float * part = args.part + (((int64_t) g*args.n_split + (args.ext_scan ? args.n_split - 1 : split))*G + hh)*(HD + 2);
                part[tid] = A[hh];
                if (tid == 0) {
                    part[HD]     = M[hh];
                    part[HD + 1] = L[hh];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            s_last = args.ext_scan ? 1 : ik_atomic_add_acq_rel(&args.counters[g], 1) == args.n_split - 1;
        }
        __syncthreads();
        IK_DBG_T(6);
        if (!s_last) {
            return;
        }
        if (tid < HD) {
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                const float * part0 = args.part + ((int64_t) g*args.n_split*G + hh)*(HD + 2);
                const int64_t ps = (int64_t) G*(HD + 2); // stride between splits
                float Mt = -INFINITY, Lt = 0.0f, At = 0.0f;
                for (int sp0 = 0; sp0 < args.n_split; sp0 += 8) {
                    float ms[8], ls[8], as[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int sp = min(sp0 + k, args.n_split - 1);
                        ms[k] = sp0 + k < args.n_split ? part0[sp*ps + HD] : -INFINITY;
                        ls[k] = part0[sp*ps + HD + 1];
                        as[k] = part0[sp*ps + tid];
                    }
                    float Mc = Mt;
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        Mc = fmaxf(Mc, ms[k]);
                    }
                    if (Mc != -INFINITY) {
                        const float c0 = Mt != -INFINITY ? expf(Mt - Mc) : 0.0f;
                        Lt *= c0;
                        At *= c0;
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            if (ms[k] != -INFINITY) {
                                const float c = expf(ms[k] - Mc);
                                Lt += ls[k]*c;
                                At += as[k]*c;
                            }
                        }
                        Mt = Mc;
                    }
                }
                M[hh] = Mt; L[hh] = Lt; A[hh] = At;
            }
        }
        if (tid == 0 && !args.ext_scan) {
            args.counters[g] = 0;
        }
    }

    // ===== the new recurrent state of the K and V channels (every block has read the old state by now) =====
    if constexpr (!MR) {
        float * st = args.state + (int64_t) args.write_cell*args.state_s1 + ch*(n_conv - 1);
#pragma unroll
        for (int j = 0; j < 7; ++j) {
            if (j < n_conv - 1) {
                st[j] = x_new[j];
            }
        }
    }

    // ===== the zero sink logit, the normalization and the sigmoid gate; threads 0..127 write the heads =====
    if (tid < HD) {
#pragma unroll
        for (int hh = 0; hh < G; ++hh) {
            const int   h     = g*G + hh;
            const float m_new = fmaxf(M[hh], 0.0f);
            const float corr  = expf(M[hh] - m_new);
            const float Lf    = L[hh]*corr + expf(0.0f - m_new);
            const float o     = A[hh]*corr/Lf * (1.0f/(1.0f + expf(-gate_v[hh])));
            if constexpr (q8_out) {
                ik_warp_store_q8_1(o, (block_q8_1 *) args.dst + ((int64_t) h*HD)/QK8_1 + warp, lane, true);
            } else {
                ((float *) args.dst)[(int64_t) h*HD + tid] = o;
            }
        }
    }
    IK_DBG_T(7);
}

// ---------------------------------------------------------------------------------------------------------------------
// inkling_attn_scan: the scan of a long Q8_0 cache as its own launch. A (cell, kv head) row is 136 bytes, so it lies
// across two 128-byte lines that it shares with the rows of the neighbouring heads. With one block per kv head (the
// fused kernel) those lines are fetched once per head by different SMs at different times: about twice the bytes of
// the cache go over the memory bus once a layer's cache is larger than the L2. Here a block takes a range of cells for
// every kv head: warp g scans head g, the warps advance over the same cells together, and a line is fetched once.
// The blocks leave their parts where the fused kernel (ext_scan) picks them up as if they were its own splits.
#define IK_SCAN_MAX_KV 8
#define IK_SCAN_TILE   (IK_SCAN_MAX_KV*4*(int) sizeof(block_q8_0)) // uint4 words of 16 cells
#ifndef IK_SCAN_MIN_BLOCKS
#define IK_SCAN_MIN_BLOCKS 2
#endif
template <int G>
static __global__ void __launch_bounds__(IK_SCAN_MAX_KV*WARP_SIZE, IK_SCAN_MIN_BLOCKS) inkling_attn_scan_kernel(const ik_attn_args args) {
    constexpr int HD = IK_ATTN_HD;
    const int split = blockIdx.x;
    const int tid   = threadIdx.x;
    const int lane  = tid % WARP_SIZE;
    const int g     = tid / WARP_SIZE; // the warp's kv head

    // The 16 cells of a step, all heads: read with 16-byte loads in cache order (the 4-byte loads of the fragments
    // straight from the cache reach 2/3 of the bandwidth of this), then taken apart from shared memory.
    __shared__ uint4 s_tk[IK_SCAN_TILE];
    __shared__ uint4 s_tv[IK_SCAN_TILE];
    __shared__ float s_r[IK_SCAN_MAX_KV][G][32];
    __shared__ float s_sink[IK_SCAN_MAX_KV][G];
    float (* s_q)[G][HD] = (float (*)[G][HD]) s_tk; // the clipped Q, until the lanes have their operands

    const int64_t kv_idx = args.kv_idxs[0];
    float tau = 1.0f;
    if (args.pos) {
        const float t = args.pos[0] + 1.0f;
        tau = 1.0f + args.tau_alpha*logf(fmaxf(1.0f, t/args.tau_floor));
    }
    // the clipped Q and R of the warp's heads, scaled by tau (a lane has 4 dims of Q and one of R); warp-local
    const float * pr0 = args.proj;
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        const int h = g*G + hh;
        const float4 q4 = *(const float4 *) (pr0 + (int64_t) h*HD + 4*lane);
        const float4 c4 = *(const float4 *) (args.consts + (int64_t) h*HD + 4*lane);
        const float  qv[4] = { q4.x*c4.x, q4.y*c4.y, q4.z*c4.z, q4.w*c4.w };
        const float  rv = lane < args.n_rel ? pr0[args.off_rel + (int64_t) h*args.n_rel + lane]*args.consts[args.c_relclip + (int64_t) h*args.n_rel + lane] : 0.0f;
        const float  ssq = warp_reduce_sum(qv[0]*qv[0] + qv[1]*qv[1] + qv[2]*qv[2] + qv[3]*qv[3]);
        const float  ssr = warp_reduce_sum(rv*rv);
        const float  fq  = fminf(fmaxf(args.qk_clip/(sqrtf(ssq) + args.clip_eps), 0.0f), 1.0f);
        const float  fr  = fminf(fmaxf(args.rel_clip/(sqrtf(ssr) + args.clip_eps), 0.0f), 1.0f);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            s_q[g][hh][4*lane + e] = qv[e]*fq*tau;
        }
        s_r[g][hh][lane] = lane < args.n_rel ? rv*fr*tau : 0.0f;
        if (lane == 0) {
            s_sink[g][hh] = pr0[args.off_sink + h];
        }
    }
    __syncwarp();
    ik_mma_lane<G> st;
    ik_mma_init<G>(st, s_q[g], s_sink[g]);
    __syncthreads(); // the tiles take the place of Q

    const int kv0 = split*args.kv_per_split;
    const int kv1 = min(args.n_kv, kv0 + args.kv_per_split);
    const int     n_thr  = args.n_head_kv*WARP_SIZE;
    const int     n_tile = (int) args.kc_nb2;                           // uint4 words of 16 cells
    const int64_t end    = (int64_t) args.n_kv*args.kc_nb2 - 16;        // the last word of the cache
    const char *  kb     = (const char *) s_tk + g*args.kc_nb1;
    const char *  vb     = (const char *) s_tv + g*args.vc_nb1;
    if (!(args.dbg_skip & 2)) {
        for (int kt = kv0; kt < kv1; kt += 16) {
            const int64_t off = (int64_t) kt*args.kc_nb2;
            uint4 tk[5], tv[5];
#pragma unroll
            for (int j = 0; j < 5; ++j) { // 136/32 words a thread
                const int     c = min(j*n_thr + tid, n_tile - 1);
                const int64_t o = min(off + 16*(int64_t) c, end);       // (past the end: cells that the mask leaves out)
                tk[j] = *(const uint4 *) (args.k_cache + o);
                tv[j] = *(const uint4 *) (args.v_cache + o);
            }
#pragma unroll
            for (int j = 0; j < 5; ++j) {
                const int c = j*n_thr + tid;
                if (c < n_tile) {
                    s_tk[c] = tk[j];
                    s_tv[c] = tv[j];
                }
            }
            __syncthreads();
            ik_mma_step<G>(args, st, s_r[g], kb, args.kc_nb2, vb, args.vc_nb2, kt, 15, kt, kv1, kv_idx);
            __syncthreads();
        }
    }
    float * part = args.part + (((int64_t) g*args.n_split + split)*G + min(lane & 3, G - 1))*(HD + 2);
    ik_mma_finish<G>(st, part);
    if (lane < G) {
        float * part_h = args.part + (((int64_t) g*args.n_split + split)*G + lane)*(HD + 2);
        part_h[HD]     = st.mm;
        part_h[HD + 1] = st.ll;
    }
}

static void inkling_attn_scan_launch(const ik_attn_args & a, const int n_blocks, cudaStream_t stream) {
    const dim3 grid(n_blocks, 1, 1);
    const int  block = a.n_head_kv*WARP_SIZE;
    switch (a.n_head/a.n_head_kv) {
        case 1: inkling_attn_scan_kernel<1><<<grid, block, 0, stream>>>(a); break;
        case 2: inkling_attn_scan_kernel<2><<<grid, block, 0, stream>>>(a); break;
        case 3: inkling_attn_scan_kernel<3><<<grid, block, 0, stream>>>(a); break;
        default: inkling_attn_scan_kernel<4><<<grid, block, 0, stream>>>(a); break;
    }
}

template <typename cache_t, bool q8_out, bool MMA = false, bool MR = false>
static void inkling_attn2_launch(const ik_attn_args & a, const dim3 grid, cudaStream_t stream) {
    switch (a.n_head/a.n_head_kv) {
        case 1: inkling_attn2_kernel<cache_t, q8_out, 1, MMA, MR><<<grid, IK_A2_BLOCK, 0, stream>>>(a); break;
        case 2: inkling_attn2_kernel<cache_t, q8_out, 2, MMA, MR><<<grid, IK_A2_BLOCK, 0, stream>>>(a); break;
        case 3: inkling_attn2_kernel<cache_t, q8_out, 3, MMA, MR><<<grid, IK_A2_BLOCK, 0, stream>>>(a); break;
        default: inkling_attn2_kernel<cache_t, q8_out, 4, MMA, MR><<<grid, IK_A2_BLOCK, 0, stream>>>(a); break;
    }
}

// The K and V of every row of a window: the convolution over [state, the rows so far], the post-normalization, the cache
// store and the rollback planes of the recurrent state. One block per kv head, a thread per K or V channel walks the rows
// in order, so each value is computed exactly as the single-token kernel computes it (same statements, same reductions).
template <typename cache_t, int HAS_TAPS = -1> // HAS_TAPS >= 0: the shape constants of the specialized single-token instance
static __global__ void __launch_bounds__(2*IK_ATTN_HD) inkling_attn_kv_rows_kernel(const ik_attn_args args) {
    constexpr int HD = IK_ATTN_HD;
    const int g      = blockIdx.x;
    const int tid    = threadIdx.x;
    const int lane   = tid % WARP_SIZE;
    const int warp   = tid / WARP_SIZE;
    const int which  = tid/HD;    // 0: a K channel, 1: a V channel
    const int ci     = tid % HD;
    const int  n_conv   = HAS_TAPS >= 0 ? 4 : args.n_conv;
    const bool has_taps = HAS_TAPS >= 0 ? HAS_TAPS != 0 : args.has_taps != 0;
    __shared__ float s_red[2][IK_A2_WARPS];

    const int64_t ch   = (int64_t) which*HD*args.n_head_kv + (int64_t) g*HD + ci; // state channel
    const int64_t off  = which ? args.off_v : args.off_k;
    const int64_t offt = which ? args.off_vtaps : args.off_ktaps;
    const int     col  = args.proj_ncols == 3 ? (which ? 2 : 1) : 0;
    const float * w    = args.consts + (which ? args.c_vconv : args.c_kconv) + ((int64_t) g*HD + ci)*n_conv;
    float w_c[8], x[8];
    {
        const float * past = args.state + (int64_t) args.rs_ids[0]*args.state_s1 + ch*(n_conv - 1);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w_c[j] = j < n_conv ? w[j] : 0.0f;
            x[j]   = j < n_conv - 1 ? past[j] : 0.0f;
        }
    }
    const float scale = 1.0f/sqrtf((has_taps ? 2 : 1)*n_conv + 1);
    for (int row = 0; row < args.mr_rows; ++row) {
        const float * pr = args.proj + ((int64_t) col*args.mr_rows + row)*args.proj_s1;
        const float cur_in = pr[off + (int64_t) g*HD + ci];
        float taps[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            taps[j] = has_taps && j < n_conv ? pr[offt + (int64_t) g*n_conv + j] : 0.0f;
            if (j == n_conv - 1) {
                x[j] = cur_in;
            }
        }
        float out = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < n_conv) {
                out += x[j]*w_c[j];
            }
        }
        if (has_taps) {
#pragma unroll
            for (int lag = 0; lag < 8; ++lag) {
                if (lag < n_conv) {
                    out += x[n_conv - 1 - lag]*taps[lag];
                }
            }
        }
        float y = (x[n_conv - 1] + out)*scale;
        const float ss = warp_reduce_sum(y*y);
        if (lane == 0) {
            s_red[which][warp % 4] = ss;
        }
        __syncthreads();
        const float sst = s_red[which][0] + s_red[which][1] + s_red[which][2] + s_red[which][3];
        float factor;
        if (which == 0) {
            const float norm = sqrtf(sst) + args.clip_eps;
            factor = fminf(fmaxf(args.qk_clip/norm, 0.0f), 1.0f);
        } else {
            factor = rsqrtf(sst/HD + args.norm_eps);
        }
        y *= factor;
        const int64_t kv_idx = args.kv_idxs[row];
        char * crow = (which ? args.v_cache : args.k_cache) + kv_idx*(which ? args.vc_nb2 : args.kc_nb2) + g*(which ? args.vc_nb1 : args.kc_nb1);
        if constexpr (std::is_same<cache_t, half>::value) {
            ((half *) crow)[ci] = __float2half(y);
        } else {
            const float amax = warp_reduce_max(fabsf(y));
            const float d    = amax/127.0f;
            const float id   = d != 0.0f ? 1.0f/d : 0.0f;
            const int   q    = (int) roundf(y*id);
            block_q8_0 * b = (block_q8_0 *) crow + (warp % 4);
            b->qs[lane] = (int8_t) q;
            if (lane == 0) {
                b->d = __float2half(d);
            }
        }
        __syncthreads(); // s_red is reused by the next row
        const int k = args.mr_rows - 1 - row;
        if (k < args.mr_planes) {
            float * st = args.state + ((int64_t) args.write_cell + (int64_t) k*args.mr_plane_cells)*args.state_s1 + ch*(n_conv - 1);
#pragma unroll
            for (int j = 0; j < 7; ++j) {
                if (j < n_conv - 1) {
                    st[j] = x[j + 1];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 7; ++j) {
            x[j] = x[j + 1];
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
    GGML_ASSERT(a.n_rel <= 16 && a.n_conv <= 8);

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

    const int n_rows = dst->ne[1];
    a.mr_kv_before   = dst->src[10] ? (const int32_t *) dst->src[10]->data : nullptr;
    a.mr_rows        = n_rows;
    a.mr_planes      = n_rows > 1 ? (ggml_get_op_params_i32(dst, 14) & 0xff) : 1;
    a.mr_plane_cells = ggml_get_op_params_i32(dst, 15);
    a.mr_dst_rs      = dst->nb[1];
    a.mr_part_rs     = 0;
    a.proj       = (const float *) proj->data; a.proj_s1 = proj->nb[1]/sizeof(float); a.proj_ncols = proj->ne[1]/n_rows;
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

    // At medium context on GA10x, fewer partial results outweigh the extra Q8 cells per block.
    static const bool v1 = getenv("IK_ATTN_V1") && atoi(getenv("IK_ATTN_V1")) != 0;
    static const int split_override = getenv("IK_ATTN_CELLS_PER_SPLIT") ? atoi(getenv("IK_ATTN_CELLS_PER_SPLIT")) : 0;
    const bool medium_q8 = ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_AMPERE_GA10X &&
            k_cache->type == GGML_TYPE_Q8_0 && a.n_kv > 512 && a.n_kv <= 2304;
    const int cells_per_split = split_override > 0 ? split_override : v1 ? 32 : medium_q8 ? 128 : 64;
    static const int dbg_skip        = getenv("IK_ATTN_SKIP") ? atoi(getenv("IK_ATTN_SKIP")) : 0;
    a.dbg_skip     = dbg_skip;
    static long long * dbg_times = [] () -> long long * {
        if (!getenv("IK_ATTN_TIMING")) {
            return nullptr;
        }
        long long * t = nullptr;
        CUDA_CHECK(cudaMallocManaged(&t, 16*sizeof(long long)));
        memset(t, 0, 16*sizeof(long long));
        static long long * t_static = t;
        atexit([] {
            cudaDeviceSynchronize();
            fprintf(stderr, "inkling_attn phases of block (0,0) (cycles from T0): loads issued %lld, reductions done %lld, prologue done %lld, loop done %lld, warps combined %lld, arrival counted %lld, end %lld\n",
                t_static[1]-t_static[0], t_static[2]-t_static[0], t_static[3]-t_static[0], t_static[4]-t_static[0], t_static[5]-t_static[0], t_static[6]-t_static[0], t_static[7]-t_static[0]);
        });
        return t;
    }();
    a.dbg_times    = dbg_times;
    a.n_split      = a.n_kv <= cells_per_split ? 1 : std::min(64, (a.n_kv + cells_per_split - 1)/cells_per_split);
    // The tensor-core scan (Q8_0 cache): a warp takes steps of 16 cells; few blocks, all resident at once.
    static const int mma_on    = getenv("IK_ATTN_MMA") ? atoi(getenv("IK_ATTN_MMA")) : 1;
    static const int mma_min   = getenv("IK_ATTN_MMA_MIN") ? atoi(getenv("IK_ATTN_MMA_MIN")) : 1;
    static const int mma_spw   = getenv("IK_ATTN_MMA_SPW") ? atoi(getenv("IK_ATTN_MMA_SPW")) : 1;         // steps per warp before another split
    static const int mma_split = getenv("IK_ATTN_MMA_SPLITS") ? atoi(getenv("IK_ATTN_MMA_SPLITS")) : 10;  // most splits (blocks per kv head)
    const bool mma = mma_on && !v1 && a.n_kv >= mma_min && k_cache->type == GGML_TYPE_Q8_0 && v_cache->type == GGML_TYPE_Q8_0 &&
            a.kc_nb1 == 4*(int64_t) sizeof(block_q8_0) && a.vc_nb1 == a.kc_nb1 && a.kc_nb2 % 4 == 0 && a.vc_nb2 % 4 == 0 &&
            (uintptr_t) a.k_cache % 4 == 0 && (uintptr_t) a.v_cache % 4 == 0;
    // A long cache: the scan as its own launch of small blocks (see inkling_attn_scan_kernel).
    // Measured slower than the fused kernel's own scan (158 vs 164 tok/s at 32K), so it is opt-in: IK_ATTN_SCAN_MIN=2048.
    static const int scan_min   = getenv("IK_ATTN_SCAN_MIN") ? atoi(getenv("IK_ATTN_SCAN_MIN")) : 0;          // cells; 0: never
    static const int scan_split = getenv("IK_ATTN_SCAN_SPLITS") ? atoi(getenv("IK_ATTN_SCAN_SPLITS")) : 82;   // most blocks
    static const int scan_spw   = getenv("IK_ATTN_SCAN_SPW") ? atoi(getenv("IK_ATTN_SCAN_SPW")) : 1;         // steps per warp before another block
    const bool ext = mma && scan_min > 0 && a.n_kv >= scan_min && a.n_head_kv <= IK_SCAN_MAX_KV && a.kc_nb2 == a.vc_nb2 &&
            a.kc_nb2 == a.n_head_kv*a.kc_nb1 && a.kc_nb2 % 16 == 0 && (uintptr_t) a.k_cache % 16 == 0 && (uintptr_t) a.v_cache % 16 == 0;
    a.ext_scan = 0;
    int n_ext = 0;
    if (ext) {
        n_ext = std::max(1, std::min(scan_split, (a.n_kv + 16*scan_spw - 1)/(16*scan_spw)));
        a.n_split = n_ext + 1; // the parts of the scan, then the fused kernel's own (the current cell)
    } else if (mma) {
        a.n_split = std::max(1, std::min(mma_split, (a.n_kv + 16*IK_A2_WARPS*mma_spw - 1)/(16*IK_A2_WARPS*mma_spw)));
    }
    static const bool fixed_enabled = !getenv("IK_ATTN_FIXED") || atoi(getenv("IK_ATTN_FIXED"));
    static const bool short_enabled = !getenv("IK_ATTN_SHORT") || atoi(getenv("IK_ATTN_SHORT"));
    // Windows (IK_MR) take the same shape-specialized instance per row, so rows stay bit-identical to single tokens;
    // the short 256-cell schedule groups the partial sums differently and has no window form: it is off under IK_MR
    // (IK_ATTN_FIXED_MR=0: the generic instances for windows and for single tokens under IK_MR, as before the merge).
    static const bool fixed_mr = !getenv("IK_ATTN_FIXED_MR") || atoi(getenv("IK_ATTN_FIXED_MR"));
    const bool fixed = fixed_enabled && mma && !ext && (fixed_mr || (n_rows == 1 && !ik_mr_enabled())) && dst->type == GGML_TYPE_Q8_1 &&
            ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_AMPERE_GA10X && !a.dbg_skip && !a.dbg_times &&
            a.n_head == 24 && a.n_head_kv == 8 && a.n_conv == 4 && a.n_rel == 16 && a.n_extent == 1024 &&
            a.kc_nb2 == 32*sizeof(block_q8_0) && a.vc_nb2 == a.kc_nb2;
    const bool short_scan = fixed && short_enabled && a.n_kv == 256 && n_rows == 1 && !ik_mr_enabled();
    if (short_scan) {
        a.n_split = 1;
    }
    a.kv_per_split = ext ? (a.n_kv + n_ext - 1)/n_ext : (a.n_kv + a.n_split - 1)/a.n_split;
    a.mr_mma       = mma;
    a.mr_mma_split = mma_split;
    a.mr_mma_step  = 16*IK_A2_WARPS*mma_spw;
    a.mr_cps       = split_override > 0 ? split_override : 64;
    a.mr_medium    = split_override <= 0 && ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_AMPERE_GA10X && k_cache->type == GGML_TYPE_Q8_0;

    cudaStream_t stream = ctx.stream();
    const int gqa = a.n_head/a.n_head_kv;

    ggml_cuda_pool_alloc<float> part(ctx.pool());
    a.part = nullptr; a.counters = nullptr;
    if (a.n_split > 1) {
        a.mr_part_rs = (int64_t) a.n_head_kv*a.n_split*gqa*(IK_ATTN_HD + 2);
        part.alloc((size_t) a.mr_part_rs*n_rows);
        a.part     = part.get();
        a.counters = ext ? nullptr : n_rows > 1 ? ik_counters(ctx, 64, a.n_head_kv*n_rows) : ik_counters(ctx, 0, a.n_head_kv); // self-resetting arrival counters
    }

    const bool q8 = dst->type == GGML_TYPE_Q8_1;
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || q8);
    GGML_ASSERT(k_cache->type == v_cache->type && (k_cache->type == GGML_TYPE_F16 || k_cache->type == GGML_TYPE_Q8_0));

    if (n_rows > 1) {
        // a window: the K/V of all rows first (their cache cells and the state planes), then a block set per row
        GGML_ASSERT(!ext && !v1 && n_rows <= 8 && a.n_head_kv*n_rows <= 64);
        const dim3 grid_kv(a.n_head_kv, 1, 1);
        const dim3 grid_mr(a.n_head_kv, a.n_split, n_rows);
        if (k_cache->type == GGML_TYPE_F16) {
            inkling_attn_kv_rows_kernel<half><<<grid_kv, 2*IK_ATTN_HD, 0, stream>>>(a);
            if (q8) { inkling_attn2_launch<half, true, false, true>(a, grid_mr, stream); } else { inkling_attn2_launch<half, false, false, true>(a, grid_mr, stream); }
        } else {
            if (!fixed)          { inkling_attn_kv_rows_kernel<block_q8_0><<<grid_kv, 2*IK_ATTN_HD, 0, stream>>>(a); }
            else if (a.has_taps) { inkling_attn_kv_rows_kernel<block_q8_0, 1><<<grid_kv, 2*IK_ATTN_HD, 0, stream>>>(a); }
            else                 { inkling_attn_kv_rows_kernel<block_q8_0, 0><<<grid_kv, 2*IK_ATTN_HD, 0, stream>>>(a); }
            if (fixed) {
                if (a.has_taps) { inkling_attn2_kernel<block_q8_0, true, 3, true, true, 1><<<grid_mr, IK_A2_BLOCK, 0, stream>>>(a); }
                else            { inkling_attn2_kernel<block_q8_0, true, 3, true, true, 0><<<grid_mr, IK_A2_BLOCK, 0, stream>>>(a); }
            } else if (mma) {
                if (q8) { inkling_attn2_launch<block_q8_0, true, true, true>(a, grid_mr, stream); } else { inkling_attn2_launch<block_q8_0, false, true, true>(a, grid_mr, stream); }
            } else {
                if (q8) { inkling_attn2_launch<block_q8_0, true, false, true>(a, grid_mr, stream); } else { inkling_attn2_launch<block_q8_0, false, false, true>(a, grid_mr, stream); }
            }
        }
        return;
    }
    if (ext) {
        inkling_attn_scan_launch(a, n_ext, stream);
        a.ext_scan = 1;
        const dim3 grid1(a.n_head_kv, 1, 1);
        if (q8) { inkling_attn2_launch<block_q8_0, true>(a, grid1, stream); } else { inkling_attn2_launch<block_q8_0, false>(a, grid1, stream); }
        return;
    }
    const dim3 grid(a.n_head_kv, a.n_split, 1);
    if (fixed) {
        if (short_scan) {
            if (a.has_taps) { inkling_attn2_kernel<block_q8_0, true, 3, true, false, 1, true><<<grid, IK_A2_BLOCK, 0, stream>>>(a); }
            else            { inkling_attn2_kernel<block_q8_0, true, 3, true, false, 0, true><<<grid, IK_A2_BLOCK, 0, stream>>>(a); }
        } else {
            if (a.has_taps) { inkling_attn2_kernel<block_q8_0, true, 3, true, false, 1><<<grid, IK_A2_BLOCK, 0, stream>>>(a); }
            else            { inkling_attn2_kernel<block_q8_0, true, 3, true, false, 0><<<grid, IK_A2_BLOCK, 0, stream>>>(a); }
        }
        return;
    }
    if (!v1) {
        if (k_cache->type == GGML_TYPE_F16) {
            if (q8) { inkling_attn2_launch<half, true>(a, grid, stream); } else { inkling_attn2_launch<half, false>(a, grid, stream); }
        } else {
            if (mma) {
                if (q8) { inkling_attn2_launch<block_q8_0, true, true>(a, grid, stream); } else { inkling_attn2_launch<block_q8_0, false, true>(a, grid, stream); }
            } else {
                if (q8) { inkling_attn2_launch<block_q8_0, true>(a, grid, stream); } else { inkling_attn2_launch<block_q8_0, false>(a, grid, stream); }
            }
        }
        return;
    }
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

// ---------------------------------------------------------------------------------------------------------------------
// inkling_moe_up / inkling_moe_down: the expert matvecs of one token with their epilogues. A warp computes the dot
// products of several rows with a Q8_1 activation (the loads of all rows are in flight together).

#define IK_MOE_BLOCK 256
#define IK_MOE_WARPS (IK_MOE_BLOCK/WARP_SIZE)
#define IK_MOE_MAX_K 4096

template <ggml_type type> struct ik_vec_dot;
template <> struct ik_vec_dot<GGML_TYPE_Q8_0> {
    static constexpr int qi = QI8_0, vdr = VDR_Q8_0_Q8_1_MMVQ;
    static __device__ __forceinline__ float dot(const void * vbq, const block_q8_1 * y, const int kbx, const int iqs) { return vec_dot_q8_0_q8_1(vbq, y, kbx, iqs); }
};
template <> struct ik_vec_dot<GGML_TYPE_IQ4_NL> {
    static constexpr int qi = QI4_NL, vdr = VDR_IQ4_NL_Q8_1_MMVQ;
    static __device__ __forceinline__ float dot(const void * vbq, const block_q8_1 * y, const int kbx, const int iqs) { return vec_dot_iq4_nl_q8_1(vbq, y, kbx, iqs); }
};

// acc[r] = rows[r] . y over n_blocks blocks of 32, the result in every lane
template <ggml_type type, int n_rows>
static __device__ __forceinline__ void ik_warp_dot_rows(const char * const * rows, const block_q8_1 * __restrict__ y, const int n_blocks, float * acc) {
    constexpr int lpb = ik_vec_dot<type>::qi/ik_vec_dot<type>::vdr; // lanes per block
    const int lane = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int r = 0; r < n_rows; ++r) {
        acc[r] = 0.0f;
    }
    for (int unit = lane; unit < n_blocks*lpb; unit += WARP_SIZE) {
        const int kbx = unit/lpb;
        const int iqs = ik_vec_dot<type>::vdr*(unit % lpb);
#pragma unroll
        for (int r = 0; r < n_rows; ++r) {
            acc[r] += ik_vec_dot<type>::dot(rows[r], y + kbx, kbx, iqs);
        }
    }
#pragma unroll
    for (int r = 0; r < n_rows; ++r) {
        acc[r] = warp_reduce_sum(acc[r]);
    }
}

// one block per (32 channels, expert slot): the 32 gate and 32 up rows over W warps (2*32/W rows per warp), the SITU
// activation, one Q8_1 block out
template <ggml_type type, int W>
static __global__ void __launch_bounds__(W*WARP_SIZE) inkling_moe_up_kernel(const char * __restrict__ as, const int64_t nb1, const int64_t nb2,
        const float * __restrict__ x, const int32_t * __restrict__ ids, block_q8_1 * __restrict__ dst,
        const int n_embd, const int n_ff, const float beta, const float lb) {
    constexpr int CPW = 32/W; // channels per warp
    __shared__ block_q8_1 s_x[IK_MOE_MAX_K/QK8_1];
    __shared__ float      s_out[32];

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int nblk = n_embd/QK8_1;

    // the activation quantized like the mmvq path does
    for (int b = warp; b < nblk; b += W) {
        ik_warp_store_q8_1(x[b*QK8_1 + lane], &s_x[b], lane, true);
    }
    __syncthreads();

    const int e_slot = blockIdx.y;
    const int e      = ids[e_slot];
    const int c0     = blockIdx.x*32;
    const char * base = as + (int64_t) e*nb2;
    const char * rows[2*CPW];
#pragma unroll
    for (int j = 0; j < CPW; ++j) {
        rows[j]       = base + (int64_t) (c0 + CPW*warp + j)*nb1;
        rows[CPW + j] = base + (int64_t) (n_ff + c0 + CPW*warp + j)*nb1;
    }
    float acc[2*CPW];
    ik_warp_dot_rows<type, 2*CPW>(rows, s_x, nblk, acc);

    if (lane < CPW) {
        float g = 0.0f, u = 0.0f;
#pragma unroll
        for (int j = 0; j < CPW; ++j) {
            if (lane == j) {
                g = acc[j];
                u = acc[CPW + j];
            }
        }
        const float sig = 1.0f/(1.0f + expf(-g));
        const float act = tanhf(g*(1.0f/beta))*beta*sig;
        if (lb > 0.0f) {
            u = tanhf(u*(1.0f/lb))*lb;
        }
        s_out[CPW*warp + lane] = act*u;
    }
    __syncthreads();
    if (warp == 0) {
        ik_warp_store_q8_1(s_out[lane], dst + ((int64_t) e_slot*n_ff + c0)/QK8_1, lane, true);
    }
}

// one block per C channels: warp e computes the C rows of expert slot e, the weighted sum over the experts, then the FFN
// short convolution with the residual (channels chan_offset.. of the recurrent state)
template <ggml_type type, int C>
static __global__ void __launch_bounds__(IK_MOE_BLOCK) inkling_moe_down_kernel(const char * __restrict__ as, const int64_t nb1, const int64_t nb2,
        const block_q8_1 * __restrict__ h, const int h_blocks, const int32_t * __restrict__ ids, const float * __restrict__ weights, const int w_stride,
        const float * __restrict__ conv_w, float * __restrict__ state, const int64_t state_s1, const int32_t * __restrict__ rs_ids,
        const int write_cell, const int chan_offset, const float scale, const float * __restrict__ residual, float * __restrict__ dst,
        const int n_used, const int n_conv) {
    __shared__ float s_dot[8][C];

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int c0   = blockIdx.x*C;

    if (warp < n_used) {
        const int e = ids[warp];
        const char * base = as + (int64_t) e*nb2;
        const char * rows[C];
#pragma unroll
        for (int j = 0; j < C; ++j) {
            rows[j] = base + (int64_t) (c0 + j)*nb1;
        }
        float acc[C];
        ik_warp_dot_rows<type, C>(rows, h + (int64_t) warp*h_blocks, h_blocks, acc);
#pragma unroll
        for (int j = 0; j < C; ++j) {
            if (lane == j) {
                s_dot[warp][j] = acc[j];
            }
        }
    }
    __syncthreads();

    if (tid < C) {
        const int c = c0 + tid;
        float sum = 0.0f;
        for (int e = 0; e < n_used; ++e) {
            sum += s_dot[e][tid]*weights[e*w_stride];
        }
        const int64_t ch = chan_offset + c;
        const float * past = state + (int64_t) rs_ids[0]*state_s1 + ch*(n_conv - 1);
        const float * w    = conv_w + (int64_t) c*n_conv;
        float xs[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            xs[j] = j < n_conv - 1 ? past[j] : (j == n_conv - 1 ? sum : 0.0f);
        }
        float out = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < n_conv) {
                out += xs[j]*w[j];
            }
        }
        float * st = state + (int64_t) write_cell*state_s1 + ch*(n_conv - 1);
#pragma unroll
        for (int j = 0; j < 7; ++j) {
            if (j < n_conv - 1) {
                st[j] = xs[j + 1];
            }
        }
        dst[c] = residual[c] + (sum + out)*scale;
    }
}

void ggml_cuda_op_inkling_moe_up(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * as  = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];

    const int n_embd = as->ne[0];
    const int n_ff   = dst->ne[0];
    const int n_used = dst->ne[1];
    GGML_ASSERT(n_embd % QK8_1 == 0 && n_embd <= IK_MOE_MAX_K && n_ff % 32 == 0 && n_used <= 8);
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_1 && ggml_is_contiguous(dst));

    const float beta = ggml_get_op_params_f32(dst, 0);
    const float lb   = ggml_get_op_params_f32(dst, 1);

    const dim3 grid(n_ff/32, n_used, 1);
    cudaStream_t stream = ctx.stream();
    static const int W = getenv("IK_MOE_UP_WARPS") ? atoi(getenv("IK_MOE_UP_WARPS")) : 8;
    auto launch = [&](auto type_c) {
        constexpr ggml_type T = decltype(type_c)::value;
        const char * as_d = (const char *) as->data; const float * x_d = (const float *) x->data;
        const int32_t * ids_d = (const int32_t *) ids->data; block_q8_1 * dst_d = (block_q8_1 *) dst->data;
        switch (W) {
            case 16: inkling_moe_up_kernel<T, 16><<<grid, 16*WARP_SIZE, 0, stream>>>(as_d, as->nb[1], as->nb[2], x_d, ids_d, dst_d, n_embd, n_ff, beta, lb); break;
            case 32: inkling_moe_up_kernel<T, 32><<<grid, 32*WARP_SIZE, 0, stream>>>(as_d, as->nb[1], as->nb[2], x_d, ids_d, dst_d, n_embd, n_ff, beta, lb); break;
            default: inkling_moe_up_kernel<T,  8><<<grid,  8*WARP_SIZE, 0, stream>>>(as_d, as->nb[1], as->nb[2], x_d, ids_d, dst_d, n_embd, n_ff, beta, lb); break;
        }
    };
    switch (as->type) {
        case GGML_TYPE_IQ4_NL: launch(std::integral_constant<ggml_type, GGML_TYPE_IQ4_NL>{}); break;
        case GGML_TYPE_Q8_0:   launch(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});   break;
        default: GGML_ABORT("inkling_moe_up: unsupported type");
    }
}

void ggml_cuda_op_inkling_moe_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * as       = dst->src[0];
    const ggml_tensor * h        = dst->src[1];
    const ggml_tensor * ids      = dst->src[2];
    const ggml_tensor * weights  = dst->src[3];
    const ggml_tensor * conv_w   = dst->src[4];
    ggml_tensor       * state    = dst->src[5];
    const ggml_tensor * rs_ids   = dst->src[6];
    const ggml_tensor * residual = dst->src[7];

    const int n_ff   = as->ne[0];
    const int n_embd = dst->ne[0];
    const int n_used = ids->ne[0];
    const int n_conv = conv_w->ne[0];
    GGML_ASSERT(n_ff % QK8_1 == 0 && n_embd % 8 == 0 && n_used <= 8 && n_conv <= 8);

    const int   write_cell  = ggml_get_op_params_i32(dst, 0);
    const int   chan_offset = ggml_get_op_params_i32(dst, 1);
    const float scale       = ggml_get_op_params_f32(dst, 2);

    cudaStream_t stream = ctx.stream();
    static const int C = getenv("IK_MOE_DOWN_CH") ? atoi(getenv("IK_MOE_DOWN_CH")) : 8;
    auto launch = [&](auto kernel, int c) {
        const dim3 grid(n_embd/c, 1, 1);
        kernel<<<grid, IK_MOE_BLOCK, 0, stream>>>((const char *) as->data, as->nb[1], as->nb[2],
                (const block_q8_1 *) h->data, n_ff/QK8_1, (const int32_t *) ids->data, (const float *) weights->data, (int) (weights->nb[1]/sizeof(float)),
                (const float *) conv_w->data, (float *) state->data, (int64_t) (state->nb[1]/sizeof(float)), (const int32_t *) rs_ids->data,
                write_cell, chan_offset, scale, (const float *) residual->data, (float *) dst->data, n_used, n_conv);
    };
    switch (as->type) {
        case GGML_TYPE_IQ4_NL: if (C == 4) launch(inkling_moe_down_kernel<GGML_TYPE_IQ4_NL, 4>, 4); else if (C == 2) launch(inkling_moe_down_kernel<GGML_TYPE_IQ4_NL, 2>, 2); else launch(inkling_moe_down_kernel<GGML_TYPE_IQ4_NL, 8>, 8); break;
        case GGML_TYPE_Q8_0:   if (C == 4) launch(inkling_moe_down_kernel<GGML_TYPE_Q8_0, 4>, 4);   else if (C == 2) launch(inkling_moe_down_kernel<GGML_TYPE_Q8_0, 2>, 2);   else launch(inkling_moe_down_kernel<GGML_TYPE_Q8_0, 8>, 8);   break;
        default: GGML_ABORT("inkling_moe_down: unsupported type");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// inkling_moe_sum_conv: the weighted expert sum, the FFN short convolution and the residual, one thread per channel

static __global__ void inkling_moe_sum_conv_kernel(const float * __restrict__ a, const int64_t a_s1, const float * __restrict__ w, const int w_s1,
        const float * __restrict__ conv_w, float * __restrict__ state, const int64_t state_s1, const int32_t * __restrict__ rs_ids,
        const int write_cell, const int chan_offset, const float scale, const float * __restrict__ residual, float * __restrict__ dst,
        const int n_embd, const int n_used, const int n_conv) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n_embd) {
        return;
    }
    const int64_t ch   = chan_offset + c;
    const float * past = state + (int64_t) rs_ids[0]*state_s1 + ch*(n_conv - 1);
    const float * cw   = conv_w + (int64_t) c*n_conv;

    // all the loads first
    float av[8], xs[8], wv[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        av[e] = e < n_used ? a[c + e*a_s1] : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        xs[j] = j < n_conv - 1 ? past[j] : 0.0f;
        wv[j] = j < n_conv ? cw[j] : 0.0f;
    }
    const float res = residual[c];

    float sum = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        sum += av[e]*w[e*w_s1];
    }
    xs[n_conv - 1] = sum;
    float out = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < n_conv) {
            out += xs[j]*wv[j];
        }
    }
    float * st = state + (int64_t) write_cell*state_s1 + ch*(n_conv - 1);
#pragma unroll
    for (int j = 0; j < 7; ++j) {
        if (j < n_conv - 1) {
            st[j] = xs[j + 1];
        }
    }
    dst[c] = res + (sum + out)*scale;
}

// the rows of a window: a thread per channel walks the tokens in order (the statements of the single-token kernel)
static __global__ void inkling_moe_sum_conv_rows_kernel(const float * __restrict__ a, const int64_t a_s1, const int64_t a_s2,
        const float * __restrict__ w, const int w_s1, const int w_s2,
        const float * __restrict__ conv_w, float * __restrict__ state, const int64_t state_s1, const int32_t * __restrict__ rs_ids,
        const int write_cell, const int plane_cells, const int n_planes, const int chan_offset, const float scale,
        const float * __restrict__ residual, const int64_t res_s1, float * __restrict__ dst, const int64_t dst_s1,
        const int n_embd, const int n_used, const int n_conv, const int n_rows) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n_embd) {
        return;
    }
    const int64_t ch   = chan_offset + c;
    const float * past = state + (int64_t) rs_ids[0]*state_s1 + ch*(n_conv - 1);
    const float * cw   = conv_w + (int64_t) c*n_conv;
    float xs[8], wv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        xs[j] = j < n_conv - 1 ? past[j] : 0.0f;
        wv[j] = j < n_conv ? cw[j] : 0.0f;
    }
    for (int row = 0; row < n_rows; ++row) {
        float av[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            av[e] = e < n_used ? a[c + e*a_s1 + row*a_s2] : 0.0f;
        }
        const float res = residual[c + row*res_s1];
        float sum = 0.0f;
        for (int e = 0; e < n_used; ++e) {
            sum += av[e]*w[e*w_s1 + row*w_s2];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j == n_conv - 1) {
                xs[j] = sum;
            }
        }
        float out = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < n_conv) {
                out += xs[j]*wv[j];
            }
        }
        const int k = n_rows - 1 - row;
        if (k < n_planes) {
            float * st = state + ((int64_t) write_cell + (int64_t) k*plane_cells)*state_s1 + ch*(n_conv - 1);
#pragma unroll
            for (int j = 0; j < 7; ++j) {
                if (j < n_conv - 1) {
                    st[j] = xs[j + 1];
                }
            }
        }
        dst[c + row*dst_s1] = res + (sum + out)*scale;
#pragma unroll
        for (int j = 0; j < 7; ++j) {
            xs[j] = xs[j + 1];
        }
    }
}

void ggml_cuda_op_inkling_moe_sum_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a        = dst->src[0];
    const ggml_tensor * w        = dst->src[1];
    const ggml_tensor * conv_w   = dst->src[2];
    ggml_tensor       * state    = dst->src[3];
    const ggml_tensor * rs_ids   = dst->src[4];
    const ggml_tensor * residual = dst->src[5];

    const int n_embd = dst->ne[0];
    const int n_used = a->ne[1];
    const int n_conv = conv_w->ne[0];
    GGML_ASSERT(n_used <= 8 && n_conv <= 8);

    const int   write_cell  = ggml_get_op_params_i32(dst, 0);
    const int   chan_offset = ggml_get_op_params_i32(dst, 1);
    const float scale       = ggml_get_op_params_f32(dst, 2);

    const int block = 256;
    if (dst->ne[1] > 1) {
        inkling_moe_sum_conv_rows_kernel<<<(n_embd + block - 1)/block, block, 0, ctx.stream()>>>((const float *) a->data, a->nb[1]/sizeof(float), a->nb[2]/sizeof(float),
                (const float *) w->data, (int) (w->nb[1]/sizeof(float)), (int) (w->nb[2]/sizeof(float)), (const float *) conv_w->data,
                (float *) state->data, state->nb[1]/sizeof(float), (const int32_t *) rs_ids->data, write_cell, ggml_get_op_params_i32(dst, 4),
                ggml_get_op_params_i32(dst, 3), chan_offset, scale, (const float *) residual->data, residual->nb[1]/sizeof(float),
                (float *) dst->data, dst->nb[1]/sizeof(float), n_embd, n_used, n_conv, (int) dst->ne[1]);
        return;
    }
    inkling_moe_sum_conv_kernel<<<(n_embd + block - 1)/block, block, 0, ctx.stream()>>>((const float *) a->data, a->nb[1]/sizeof(float),
            (const float *) w->data, (int) (w->nb[1]/sizeof(float)), (const float *) conv_w->data, (float *) state->data, state->nb[1]/sizeof(float),
            (const int32_t *) rs_ids->data, write_cell, chan_offset, scale, (const float *) residual->data, (float *) dst->data, n_embd, n_used, n_conv);
}

// ---------------------------------------------------------------------------------------------------------------------
// inkling_router: the FFN input (mix + norm, computed by every block) and the routing of one token: a warp per expert row
// (8 experts per block), the last block selects the n_used experts and normalizes their weights

#define IK_ROUTER_MAX_EMBD   4096
#define IK_ROUTER_MAX_EXPERT 256
#define IK_ROUTER_BLOCK      256
#define IK_ROUTER_WARPS      (IK_ROUTER_BLOCK/WARP_SIZE)

struct ik_router_args {
    const float * query;              // nullptr: no mix
    ik_state_ptrs st; int n_states;
    const float * norm_w;             // [n_embd], followed by the selection bias [n_expert]
    const float * gate_inp;           // [n_embd, n_expert]
    const float * probs_b;            // the selection bias
    int32_t     * route;              // [2*n_used]: ids, then the weights as float bits
    float       * dst;                // [n_embd]
    block_q8_1  * xq;                 // nullptr or the quantized result
    float       * logits;             // [n_expert] scratch
    int         * counter;
    int n_embd, n_expert, n_used;
    bool warp_topk;
    float eps, scale, norm_eps, w_scale, w_eps;
    // the optional attention-output convolution producing the last state
    const float   * wo_out;          // nullptr: no convolution
    const float   * conv_w;          // [n_conv, n_embd]
    float         * cv_state; int64_t cv_state_s1; const int32_t * cv_rs_ids; int cv_write_cell, cv_chan_offset, cv_n_conv; float cv_scale;
    const float   * cv_residual;     // nullptr: none
    float         * attn_out;        // the last state, written
    int64_t mr_st_rs, mr_route_rs, mr_dst_rs, mr_xq_rs; // multi-row launches (row: blockIdx.y): elements between the rows
    int dbg_skip; // timing experiments (IK_ROUTER_SKIP): 1 no arrival/selection, 2 no router rows, 4 no mix
    long long * dbg_times; // timing experiments (IK_ROUTER_TIMING): clock64 at the phase boundaries of block 0
};

#define IK_RDBG_T(k) do { if (args.dbg_times && tid == 0 && blockIdx.x == 0) { args.dbg_times[k] = clock64(); } } while (0)

template <int W> // warps (experts) per block
static __global__ void __launch_bounds__(W*WARP_SIZE) inkling_router_kernel(const ik_router_args args) {
    constexpr int BLOCK = W*WARP_SIZE;
    __shared__ float s_x[IK_ROUTER_MAX_EMBD];
    __shared__ float s_red[W][2*IK_MAX_STATES];
    __shared__ float s_rms[IK_MAX_STATES];
    __shared__ float s_p[IK_MAX_STATES];
    __shared__ float s_r[W];
    __shared__ int   s_last;

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int n_embd   = args.n_embd;
    const int n_embd4  = n_embd/4;
    const int n_states = args.n_states;

    const int e = blockIdx.x*W + warp;
    constexpr int NV = IK_ROUTER_MAX_EMBD/4/WARP_SIZE; // float4 per lane of the router row
    IK_RDBG_T(0);

    if (args.dbg_skip & 4) {
        if (tid < WARP_SIZE) { s_rms[0] = 1.0f; s_p[0] = 1.0f; }
        __syncthreads();
        for (int k = tid; k < n_embd4; k += BLOCK) {
            ((float4 *) s_x)[k] = ((const float4 *) args.st.p[0])[k];
        }
        __syncthreads();
    } else {
    // --- the mix, pass 1: the sums of squares and the query dots of every state (the states are re-read in pass 2,
    //     from L2, instead of being kept in registers: that keeps the router row loads above in flight)
    constexpr int KV = (IK_ROUTER_MAX_EMBD/4 + BLOCK - 1)/BLOCK; // float4 per thread
    float acc[2*IK_MAX_STATES];
#pragma unroll
    for (int i = 0; i < 2*IK_MAX_STATES; ++i) {
        acc[i] = 0.0f;
    }
    if (args.query) {
#pragma unroll
        for (int j = 0; j < KV; ++j) {
            const int k = tid + j*BLOCK;
            if (k < n_embd4) {
                const float4 q = ((const float4 *) args.query)[k];
#pragma unroll
                for (int i = 0; i < IK_MAX_STATES; ++i) {
                    if (i < n_states) {
                        const float4 x = ((const float4 *) args.st.p[i])[k];
                        acc[2*i + 0] += x.x*x.x + x.y*x.y + x.z*x.z + x.w*x.w;
                        acc[2*i + 1] += x.x*q.x + x.y*q.y + x.z*q.z + x.w*q.w;
                    }
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
            s_red[warp][lane] = v;
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
            float ss = 0.0f, dt = 0.0f;
            if (lane < n_states) {
#pragma unroll
                for (int w = 0; w < W; ++w) {
                    ss += s_red[w][2*lane + 0];
                    dt += s_red[w][2*lane + 1];
                }
            }
            const float r  = rsqrtf(ss/n_embd + args.eps);
            float       sc = lane < n_states ? dt*r*args.scale : -INFINITY;
            const float mx = warp_reduce_max(sc);
            sc = lane < n_states ? expf(sc - mx) : 0.0f;
            const float sum = warp_reduce_sum(sc);
            if (lane < n_states) {
                s_rms[lane] = r;
                s_p[lane]   = sc/sum;
            }
        }
    } else if (tid == 0) {
        s_rms[0] = 1.0f;
        s_p[0]   = 1.0f;
    }
    __syncthreads();
    IK_RDBG_T(1); // pass 1 reduced

    // --- pass 2: the mixed input, its RMS norm (the operation order of the rms_norm + mul kernel) into shared memory
    float4 y[KV];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < KV; ++j) {
        const int k = tid + j*BLOCK;
        y[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (k < n_embd4) {
#pragma unroll
            for (int i = 0; i < IK_MAX_STATES; ++i) {
                if (i < n_states) {
                    const float4 x = ((const float4 *) args.st.p[i])[k];
                    y[j].x += (x.x*s_rms[i])*s_p[i];
                    y[j].y += (x.y*s_rms[i])*s_p[i];
                    y[j].z += (x.z*s_rms[i])*s_p[i];
                    y[j].w += (x.w*s_rms[i])*s_p[i];
                }
            }
            ss += y[j].x*y[j].x + y[j].y*y[j].y + y[j].z*y[j].z + y[j].w*y[j].w;
        }
    }
    ss = ik_block_sum<BLOCK>(ss, s_r);
    const float factor = rsqrtf(ss/n_embd + args.norm_eps);
#pragma unroll
    for (int j = 0; j < KV; ++j) {
        const int k = tid + j*BLOCK;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (k < n_embd4) {
            const float4 w = ((const float4 *) args.norm_w)[k];
            o = make_float4(factor*y[j].x*w.x, factor*y[j].y*w.y, factor*y[j].z*w.z, factor*y[j].w*w.w);
            ((float4 *) s_x)[k] = o;
            if (blockIdx.x == 0) {
                ((float4 *) args.dst)[k] = o;
            }
        }
        if (args.xq && blockIdx.x == 0) {
            // 8 consecutive lanes hold one Q8_1 block (all lanes take part in the shuffles)
            ik_store_q8_1_vec4(o, args.xq, k, k < n_embd4);
        }
    }
    __syncthreads();
    }

    IK_RDBG_T(2); // the input is in shared memory
    if (args.dbg_skip & 2) {
        return;
    }
    // --- the logit of this warp's expert: the row loads in flight together, then the dot with the shared input
    {
        const float4 * w4 = (const float4 *) (args.gate_inp + (int64_t) e*n_embd);
        float4 wrow[NV];
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int k = lane + i*WARP_SIZE;
            wrow[i] = k < n_embd4 ? w4[k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < NV; ++i) {
            const int k = lane + i*WARP_SIZE;
            if (k < n_embd4) {
                const float4 xk = ((const float4 *) s_x)[k];
                dot += wrow[i].x*xk.x + wrow[i].y*xk.y + wrow[i].z*xk.z + wrow[i].w*xk.w;
            }
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            args.logits[e] = dot;
        }
    }
    IK_RDBG_T(3); // logits computed
    if (args.dbg_skip & 1) {
        return;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const int old = atomicAdd(args.counter, 1);
        s_last = old == (int) gridDim.x - 1;
    }
    __syncthreads();
    IK_RDBG_T(4); // arrival counted
    if (!s_last) {
        return;
    }
    __threadfence();
    if (args.dbg_times && tid == 0) { args.dbg_times[5] = clock64(); } // the last block starts selecting (its own clock)

    // --- the selection, by the first warp of the last block: sigmoid, bias, n_used rounds of argmax (ties: the smaller id)
    if (warp == 0) {
        constexpr int EPT = IK_ROUTER_MAX_EXPERT/WARP_SIZE;
        float prob[EPT], sel[EPT];
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            const int ex = lane + i*WARP_SIZE;
            prob[i] = -INFINITY;
            sel[i]  = -INFINITY;
            if (ex < args.n_expert) {
                prob[i] = 1.0f/(1.0f + expf(-args.logits[ex]));
                sel[i]  = prob[i] + args.probs_b[ex];
            }
        }
        float wsum = 0.0f;
        float wts[8];
        for (int k = 0; k < args.n_used; ++k) {
            float max_s = sel[0];
            int   max_e = lane;
#pragma unroll
            for (int i = 1; i < EPT; ++i) {
                const int ex = lane + i*WARP_SIZE;
                if (ex < args.n_expert && sel[i] > max_s) {
                    max_s = sel[i];
                    max_e = ex;
                }
            }
#pragma unroll
            for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
                const float vs = __shfl_xor_sync(0xffffffff, max_s, mask);
                const int   ve = __shfl_xor_sync(0xffffffff, max_e, mask);
                if (vs > max_s || (vs == max_s && ve < max_e)) {
                    max_s = vs;
                    max_e = ve;
                }
            }
            float pmax = 0.0f;
#pragma unroll
            for (int i = 0; i < EPT; ++i) {
                if (lane + i*WARP_SIZE == max_e) {
                    pmax   = prob[i];
                    sel[i] = -INFINITY;
                }
            }
            pmax = warp_reduce_sum(pmax);
            if (lane == k) {
                args.route[k] = max_e;
            }
            wts[k] = pmax;
            wsum  += pmax;
        }
        const float inv = 1.0f/(wsum + args.w_eps);
        if (lane < args.n_used) {
            float w = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (k == lane) {
                    w = wts[k];
                }
            }
            w = w*inv*args.w_scale;
            args.route[args.n_used + lane] = __float_as_int(w);
        }
        if (lane == 0) {
            *args.counter = 0;
        }
    }
    if (args.dbg_times && tid == 0) { args.dbg_times[6] = clock64(); } // selection done (last block's clock)
}

// ---------------------------------------------------------------------------------------------------------------------
// inkling_router v2: n_expert/8 blocks of 256 threads (one expert row per warp). Every block issues all its loads in one
// round trip (the states, query and norm weight into registers, its 8 router rows into shared memory with cp.async),
// computes the mix weights and the RMS factor of the mixed input from the Gram matrix of the states (one pass), the
// normalized input into shared memory (block 0 also writes the F32 / Q8_1 outputs), and its 8 logits; the last block
// to arrive ranks the experts in parallel (rank = number of larger selection scores) instead of n_used argmax rounds.

#define IK_R2_BLOCK      256
#define IK_R2_WARPS      (IK_R2_BLOCK/WARP_SIZE)
#define IK_R2_MAX_EMBD   2304
#define IK_R2_CPT        (IK_R2_MAX_EMBD/IK_R2_BLOCK)  // channels per thread
#define IK_R2_MAX_STATES 5
#define IK_R2_MAX_EXPERT 256

static __device__ __forceinline__ unsigned long long ik_sort_warp(unsigned long long key, int lane) {
#pragma unroll
    for (int size = 2; size <= 32; size *= 2) {
#pragma unroll
        for (int delta = size/2; delta; delta /= 2) {
            const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, delta);
            const bool higher = ((lane & size) == 0) == ((lane & delta) == 0);
            key = higher ? max(key, other) : min(key, other);
        }
    }
    return key;
}

template <int NS, int NC, bool MR = false> // NC: taps of the fused attention-output convolution (0: none); MR: a row per blockIdx.y
static __global__ void __launch_bounds__(IK_R2_BLOCK, 1) inkling_router2_kernel(const ik_router_args args_in) {
    ik_router_args args_m;
    if constexpr (MR) {
        const int row = blockIdx.y;
        args_m = args_in;
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            args_m.st.p[i] += row*args_in.mr_st_rs;
        }
        args_m.route   += row*args_in.mr_route_rs;
        args_m.dst     += row*args_in.mr_dst_rs;
        args_m.xq       = args_in.xq ? args_in.xq + row*args_in.mr_xq_rs : nullptr;
        args_m.logits  += row*args_in.n_expert;
        args_m.counter += row;
    }
    const ik_router_args & args = MR ? args_m : args_in;
    constexpr int NG = NS*(NS + 1)/2;          // Gram matrix (upper triangle)
    constexpr int NP = NG + NS;                 // + the dots with the query
    constexpr int C4 = (IK_R2_MAX_EMBD/4 + IK_R2_BLOCK - 1)/IK_R2_BLOCK; // float4 slots per thread
    extern __shared__ float4 r2_smem[];
    float4 * s_rows = r2_smem;                                          // [IK_R2_WARPS][n_embd/4]
    float4 * s_x    = r2_smem + IK_R2_WARPS*(IK_R2_MAX_EMBD/4);         // [n_embd/4]
    __shared__ float s_red[IK_R2_WARPS][NP];
    __shared__ float s_tot[NP];
    __shared__ float s_w[2*NS + 1];            // rms[NS], p[NS], factor
    __shared__ float s_sel[IK_R2_MAX_EXPERT];
    __shared__ float s_prob[IK_R2_MAX_EXPERT];
    __shared__ float s_selected[8];
    __shared__ int   s_last;

    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int n_embd  = args.n_embd;
    const int n_embd4 = n_embd/4;
    const int e0 = blockIdx.x*IK_R2_WARPS;
    IK_RDBG_T(0);

    // ===== all loads in one round trip: the states / query / norm weight into registers (float4, lane-consecutive),
    //       then the router row of this warp's expert into shared memory with cp.async (needed last) =====
    float4 x[NS][C4], q[C4], nw[C4];
    constexpr bool has_conv = NC > 0;
#pragma unroll
    for (int j = 0; j < C4; ++j) {
        const int  k  = tid + j*IK_R2_BLOCK;
        const bool ok = k < n_embd4;
        const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            x[i][j] = ok && (i < NS - 1 || !has_conv) ? ((const float4 *) args.st.p[i])[k] : z;
        }
        q[j]  = ok && args.query ? ((const float4 *) args.query)[k] : z;
        nw[j] = ok ? ((const float4 *) args.norm_w)[k] : z;
    }
    {
        const float4 * row = (const float4 *) (args.gate_inp + (int64_t) (e0 + warp)*n_embd);
        for (int k = lane; k < n_embd4; k += WARP_SIZE) {
            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(s_rows + warp*(IK_R2_MAX_EMBD/4) + k), row + k);
        }
        cp_async_commit_group();
    }
    if constexpr (has_conv) {
        // the attention-output convolution + residual: the last state (block b writes the channels of the Q8_1 blocks
        // (k/8) % gridDim.x == b); 4 channels per float4 slot: 4 wo values, 4*NC taps,
        // 4*(NC-1) past values (all 16-byte aligned: chan_offset % 4 == 0)
        const float scale = args.cv_scale;
        const int   cell  = args.cv_rs_ids[0];
        static_assert(NC == 4, "the fused convolution is written for 4 taps");
#pragma unroll
        for (int j = 0; j < C4; ++j) {
            const int  k  = tid + j*IK_R2_BLOCK;
            const bool ok = k < n_embd4;
            if (!ok) {
                continue;
            }
            const float4 wo  = ((const float4 *) args.wo_out)[k];
            const float4 res = args.cv_residual ? ((const float4 *) args.cv_residual)[k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 * cw4   = (const float4 *) (args.conv_w + (int64_t) 4*k*NC);                                             // 4 float4: taps of channels 4k..4k+3
            const float4 * past4 = (const float4 *) (args.cv_state + (int64_t) cell*args.cv_state_s1 + ((int64_t) args.cv_chan_offset + 4*k)*(NC - 1)); // 3 float4: 12 past values
            const float4 w0 = cw4[0], w1 = cw4[1], w2 = cw4[2], w3 = cw4[3];
            const float4 p0 = past4[0], p1 = past4[1], p2 = past4[2];
            // channel c: past (p[3c], p[3c+1], p[3c+2]), current wo[c]; taps w[c]
            const float pa[12] = { p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w, p2.x, p2.y, p2.z, p2.w };
            const float ww[16] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w, w3.x, w3.y, w3.z, w3.w };
            const float wo4[4] = { wo.x, wo.y, wo.z, wo.w }, res4[4] = { res.x, res.y, res.z, res.w };
            float cur4[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float out = pa[3*c]*ww[4*c] + pa[3*c + 1]*ww[4*c + 1] + pa[3*c + 2]*ww[4*c + 2] + wo4[c]*ww[4*c + 3];
                cur4[c] = res4[c] + (wo4[c] + out)*scale;
            }
            x[NS - 1][j] = make_float4(cur4[0], cur4[1], cur4[2], cur4[3]);
            if ((k/8) % gridDim.x == blockIdx.x) {
                ((float4 *) args.attn_out)[k] = x[NS - 1][j];
            }
        }
    }

    // ===== the Gram matrix of the states and their dots with the query: block reduction =====
    {
        float acc[NP];
        int n = 0;
#pragma unroll
        for (int i = 0; i < NS; ++i) {
#pragma unroll
            for (int k = i; k < NS; ++k) {
                float v = 0.0f;
#pragma unroll
                for (int j = 0; j < C4; ++j) {
                    v += x[i][j].x*x[k][j].x + x[i][j].y*x[k][j].y + x[i][j].z*x[k][j].z + x[i][j].w*x[k][j].w;
                }
                acc[n++] = v;
            }
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            float v = 0.0f;
#pragma unroll
            for (int j = 0; j < C4; ++j) {
                v += x[i][j].x*q[j].x + x[i][j].y*q[j].y + x[i][j].z*q[j].z + x[i][j].w*q[j].w;
            }
            acc[NG + i] = v;
        }
        IK_RDBG_T(7); // the loads landed (the products need them)
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            acc[i] = warp_reduce_sum(acc[i]);
        }
        IK_RDBG_T(8); // reduced
        if (lane < NP) {
            float v = 0.0f;
#pragma unroll
            for (int i = 0; i < NP; ++i) {
                if (i == lane) {
                    v = acc[i];
                }
            }
            s_red[warp][lane] = v;
        }
    }
    __syncthreads();
    if (tid < NP) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < IK_R2_WARPS; ++w) {
            v += s_red[w][tid];
        }
        s_tot[tid] = v;
    }
    __syncthreads();
    IK_RDBG_T(1);
    if (warp == 0) {
        // lane i < NS: state i
        auto gram = [&](const int i, const int k) -> float {
            const int lo = min(i, k), hi = max(i, k);
            return s_tot[lo*NS - lo*(lo - 1)/2 + (hi - lo)];
        };
        float r = 1.0f, wgt = 0.0f;
        if (args.query) {
            const float ss = lane < NS ? gram(lane, lane) : 0.0f;
            const float dt = lane < NS ? s_tot[NG + lane] : 0.0f;
            r = rsqrtf(ss/n_embd + args.eps);
            float       sc = lane < NS ? dt*r*args.scale : -INFINITY;
            const float mx = warp_reduce_max(sc);
            sc = lane < NS ? expf(sc - mx) : 0.0f;
            const float sum = warp_reduce_sum(sc);
            wgt = lane < NS ? r*(sc/sum) : 0.0f;
        } else {
            wgt = lane == 0 ? 1.0f : 0.0f;
        }
        // |y|^2 = sum_ik w_i w_k G_ik
        float ssy = 0.0f;
#pragma unroll
        for (int k = 0; k < NS; ++k) {
            const float wk = __shfl_sync(0xffffffff, wgt, k);
            if (lane < NS) {
                ssy += wgt*wk*gram(lane, k);
            }
        }
        ssy = warp_reduce_sum(ssy);
        if (lane < NS) {
            s_w[lane]      = r;                              // the operation order of res_mix_norm: (x*rms)*p
            s_w[NS + lane] = wgt/(r == 0.0f ? 1.0f : r);
        }
        if (lane == 0) {
            s_w[2*NS] = rsqrtf(ssy/n_embd + args.norm_eps);
        }
    }
    __syncthreads();
    IK_RDBG_T(9); // weights ready

    // ===== the normalized input: shared memory for the logits; the F32 and Q8_1 outputs spread over the blocks
    //       (block b stores the Q8_1 blocks k/8 with (k/8) % gridDim.x == b) =====
    {
        const float factor = s_w[2*NS];
#pragma unroll
        for (int j = 0; j < C4; ++j) {
            const int  k  = tid + j*IK_R2_BLOCK;
            const bool ok = k < n_embd4;
            float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int i = 0; i < NS; ++i) {
                const float f0 = s_w[i], f1 = s_w[NS + i];
                y.x += (x[i][j].x*f0)*f1;
                y.y += (x[i][j].y*f0)*f1;
                y.z += (x[i][j].z*f0)*f1;
                y.w += (x[i][j].w*f0)*f1;
            }
            const float4 v = make_float4(factor*y.x*nw[j].x, factor*y.y*nw[j].y, factor*y.z*nw[j].z, factor*y.w*nw[j].w);
            const bool mine = ok && (k/8) % gridDim.x == blockIdx.x;
            if (ok) {
                s_x[k] = v;
            }
            if (mine) {
                ((float4 *) args.dst)[k] = v;
            }
            if (args.xq) {
                ik_store_q8_1_vec4(v, args.xq, k, mine); // 8 consecutive lanes hold one Q8_1 block
            }
        }
    }
    IK_RDBG_T(10); // y stored
    cp_async_wait_all();
    __syncthreads();
    IK_RDBG_T(2);

    // ===== the logit of this warp's expert =====
    {
        const float4 * row = s_rows + warp*(IK_R2_MAX_EMBD/4);
        float dot = 0.0f;
        for (int k = lane; k < n_embd4; k += WARP_SIZE) {
            const float4 w = row[k];
            const float4 v = s_x[k];
            dot += w.x*v.x + w.y*v.y + w.z*v.z + w.w*v.w;
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            args.logits[e0 + warp] = dot;
        }
    }
    IK_RDBG_T(3);
    __syncthreads();
    if (tid == 0) {
        s_last = ik_atomic_add_acq_rel(args.counter, 1) == (int) gridDim.x - 1; // releases the logits, acquires the others'
    }
    __syncthreads();
    IK_RDBG_T(4);
    if (!s_last) {
        return;
    }
    if (args.dbg_times && tid == 0) { args.dbg_times[5] = clock64(); }

    if constexpr (has_conv) {
        // All blocks read every channel. Only the last arrival may rotate the
        // history, since the sequence can read and write the same state cell.
        const float * src = args.cv_state + (int64_t) args.cv_rs_ids[0]*args.cv_state_s1;
        float * dst = args.cv_state + (int64_t) args.cv_write_cell*args.cv_state_s1;
        for (int c = tid; c < n_embd; c += IK_R2_BLOCK) {
            const int off = (args.cv_chan_offset + c)*(NC - 1);
            dst[off]     = src[off + 1];
            dst[off + 1] = src[off + 2];
            dst[off + 2] = args.wo_out[c];
        }
    }

    // ===== the selection by the last block: the rank of every expert (ties: the smaller id first) =====
    const int n_expert = args.n_expert;
    if (tid < n_expert) {
        const float p = 1.0f/(1.0f + expf(-args.logits[tid]));
        s_prob[tid] = p;
        s_sel[tid]  = p + args.probs_b[tid];
    }
    __syncthreads();
    if (!MR && args.warp_topk && !__syncthreads_or(tid < n_expert && isnan(s_sel[tid]))) {
        __shared__ unsigned long long winners[32];
        if (tid < 128) {
            const float score = s_sel[tid];
            const unsigned bits = __float_as_uint(score == 0.0f ? 0.0f : score);
            const unsigned ordered = bits ^ (bits & 0x80000000u ? 0xffffffffu : 0x80000000u);
            // Descending keys break tied scores in favor of the smaller expert id.
            const unsigned long long key = (static_cast<unsigned long long>(ordered) << 32) | (127u - tid);
            const auto best = ik_sort_warp(key, lane);
            if (lane < 8) { winners[warp*8 + lane] = best; }
        }
        __syncthreads();
        if (warp == 0) {
            const auto best = ik_sort_warp(winners[lane], lane);
            const int id = 127 - (uint32_t) best;
            const float prob = s_prob[id];
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) { sum += __shfl_sync(0xffffffffu, prob, i); }
            if (lane < 8) {
                args.route[lane] = id;
                args.route[8 + lane] = __float_as_int(prob/(sum + args.w_eps)*args.w_scale);
            }
            if (lane == 0) { *args.counter = 0; }
        }
        return;
    }
    int rank = n_expert;
    if (tid < n_expert) {
        const float mine = s_sel[tid];
        rank = 0;
#pragma unroll 16
        for (int e = 0; e < n_expert; ++e) {
            const float o = s_sel[e];
            rank += (o > mine || (o == mine && e < tid)) ? 1 : 0;
        }
        if (rank < args.n_used) {
            args.route[rank] = tid;
            s_selected[rank] = s_prob[tid];
        }
    }
    __syncthreads();
    if (rank < args.n_used) {
        // Normalize in top-k order, independent of which warp won the atomic race.
        float sum = 0.0f;
        for (int i = 0; i < args.n_used; ++i) {
            sum += s_selected[i];
        }
        const float w = s_prob[tid]/(sum + args.w_eps)*args.w_scale;
        args.route[args.n_used + rank] = __float_as_int(w);
    }
    if (tid == 0) {
        *args.counter = 0;
    }
    if (args.dbg_times && tid == 0) { args.dbg_times[6] = clock64(); }
}

template <int NS, int NC>
static void inkling_router2_launch_nc(ggml_backend_cuda_context & ctx, const ik_router_args & a) {
    const size_t smem = (size_t) (IK_R2_WARPS + 1)*(IK_R2_MAX_EMBD/4)*sizeof(float4);
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = { false };
    if (!attr_set[ctx.device]) {
        CUDA_CHECK(cudaFuncSetAttribute(inkling_router2_kernel<NS, NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        attr_set[ctx.device] = true;
    }
    inkling_router2_kernel<NS, NC><<<a.n_expert/IK_R2_WARPS, IK_R2_BLOCK, smem, ctx.stream()>>>(a);
}

template <int NS>
static void inkling_router2_launch_rows(ggml_backend_cuda_context & ctx, const ik_router_args & a, const int n_rows) {
    const size_t smem = (size_t) (IK_R2_WARPS + 1)*(IK_R2_MAX_EMBD/4)*sizeof(float4);
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = { false };
    if (!attr_set[ctx.device]) {
        CUDA_CHECK(cudaFuncSetAttribute(inkling_router2_kernel<NS, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        attr_set[ctx.device] = true;
    }
    inkling_router2_kernel<NS, 0, true><<<dim3(a.n_expert/IK_R2_WARPS, n_rows), IK_R2_BLOCK, smem, ctx.stream()>>>(a);
}

template <int NS>
static void inkling_router2_launch(ggml_backend_cuda_context & ctx, const ik_router_args & a) {
    if (a.wo_out) {
        GGML_ASSERT(a.cv_n_conv == 4);
        inkling_router2_launch_nc<NS, 4>(ctx, a);
    } else {
        inkling_router2_launch_nc<NS, 0>(ctx, a);
    }
}

void ggml_cuda_op_inkling_router(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int n_states = ggml_get_op_params_i32(dst, 6);
    const ggml_tensor * query    = dst->src[0];
    const ggml_tensor * consts   = dst->src[1 + n_states];
    const ggml_tensor * gate_inp = dst->src[2 + n_states];
    ggml_tensor       * route    = dst->src[3 + n_states];
    ggml_tensor       * xq       = dst->src[4 + n_states];

    ik_router_args a;
    a.query    = query ? (const float *) query->data : nullptr;
    a.n_states = n_states;
    for (int i = 0; i < n_states; ++i) {
        a.st.p[i]   = (const float *) dst->src[1 + i]->data;
        a.st.nb1[i] = 0;
        GGML_ASSERT(((uintptr_t) a.st.p[i]) % 16 == 0);
    }
    a.wo_out = nullptr; a.conv_w = nullptr; a.cv_state = nullptr; a.cv_rs_ids = nullptr; a.cv_residual = nullptr; a.attn_out = nullptr;
    a.cv_state_s1 = 0; a.cv_write_cell = 0; a.cv_chan_offset = 0; a.cv_n_conv = 0; a.cv_scale = 0.0f;
    const bool has_conv = ggml_get_op_params_i32(dst, 10) != 0;
    if (has_conv) {
        const ggml_tensor * wo_out   = dst->src[5 + n_states];
        const ggml_tensor * conv_w   = dst->src[6 + n_states];
        ggml_tensor       * state    = dst->src[7 + n_states];
        const ggml_tensor * rs_ids   = dst->src[8 + n_states];
        const ggml_tensor * residual = dst->src[9 + n_states];
        a.wo_out         = (const float *) wo_out->data;
        a.conv_w         = (const float *) conv_w->data;
        a.cv_state       = (float *) state->data;
        a.cv_state_s1    = state->nb[1]/sizeof(float);
        a.cv_rs_ids      = (const int32_t *) rs_ids->data;
        a.cv_write_cell  = ggml_get_op_params_i32(dst, 7);
        a.cv_chan_offset = ggml_get_op_params_i32(dst, 8);
        a.cv_n_conv      = conv_w->ne[0];
        a.cv_scale       = ggml_get_op_params_f32(dst, 9);
        a.cv_residual    = residual ? (const float *) residual->data : nullptr;
        a.attn_out       = (float *) dst->src[n_states]->data;
        GGML_ASSERT(a.cv_n_conv <= 8 && a.cv_chan_offset % 4 == 0 && ((uintptr_t) a.wo_out) % 16 == 0 && ((uintptr_t) a.attn_out) % 16 == 0 &&
                (a.cv_residual == nullptr || ((uintptr_t) a.cv_residual) % 16 == 0));
    }
    a.norm_w   = (const float *) consts->data;
    a.gate_inp = (const float *) gate_inp->data;
    a.probs_b  = (const float *) consts->data + dst->ne[0];
    a.route    = (int32_t *) route->data;
    a.dst      = (float *) dst->data;
    a.xq       = xq ? (block_q8_1 *) xq->data : nullptr;
    a.n_embd   = dst->ne[0];
    a.n_expert = gate_inp->ne[1];
    a.n_used   = ggml_get_op_params_i32(dst, 3);
    a.eps      = ggml_get_op_params_f32(dst, 0);
    a.scale    = ggml_get_op_params_f32(dst, 1);
    a.norm_eps = ggml_get_op_params_f32(dst, 2);
    a.w_scale  = ggml_get_op_params_f32(dst, 4);
    a.w_eps    = ggml_get_op_params_f32(dst, 5);
    GGML_ASSERT(a.n_embd % 4 == 0 && a.n_embd <= IK_ROUTER_MAX_EMBD && a.n_expert % 8 == 0 && a.n_expert <= IK_ROUTER_MAX_EXPERT);
    GGML_ASSERT(a.n_used <= 8 && n_states <= IK_MAX_STATES);
    GGML_ASSERT(((uintptr_t) a.query) % 16 == 0 && ((uintptr_t) a.norm_w) % 16 == 0 && ((uintptr_t) a.gate_inp) % 16 == 0 && ((uintptr_t) a.dst) % 16 == 0 && (a.xq == nullptr || ((uintptr_t) a.xq) % 16 == 0));

    const int n_rows = dst->ne[1];
    ggml_cuda_pool_alloc<float> logits(ctx.pool(), (size_t) a.n_expert*n_rows);
    a.logits  = logits.get();
    a.counter = n_rows > 1 ? ik_counters(ctx, 128, n_rows) : ik_counters(ctx, 8, 1);
    a.mr_st_rs    = dst->src[1]->nb[1]/sizeof(float);
    a.mr_route_rs = route->nb[1]/sizeof(int32_t);
    a.mr_dst_rs   = dst->nb[1]/sizeof(float);
    a.mr_xq_rs    = xq ? xq->nb[1]/sizeof(block_q8_1) : 0;
    static const bool warp_topk = !getenv("IK_ROUTER_WARP_TOPK") || atoi(getenv("IK_ROUTER_WARP_TOPK"));
    static const int dbg_skip = getenv("IK_ROUTER_SKIP") ? atoi(getenv("IK_ROUTER_SKIP")) : 0;
    a.dbg_skip = dbg_skip;
    static long long * dbg_times = [] () -> long long * {
        if (!getenv("IK_ROUTER_TIMING")) {
            return nullptr;
        }
        long long * t = nullptr;
        CUDA_CHECK(cudaMallocManaged(&t, 16*sizeof(long long)));
        memset(t, 0, 16*sizeof(long long));
        static long long * t_static = t;
        atexit([] {
            cudaDeviceSynchronize();
            fprintf(stderr, "inkling_router phases of block 0 (cycles from T0): loads landed %lld, warp-reduced %lld, block-reduced %lld, weights %lld, y stored %lld, rows ready %lld, logits %lld, arrival %lld; last block: selection %lld cycles\n",
                t_static[7]-t_static[0], t_static[8]-t_static[0], t_static[1]-t_static[0], t_static[9]-t_static[0], t_static[10]-t_static[0], t_static[2]-t_static[0], t_static[3]-t_static[0], t_static[4]-t_static[0], t_static[6]-t_static[5]);
        });
        return t;
    }();
    a.dbg_times = dbg_times;
    a.warp_topk = warp_topk && !a.dbg_skip && !a.dbg_times && a.n_expert == 128 && a.n_used == 8 &&
            ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_AMPERE_GA10X;

    static const bool v1 = getenv("IK_ROUTER_V1") && atoi(getenv("IK_ROUTER_V1")) != 0;
    GGML_ASSERT(!has_conv || (a.n_embd <= IK_R2_MAX_EMBD && a.n_expert <= IK_R2_MAX_EXPERT && n_states <= IK_R2_MAX_STATES));
    if (n_rows > 1) {
        GGML_ASSERT(!has_conv && n_rows <= 8 && a.n_embd <= IK_R2_MAX_EMBD && a.n_expert <= IK_R2_MAX_EXPERT && n_states <= IK_R2_MAX_STATES);
        for (int i = 0; i < n_states; ++i) {
            GGML_ASSERT((int64_t) (dst->src[1 + i]->nb[1]/sizeof(float)) == a.mr_st_rs);
        }
        switch (n_states) {
            case 1:  inkling_router2_launch_rows<1>(ctx, a, n_rows); break;
            case 2:  inkling_router2_launch_rows<2>(ctx, a, n_rows); break;
            case 3:  inkling_router2_launch_rows<3>(ctx, a, n_rows); break;
            case 4:  inkling_router2_launch_rows<4>(ctx, a, n_rows); break;
            default: inkling_router2_launch_rows<5>(ctx, a, n_rows); break;
        }
        return;
    }
    if ((!v1 || has_conv) && a.n_embd <= IK_R2_MAX_EMBD && a.n_expert <= IK_R2_MAX_EXPERT && n_states <= IK_R2_MAX_STATES) {
        switch (n_states) {
            case 1:  inkling_router2_launch<1>(ctx, a); break;
            case 2:  inkling_router2_launch<2>(ctx, a); break;
            case 3:  inkling_router2_launch<3>(ctx, a); break;
            case 4:  inkling_router2_launch<4>(ctx, a); break;
            default: inkling_router2_launch<5>(ctx, a); break;
        }
        return;
    }
    static const int W = getenv("IK_ROUTER_WARPS") ? atoi(getenv("IK_ROUTER_WARPS")) : 8;
    switch (W) {
        case 2:  inkling_router_kernel<2><<<a.n_expert/2, 2*WARP_SIZE, 0, ctx.stream()>>>(a); break;
        case 8:  inkling_router_kernel<8><<<a.n_expert/8, 8*WARP_SIZE, 0, ctx.stream()>>>(a); break;
        default: inkling_router_kernel<4><<<a.n_expert/4, 4*WARP_SIZE, 0, ctx.stream()>>>(a); break;
    }
}
