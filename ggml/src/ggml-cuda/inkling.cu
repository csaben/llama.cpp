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

// n_embd % 4 == 0 and n_embd/4 <= block_size: one float4 of every state per thread, a single pass of loads
template <int block_size>
static __global__ void res_mix_f32_vec4(const float4 * __restrict__ query, const ik_state_ptrs st, float4 * __restrict__ dst,
        const int n_embd4, const int n_tok, const int n_states, const int64_t q_nb1, const float eps, const float scale) {
    constexpr int n_warps = block_size/WARP_SIZE;

    const int t  = blockIdx.x;
    const int iq = blockIdx.y;

    const int lane = threadIdx.x % WARP_SIZE;
    const int wid  = threadIdx.x / WARP_SIZE;
    const int k    = threadIdx.x;

    __shared__ float red[n_warps][2*IK_MAX_STATES];
    __shared__ float s_rms[IK_MAX_STATES];
    __shared__ float s_p[IK_MAX_STATES];

    const float4 * q = (const float4 *) ((const char *) query + iq*q_nb1);

    float4 x[IK_MAX_STATES];
    float  acc[2*IK_MAX_STATES];
#pragma unroll
    for (int i = 0; i < IK_MAX_STATES; ++i) {
        x[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        acc[2*i + 0] = 0.0f;
        acc[2*i + 1] = 0.0f;
    }

    if (k < n_embd4) {
        const float4 qk = q[k];
#pragma unroll
        for (int i = 0; i < IK_MAX_STATES; ++i) {
            if (i < n_states) {
                x[i] = ((const float4 *) ((const char *) st.p[i] + t*st.nb1[i]))[k];
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
    __syncthreads();

    if (k < n_embd4) {
        float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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
        dst[((int64_t) iq*n_tok + t)*n_embd4 + k] = y;
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

    bool aligned = n_embd % 4 == 0 && query->nb[1] % 16 == 0;
    for (int i = 0; i < n_states; ++i) {
        aligned = aligned && st.nb1[i] % 16 == 0 && ((uintptr_t) st.p[i]) % 16 == 0;
    }
    if (aligned && n_embd/4 <= 1024) {
        const int n_embd4 = n_embd/4;
        if (n_embd4 <= 256) {
            res_mix_f32_vec4<256><<<grid, 256, 0, ctx.stream()>>>((const float4 *) query->data, st, (float4 *) dst->data,
                    n_embd4, n_tok, n_states, query->nb[1], eps, scale);
        } else if (n_embd4 <= 512) {
            res_mix_f32_vec4<512><<<grid, 512, 0, ctx.stream()>>>((const float4 *) query->data, st, (float4 *) dst->data,
                    n_embd4, n_tok, n_states, query->nb[1], eps, scale);
        } else if (n_embd4 <= 640) {
            res_mix_f32_vec4<640><<<grid, 640, 0, ctx.stream()>>>((const float4 *) query->data, st, (float4 *) dst->data,
                    n_embd4, n_tok, n_states, query->nb[1], eps, scale);
        } else {
            res_mix_f32_vec4<1024><<<grid, 1024, 0, ctx.stream()>>>((const float4 *) query->data, st, (float4 *) dst->data,
                    n_embd4, n_tok, n_states, query->nb[1], eps, scale);
        }
    } else if (n_embd >= 2048) {
        res_mix_f32<1024><<<grid, 1024, 0, ctx.stream()>>>((const float *) query->data, st, (float *) dst->data,
                n_embd, n_tok, n_states, query->nb[1], eps, scale);
    } else {
        res_mix_f32<256><<<grid, 256, 0, ctx.stream()>>>((const float *) query->data, st, (float *) dst->data,
                n_embd, n_tok, n_states, query->nb[1], eps, scale);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// shortconv_res: y = res + (cur + conv([past, cur], w) + dynamic taps)*scale, one thread per (channel, token);
// the thread of token n_t - 1 - k also stores the inputs it saw into state slot k

struct ik_shortconv_args {
    const float * past; const float * w; const float * cur; const float * taps; const float * res;
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

    const float * ps = (const float *) ((const char *) args.past + c*args.past_nb1 + s*args.past_nb2);
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

    ik_shortconv_args args;
    memcpy(&args.scale,          (const float *)   dst->op_params + 0, sizeof(float));
    memcpy(&args.post,           (const int32_t *) dst->op_params + 1, sizeof(int32_t));
    int32_t n_heads_post = 1;
    memcpy(&n_heads_post,        (const int32_t *) dst->op_params + 2, sizeof(int32_t));
    memcpy(&args.post_eps,       (const float *)   dst->op_params + 3, sizeof(float));
    memcpy(&args.post_threshold, (const float *)   dst->op_params + 4, sizeof(float));

    const int d_conv = w->ne[0];
    args.n_chan  = w->ne[1];
    args.n_s     = past->ne[2];
    args.n_t     = cur->ne[1]/args.n_s;
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

// with the norm: a second launch RMS-normalizes the rows in place and scales them by the norm weight
template <int block_size>
static __global__ void rms_norm_mul_inplace_f32(float * __restrict__ y, const float * __restrict__ nw, const int n, const float eps) {
    const int t = blockIdx.x;
    __shared__ float red[block_size/WARP_SIZE];

    float * yt = y + (int64_t) t*n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += block_size) {
        ss += yt[i]*yt[i];
    }
    ss = ik_block_sum<block_size>(ss, red);
    const float scale = rsqrtf(ss/n + eps);
    for (int i = threadIdx.x; i < n; i += block_size) {
        yt[i] = yt[i]*scale*nw[i];
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

    const int block_size = 256;
    const int64_t nblocks = ((int64_t) n*n_tok*WARP_SIZE + block_size - 1)/block_size;

    loop_ctrl_apply_f32<<<nblocks, block_size, 0, ctx.stream()>>>((const float *) cur->data, (const float *) msg->data,
            (const float *) d->data, (const float *) wg->data, (const float *) wf->data, (float *) dst->data,
            n, n_tok, rank, cur->nb[1], msg->nb[1]);

    if (nw) {
        float eps;
        memcpy(&eps, (const float *) dst->op_params + 0, sizeof(float));
        rms_norm_mul_inplace_f32<1024><<<n_tok, 1024, 0, ctx.stream()>>>((float *) dst->data, (const float *) nw->data, n, eps);
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
