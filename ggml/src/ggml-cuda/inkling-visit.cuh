// The scan reads Q, relative, sink, K and V first. Its sigmoid gate is needed only at the final store.
static __device__ __forceinline__ void ik_project_row(const block_iq4_nl * w, const block_q8_1 * x, float * out, int row) {
    constexpr int blocks = 2304/QK4_NL;
    const int lane = threadIdx.x%32;
    const int iqs = 2*(lane%2);
    float partial[4]{};
#pragma unroll
    for (int start = 0; start < blocks; start += 64) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b = start + 16*j + lane/2;
            if (b < blocks) {
                const auto wi = ik_iq4_nl_decode(w, row*blocks + b, iqs);
                const float d = __fmul_rn(wi.d, __low2float(x[b].ds));
                partial[j] = __fmaf_rn(d, float(ik_iq4_nl_sumi(wi, x + b, iqs)), partial[j]);
            }
        }
    }
    const float y = warp_reduce_sum(__fadd_rn(__fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]), partial[3]));
    if (lane == row%2) { out[row] = y; }
}

template<int taps>
static __global__ void ik_dense_scan_inputs(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x) {
    const int i = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    const int row = i < a.off_gate ? i : i + 3072;
    const int col = a.proj_ncols == 3 && row >= a.off_k ? (row < a.off_v ? 1 : 2) : 0;
    ik_project_row(w, x + col*(2304/QK8_1), (float *) a.proj, row);
}

template<int taps>
static __device__ __forceinline__ void ik_prepare_cache_head(ik_attn_args a, int g, int which) {
    const int c = threadIdx.x, lane = c%32, warp = c/32;
    const int i = 128*g + c;
    const int channel = which*1024 + i;
    const float * pr = a.proj + (which ? a.off_v : a.off_k);
    const float * w = a.consts + (which ? a.c_vconv : a.c_kconv) + i*4;
    const float * past = a.state + int64_t(a.rs_ids[0])*a.state_s1 + channel*3;
    const float x[4] = {past[0], past[1], past[2], pr[i]};
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) { sum += x[j]*w[j]; }
    if constexpr (taps) {
        const float * dyn = a.proj + (which ? a.off_vtaps : a.off_ktaps) + 4*g;
#pragma unroll
        for (int j = 0; j < 4; ++j) { sum += x[3-j]*dyn[j]; }
    }
    float y = (x[3] + sum)*(1.0f/sqrtf((taps ? 2 : 1)*4 + 1));
    __shared__ float partial[4];
    float ss = warp_reduce_sum(y*y);
    if (lane == 0) { partial[warp] = ss; }
    __syncthreads();
    ss = partial[0] + partial[1] + partial[2] + partial[3];
    const float factor = which ? rsqrtf(ss/128 + a.norm_eps) : fminf(fmaxf(a.qk_clip/(sqrtf(ss) + a.clip_eps), 0.0f), 1.0f);
    y *= factor;
    const float d = warp_reduce_max(fabsf(y))/127.0f;
    const int q = (int) roundf(y*(d != 0.0f ? 1.0f/d : 0.0f));
    block_q8_0 * dst = (block_q8_0 *) (which ? a.v_cache : a.k_cache) + a.kv_idxs[0]*32 + g*4 + warp;
    dst->qs[lane] = q;
    if (lane == 0) { dst->d = __float2half(d); }
    float * st = a.state + int64_t(a.write_cell)*a.state_s1 + channel*3;
#pragma unroll
    for (int j = 0; j < 3; ++j) { st[j] = x[j+1]; }
}

template<int taps>
static __global__ void ik_project_cache_heads(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x, int * counts) {
    const int warp = threadIdx.x/32;
    if (blockIdx.x >= 512) {
        ik_project_row(w, x, (float *) a.proj, 4*(blockIdx.x - 512) + warp);
        return;
    }
    const int head = blockIdx.x/32, tile = blockIdx.x%32, which = head/8, g = head%8;
    const int col = a.proj_ncols == 3 ? which + 1 : 0;
    x += col*72;
    ik_project_row(w, x, (float *) a.proj, (which ? a.off_v : a.off_k) + 128*g + 4*tile + warp);
    if constexpr (taps) {
        if (tile == 0) { ik_project_row(w, x, (float *) a.proj, (which ? a.off_vtaps : a.off_ktaps) + 4*g + warp); }
    }
    __shared__ int last;
    __syncthreads();
    if (threadIdx.x == 0) { last = ik_atomic_add_acq_rel(counts + head, 1) == 31; }
    __syncthreads();
    if (last) {
        if (threadIdx.x == 0) { ik_st_relaxed(counts + head, 0); }
        ik_prepare_cache_head<taps>(a, g, which);
    }
}

static constexpr int IK_GATE_BLOCKS = 12;

template<int taps, bool prepared = false>
static __global__ void __launch_bounds__(256) ik_dense_gate_scan(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x,
        float * values, int * counters) {
    const int tid = threadIdx.x;
    const int head = blockIdx.x < 8 ? blockIdx.x : (blockIdx.x - 8)/IK_GATE_BLOCKS;
    block_q8_1 * dst = (block_q8_1 *) a.dst;
    if (blockIdx.x < 8) {
        a.proj_ncols = 1;
        a.dst = values;
        ik_attention_tile<block_q8_0, false, 3, true, false, taps, true, true, false, prepared>(a, head, 0);
    } else {
        for (int i = ((blockIdx.x - 8)%IK_GATE_BLOCKS)*8 + tid/32; i < 384; i += 8*IK_GATE_BLOCKS) {
            ik_project_row(w, x, (float *) a.proj, a.off_gate + head*384 + i);
        }
    }
    __shared__ int last;
    __syncthreads();
    if (tid == 0) { last = ik_atomic_add_acq_rel(counters + head, 1) == IK_GATE_BLOCKS; }
    __syncthreads();
    if (last) {
        if (tid == 0) { ik_st_relaxed(counters + head, 0); }
        for (int i = tid; i < 384; i += blockDim.x) {
            const int c = 384*head + i;
            const float y = values[c]*(1.0f/(1.0f + expf(-a.proj[a.off_gate + c])));
            ik_warp_store_q8_1(y, dst + c/QK8_1, tid%32, true);
        }
    }
}

template<int taps, bool prepared = false>
static __global__ void __launch_bounds__(384) ik_dense_warp_roles(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x, int * counters) {
    const int head = blockIdx.x, split = blockIdx.y;
    if (threadIdx.x < 256) {
        a.proj_ncols = 1;
        ik_attention_tile<block_q8_0, true, 3, true, false, taps, false, false, true, prepared>(a, head, split, counters + head, a.n_split);
    } else {
        for (int i = split*4 + (threadIdx.x - 256)/32; i < 384; i += a.n_split*4) {
            ik_project_row(w, x, (float *) a.proj, a.off_gate + 384*head + i);
        }
        asm volatile("bar.sync 2, 128;" ::: "memory");
        if (threadIdx.x == 256) { ik_atomic_add_acq_rel(counters + head, 1); }
    }
}

static __global__ void ik_gate_and_scores(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x, int tiles_per_cta) {
    const int t = threadIdx.x, warp = t/32, lane = t%32;
    if (blockIdx.x >= 8*((a.n_kv+128*tiles_per_cta-1)/(128*tiles_per_cta))) {
        const int tile = blockIdx.x - 8*((a.n_kv+128*tiles_per_cta-1)/(128*tiles_per_cta));
        ik_project_row(w, x, (float *) a.proj, a.off_gate + tile*8 + warp);
        return;
    }
    const int g = blockIdx.x%8;
    __shared__ float q[3][128], rel[3][32], sink[3], sums[3][5];
    const float tau = a.pos ? 1.0f + a.tau_alpha*logf(fmaxf(1.0f,(a.pos[0]+1.0f)/a.tau_floor)) : 1.0f;
    float v[3];
#pragma unroll
    for (int hh = 0; hh < 3; ++hh) {
        const int h = 3*g+hh;
        v[hh] = t < 128 ? a.proj[h*128+t]*a.consts[h*128+t] :
                warp == 4 && lane < 16 ? a.proj[a.off_rel+h*16+lane]*a.consts[a.c_relclip+h*16+lane] : 0.0f;
        const float sum = warp_reduce_sum(v[hh]*v[hh]);
        if (lane == 0 && warp < 5) { sums[hh][warp] = sum; }
        if (t == 0) { sink[hh] = a.proj[a.off_sink+h]; }
    }
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < 3; ++hh) {
        if (t < 128) {
            const float ss = sums[hh][0]+sums[hh][1]+sums[hh][2]+sums[hh][3];
            q[hh][t] = v[hh]*fminf(fmaxf(a.qk_clip/(sqrtf(ss)+a.clip_eps),0.0f),1.0f)*tau;
        } else if (warp == 4) {
            rel[hh][lane] = v[hh]*fminf(fmaxf(a.rel_clip/(sqrtf(sums[hh][4])+a.clip_eps),0.0f),1.0f)*tau;
        }
    }
    __syncthreads();
    ik_mma_lane<3> st;
    ik_mma_init(st, q, sink);
    a.n_rel = 16;
    a.n_extent = 1024;
    const int64_t current = a.kv_idxs[0];
    for (int i = 0; i < tiles_per_cta; ++i) {
        const int cell = ((blockIdx.x/8*tiles_per_cta+i)*8+warp)*16;
        if (cell >= a.n_kv) { break; }
        float scores[2];
        ik_mma_scores(a, st, rel, a.k_cache + g*136, 1088, 0, a.n_kv-1, cell, a.n_kv, current, scores);
        if (lane%4 < 3) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                a.scan_scores[(g*a.n_kv + cell + lane/4 + j*8)*4 + lane%4] = scores[j];
            }
        }
    }
}

template<int taps, bool short_scan>
static __global__ void __launch_bounds__(256) ik_value_scan(ik_attn_args a) {
    a.proj_ncols = 1;
    ik_attention_tile<block_q8_0, true, 3, true, false, taps, short_scan, false, false, false, true>(a, blockIdx.x, blockIdx.y);
}

template<int taps>
static void ik_launch_gate_projection(ggml_backend_cuda_context & ctx, ik_attn_args a, const ggml_tensor * proj, bool short_scan) {
    const auto * w = (const block_iq4_nl *) proj->src[0]->data;
    const auto * x = (const block_q8_1 *) proj->src[1]->data;
    int * counters = ik_counters(ctx, 192, 8);
    static const bool prepare = getenv("IK_PREPARE_KV") && atoi(getenv("IK_PREPARE_KV"));
    if (prepare) {
        ik_project_cache_heads<taps><<<512 + a.off_gate/4, 128, 0, ctx.stream()>>>(a, w, x, ik_counters(ctx, 208, 16));
        if (short_scan) {
            ggml_cuda_pool_alloc<float> values(ctx.pool(), 3072);
            ik_dense_gate_scan<taps, true><<<8*(1 + IK_GATE_BLOCKS), 256, 0, ctx.stream()>>>(a, w, x, values.get(), counters);
        } else {
            ik_dense_warp_roles<taps, true><<<dim3(8, a.n_split), 384, 0, ctx.stream()>>>(a, w, x, counters);
        }
        return;
    }
    ik_dense_scan_inputs<taps><<<(proj->ne[0] - 3072)/4, 128, 0, ctx.stream()>>>(a, w, x);
    static const int score_tiles = getenv("IK_SCORE_TILES") ? atoi(getenv("IK_SCORE_TILES")) : 0;
    if (score_tiles > 0 && a.n_kv%128 == 0) {
        const int tiles = std::min(16, score_tiles);
        ggml_cuda_pool_alloc<float> scores(ctx.pool(), size_t(8)*a.n_kv*4);
        a.scan_scores = scores.get();
        ik_gate_and_scores<<<8*((a.n_kv+128*tiles-1)/(128*tiles)) + 384, 256, 0, ctx.stream()>>>(a, w, x, tiles);
        if (short_scan) {
            ik_value_scan<taps, true><<<8, 256, 0, ctx.stream()>>>(a);
        } else {
            ik_value_scan<taps, false><<<dim3(8, a.n_split), 256, 0, ctx.stream()>>>(a);
        }
        return;
    }
    if (short_scan) {
        ggml_cuda_pool_alloc<float> values(ctx.pool(), 3072);
        ik_dense_gate_scan<taps><<<8*(1 + IK_GATE_BLOCKS), 256, 0, ctx.stream()>>>(a, w, x, values.get(), counters);
    } else {
        ik_dense_warp_roles<taps><<<dim3(8, a.n_split), 384, 0, ctx.stream()>>>(a, w, x, counters);
    }
}
