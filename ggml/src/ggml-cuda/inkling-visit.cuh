// An input tile produces up to eight rows for one KV head. The last producer runs that head's attention.
struct ik_dense_tile {
    short base, stride;
    unsigned char rows, col;
};

template<int taps, bool gates = true>
struct ik_dense_program {
    ik_dense_tile tiles[138]{};
    int count = 0;

    constexpr void add(int base, int span, int col) {
        for (int r = 0; r < span; r += 8) {
            tiles[count++] = {short(base + r), short(span), (unsigned char) (r + 8 <= span ? 8 : span - r), (unsigned char) col};
        }
    }

    constexpr ik_dense_program() {
        add(0, 384, 0);
        add(3072, 48, 0);
        add(3456, 3, 0);
        if constexpr (gates) { add(3480, 384, 0); }
        add(6552, 128, 1);
        if (taps) { add(7576, 4, 1); }
        add(7576 + 32*taps, 128, 2);
        if (taps) { add(8632, 4, 2); }
    }
};

template<int taps, bool gates = true>
static __device__ __constant__ ik_dense_program<taps, gates> ik_dense_code;

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

static constexpr int IK_GATE_BLOCKS = 12;

template<int taps>
static __global__ void __launch_bounds__(256) ik_dense_gate_scan(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x,
        float * values, int * counters) {
    const int tid = threadIdx.x;
    const int head = blockIdx.x < 8 ? blockIdx.x : (blockIdx.x - 8)/IK_GATE_BLOCKS;
    block_q8_1 * dst = (block_q8_1 *) a.dst;
    if (blockIdx.x < 8) {
        a.proj_ncols = 1;
        a.dst = values;
        ik_attention_tile<block_q8_0, false, 3, true, false, taps, true, true>(a, head, 0);
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

template<int taps>
static __global__ void __launch_bounds__(384) ik_dense_warp_roles(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x, int * counters) {
    const int head = blockIdx.x, split = blockIdx.y;
    if (threadIdx.x < 256) {
        a.proj_ncols = 1;
        ik_attention_tile<block_q8_0, true, 3, true, false, taps, false, false, true>(a, head, split, counters + head, a.n_split);
    } else {
        for (int i = split*4 + (threadIdx.x - 256)/32; i < 384; i += a.n_split*4) {
            ik_project_row(w, x, (float *) a.proj, a.off_gate + 384*head + i);
        }
        asm volatile("bar.sync 2, 128;" ::: "memory");
        if (threadIdx.x == 256) { ik_atomic_add_acq_rel(counters + head, 1); }
    }
}

template<int taps, bool late>
static __global__ void __launch_bounds__(256, 2) ik_dense_drain(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x,
        float * values, int * counts, int workers, bool interleave) {
    constexpr int gates = late ? 12 : 0;
    const int block = blockIdx.x, t = threadIdx.x, warp = t/32;
    const int task = interleave ? block/8 : block%(workers+gates), head = interleave ? block%8 : block/(workers+gates);
    __shared__ int last;
    const auto & program = ik_dense_code<taps, !late>;
    block_q8_1 * out = (block_q8_1 *)a.dst;
    if (task < workers) {
        for (int k = task; k < program.count; k += workers) {
            auto tile = program.tiles[k];
            if (warp < tile.rows) {
                const int row = tile.base + head*tile.stride + warp;
                const int col = a.proj_ncols == 3 ? tile.col : 0;
                ik_project_row(w, x + col*72, (float *)a.proj, row);
            }
        }
        __syncthreads();
        if (!t) { last = ik_atomic_add_acq_rel(counts+head, 1) + 1 == workers; }
        __syncthreads();
        if (!last) { return; }
        if (!t) { ik_st_relaxed(counts+head, 0); }
        a.proj_ncols = 1;
        if constexpr (late) {
            a.dst = values;
            ik_attention_tile<block_q8_0, false, 3, true, false, taps, true, true>(a, head, 0);
        } else {
            ik_attention_tile<block_q8_0, true, 3, true, false, taps, true>(a, head, 0);
        }
    } else if constexpr (late) {
        for (int i = (task-workers)*8 + warp; i < 384; i += gates*8) {
            ik_project_row(w, x, (float *)a.proj, a.off_gate + head*384 + i);
        }
    }
    if constexpr (late) {
        __syncthreads();
        if (!t) { last = ik_atomic_add_acq_rel(counts+8+head, 1) == gates; }
        __syncthreads();
        if (last) {
            if (!t) { ik_st_relaxed(counts+8+head, 0); }
            for (int i = t; i < 384; i += 256) {
                const int c = 384*head + i;
                float y = values[c]*(1.0f/(1.0f + expf(-a.proj[a.off_gate+c])));
                ik_warp_store_q8_1(y, out + c/32, t%32, true);
            }
        }
    }
}

template<int taps>
static void ik_launch_gate_projection(ggml_backend_cuda_context & ctx, ik_attn_args a, const ggml_tensor * proj, bool short_scan) {
    const auto * w = (const block_iq4_nl *) proj->src[0]->data;
    const auto * x = (const block_q8_1 *) proj->src[1]->data;
    int * counters = ik_counters(ctx, 192, 8);
    static const int drain = getenv("IK_DENSE_DRAIN") ? atoi(getenv("IK_DENSE_DRAIN")) : 0;
    static const int workers = getenv("IK_DRAIN_CTAS") ? std::clamp(atoi(getenv("IK_DRAIN_CTAS")), 4, 144) : 48;
    if (short_scan && drain) {
        ggml_cuda_pool_alloc<float> values(ctx.pool(), 3072);
        bool late = abs(drain) == 2;
        auto kernel = late ? ik_dense_drain<taps,true> : ik_dense_drain<taps,false>;
        kernel<<<8*(workers + (late ? 12 : 0)), 256, 0, ctx.stream()>>>(a, w, x, values.get(), ik_counters(ctx, 192, 16), workers, drain < 0);
        return;
    }
    ik_dense_scan_inputs<taps><<<(proj->ne[0] - 3072)/4, 128, 0, ctx.stream()>>>(a, w, x);
    if (short_scan) {
        ggml_cuda_pool_alloc<float> values(ctx.pool(), 3072);
        ik_dense_gate_scan<taps><<<8*(1 + IK_GATE_BLOCKS), 256, 0, ctx.stream()>>>(a, w, x, values.get(), counters);
    } else {
        ik_dense_warp_roles<taps><<<dim3(8, a.n_split), 384, 0, ctx.stream()>>>(a, w, x, counters);
    }
}
