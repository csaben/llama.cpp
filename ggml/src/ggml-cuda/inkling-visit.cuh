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

template<int taps, bool fast_tail = false>
static __global__ void __launch_bounds__(256) ik_dense_gate_scan(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x,
        float * values, int * counters) {
    const int tid = threadIdx.x;
    const int head = blockIdx.x < 8 ? blockIdx.x : (blockIdx.x - 8)/IK_GATE_BLOCKS;
    block_q8_1 * dst = (block_q8_1 *) a.dst;
    if (blockIdx.x < 8) {
        a.proj_ncols = 1;
        a.dst = values;
        ik_attention_tile<block_q8_0, false, 3, true, false, taps, true, true, false, fast_tail>(a, head, 0);
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

template<int taps, bool fast_tail = false, bool mask_tiles = false>
static __global__ void __launch_bounds__(384) ik_dense_warp_roles(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x, int * counters) {
    const int head = blockIdx.x, split = blockIdx.y;
    if (threadIdx.x < 256) {
        a.proj_ncols = 1;
        ik_attention_tile<block_q8_0, true, 3, true, false, taps, false, false, true, fast_tail, mask_tiles>(a, head, split, counters + head, a.n_split);
    } else {
        for (int i = split*4 + (threadIdx.x - 256)/32; i < 384; i += a.n_split*4) {
            ik_project_row(w, x, (float *) a.proj, a.off_gate + 384*head + i);
        }
        asm volatile("bar.sync 2, 128;" ::: "memory");
        if (threadIdx.x == 256) { ik_atomic_add_acq_rel(counters + head, 1); }
    }
}

template<int taps>
static __global__ void __launch_bounds__(256, 2) ik_coop_gate_scan(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x,
        int * counters, int blocks) {
    const int t = threadIdx.x;
    if (blockIdx.x < 8) {
        a.proj_ncols = 1;
        ik_attention_tile<block_q8_0, true, 3, true, false, taps, true, false, false, true>(a, blockIdx.x, 0, counters+blockIdx.x, blocks);
    } else {
        const int g = (blockIdx.x-8)/blocks;
        for (int i = ((blockIdx.x-8)%blocks)*8 + t/32; i < 384; i += blocks*8) {
            ik_project_row(w, x, (float *) a.proj, a.off_gate + g*384+i);
        }
        __syncthreads();
        if (t == 0) { ik_atomic_add_acq_rel(counters+g, 1); }
    }
}

template<int taps>
static void ik_launch_gate_projection(ggml_backend_cuda_context & ctx, ik_attn_args a, const ggml_tensor * proj, bool short_scan) {
    const auto * w = (const block_iq4_nl *) proj->src[0]->data;
    const auto * x = (const block_q8_1 *) proj->src[1]->data;
    int * counters = ik_counters(ctx, 192, 8);
    ik_dense_scan_inputs<taps><<<(proj->ne[0] - 3072)/4, 128, 0, ctx.stream()>>>(a, w, x);
    static const int mode = getenv("IK_SCAN_SCHEDULE") ? atoi(getenv("IK_SCAN_SCHEDULE")) : 0;
    if (short_scan && (mode & 4) && ggml_cuda_info().devices[ctx.device].supports_cooperative_launch &&
            ctx.curr_stream_no == 0 && ctx.stream_context().concurrent_events.empty()) {
        static const int requested = getenv("IK_SCAN_GATE_BLOCKS") ? atoi(getenv("IK_SCAN_GATE_BLOCKS")) : 12;
        int blocks = std::max(1, std::min(48, requested)), resident;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, ik_coop_gate_scan<taps>, 256, 0));
        if (8*(1+blocks) <= resident*ggml_cuda_info().devices[ctx.device].nsm) {
            void * params[] = {&a, &w, &x, &counters, &blocks};
            CUDA_CHECK(cudaLaunchCooperativeKernel((void *) ik_coop_gate_scan<taps>, 8*(1+blocks), 256, params, 0, ctx.stream()));
            return;
        }
    }
    if (short_scan) {
        ggml_cuda_pool_alloc<float> values(ctx.pool(), 3072);
        if (mode & 1) {
            ik_dense_gate_scan<taps, true><<<8*(1 + IK_GATE_BLOCKS), 256, 0, ctx.stream()>>>(a, w, x, values.get(), counters);
            return;
        }
        ik_dense_gate_scan<taps><<<8*(1 + IK_GATE_BLOCKS), 256, 0, ctx.stream()>>>(a, w, x, values.get(), counters);
    } else {
        const dim3 grid(8, a.n_split);
        if ((mode & 2) && !taps) {
            if (mode & 1) { ik_dense_warp_roles<taps,true,true><<<grid,384,0,ctx.stream()>>>(a,w,x,counters); }
            else { ik_dense_warp_roles<taps,false,true><<<grid,384,0,ctx.stream()>>>(a,w,x,counters); }
            return;
        }
        if (mode & 1) {
            ik_dense_warp_roles<taps,true><<<grid,384,0,ctx.stream()>>>(a,w,x,counters);
            return;
        }
        ik_dense_warp_roles<taps><<<dim3(8, a.n_split), 384, 0, ctx.stream()>>>(a, w, x, counters);
    }
}
