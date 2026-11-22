// An input tile produces up to eight rows for one KV head. The last producer runs that head's attention.
struct ik_dense_tile {
    short base, stride;
    unsigned char rows, col;
};

template<int taps>
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
        add(3480, 384, 0);
        add(6552, 128, 1);
        if (taps) { add(7576, 4, 1); }
        add(7576 + 32*taps, 128, 2);
        if (taps) { add(8632, 4, 2); }
    }
};

template<int taps>
static __device__ __constant__ ik_dense_program<taps> ik_dense_code;

template<int k>
static __device__ __forceinline__ float ik_dense_row(const block_iq4_nl * w, const block_q8_1 * x, int row) {
    constexpr int n = k/QK4_NL;
    const int lane = threadIdx.x%32;
    const int iqs = 2*(lane%2);
    float p[4]{};
#pragma unroll
    for (int start = 0; start < n; start += 64) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b = start + j*16 + lane/2;
            if (b < n) {
                const auto wi = ik_iq4_nl_decode(w, row*n + b, iqs);
                const auto d = __fmul_rn(wi.d, __low2float(x[b].ds));
                p[j] = __fmaf_rn(d, float(ik_iq4_nl_sumi(wi, x + b, iqs)), p[j]);
            }
        }
    }
    return warp_reduce_sum(__fadd_rn(__fadd_rn(__fadd_rn(p[0], p[1]), p[2]), p[3]));
}

template<int taps, bool prefix>
static __global__ void __launch_bounds__(256) ik_dense_kernel(ik_attn_args a, ik_dense_visit v) {
    const int head = blockIdx.y;
    const int worker = blockIdx.x;
    const int warp = threadIdx.x/32, lane = threadIdx.x%32;
    const auto & program = ik_dense_code<taps>;
    for (int pc = worker; pc < program.count; pc += gridDim.x) {
        const auto task = program.tiles[pc];
        if (warp < task.rows) {
            const int r = task.base + head*task.stride + warp;
            const int col = a.proj_ncols == 3 ? task.col : 0;
            const float val = ik_dense_row<2304>(v.wi, v.x + col*72, r);
            if (lane == r%2) { ((float *)a.proj)[r] = val; }
        }
    }
    __syncthreads();
    __shared__ int last;
    if (threadIdx.x == 0) { last = ik_atomic_add_acq_rel(v.counters + head, 1) + 1 == gridDim.x; }
    __syncthreads();
    if (last) {
        if (!threadIdx.x) { ik_st_relaxed(v.counters + head, 0); }
        a.proj_ncols = 1;
        ik_attention_tile<block_q8_0, true, 3, true, false, taps, true>(a, head, 0);
        __syncthreads();
        if constexpr (!prefix) {
            if (!threadIdx.x) { ik_atomic_add_acq_rel(v.counters + 8, 1); }
        }
    }
    if constexpr (prefix) {
        return;
    }
    const int block = worker + head*gridDim.x;
    const int stride = 8*gridDim.x*gridDim.y;
    if (v.prefetch) {
        const char * w = (const char *) v.wo;
        for (int r = 8*block; r < 2304; r += stride) {
            for (int byte = 128*threadIdx.x; byte < 8*96*(int)sizeof(block_iq4_nl); byte += 128*256) {
                asm volatile("prefetch.global.L2 [%0];" :: "l"(w + r*96*sizeof(block_iq4_nl) + byte));
            }
        }
    }
    if (!threadIdx.x) { while (ik_ld_acquire(v.counters + 8) != 8) {} }
    __syncthreads();
    for (int r = 8*block + warp; r < 2304; r += stride) {
        const float val = ik_dense_row<3072>(v.wo, (const block_q8_1 *) a.dst, r);
        if (lane == r%2) { ik_dense_store(v.conv, v.out, r, val); }
    }
    __syncthreads();
    if (!threadIdx.x && ik_atomic_add_acq_rel(v.counters + 9, 1) + 1 == gridDim.x*gridDim.y) {
        ik_st_relaxed(v.counters + 8, 0);
        ik_st_relaxed(v.counters + 9, 0);
    }
}

template<int taps>
static bool ik_launch_dense(ggml_backend_cuda_context & ctx, ik_attn_args a, ik_dense_visit v) {
    const auto & dev = ggml_cuda_info().devices[ctx.device];
    static const int requested = getenv("IK_DENSE_BLOCKS") ? atoi(getenv("IK_DENSE_BLOCKS")) : 10000;
    if (v.mode == 2) {
        const int blocks = std::min(std::max(1, requested), ik_dense_program<taps>().count);
        ik_dense_kernel<taps, true><<<dim3(blocks, 8),256,0,ctx.stream()>>>(a, v);
        return true;
    }
    if (!dev.supports_cooperative_launch) { return false; }
    const auto kernel = ik_dense_kernel<taps, false>;
    static const int per_sm = [&] {
        int n;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, 256, 0));
        return n;
    }();
    const int blocks = std::min(std::max(1, requested), per_sm*dev.nsm/8);
    void * params[] = {&a, &v};
    CUDA_CHECK(cudaLaunchCooperativeKernel((const void *) kernel, dim3(blocks, 8), dim3(256), params, 0, ctx.stream()));
    return true;
}
