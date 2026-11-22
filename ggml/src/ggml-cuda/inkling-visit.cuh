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

template<int taps, int min_blocks>
static __global__ void __launch_bounds__(256, min_blocks) ik_dense_queue(ik_attn_args a, ik_dense_visit v, int group) {
    const int block = blockIdx.x;
    const int workers = gridDim.x - 8;
    const int tid = threadIdx.x, warp = tid/32, lane = tid%32;
    const auto & program = ik_dense_code<taps>;
    const int tasks = (program.count + group - 1)/group;
    extern __shared__ uint4 staged[];
    const int first = 2304*block/gridDim.x, end = 2304*(block+1)/gridDim.x;
    auto stage = [&]() {
        const int bytes = (end - first)*96*sizeof(block_iq4_nl);
        for (int b = 16*tid; b < bytes; b += 16*256) {
            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared((char *) staged + b), (const char *)(v.wo + first*96) + b);
        }
        cp_async_commit_group();
    };
    if (v.prefetch == 3) { stage(); }
    if (block < 8) {
        if (v.prefetch == 2) { stage(); }
        if (!tid) { while (ik_ld_acquire(v.counters + block) != tasks) {} }
        __syncthreads();
        a.proj_ncols = 1;
        ik_attention_tile<block_q8_0, true, 3, true, false, taps, true>(a, block, 0);
        __syncthreads();
        if (!tid) {
            ik_st_relaxed(v.counters + block, 0);
            ik_atomic_add_acq_rel(v.counters + 8, 1);
        }
    } else {
        for (int i = block - 8; i < tasks*8; i += workers) {
            const int head = i/tasks;
            const int pc0 = (i%tasks)*group;
            for (int j = pc0; j < min(program.count, pc0 + group); ++j) {
                const auto tile = program.tiles[j];
                if (warp < tile.rows) {
                    const int row = tile.base + head*tile.stride + warp;
                    const int col = a.proj_ncols == 3 ? tile.col : 0;
                    const float val = ik_dense_row<2304>(v.wi, v.x + col*72, row);
                    if (lane == row%2) { ((float *)a.proj)[row] = val; }
                }
            }
            __syncthreads();
            if (!tid) { ik_atomic_add_acq_rel(v.counters + head, 1); }
        }
        if (v.prefetch == 2) { stage(); }
    }
    if (v.prefetch == 1) {
        for (int b = 128*tid; b < (end-first)*1728; b += 128*256) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"((const char *)(v.wo + first*96) + b));
        }
    }
    if (!tid) { while (ik_ld_acquire(v.counters + 8) != 8) {} }
    if (v.prefetch >= 2) { cp_async_wait_all(); }
    __syncthreads();
    const auto * w = v.prefetch >= 2 ? (const block_iq4_nl *) staged : v.wo + first*96;
    for (int row = first + warp; row < end; row += 8) {
        const float val = ik_dense_row<3072>(w, (const block_q8_1 *) a.dst, row - first);
        if (lane == row%2) { ik_dense_store(v.conv, v.out, row, val); }
    }
    __syncthreads();
    if (!tid && ik_atomic_add_acq_rel(v.counters + 9, 1) + 1 == gridDim.x) {
        ik_st_relaxed(v.counters + 8, 0);
        ik_st_relaxed(v.counters + 9, 0);
    }
}

template<int taps, int min_blocks>
static __global__ void __launch_bounds__(256, min_blocks) ik_dense_late_gate(ik_attn_args a, ik_dense_visit v) {
    const int block = blockIdx.x;
    const int workers = gridDim.x - 8;
    const int tid = threadIdx.x, warp = tid/32, lane = tid%32;
    const int last_row = taps ? 8664 : 8600;
    const int first = 2304*block/gridDim.x, end = 2304*(block+1)/gridDim.x;
    extern __shared__ uint4 staged[];
    auto stage = [&]() {
        for (int b = 16*tid; b < (end-first)*1728; b += 16*256) {
            cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared((char *) staged + b), (const char *)(v.wo + first*96) + b);
        }
        cp_async_commit_group();
    };
    if (v.prefetch == 3 || (v.prefetch == 2 && block < 8)) { stage(); }
    if (block < 8) {
        if (v.mode != 8 && !tid) { while (ik_ld_acquire(v.counters) != workers) {} }
        __syncthreads();
        a.proj_ncols = 1;
        ik_attention_tile<block_q8_0, true, 3, true, false, taps, true, true>(a, block, 0, v.counters + 1, workers);
        __syncthreads();
        if (!tid) { ik_atomic_add_acq_rel(v.counters + 8, 1); }
    } else {
        const int row0 = 8*(block - 8) + warp;
        for (int i = row0; v.mode != 8 && i < last_row - 3072; i += 8*workers) {
            const int row = i < 3480 ? i : i + 3072;
            const int col = a.proj_ncols == 3 && row >= 6552 ? (row < 7576 + 32*taps ? 1 : 2) : 0;
            const float val = ik_dense_row<2304>(v.wi, v.x + col*72, row);
            if (lane == row%2) { ((float *)a.proj)[row] = val; }
        }
        __syncthreads();
        if (!tid) { ik_atomic_add_acq_rel(v.counters, 1); }
        for (int i = row0; i < 3072; i += 8*workers) {
            const int row = i + 3480;
            const float val = ik_dense_row<2304>(v.wi, v.x, row);
            if (lane == row%2) { ((float *)a.proj)[row] = val; }
        }
        __syncthreads();
        if (!tid) { ik_atomic_add_acq_rel(v.counters + 1, 1); }
        if (v.prefetch == 2) { stage(); }
    }
    if (v.prefetch == 1) {
        for (int b = 128*tid; b < (end-first)*1728; b += 128*256) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"((const char *)(v.wo + first*96) + b));
        }
    }
    if (!tid) { while (ik_ld_acquire(v.counters + 8) != 8) {} }
    if (v.prefetch >= 2) { cp_async_wait_all(); }
    __syncthreads();
    const auto * w = v.prefetch >= 2 ? (const block_iq4_nl *) staged : v.wo + first*96;
    for (int row = first + warp; row < end; row += 8) {
        const float val = ik_dense_row<3072>(w, (const block_q8_1 *) a.dst, row - first);
        if (lane == row%2) { ik_dense_store(v.conv, v.out, row, val); }
    }
    __syncthreads();
    if (!tid && ik_atomic_add_acq_rel(v.counters + 9, 1) + 1 == gridDim.x) {
        ik_st_relaxed(v.counters, 0);
        ik_st_relaxed(v.counters + 1, 0);
        ik_st_relaxed(v.counters + 8, 0);
        ik_st_relaxed(v.counters + 9, 0);
    }
}

template<int taps>
static __global__ void ik_dense_scan_inputs(ik_attn_args a, ik_dense_visit v) {
    const int i = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    if (i >= (taps ? 8664 : 8600) - 3072) { return; }
    const int row = i < 3480 ? i : i + 3072;
    const int col = a.proj_ncols == 3 && row >= 6552 ? (row < 7576 + 32*taps ? 1 : 2) : 0;
    const float val = ik_dense_row<2304>(v.wi, v.x + col*72, row);
    if (threadIdx.x%32 == row%2) { ((float *) a.proj)[row] = val; }
}

template<int taps>
static __global__ void __launch_bounds__(256) ik_dense_gate_scan(ik_attn_args a, ik_dense_visit v, float * values, int gate_blocks) {
    const int tid = threadIdx.x, warp = tid/32, lane = tid%32;
    const int b = blockIdx.x;
    const int head = b < 8 ? b : (b - 8)/gate_blocks;
    if (b < 8) {
        a.proj_ncols = 1;
        a.dst = values;
        ik_attention_tile<block_q8_0, false, 3, true, false, taps, true, false, true>(a, head, 0);
    } else {
        for (int i = ((b-8)%gate_blocks)*8 + warp; i < 384; i += 8*gate_blocks) {
            const int row = 3480 + head*384 + i;
            const float val = ik_dense_row<2304>(v.wi, v.x, row);
            if (lane == row%2) { ((float *) a.proj)[row] = val; }
        }
    }
    __shared__ int last;
    __syncthreads();
    if (!tid) { last = ik_atomic_add_acq_rel(v.counters + head, 1) == gate_blocks; }
    __syncthreads();
    if (last) {
        if (!tid) { ik_st_relaxed(v.counters + head, 0); }
        for (int i = tid; i < 384; i += 256) {
            const int c = 384*head + i;
            const float y = values[c]*(1.0f/(1.0f + expf(-a.proj[3480 + c])));
            ik_warp_store_q8_1(y, (block_q8_1 *) v.activation + c/32, lane, true);
        }
    }
}

static __global__ void ik_dense_output(ik_dense_visit v) {
    const int row = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    const float y = ik_dense_row<3072>(v.wo, v.activation, row);
    if (threadIdx.x%32 == row%2) { ik_dense_store(v.conv, v.out, row, y); }
}

template<int taps>
static bool ik_launch_dense(ggml_backend_cuda_context & ctx, ik_attn_args a, ik_dense_visit v) {
    const auto & dev = ggml_cuda_info().devices[ctx.device];
    static const int requested = getenv("IK_DENSE_BLOCKS") ? atoi(getenv("IK_DENSE_BLOCKS")) : 10000;
    if (v.mode == 7) {
        ggml_cuda_pool_alloc<float> values(ctx.pool(), 3072);
        static const int gates = getenv("IK_DENSE_GATE_BLOCKS") ? std::max(1, std::min(48, atoi(getenv("IK_DENSE_GATE_BLOCKS")))) : 24;
        ik_dense_scan_inputs<taps><<<((taps ? 8664 : 8600) - 3072)/4,128,0,ctx.stream()>>>(a, v);
        ik_dense_gate_scan<taps><<<8*(1+gates),256,0,ctx.stream()>>>(a, v, values.get(), gates);
        ik_dense_output<<<2304/4,128,0,ctx.stream()>>>(v);
        return true;
    }
    if (v.mode == 2) {
        const int blocks = std::min(std::max(1, requested), ik_dense_program<taps>().count);
        ik_dense_kernel<taps, true><<<dim3(blocks, 8),256,0,ctx.stream()>>>(a, v);
        return true;
    }
    if (!dev.supports_cooperative_launch) { return false; }
    if (v.mode >= 3) {
        const void * kernel = v.mode == 3 ? (const void *) ik_dense_queue<taps, 2> : (const void *) ik_dense_queue<taps, 4>;
        if (v.mode >= 5) { kernel = v.mode == 5 ? (const void *) ik_dense_late_gate<taps, 2> : (const void *) ik_dense_late_gate<taps, 3>; }
        static const int group_env = getenv("IK_DENSE_GROUP") ? atoi(getenv("IK_DENSE_GROUP")) : 2;
        int group = std::max(1, std::min(group_env, 16));
        int active;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 256, 0));
        int grid = std::min(std::max(1, requested)*8, active*dev.nsm);
        const int sh = v.prefetch >= 2 ? ((2304+grid-1)/grid)*1728 : 0;
        if (sh) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 256, sh));
            if (grid > active*dev.nsm) {
                // Try smaller resident grids and opt in to the shared memory they require.
                for (grid = std::min(grid, 3*dev.nsm); grid >= dev.nsm; grid -= dev.nsm) {
                    const int need = ((2304+grid-1)/grid)*1728;
                    cudaFuncAttributes attr;
                    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
                    if (need + attr.sharedSizeBytes > dev.smpbo) { continue; }
                    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, need));
                    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 256, need));
                    if (grid <= active*dev.nsm) { break; }
                }
            }
        }
        if (grid < 9) { return false; }
        const int shared = v.prefetch >= 2 ? ((2304+grid-1)/grid)*1728 : 0;
        if (v.mode == 8) {
            ik_dense_scan_inputs<taps><<<((taps ? 8664 : 8600) - 3072)/4,128,0,ctx.stream()>>>(a, v);
        }
        void * params[] = {&a, &v, &group};
        CUDA_CHECK(cudaLaunchCooperativeKernel((const void *)kernel, dim3(grid), dim3(256), params, shared, ctx.stream()));
        return true;
    }
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
