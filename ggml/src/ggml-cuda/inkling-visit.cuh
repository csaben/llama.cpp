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

template<int taps, bool short_scan, int role = 0>
static __global__ void __launch_bounds__(256) ik_dense_gate_scan(ik_attn_args a, ik_dense_visit v, float * values, int gate_blocks) {
    const int tid = threadIdx.x, warp = tid/32, lane = tid%32;
    const int b = blockIdx.x + (role == 2 ? 8*a.n_split : 0);
    const int scans = 8*a.n_split;
    const int head = b < scans ? b%8 : (b - scans)/gate_blocks;
    if (role == 1 || (role == 0 && b < scans)) {
        a.proj_ncols = 1;
        a.dst = values;
        ik_attention_tile<block_q8_0, false, 3, true, false, taps, short_scan, false, true>(a, head, b/8);
    } else {
        for (int i = ((b-scans)%gate_blocks)*(blockDim.x/32) + warp; i < 384; i += (blockDim.x/32)*gate_blocks) {
            const int row = 3480 + head*384 + i;
            const float val = ik_dense_row<2304>(v.wi, v.x, row);
            if (lane == row%2) { ((float *) a.proj)[row] = val; }
        }
    }
    __shared__ int last;
    __syncthreads();
    if (!tid) { last = ik_atomic_add_acq_rel(v.counters + head, 1) == gate_blocks + a.n_split - 1; }
    __syncthreads();
    if (last) {
        if (!tid) { ik_st_relaxed(v.counters + head, 0); }
        for (int i = tid; i < 384; i += blockDim.x) {
            const int c = 384*head + i;
            const float y = values[c]*(1.0f/(1.0f + expf(-a.proj[3480 + c])));
            ik_warp_store_q8_1(y, (block_q8_1 *) v.activation + c/32, lane, true);
        }
    }
}

template<int taps, int producers>
static __global__ void __launch_bounds__(256 + 32*producers) ik_dense_warp_roles(ik_attn_args a, ik_dense_visit v) {
    const int head = blockIdx.x, split = blockIdx.y;
    if (threadIdx.x < 256) {
        a.proj_ncols = 1;
        ik_attention_tile<block_q8_0, true, 3, true, false, taps, false, true, false, true>(
            a, head, split, v.counters + head, a.n_split);
    } else {
        const int warp = (threadIdx.x-256)/32, lane = threadIdx.x%32;
        for (int i = split*producers + warp; i < 384; i += a.n_split*producers) {
            const int row = 3480 + 384*head + i;
            const float val = ik_dense_row<2304>(v.wi, v.x, row);
            if (lane == row%2) { ((float *)a.proj)[row] = val; }
        }
        asm volatile("bar.sync 2, %0;" :: "n"(32*producers) : "memory");
        if (threadIdx.x == 256) { ik_atomic_add_acq_rel(v.counters + head, 1); }
    }
}

static __global__ void ik_dense_output(ik_dense_visit v) {
    const int row = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    const float y = ik_dense_row<3072>(v.wo, v.activation, row);
    if (threadIdx.x%32 == row%2) { ik_dense_store(v.conv, v.out, row, y); }
}

static void ik_router_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ik_dense_visit * visit);

struct ik_dense_mix_args {
    const float4 * query;
    ik_state_ptrs st;
    ik_res_mix_norm norm;
    int cols, states;
    int64_t q_nb1;
    float eps, scale;
};

template<int taps>
static __global__ void __launch_bounds__(640, 1) ik_dense_mix_input(ik_attn_args a, ik_dense_visit v, ik_dense_mix_args m, int preload) {
    const int block = blockIdx.x, tid = threadIdx.x;
    const int workers = gridDim.x - m.cols;
    if (block < m.cols) {
        ik_mix_tile<640, true>(m.query, m.st, (void *)v.x, 576, 1, m.states, m.q_nb1, m.eps, m.scale, m.norm, 0, block);
        __syncthreads();
        if (!tid) { ik_atomic_add_acq_rel(v.counters + 12, 1); }
        return;
    }
    const int b = block - m.cols;
    constexpr int rows = (taps ? 8664 : 8600) - 3072;
    const int first = rows*b/workers, end = rows*(b+1)/workers;
    const int take = min(preload, end-first);
    extern __shared__ uint4 tile[];
    for (int i = tid; i < take*81; i += 640) {
        const int logical = first + i/81;
        const int row = logical < 3480 ? logical : logical + 3072;
        const auto * w = (const uint4 *)(v.wi + 72*row);
        cp_async_cg_16<0>(ggml_cuda_cvta_generic_to_shared(tile + i), w + i%81);
    }
    if (take) { cp_async_commit_group(); }
    if (!tid) { while (ik_ld_acquire(v.counters + 12) != m.cols) {} }
    if (take) { cp_async_wait_all(); }
    __syncthreads();
    for (int i = tid/32; i < end-first; i += 20) {
        const int logical = first + i;
        const int row = logical < 3480 ? logical : logical + 3072;
        const int col = m.cols == 3 && row >= 6552 ? (row < 7576 + 32*taps ? 1 : 2) : 0;
        const auto * w = i < take ? (const block_iq4_nl *)tile + i*72 : v.wi + row*72;
        const float y = ik_dense_row<2304>(w, v.x + col*72, 0);
        if (tid%32 == row%2) { ((float *)a.proj)[row] = y; }
    }
    __syncthreads();
    if (!tid && ik_atomic_add_acq_rel(v.counters + 13, 1) + 1 == workers) {
        ik_st_relaxed(v.counters + 12, 0);
        ik_st_relaxed(v.counters + 13, 0);
    }
}

template<int taps>
static void ik_launch_mix_input(ggml_backend_cuda_context & ctx, ik_attn_args a, ik_dense_visit v) {
    const auto * mix = v.mix;
    ik_dense_mix_args m{};
    m.cols = mix->ne[2];
    m.states = ik_res_mix_state_count(mix);
    GGML_ASSERT(m.states >= 1 && m.states <= IK_MAX_STATES);
    m.query = mix->src[0] ? (const float4 *)mix->src[0]->data : nullptr;
    m.q_nb1 = m.query ? mix->src[0]->nb[1] : 0;
    for (int i = 0; i < m.states; ++i) {
        m.st.p[i] = (const float *)mix->src[1+i]->data;
        m.st.nb1[i] = mix->src[1+i]->nb[1];
    }
    const auto * norm = mix->src[1 + m.states];
    m.norm.nw = (const float *) norm->data;
    m.norm.nw_nb1 = ggml_nrows(norm) > 1 ? norm->nb[1] : 0;
    m.eps = ggml_get_op_params_f32(mix, 0);
    m.scale = ggml_get_op_params_f32(mix, 1);
    m.norm.eps = ggml_get_op_params_f32(mix, 2);
    static const int rows = getenv("IK_MIX_PRELOAD") ? atoi(getenv("IK_MIX_PRELOAD")) : 20;
    int preload = std::min(std::max(rows, 0), 56);
    const int shared = preload*1296;
    const auto kernel = ik_dense_mix_input<taps>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    int active;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, 640, shared));
    GGML_ASSERT(active > 0);
    const int blocks = active*ggml_cuda_info().devices[ctx.device].nsm;
    void * params[] = {&a, &v, &m, &preload};
    CUDA_CHECK(cudaLaunchCooperativeKernel((const void *)kernel, dim3(blocks), dim3(640), params, shared, ctx.stream()));
}

template<int taps>
static bool ik_launch_dense(ggml_backend_cuda_context & ctx, ik_attn_args a, ik_dense_visit v) {
    const auto & dev = ggml_cuda_info().devices[ctx.device];
    static const int requested = getenv("IK_DENSE_BLOCKS") ? atoi(getenv("IK_DENSE_BLOCKS")) : 10000;
    if (v.mode == 7 || v.mode >= 10) {
        ggml_cuda_pool_alloc<float> values(ctx.pool(), 3072);
        static const int gates = getenv("IK_DENSE_GATE_BLOCKS") ? std::max(1, std::min(48, atoi(getenv("IK_DENSE_GATE_BLOCKS")))) : 24;
        if (v.mix) {
            ik_launch_mix_input<taps>(ctx, a, v);
        } else {
            ik_dense_scan_inputs<taps><<<((taps ? 8664 : 8600) - 3072)/4,128,0,ctx.stream()>>>(a, v);
        }
        const int blocks = 8*(a.n_split + gates);
        if (v.mode >= 14 && a.n_kv > 256) {
            const dim3 grid(8, a.n_split);
            if (v.mode == 14) { ik_dense_warp_roles<taps, 4><<<grid,384,0,ctx.stream()>>>(a, v); }
            else { ik_dense_warp_roles<taps, 8><<<grid,512,0,ctx.stream()>>>(a, v); }
        } else if (v.mode == 11 || v.mode == 12) {
            const auto side = ctx.stream(ctx.device, 1);
            cudaEvent_t fork, join;
            CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(fork, ctx.stream()));
            CUDA_CHECK(cudaStreamWaitEvent(side, fork));
            const int threads = v.mode == 11 ? 128 : 256;
            ik_dense_gate_scan<taps, true, 2><<<8*gates,threads,0,side>>>(a, v, values.get(), gates);
            if (a.n_kv == 256 && a.n_split == 1) {
                ik_dense_gate_scan<taps, true, 1><<<8,256,0,ctx.stream()>>>(a, v, values.get(), gates);
            } else {
                ik_dense_gate_scan<taps, false, 1><<<8*a.n_split,256,0,ctx.stream()>>>(a, v, values.get(), gates);
            }
            CUDA_CHECK(cudaEventRecord(join, side));
            CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), join));
            CUDA_CHECK(cudaEventDestroy(fork));
            CUDA_CHECK(cudaEventDestroy(join));
        } else if (a.n_kv == 256 && a.n_split == 1) {
            ik_dense_gate_scan<taps, true><<<blocks,256,0,ctx.stream()>>>(a, v, values.get(), gates);
        } else {
            ik_dense_gate_scan<taps, false><<<blocks,256,0,ctx.stream()>>>(a, v, values.get(), gates);
        }
        if (v.router) {
            ik_router_dispatch(ctx, v.router, &v);
        } else {
            ik_dense_output<<<2304/4,128,0,ctx.stream()>>>(v);
        }
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
