// GPU0 experiment. Included at the graph evaluator so both replays use the live allocations.
#include "ggml-cuda/vecdotq.cuh"

struct ik_floor_op {
    ggml_tensor * mm;
    ggml_tensor * attn;
    int ids[8];
    const char * packed;
};

static bool ik_floor_graph(const ggml_cgraph * g) {
    int dense = 0, expert = 0, visits = 0;
    for (int i = 0; i < g->n_nodes; ++i) {
        const auto * n = g->nodes[i];
        if (n->op == GGML_OP_INKLING_ATTN) {
            if (n->ne[0] != 3072 || ggml_nelements(n->src[6]) != 1) { return false; }
            ++visits;
        } else if (n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_MUL_MAT_ID) {
            if (!ggml_is_contiguous(n) || !ggml_is_contiguous(n->src[0]) || ggml_nelements(n) > 201024 ||
                    ggml_nelements(n->src[1]) > 32768 || !ggml_is_quantized(n->src[0]->type)) { return false; }
            if (n->op == GGML_OP_MUL_MAT_ID) {
                if (n->src[0]->ne[2] != 128 || ggml_nelements(n->src[2]) != 8 || !ggml_is_contiguous(n->src[2])) { return false; }
                ++expert;
            } else {
                ++dense;
            }
        }
    }
    return visits == 48 && expert == 96 && dense == 97;
}

static ik_floor_op ik_floor_capture(ggml_backend_cuda_context & ctx, ggml_cgraph * g, int i) {
    ik_floor_op op{g->nodes[i], nullptr, {}, nullptr};
    if (op.mm->op == GGML_OP_MUL_MAT_ID) {
        CUDA_CHECK(cudaMemcpyAsync(op.ids, op.mm->src[2]->data, sizeof(op.ids), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        for (int j = 0; j < 8; ++j) {
            GGML_ASSERT(op.ids[j] >= 0 && op.ids[j] < 128 && std::find(op.ids, op.ids + j, op.ids[j]) == op.ids + j);
        }
    } else if (op.mm->ne[1] == 3) {
        ++i;
        while (i < g->n_nodes && ggml_cuda_is_view_or_noop(g->nodes[i])) { ++i; }
        GGML_ASSERT(i < g->n_nodes && g->nodes[i]->op == GGML_OP_INKLING_ATTN);
        op.attn = g->nodes[i];
    }
    if (ggml_cuda_env_flag("IK_FLOOR_PACK") && op.mm->op == GGML_OP_MUL_MAT && op.mm->src[0]->ne[1] < 200000) {
        auto * x = op.mm->src[1];
        if (x->op == GGML_OP_INKLING_ATTN) {
            GGML_ASSERT(x->src[11]);
            op.packed = (const char *) x->src[11]->data + ggml_nbytes(x->src[0]->src[0]);
        } else {
            while (i < g->n_nodes && g->nodes[i]->op != GGML_OP_INKLING_ATTN) { ++i; }
            GGML_ASSERT(i < g->n_nodes && g->nodes[i]->src[0] == op.mm && g->nodes[i]->src[11]);
            op.packed = (const char *) g->nodes[i]->src[11]->data;
        }
    }
    return op;
}

struct ik_floor_tile {
    const char * w;
    const block_q8_1 * x;
    int out, kind, rows, row_bytes;
};

template<int type, int k, bool packed = false>
static __device__ __forceinline__ float ik_floor_dot(const char * w, const block_q8_1 * x, int row) {
    constexpr int qk = type == GGML_TYPE_IQ4_NL ? QK4_NL : type == GGML_TYPE_Q2_K ? QK_K : QK2_0;
    constexpr int lanes = type == GGML_TYPE_IQ4_NL ? QI4_NL/VDR_IQ4_NL_Q8_1_MMVQ : type == GGML_TYPE_Q2_K ? QI2_K/VDR_Q2_K_Q8_1_MMVQ : QI2_0;
    const int lane = threadIdx.x % 32;
    float sum = 0;
#pragma unroll
    for (int j = 0; j < k/qk; j += 32/lanes) {
        const int b = j + lane/lanes;
        if (b < k/qk) {
            const auto * y = x + b*(qk/QK8_1);
            const int bi = row*(k/qk) + b;
            if constexpr (type == GGML_TYPE_IQ4_NL) {
                if constexpr (packed) {
                    constexpr int n = k/QK4_NL;
                    const char * base = w + row*n*sizeof(block_iq4_nl);
                    const uint2 codes = ((const uint2 *)base)[2*b + lane%2];
                    ik_iq4_nl_w wi;
                    wi.v[0] = get_int_from_table_16(codes.x, kvalues_iq4nl);
                    wi.v[1] = get_int_from_table_16(codes.y, kvalues_iq4nl);
                    wi.d = __half2float(((const half *)(base + 16*n))[b]);
                    sum += ik_iq4_nl_dot(wi, y, (lane%2)*2);
                } else {
                    sum += vec_dot_iq4_nl_q8_1(w, y, bi, (lane%lanes)*VDR_IQ4_NL_Q8_1_MMVQ);
                }
            } else if constexpr (type == GGML_TYPE_Q2_K) {
                sum += vec_dot_q2_K_q8_1(w, y, bi, (lane%lanes)*VDR_Q2_K_Q8_1_MMVQ);
            } else {
                sum += vec_dot_q2_0_q8_1(w, y, bi, lane%lanes);
            }
        }
    }
    return warp_reduce_sum(sum);
}

template<bool read_only>
static __global__ void ik_floor_stream(const ik_floor_tile * tiles, int n, float * out) {
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        const auto t = tiles[i];
        if constexpr (read_only) {
            const int bytes = t.rows*t.row_bytes;
            int j = 16*threadIdx.x;
            for (; j + 16 <= bytes; j += 16*blockDim.x) {
                const char * p = t.w + j;
                asm volatile(" { .reg .b32 a,b,c,d; ld.volatile.global.v4.b32 {a,b,c,d}, [%0]; } " :: "l"(p) : "memory");
            }
            for (int b = bytes/16*16 + threadIdx.x; b < bytes; b += blockDim.x) {
                asm volatile(" { .reg .b32 a; ld.volatile.global.u8 a, [%0]; } " :: "l"(t.w + b) : "memory");
            }
        } else {
            for (int r = threadIdx.x/32; r < t.rows; r += blockDim.x/32) {
                float v;
                switch (t.kind) {
                    case 0: v = ik_floor_dot<GGML_TYPE_IQ4_NL, 2304>(t.w, t.x, r); break;
                    case 1: v = ik_floor_dot<GGML_TYPE_IQ4_NL, 3072>(t.w, t.x, r); break;
                    case 2: v = ik_floor_dot<GGML_TYPE_IQ4_NL, 1152>(t.w, t.x, r); break;
                    case 3: v = ik_floor_dot<GGML_TYPE_Q2_K, 2304>(t.w, t.x, r); break;
                    case 5: v = ik_floor_dot<GGML_TYPE_IQ4_NL, 2304, true>(t.w, t.x, r); break;
                    case 6: v = ik_floor_dot<GGML_TYPE_IQ4_NL, 3072, true>(t.w, t.x, r); break;
                    default: v = ik_floor_dot<GGML_TYPE_Q2_0, 1152>(t.w, t.x, r); break;
                }
                if (threadIdx.x%32 == 0) { out[t.out + r] = v; }
            }
        }
    }
}

static __global__ void ik_floor_inputs(block_q8_1 * x) {
    const int b = blockIdx.x;
    const int lane = threadIdx.x;
    const int q = ((lane*7 + b*3)%31) - 15;
    x[b].qs[lane] = q;
    const float sum = warp_reduce_sum(float(q));
    if (!lane) {
        const float d = (1 + b%7)/64.f;
        x[b].ds = make_half2(d, d*sum);
    }
}

static void ik_floor_native(ggml_backend_cuda_context & ctx, const ik_floor_op & op, block_q8_1 * x, int * ids, float * out) {
    auto mm = *op.mm, act = *mm.src[1];
    act.type = GGML_TYPE_Q8_1;
    act.nb[0] = sizeof(block_q8_1);
    act.nb[1] = ggml_row_size(act.type, act.ne[0]);
    act.nb[2] = act.ne[1]*act.nb[1];
    act.nb[3] = act.ne[2]*act.nb[2];
    act.data = x;
    mm.src[1] = &act;
    mm.data = out;
    if (mm.op == GGML_OP_MUL_MAT_ID) {
        auto route = *mm.src[2];
        route.data = ids;
        mm.src[2] = &route;
        if (!ggml_cuda_inkling_q2_down(ctx, &mm)) {
            ggml_cuda_mul_mat_vec_q(ctx, mm.src[0], &act, &route, &mm);
        }
    } else if (op.attn) {
        ggml_cuda_inkling_segmented_projection(ctx, &mm, op.attn);
    } else {
        ggml_cuda_mul_mat_vec_q(ctx, mm.src[0], &act, nullptr, &mm);
    }
}

template<typename F>
static void ik_floor_time(cudaStream_t stream, const char * name, F launch) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 16; ++i) { launch(); }
    for (int rep = 0; rep < 5; ++rep) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < 64; ++i) { launch(); }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        fprintf(stderr, "floor time %s %d %.6f ms\n", name, rep, ms/64);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

static void ik_floor_measure(ggml_backend_cuda_context & ctx, const std::vector<ik_floor_op> & ops) {
    GGML_ASSERT(ops.size() == 193);
    const auto stream = ctx.stream();
    ggml_cuda_pool_alloc<block_q8_1> x(ctx.pool(), 1024);
    ggml_cuda_pool_alloc<int> ids(ctx.pool(), 8*ops.size());
    ik_floor_inputs<<<1024,32,0,stream>>>(x.get());
    const int tile_rows = ggml_cuda_env_int("IK_FLOOR_TILE", 16);
    GGML_ASSERT(tile_rows > 0 && tile_rows <= 256);
    std::vector<ik_floor_tile> tiles;
    std::vector<int> offsets, boundaries;
    size_t bytes = 0;
    int outputs = 0;
    for (size_t i = 0; i < ops.size(); ++i) {
        const auto & op = ops[i];
        const auto * w = op.mm->src[0];
        const int k = w->ne[0], rows = w->ne[1], row_bytes = w->nb[1];
        const bool routed = op.mm->op == GGML_OP_MUL_MAT_ID;
        const int slots = routed ? 8 : 1;
        const int qrows = op.attn ? ggml_get_op_params_i32(op.attn, 0)*(2*ggml_get_op_params_i32(op.attn, 2) + ggml_get_op_params_i32(op.attn, 3) + 1) : rows;
        const int krows = (rows - qrows)/2;
        GGML_ASSERT(qrows > 0 && qrows <= rows && (!op.attn || krows > 0));
        int kind = -1;
        if (w->type == GGML_TYPE_IQ4_NL) { kind = k == 2304 ? 0 : k == 3072 ? 1 : k == 1152 ? 2 : -1; }
        if (w->type == GGML_TYPE_Q2_K && k == 2304) { kind = 3; }
        if (w->type == GGML_TYPE_Q2_0 && k == 1152) { kind = 4; }
        GGML_ASSERT(kind >= 0);
        if (op.packed) { GGML_ASSERT(kind < 2); kind += 5; }
        offsets.push_back(outputs);
        boundaries.push_back(tiles.size());
        fprintf(stderr, "floor op %zu %s %s k=%d rows=%d bytes=%zu ids=", i, w->name, ggml_type_name(w->type), k, rows, size_t(slots)*rows*row_bytes);
        for (int slot = 0; slot < slots; ++slot) {
            const int id = routed ? op.ids[slot] : 0;
            if (routed) { fprintf(stderr, "%s%d", slot ? "," : "", id); }
            for (int r = 0; r < rows;) {
                const int col = r < qrows ? 0 : r < qrows + krows ? 1 : 2;
                const int end = col == 0 ? qrows : col == 1 ? qrows + krows : rows;
                const int count = std::min(tile_rows, end - r);
                const int xcol = routed ? slot%op.mm->src[1]->ne[1] : col;
                const char * ptr = (op.packed ? op.packed : (const char *) w->data) + id*w->nb[2] + r*w->nb[1];
                GGML_ASSERT(uintptr_t(ptr)%16 == 0);
                tiles.push_back({ptr, x.get() + xcol*(k/QK8_1), outputs + (routed ? slot : col)*rows + r, kind, count, row_bytes});
                r += count;
            }
        }
        fprintf(stderr, "\n");
        CUDA_CHECK(cudaMemcpyAsync(ids.get() + 8*i, op.ids, sizeof(op.ids), cudaMemcpyHostToDevice, stream));
        outputs += ggml_nelements(op.mm);
        bytes += size_t(slots)*rows*row_bytes;
    }
    boundaries.push_back(tiles.size());
    ggml_cuda_pool_alloc<float> ref(ctx.pool(), outputs), out(ctx.pool(), outputs);
    ggml_cuda_pool_alloc<ik_floor_tile> device_tiles(ctx.pool(), tiles.size());
    CUDA_CHECK(cudaMemcpyAsync(device_tiles.get(), tiles.data(), tiles.size()*sizeof(tiles[0]), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(ref.get(), 0, outputs*sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(out.get(), 0, outputs*sizeof(float), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    for (size_t i = 0; i < ops.size(); ++i) {
        ik_floor_native(ctx, ops[i], x.get(), ids.get() + 8*i, ref.get() + offsets[i]);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    GGML_ASSERT(nodes == 193);
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    int sm = ggml_cuda_info().devices[ctx.device].nsm;
    const int threads = ggml_cuda_env_int("IK_FLOOR_THREADS", 128);
    int active = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, ik_floor_stream<false>, threads, 0));
    int blocks = sm*std::min(active, ggml_cuda_env_int("IK_FLOOR_CTA", active));
    GGML_ASSERT(blocks >= sm && threads%32 == 0);
    fprintf(stderr, "floor inventory bytes=%zu ops=%zu nodes=%zu tiles=%zu tile_rows=%d sm=%d threads=%d grid=%d\n", bytes, ops.size(), nodes, tiles.size(), tile_rows, sm, threads, blocks);
    auto native = [&] { CUDA_CHECK(cudaGraphLaunch(exec, stream)); };
    auto fused = [&] { ik_floor_stream<false><<<blocks,threads,0,stream>>>(device_tiles.get(), tiles.size(), out.get()); };
    auto read = [&] { ik_floor_stream<true><<<blocks,threads,0,stream>>>(device_tiles.get(), tiles.size(), out.get()); };
    native();
    fused();
    std::vector<float> a(outputs), b(outputs);
    CUDA_CHECK(cudaMemcpyAsync(a.data(), ref.get(), outputs*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(b.data(), out.get(), outputs*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int checked = 0, mismatches = 0;
    float max_abs = 0, max_scaled = 0;
    for (const auto & t : tiles) {
        for (int r = t.out; r < t.out + t.rows; ++r) {
            const float diff = fabsf(a[r] - b[r]);
            GGML_ASSERT(std::isfinite(a[r]) && std::isfinite(b[r]));
            max_abs = std::max(max_abs, diff);
            max_scaled = std::max(max_scaled, diff/(1 + fabsf(a[r])));
            mismatches += diff > 2e-5f*(1 + fabsf(a[r]));
            ++checked;
        }
    }
    fprintf(stderr, "floor check rows=%d mismatches=%d max_abs=%g max_scaled=%g\n", checked, mismatches, max_abs, max_scaled);
    GGML_ASSERT(mismatches == 0);
    cudaGraph_t tile_graph;
    cudaGraphExec_t tile_exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    for (size_t i = 0; i < ops.size(); ++i) {
        const int n = boundaries[i + 1] - boundaries[i];
        ik_floor_stream<false><<<std::min(blocks, n),threads,0,stream>>>(device_tiles.get() + boundaries[i], n, out.get());
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &tile_graph));
    CUDA_CHECK(cudaGraphGetNodes(tile_graph, nullptr, &nodes));
    GGML_ASSERT(nodes == 193);
    CUDA_CHECK(cudaGraphInstantiate(&tile_exec, tile_graph, nullptr, nullptr, 0));
    ik_floor_time(stream, "stock-graph", native);
    ik_floor_time(stream, "persistent", fused);
    ik_floor_time(stream, "read", read);
    ik_floor_time(stream, "same-dot-193", [&] { CUDA_CHECK(cudaGraphLaunch(tile_exec, stream)); });
    CUDA_CHECK(cudaGraphExecDestroy(tile_exec));
    CUDA_CHECK(cudaGraphDestroy(tile_graph));
    ik_floor_time(stream, "stock-repeat", native);
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
}
