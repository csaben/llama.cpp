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

struct ik_gate_input {
    int2 lo, hi;
    float d;
};

template<int mode>
static __device__ __forceinline__ void ik_gate_rows(const block_iq4_nl * w, const block_q8_1 * x, float * out, int base, int first, int stride) {
    constexpr int rows = mode == 1 || mode == 4 ? 1 : mode == 2 || mode == 5 || mode == 7 ? 2 : 4;
    constexpr bool cached = mode <= 3 || mode >= 7;
    const int lane = threadIdx.x%32, iqs = 2*(lane%2);
    ik_gate_input a[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int b = 16*j + lane/2;
        if (cached && b < 72) {
            const int * q = (const int *)x[b].qs+iqs;
            a[j] = {make_int2(q[0],q[1]), make_int2(q[4],q[5]), __low2float(x[b].ds)};
        }
    }
    for (int index = mode >= 7 ? first*rows : first; index < 384; index += rows*stride) {
        float partial[rows][4]{};
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            const int b = 16*j+lane/2;
#pragma unroll
            for (int r = 0; r < rows; ++r) {
                const int i = index + (mode >= 7 ? r : r*stride);
                if (b < 72 && i < 384) {
                    const auto wi = ik_iq4_nl_decode(w,(base+i)*72+b,iqs);
                    const float scale = __fmul_rn(wi.d, cached ? a[j].d : __low2float(x[b].ds));
                    int dot = 0;
                    if constexpr (cached) {
                        dot = ggml_cuda_dp4a(wi.v[0].x,a[j].lo.x,dot);
                        dot = ggml_cuda_dp4a(wi.v[0].y,a[j].hi.x,dot);
                        dot = ggml_cuda_dp4a(wi.v[1].x,a[j].lo.y,dot);
                        dot = ggml_cuda_dp4a(wi.v[1].y,a[j].hi.y,dot);
                    } else { dot = ik_iq4_nl_sumi(wi,x+b,iqs); }
                    partial[r][j%4] = __fmaf_rn(scale,(float)dot,partial[r][j%4]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            const int i=index+(mode >= 7 ? r : r*stride);
            const float * v=partial[r];
            const float y=warp_reduce_sum(__fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),v[2]),v[3]));
            if (i < 384 && lane == (base+i)%2) { out[base+i]=y; }
        }
    }
}

template<int taps>
static __global__ void ik_dense_scan_inputs(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x) {
    const int i = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    const int row = i < a.off_gate ? i : i + 3072;
    const int col = a.proj_ncols == 3 && row >= a.off_k ? (row < a.off_v ? 1 : 2) : 0;
    ik_project_row(w, x + col*(2304/QK8_1), (float *) a.proj, row);
}

template<int taps, bool fast = false, int mode = 0>
static __global__ void __launch_bounds__(fast ? 384 : 256) ik_dense_gate_scan(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x,
        float * values, int * counters) {
    constexpr int gates = fast ? 6 : 12;
    constexpr int warps = fast ? 12 : 8;
    const int tid = threadIdx.x;
    const int head = blockIdx.x < 8 ? blockIdx.x : (blockIdx.x - 8)/gates;
    block_q8_1 * dst = (block_q8_1 *) a.dst;
    if (blockIdx.x < 8) {
        a.proj_ncols = 1;
        a.dst = values;
        if (tid < 256) {
            ik_attention_tile<block_q8_0, false, 3, true, false, taps, true, true, false, fast>(a, head, 0);
        }
    } else if constexpr (mode) {
        ik_gate_rows<mode>(w,x,(float *)a.proj,a.off_gate+384*head,((blockIdx.x-8)%gates)*warps+tid/32,warps*gates);
    } else {
        for (int i = ((blockIdx.x - 8)%gates)*warps + tid/32; i < 384; i += warps*gates) {
            ik_project_row(w, x, (float *) a.proj, a.off_gate + head*384 + i);
        }
    }
    __shared__ int last;
    __syncthreads();
    if (tid == 0) { last = ik_atomic_add_acq_rel(counters + head, 1) == gates; }
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

template<int taps, bool fast = false, int mode = 0>
static __global__ void __launch_bounds__(384) ik_dense_warp_roles(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x, int * counters) {
    const int head = blockIdx.x, split = blockIdx.y;
    if (threadIdx.x < 256) {
        a.proj_ncols = 1;
        ik_attention_tile<block_q8_0, true, 3, true, false, taps, false, false, true, fast>(a, head, split, counters + head, a.n_split);
    } else {
        if constexpr (mode) {
            ik_gate_rows<mode>(w,x,(float *)a.proj,a.off_gate+384*head,split*4+(threadIdx.x-256)/32,a.n_split*4);
        } else {
            for (int i = split*4 + (threadIdx.x - 256)/32; i < 384; i += a.n_split*4) {
                ik_project_row(w, x, (float *) a.proj, a.off_gate + 384*head + i);
            }
        }
        asm volatile("bar.sync 2, 128;" ::: "memory");
        if (threadIdx.x == 256) { ik_atomic_add_acq_rel(counters + head, 1); }
    }
}

template<int taps>
static void ik_launch_gate_projection(ggml_backend_cuda_context & ctx, ik_attn_args a, const ggml_tensor * proj, bool short_scan) {
    const auto * w = (const block_iq4_nl *) proj->src[0]->data;
    const auto * x = (const block_q8_1 *) proj->src[1]->data;
    int * counters = ik_counters(ctx, 192, 8);
    ik_dense_scan_inputs<taps><<<(proj->ne[0] - 3072)/4, 128, 0, ctx.stream()>>>(a, w, x);
    static const int grouped = getenv("IK_GATE_REGISTERS") ? atoi(getenv("IK_GATE_REGISTERS")) : 0;
    if (grouped > 0 && grouped <= 8 && ggml_cuda_inkling_fast_glue(ctx)) {
        auto launch = [&](auto choice) {
            constexpr int m = decltype(choice)::value;
            if (short_scan) {
                ggml_cuda_pool_alloc<float> values(ctx.pool(), 3072);
                ik_dense_gate_scan<taps,true,m><<<56,384,0,ctx.stream()>>>(a,w,x,values.get(),counters);
            } else {
                ik_dense_warp_roles<taps,!taps,m><<<dim3(8,a.n_split),384,0,ctx.stream()>>>(a,w,x,counters);
            }
        };
        switch (grouped) {
            case 1: launch(std::integral_constant<int,1>{}); break;
            case 2: launch(std::integral_constant<int,2>{}); break;
            case 3: launch(std::integral_constant<int,3>{}); break;
            case 4: launch(std::integral_constant<int,4>{}); break;
            case 5: launch(std::integral_constant<int,5>{}); break;
            case 6: launch(std::integral_constant<int,6>{}); break;
            case 7: launch(std::integral_constant<int,7>{}); break;
            case 8: launch(std::integral_constant<int,8>{}); break;
        }
        return;
    }
    if (short_scan) {
        ggml_cuda_pool_alloc<float> values(ctx.pool(), 3072);
        if (ggml_cuda_inkling_fast_glue(ctx)) {
            ik_dense_gate_scan<taps, true><<<8*7, 384, 0, ctx.stream()>>>(a, w, x, values.get(), counters);
        } else {
            ik_dense_gate_scan<taps><<<8*13, 256, 0, ctx.stream()>>>(a, w, x, values.get(), counters);
        }
    } else if (!taps && ggml_cuda_inkling_fast_glue(ctx)) {
        ik_dense_warp_roles<taps, true><<<dim3(8, a.n_split), 384, 0, ctx.stream()>>>(a, w, x, counters);
    } else {
        ik_dense_warp_roles<taps><<<dim3(8, a.n_split), 384, 0, ctx.stream()>>>(a, w, x, counters);
    }
}
