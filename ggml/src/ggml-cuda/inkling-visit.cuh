// The scan reads Q, relative, sink, K and V first. Its sigmoid gate is needed only at the final store.
template<bool sig>
static __device__ __forceinline__ void ik_project_row_sig(const block_iq4_nl * w, const block_q8_1 * x, float * out, int row) {
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
    if (lane == row%2) { out[row] = sig ? 1.0f/(1.0f+expf(-y)) : y; }
}

static __device__ __forceinline__ void ik_project_row(const block_iq4_nl * w, const block_q8_1 * x,float * out,int row) {
    ik_project_row_sig<false>(w,x,out,row);
}

template<int taps>
static __global__ void ik_dense_scan_inputs(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x) {
    const int i = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    const int row = i < a.off_gate ? i : i + 3072;
    const int col = a.proj_ncols == 3 && row >= a.off_k ? (row < a.off_v ? 1 : 2) : 0;
    ik_project_row(w, x + col*(2304/QK8_1), (float *) a.proj, row);
}

static __device__ __forceinline__ int2 ik_gate_lookup(uint32_t word) {
    const uint32_t * table = (const uint32_t *) kvalues_iq4nl;
    const uint32_t select = 0x32103210u | ((word & 0x88888888u) >> 1);
    const uint32_t lo = __byte_perm(__byte_perm(table[0],table[1],word), __byte_perm(table[2],table[3],word), select);
    word >>= 16;
    const uint32_t hi = __byte_perm(__byte_perm(table[0],table[1],word), __byte_perm(table[2],table[3],word), select>>16);
    return make_int2(lo,hi);
}

template<int rows, bool packed>
static __device__ __forceinline__ void ik_gate_dot_tile(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x, int row) {
    const int lane = threadIdx.x%32, iqs = 2*(lane%2);
    float part[rows][4]{};
#pragma unroll
    for (int tile = 0; tile < 5; ++tile) {
        const int b = 16*tile + lane/2;
        if (b < 72) {
#pragma unroll
            for (int r = 0; r < rows; ++r) {
                const int y = row+r;
                ik_iq4_nl_w wi;
                if constexpr (packed) {
                    const int k = (y/8)*576 + tile*128 + (y%8)*(tile == 4 ? 8 : 16) + lane/2;
                    const uint2 codes = ((const uint2 *) a.gate_tiles)[k*2+lane%2];
                    wi.v[0] = ik_gate_lookup(codes.x);
                    wi.v[1] = ik_gate_lookup(codes.y);
                    wi.d = __half2float(((const half *) (a.gate_tiles+3072*72*16))[y*72+b]);
                } else {
                    wi = ik_iq4_nl_decode(w, (a.off_gate+y)*72+b, iqs);
                }
                const float d = __fmul_rn(wi.d,__low2float(x[b].ds));
                part[r][tile%4] = __fmaf_rn(d,float(ik_iq4_nl_sumi(wi,x+b,iqs)),part[r][tile%4]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < rows; ++r) {
        const float y = warp_reduce_sum(__fadd_rn(__fadd_rn(__fadd_rn(part[r][0],part[r][1]),part[r][2]),part[r][3]));
        if (lane == (row+r)%2) { ((float *)a.proj)[a.off_gate+row+r] = y; }
    }
}

static constexpr int IK_GATE_BLOCKS = 12;

template<int taps, bool fast_tail = false, int rows = 0, bool packed = false, int sched = 0>
static __global__ void __launch_bounds__(256) ik_dense_gate_scan(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x,
        float * values, int * counters) {
    const int tid = threadIdx.x;
    const int head = blockIdx.x < 8 ? blockIdx.x : (blockIdx.x - 8)/IK_GATE_BLOCKS;
    block_q8_1 * dst = (block_q8_1 *) a.dst;
    if (blockIdx.x < 8) {
        a.proj_ncols = 1;
        a.dst = values;
        ik_attention_tile<block_q8_0, false, 3, true, false, taps, true, true, false, fast_tail, false, sched>(a, head, 0);
    } else {
        if constexpr (rows != 0) {
            for (int i = (((blockIdx.x-8)%IK_GATE_BLOCKS)*8+tid/32)*rows; i < 384; i += 8*IK_GATE_BLOCKS*rows) {
                ik_gate_dot_tile<rows,packed>(a,w,x,head*384+i);
            }
        } else {
        for (int i = ((blockIdx.x - 8)%IK_GATE_BLOCKS)*8 + tid/32; i < 384; i += 8*IK_GATE_BLOCKS) {
            ik_project_row(w, x, (float *) a.proj, a.off_gate + head*384 + i);
        }
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

template<int taps, bool fast_tail = false, bool mask_tiles = false, int rows = 0, bool packed = false, int sched = 0>
static __global__ void __launch_bounds__(384) ik_dense_warp_roles(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x, int * counters) {
    const int head = blockIdx.x, split = blockIdx.y;
    if (threadIdx.x < 256) {
        a.proj_ncols = 1;
        ik_attention_tile<block_q8_0, true, 3, true, false, taps, false, false, true, fast_tail, mask_tiles, sched>(a, head, split, counters + head, a.n_split);
    } else {
        if constexpr (rows != 0) {
            for (int i = (split*4+(threadIdx.x-256)/32)*rows; i < 384; i += a.n_split*4*rows) {
                ik_gate_dot_tile<rows,packed>(a,w,x,head*384+i);
            }
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

#include "inkling-short.cuh"

template<int taps>
static __global__ void ik_scan_inputs_with_gate(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x, int early) {
    const int i = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    const int input_rows = taps ? 5592 : 5528;
    const int row = i < input_rows ? (i < a.off_gate ? i : i+3072) :
        a.off_gate + ((i-input_rows)/early)*384 + (i-input_rows)%early;
    const int col = a.proj_ncols == 3 && row >= a.off_k ? (row < a.off_v ? 1 : 2) : 0;
    ik_project_row(w,x + col*72,(float *)a.proj,row);
}

static __device__ __forceinline__ void ik_gpu_stamp(unsigned long long * ptr, int i) {
    if (ptr && threadIdx.x == 0) {
        unsigned long long v;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
        ptr[3*blockIdx.x+i] = v;
    }
}

template<int taps, bool short_scan>
static __global__ void __launch_bounds__(256) ik_wave_gate_scan(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x,
        float * values, int * ctr, int gates, int early, bool interleave, unsigned long long * times) {
    const int scans = a.n_split;
    int g, k;
    if (interleave) {
        g = blockIdx.x/(scans+gates); k = blockIdx.x%(scans+gates);
    } else {
        g = blockIdx.x < 8*scans ? blockIdx.x%8 : (blockIdx.x-8*scans)/gates;
        k = blockIdx.x < 8*scans ? blockIdx.x/8 : scans + (blockIdx.x-8*scans)%gates;
    }
    ik_gpu_stamp(times,0);
    block_q8_1 * dst = (block_q8_1 *)a.dst;
    if (k < scans) {
        a.proj_ncols = 1;
        a.dst = values;
        ik_attention_tile<block_q8_0,false,3,true,false,taps,short_scan,true,false,false,!short_scan && !taps>(a,g,k);
    } else {
        for (int i = early+(k-scans)*8+threadIdx.x/32; i < 384; i += gates*8) {
            ik_project_row(w,x,(float *)a.proj,a.off_gate+g*384+i);
        }
    }
    __shared__ int last;
    __syncthreads();
    ik_gpu_stamp(times,1);
    if (threadIdx.x == 0) { last = ik_atomic_add_acq_rel(ctr+g,1) == gates+scans-1; }
    __syncthreads();
    if (last) {
        if (threadIdx.x == 0) { ik_st_relaxed(ctr+g,0); }
        for (int i = threadIdx.x; i < 384; i += 256) {
            const int c = 384*g+i;
            const float v = values[c]*(1.0f/(1.0f+expf(-a.proj[a.off_gate+c])));
            ik_warp_store_q8_1(v,dst+c/32,threadIdx.x%32,true);
        }
    }
    ik_gpu_stamp(times,2);
}

static unsigned long long * ik_wave_times(int taps, bool short_scan) {
    static unsigned long long * times = []() -> unsigned long long * {
        if (!getenv("IK_WAVE_TIMING")) { return nullptr; }
        unsigned long long * out;
        CUDA_CHECK(cudaMallocManaged(&out,4*2048*3*sizeof(*out)));
        memset(out,0,4*2048*3*sizeof(*out));
        static unsigned long long * ptr = out;
        atexit([]{
            cudaDeviceSynchronize();
            for (int i = 0; i < 4*2048; ++i) {
                if (ptr[3*i]) { fprintf(stderr,"r4 phase %d %d %llu %llu %llu\n",i/2048,i%2048,ptr[3*i],ptr[3*i+1],ptr[3*i+2]); }
            }
        });
        return out;
    }();
    return times ? times + (2*short_scan+taps)*2048*3 : nullptr;
}

template<int taps, int threads, int sched>
static __global__ void __launch_bounds__(threads) ik_short_geometry(ik_attn_args a, const block_iq4_nl * w, const block_q8_1 * x,
        float * values, int * ctr, int gates, bool share, bool permute) {
    const int t=threadIdx.x, bx=blockIdx.x;
    const bool scan=permute ? bx%(gates+1)==0 : bx<8;
    const int g=permute ? bx/(gates+1) : (scan ? bx : (bx-8)/gates);
    const int chunk=permute ? bx%(gates+1)-1 : (bx-8)%gates;
    block_q8_1 * dst=(block_q8_1 *)a.dst;
    if (scan && t<256) {
        a.proj_ncols=1; a.dst=values;
        ik_attention_tile<block_q8_0,false,3,true,false,taps,true,true,false,false,false,sched,(threads>256)>(a,g,0);
    } else if (!scan || share) {
        const int extra=share ? (threads-256)/32 : 0;
        const int warp=scan ? (t-256)/32 : extra+chunk*(threads/32)+t/32;
        for (int i=warp;i<384;i+=extra+gates*(threads/32)) {
            ik_project_row(w,x,(float *)a.proj,a.off_gate+384*g+i);
        }
    }
    __shared__ int last;
    __syncthreads();
    if (t==0) { last=ik_atomic_add_acq_rel(ctr+g,1)==gates; }
    __syncthreads();
    if (last) {
        if (t==0) { ik_st_relaxed(ctr+g,0); }
        for (int i=t; i<384; i+=threads) {
            const int c=384*g+i;
            const float v=values[c]*(1.0f/(1.0f+expf(-a.proj[a.off_gate+c])));
            ik_warp_store_q8_1(v,dst+c/32,t%32,true);
        }
    }
}

template<int gates,int sched>
static __global__ void __launch_bounds__(512+gates*32) ik_ahead_gate_scan(ik_attn_args a,const block_iq4_nl * w,const block_q8_1 * x,int * ctr) {
    const int t=threadIdx.x,g=blockIdx.x,split=blockIdx.y;
    if (t<512) {
        a.proj_ncols=1;
        ik_attention_tile<block_q8_0,true,3,true,false,0,false,false,true,false,true,sched,false,true>(a,g,split,ctr+g,a.n_split);
    } else {
        for (int i=split*gates+(t-512)/32;i<384;i+=a.n_split*gates) {
            ik_project_row(w,x,(float *)a.proj,a.off_gate+g*384+i);
        }
        asm volatile("bar.sync 2, %0;" :: "n"(32*gates) : "memory");
        if (t==512) { ik_atomic_add_acq_rel(ctr+g,1); }
    }
}

template<bool linear, bool presig>
static __device__ __forceinline__ void ik_join_tile(const ik_attn_args & a, int g, int qb) {
    const int h=qb/4, lane=threadIdx.x%32, c=(qb%4)*32+lane, count=a.n_split;
    const int group=g*3+h;
    const float * meta=linear ? a.part+8*3*count*128+group*count*2 : a.part+(g*count*3+h)*130+128;
    const int stride=linear ? 2 : 390;
    const float * data=linear ? a.part+(group*count)*128+c : a.part+(g*count*3+h)*130+c;
    float mx=-INFINITY, av=0, den=0;
    for (int base=0;base<count;base+=8) {
        const int sp=base+lane;
        const bool live=lane<8 && sp<count;
        const float m=live ? meta[sp*stride] : -INFINITY;
        const float l=live ? meta[sp*stride+1] : 0;
        float value[8];
#pragma unroll
        for (int k=0;k<8;++k) {
            value[k]=base+k<count ? data[(base+k)*(linear ? 128 : 390)] : 0;
        }
        const float next=fmaxf(mx,warp_reduce_max(m));
        if (next!=-INFINITY) {
            const float prev=mx!=-INFINITY ? expf(mx-next) : 0.0f;
            av*=prev; den*=prev;
            const float coeff=m!=-INFINITY ? expf(m-next) : 0.0f;
            const unsigned active=__ballot_sync(0xffffffffu,m!=-INFINITY);
#pragma unroll
            for (int k=0;k<8;++k) {
                if (active&(1u<<k)) {
                    const float v=__shfl_sync(0xffffffffu,coeff,k);
                    av+=value[k]*v;
                    den+=__shfl_sync(0xffffffffu,l,k)*v;
                }
            }
        }
        mx=next;
    }
    const int i=384*g+qb*32+lane;
    const float peak=fmaxf(mx,0.0f), corr=expf(mx-peak);
    const float gv=a.proj[a.off_gate+i];
    const float val=av*corr/(den*corr+expf(-peak))*(presig ? gv : 1.0f/(1.0f+expf(-gv)));
    ik_warp_store_q8_1(val,(block_q8_1 *)a.dst+i/32,lane,true);
}

template<int taps, int mode>
static __global__ void __launch_bounds__(384) ik_join_gate_scan(ik_attn_args a,const block_iq4_nl * w,const block_q8_1 * x,int * ctr) {
    constexpr bool linear=(mode&1)!=0, coop=(mode&2)!=0, presig=(mode&4)!=0;
    const int t=threadIdx.x,g=blockIdx.x,sp=blockIdx.y;
    if (t<256) {
        a.proj_ncols=1;
        ik_attention_tile<block_q8_0,true,3,true,false,taps,false,false,true,false,!taps,8,false,false,linear ? 2 : 1>(a,g,sp);
    } else {
        for (int i=sp*4+(t-256)/32;i<384;i+=a.n_split*4) {
            const int row=a.off_gate+g*384+i;
            ik_project_row_sig<presig>(w,x,(float *)a.proj,row);
        }
    }
    if constexpr (coop) {
        cooperative_groups::this_grid().sync();
        for (int qb=sp+(t/32)*a.n_split;qb<12;qb+=12*a.n_split) {
            ik_join_tile<linear,presig>(a,g,qb);
        }
    } else {
        __shared__ int last;
        __syncthreads();
        if (t==0) { last=ik_atomic_add_acq_rel(ctr+g,1)==a.n_split-1; }
        __syncthreads();
        if (last) {
            if (!t) { ik_st_relaxed(ctr+g,0); }
            ik_join_tile<linear,presig>(a,g,t/32);
        }
    }
}

template<int taps>
static void ik_launch_gate_projection(ggml_backend_cuda_context & ctx, ik_attn_args a, const ggml_tensor * proj, bool short_scan) {
    const auto * w = (const block_iq4_nl *) proj->src[0]->data;
    const auto * x = (const block_q8_1 *) proj->src[1]->data;
    int * counters = ik_counters(ctx, 192, 8);
    static const int wave = getenv("IK_GATE_WAVE") ? atoi(getenv("IK_GATE_WAVE")) : 0;
    if (wave) {
        static const int early = getenv("IK_GATE_EARLY") ? std::min(384, std::max(0,atoi(getenv("IK_GATE_EARLY"))/8)*8) : 0;
        static const bool interleave = getenv("IK_WAVE_INTERLEAVE") && atoi(getenv("IK_WAVE_INTERLEAVE"));
        const int gates = early == 384 ? 0 : std::min(48,wave);
        if (early) {
            ik_scan_inputs_with_gate<taps><<<(proj->ne[0]-3072+8*early)/4,128,0,ctx.stream()>>>(a,w,x,early);
        } else {
            ik_dense_scan_inputs<taps><<<(proj->ne[0]-3072)/4,128,0,ctx.stream()>>>(a,w,x);
        }
        ggml_cuda_pool_alloc<float> values(ctx.pool(),3072);
        auto * times = ik_wave_times(taps,short_scan);
        if (short_scan) {
            ik_wave_gate_scan<taps,true><<<8*(a.n_split+gates),256,0,ctx.stream()>>>(a,w,x,values.get(),counters,gates,early,interleave,times);
        } else {
            ik_wave_gate_scan<taps,false><<<8*(a.n_split+gates),256,0,ctx.stream()>>>(a,w,x,values.get(),counters,gates,early,interleave,times);
        }
        return;
    }
    ik_dense_scan_inputs<taps><<<(proj->ne[0] - 3072)/4, 128, 0, ctx.stream()>>>(a, w, x);
    static const int join=getenv("IK_JOIN_GATE") ? atoi(getenv("IK_JOIN_GATE")) : 0;
    if (join && !short_scan && a.n_split>1) {
        auto run=[&](auto opt) {
            constexpr int bits=decltype(opt)::value;
            const auto kernel=ik_join_gate_scan<taps,bits>;
            if constexpr ((bits&2)!=0) {
                int resident;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,384,0));
                if (!ggml_cuda_info().devices[ctx.device].supports_cooperative_launch || 8*a.n_split>resident*ggml_cuda_info().devices[ctx.device].nsm) { return false; }
                void * params[]={&a,&w,&x,&counters};
                CUDA_CHECK(cudaLaunchCooperativeKernel((void *)kernel,dim3(8,a.n_split),384,params,0,ctx.stream()));
            } else { kernel<<<dim3(8,a.n_split),384,0,ctx.stream()>>>(a,w,x,counters); }
            return true;
        };
        if (join==1 && run(std::integral_constant<int,0>{})) { return; }
        if (join==2 && run(std::integral_constant<int,1>{})) { return; }
        if (join==3 && run(std::integral_constant<int,2>{})) { return; }
        if (join==4 && run(std::integral_constant<int,3>{})) { return; }
        if (join==5 && run(std::integral_constant<int,4>{})) { return; }
        if (join==6 && run(std::integral_constant<int,5>{})) { return; }
        if (join==7 && run(std::integral_constant<int,6>{})) { return; }
        if (join==8 && run(std::integral_constant<int,7>{})) { return; }
    }
    static const int ahead=getenv("IK_SCORE_AHEAD") ? atoi(getenv("IK_SCORE_AHEAD")) : 0;
    if (ahead && !taps && !short_scan && a.kv_per_split<=256) {
        auto run=[&](auto n,auto order) {
            constexpr int gates=decltype(n)::value, sched=decltype(order)::value;
            ik_ahead_gate_scan<gates,sched><<<dim3(8,a.n_split),512+32*gates,0,ctx.stream()>>>(a,w,x,counters);
        };
        if (ahead==1) { run(std::integral_constant<int,4>{},std::integral_constant<int,0>{}); }
        else if (ahead==2) { run(std::integral_constant<int,4>{},std::integral_constant<int,8>{}); }
        else if (ahead==3) { run(std::integral_constant<int,8>{},std::integral_constant<int,8>{}); }
        else { run(std::integral_constant<int,2>{},std::integral_constant<int,8>{}); }
        return;
    }
    static const int cached=getenv("IK_STAGE_CACHE") ? atoi(getenv("IK_STAGE_CACHE")) : 0;
    if (cached && !short_scan && (uintptr_t)a.k_cache%8==0 && (uintptr_t)a.v_cache%8==0) {
        auto launch=[&](auto setting) {
            constexpr int mode=decltype(setting)::value;
            constexpr int smem=8*272*sizeof(uint2)*((mode&3)==3 ? 2 : 1);
            const auto kernel=ik_dense_warp_roles<taps,false,!taps,0,false,(mode<<4)>;
            static bool attr[GGML_CUDA_MAX_DEVICES]{};
            if (!attr[ctx.device]) {
                CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
                attr[ctx.device]=true;
            }
            kernel<<<dim3(8,a.n_split),384,smem,ctx.stream()>>>(a,w,x,counters);
        };
        if (cached==1) { launch(std::integral_constant<int,1>{}); return; }
        if (cached==2) { launch(std::integral_constant<int,2>{}); return; }
        if (cached==3) { launch(std::integral_constant<int,3>{}); return; }
        if (cached==5) { launch(std::integral_constant<int,5>{}); return; }
        if (cached==6) { launch(std::integral_constant<int,6>{}); return; }
        if (cached==7) { launch(std::integral_constant<int,7>{}); return; }
    }
    static const int geometry=getenv("IK_GATE_THREADS") ? atoi(getenv("IK_GATE_THREADS")) : 0;
    if (geometry && short_scan) {
        static const int gates=getenv("IK_GEOMETRY_BLOCKS") ? std::min(48,std::max(1,atoi(getenv("IK_GEOMETRY_BLOCKS")))) : 8;
        static const int opts=getenv("IK_GATE_OPTS") ? atoi(getenv("IK_GATE_OPTS")) : 1;
        ggml_cuda_pool_alloc<float> values(ctx.pool(),3072);
        auto run=[&](auto width,auto order) {
            constexpr int nt=decltype(width)::value;
            ik_short_geometry<taps,nt,decltype(order)::value><<<8*(gates+1),nt,0,ctx.stream()>>>(a,w,x,values.get(),counters,gates,opts&2,opts&4);
        };
        auto choose=[&](auto order) {
            if (geometry==256) { run(std::integral_constant<int,256>{},order); return; }
            if (geometry==288) { run(std::integral_constant<int,288>{},order); return; }
            if (geometry==320) { run(std::integral_constant<int,320>{},order); return; }
            if (geometry==384) { run(std::integral_constant<int,384>{},order); return; }
            if (geometry==512) { run(std::integral_constant<int,512>{},order); return; }
        };
        if (opts&1) { choose(std::integral_constant<int,1>{}); } else { choose(std::integral_constant<int,0>{}); }
        return;
    }
    static const int local = getenv("IK_LOCAL_SCAN") ? atoi(getenv("IK_LOCAL_SCAN")) : 0;
    if (local && short_scan) {
        static const int gates = getenv("IK_LOCAL_GATES") ? std::min(48,std::max(1,atoi(getenv("IK_LOCAL_GATES")))) : 12;
        ggml_cuda_pool_alloc<float> values(ctx.pool(),3072);
        auto run = [&](auto width, auto late) {
            constexpr int n = decltype(width)::value;
            ik_local_gate_scan<taps,n,decltype(late)::value><<<8*(1+gates),n,0,ctx.stream()>>>(a,w,x,values.get(),counters,gates);
        };
        if (local == 1) { run(std::integral_constant<int,256>{},std::false_type{}); }
        else if (local == 2) { run(std::integral_constant<int,288>{},std::false_type{}); }
        else if (local == 3) { run(std::integral_constant<int,384>{},std::false_type{}); }
        else if (local == 4) { run(std::integral_constant<int,256>{},std::true_type{}); }
        else if (local == 5) { run(std::integral_constant<int,288>{},std::true_type{}); }
        else { run(std::integral_constant<int,384>{},std::true_type{}); }
        return;
    }
    static const int mma = getenv("IK_MMA_ORDER") ? atoi(getenv("IK_MMA_ORDER")) : 0;
    if (mma) {
        auto run = [&](auto schedule) {
            constexpr int n = decltype(schedule)::value;
            if (short_scan) {
                ggml_cuda_pool_alloc<float> values(ctx.pool(),3072);
                ik_dense_gate_scan<taps,false,0,false,n><<<104,256,0,ctx.stream()>>>(a,w,x,values.get(),counters);
            } else {
                ik_dense_warp_roles<taps,false,!taps,0,false,n><<<dim3(8,a.n_split),384,0,ctx.stream()>>>(a,w,x,counters);
            }
        };
        if (mma == 1) { run(std::integral_constant<int,1>{}); return; }
        if (mma == 2) { run(std::integral_constant<int,2>{}); return; }
        if (mma == 3) { run(std::integral_constant<int,3>{}); return; }
        if (mma == 4) { run(std::integral_constant<int,4>{}); return; }
        if (mma == 5) { run(std::integral_constant<int,5>{}); return; }
        if (mma == 7) { run(std::integral_constant<int,7>{}); return; }
        if (mma == 8) { run(std::integral_constant<int,8>{}); return; }
        if (mma == 12) { run(std::integral_constant<int,12>{}); return; }
        if (mma == 15) { run(std::integral_constant<int,15>{}); return; }
    }
    static const int stripes = getenv("IK_SHORT_STRIPES") ? atoi(getenv("IK_SHORT_STRIPES")) : 0;
    if (short_scan && stripes > 0) {
        static const int gate_blocks = getenv("IK_STRIPE_GATES") ? std::max(1,std::min(48,atoi(getenv("IK_STRIPE_GATES")))) : 12;
        ggml_cuda_pool_alloc<ik_short_head> parts(ctx.pool(), 8);
        auto launch = [&](auto width) {
            constexpr int n = decltype(width)::value;
            ik_short_stripes<taps,n><<<8*(1+n+gate_blocks),256,0,ctx.stream()>>>(a,w,x,parts.get(),counters,gate_blocks);
        };
        if (stripes == 8) { launch(std::integral_constant<int,8>{}); }
        else if (stripes == 4) { launch(std::integral_constant<int,4>{}); }
        else if (stripes == 2) { launch(std::integral_constant<int,2>{}); }
        else { launch(std::integral_constant<int,1>{}); }
        return;
    }
    static const int mode = getenv("IK_SCAN_SCHEDULE") ? atoi(getenv("IK_SCAN_SCHEDULE")) : 0;
    static const int gate_rows = getenv("IK_GATE_ROWS") ? atoi(getenv("IK_GATE_ROWS")) : 0;
    if (gate_rows || a.gate_tiles) {
        auto launch = [&](auto row_count, auto packed_flag) {
            constexpr int rows = decltype(row_count)::value;
            constexpr bool packed = decltype(packed_flag)::value;
            if (short_scan) {
                ggml_cuda_pool_alloc<float> values(ctx.pool(),3072);
                ik_dense_gate_scan<taps,false,rows,packed><<<8*(1+IK_GATE_BLOCKS),256,0,ctx.stream()>>>(a,w,x,values.get(),counters);
            } else {
                ik_dense_warp_roles<taps,false,!taps,rows,packed><<<dim3(8,a.n_split),384,0,ctx.stream()>>>(a,w,x,counters);
            }
        };
        auto select_rows = [&](auto packed) {
            if (gate_rows == 4) { launch(std::integral_constant<int,4>{},packed); }
            else if (gate_rows == 2) { launch(std::integral_constant<int,2>{},packed); }
            else { launch(std::integral_constant<int,1>{},packed); }
        };
        if (a.gate_tiles) { select_rows(std::true_type{}); }
        else { select_rows(std::false_type{}); }
        return;
    }
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
