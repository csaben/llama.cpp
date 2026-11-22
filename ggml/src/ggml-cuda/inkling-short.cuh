struct ik_short_partial {
    float acc[3][128];
    float m[3], l[3];
};

struct ik_short_head {
    ik_short_partial warp[8];
    ik_short_partial current;
    block_q8_0 kv[2][4];
};

static __device__ __forceinline__ void ik_short_query(const ik_attn_args & a, int g, float (&q)[3][128],
        float (&rel)[3][32], float (&sink)[3]) {
    const int t = threadIdx.x, w = t/32, lane = t%32;
    __shared__ float sums[3][5];
    const float tau = a.pos ? 1.0f + a.tau_alpha*logf(fmaxf(1.0f,(a.pos[0]+1.0f)/a.tau_floor)) : 1.0f;
    float v[3];
#pragma unroll
    for (int h = 0; h < 3; ++h) {
        const int row = 3*g+h;
        v[h] = t < 128 ? a.proj[row*128+t]*a.consts[row*128+t] : w == 4 && lane < 16 ?
                a.proj[a.off_rel+row*16+lane]*a.consts[a.c_relclip+row*16+lane] : 0.0f;
        const float ss = warp_reduce_sum(v[h]*v[h]);
        if (lane == 0 && w < 5) { sums[h][w] = ss; }
        if (t == 0) { sink[h] = a.proj[a.off_sink+row]; }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 3; ++h) {
        if (t < 128) {
            const float ss = sums[h][0]+sums[h][1]+sums[h][2]+sums[h][3];
            q[h][t] = v[h]*fminf(fmaxf(a.qk_clip/(sqrtf(ss)+a.clip_eps),0.0f),1.0f)*tau;
        } else if (w == 4) {
            rel[h][lane] = v[h]*fminf(fmaxf(a.rel_clip/(sqrtf(sums[h][4])+a.clip_eps),0.0f),1.0f)*tau;
        }
    }
    __syncthreads();
}

template<int taps>
static __device__ __forceinline__ void ik_short_current(const ik_attn_args & a, int g, int64_t index,
        ik_short_head & p, const float (&q)[3][128], const float (&rel)[3][32], const float (&sink)[3]) {
    const int t = threadIdx.x, lane = t%32, w = t/32, which = t/128, c = t%128, i = g*128+c;
    const int channel = which*1024+i;
    const float * hist = a.state + int64_t(a.rs_ids[0])*a.state_s1+channel*3;
    const float * weights = a.consts + (which ? a.c_vconv : a.c_kconv)+i*4;
    const float x[4] = {hist[0],hist[1],hist[2],a.proj[(which ? a.off_v : a.off_k)+i]};
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) { sum += x[j]*weights[j]; }
    if constexpr (taps) {
#pragma unroll
        for (int j = 0; j < 4; ++j) { sum += x[3-j]*a.proj[(which ? a.off_vtaps : a.off_ktaps)+4*g+j]; }
    }
    float y = (x[3]+sum)*(1.0f/sqrtf((taps ? 2 : 1)*4+1));
    __shared__ float ss[2][4], kv[2][128];
    float v = warp_reduce_sum(y*y);
    if (lane == 0) { ss[which][w%4] = v; }
    __syncthreads();
    v = ss[which][0]+ss[which][1]+ss[which][2]+ss[which][3];
    y *= which ? rsqrtf(v/128+a.norm_eps) : fminf(fmaxf(a.qk_clip/(sqrtf(v)+a.clip_eps),0.0f),1.0f);
    const float d = warp_reduce_max(fabsf(y))/127.0f;
    const int byte = (int)roundf(y*(d != 0.0f ? 1.0f/d : 0.0f));
    const half dh = __float2half(d);
    p.kv[which][w%4].qs[lane] = byte;
    if (lane == 0) { p.kv[which][w%4].d = dh; }
    kv[which][c] = __half2float(dh)*byte;
    float * st = a.state + int64_t(a.write_cell)*a.state_s1 + channel*3;
#pragma unroll
    for (int j = 0; j < 3; ++j) { st[j] = x[j+1]; }
    __syncthreads();
    if (w == 0) {
        const float tv = 0.5f*a.consts[a.c_table+lane%16];
        float dot[3];
#pragma unroll
        for (int h = 0; h < 3; ++h) {
            const int k = 4*lane;
            dot[h] = a.kq_scale*(q[h][k]*kv[0][k]+q[h][k+1]*kv[0][k+1]+q[h][k+2]*kv[0][k+2]+q[h][k+3]*kv[0][k+3])+tv*rel[h][lane%16];
        }
        float a0 = lane >= 16 ? dot[2] : dot[0], a1 = lane >= 16 ? 0.0f : dot[1];
        a0 += __shfl_xor_sync(0xffffffffu,lane >= 16 ? dot[0] : dot[2],16);
        a1 += __shfl_xor_sync(0xffffffffu,lane >= 16 ? dot[1] : 0.0f,16);
        float z = lane & 8 ? a1 : a0;
        z += __shfl_xor_sync(0xffffffffu,lane & 8 ? a0 : a1,8);
        z += __shfl_xor_sync(0xffffffffu,z,4);
        z += __shfl_xor_sync(0xffffffffu,z,2);
        z += __shfl_xor_sync(0xffffffffu,z,1);
        const float mask = a.mask_f16 ? __half2float(((const half *)a.mask)[index]) : ((const float *)a.mask)[index];
#pragma unroll
        for (int h = 0; h < 3; ++h) {
            const float score = __shfl_sync(0xffffffffu,z,8*h)-sink[h]+mask;
            const float mx = mask == -INFINITY ? -INFINITY : fmaxf(-INFINITY, score);
            const float prob = mask == -INFINITY ? 0.0f : expf(score-mx);
            if (lane == 0) { p.current.m[h] = mx; p.current.l[h] = prob; }
#pragma unroll
            for (int j = 0; j < 4; ++j) { p.current.acc[h][4*lane+j] = prob*kv[1][4*lane+j]; }
        }
    }
}

static __device__ __forceinline__ void ik_short_join(const ik_attn_args & a, int g, int64_t index, const ik_short_head & p) {
    const int t = threadIdx.x, lane = t%32, w = t/32, owner = (index/16)%8;
    __shared__ float coeff[3][8], den[3], maxv[3], prev[3], curr[3];
    __shared__ unsigned valid[3];
    if (w < 3) {
        float m = lane < 8 ? p.warp[lane].m[w] : -INFINITY;
        float l = lane < 8 ? p.warp[lane].l[w] : 0.0f;
        if (lane == owner) {
            const float cm = p.current.m[w];
            if (cm != -INFINITY) {
                const float mx = fmaxf(m,cm);
                const float co = m == -INFINITY ? 0.0f : expf(m-mx), cn = expf(cm-mx);
                prev[w] = co; curr[w] = cn;
                l = l*co + p.current.l[w]*cn; m = mx;
            } else { prev[w] = 1.0f; curr[w] = 0.0f; }
        }
        const float mx = warp_reduce_max(m);
        const unsigned active = __ballot_sync(0xffffffffu,m != -INFINITY);
        const float c = m != -INFINITY ? expf(m-mx) : 0.0f;
        if (lane < 8) { coeff[w][lane] = c; }
        float ld = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float lj = __shfl_sync(0xffffffffu,l,j), cj = __shfl_sync(0xffffffffu,c,j);
            if (active & (1u<<j)) { ld += lj*cj; }
        }
        if (lane == 0) { valid[w] = active; den[w] = ld; maxv[w] = mx; }
    }
    if (t < 68) {
        const int which = t/34, k = t%34;
        char * cache = (which ? a.v_cache : a.k_cache) + index*1088 + g*136;
        ((uint32_t *)cache)[k] = ((const uint32_t *)p.kv[which])[k];
    }
    __syncthreads();
    for (int i = t; i < 384; i += 256) {
        const int h = i/128, c = i%128;
        float v = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (valid[h] & (1u<<j)) {
                float x = p.warp[j].acc[h][c];
                if (j == owner && p.current.m[h] != -INFINITY) {
                    x = (p.warp[j].m[h] == -INFINITY ? 0.0f : x*prev[h]) + p.current.acc[h][c]*curr[h];
                }
                v += x*coeff[h][j];
            }
        }
        const float mx = fmaxf(maxv[h],0.0f), corr = expf(maxv[h]-mx);
        const float out = v*corr/(den[h]*corr+expf(-mx))*(1.0f/(1.0f+expf(-a.proj[a.off_gate+384*g+i])));
        ik_warp_store_q8_1(out,(block_q8_1 *)a.dst+(384*g+i)/32,lane,true);
    }
}

template<int taps, int stripes>
static __global__ void __launch_bounds__(256) ik_short_stripes(ik_attn_args a, const block_iq4_nl * weights,
        const block_q8_1 * x, ik_short_head * parts, int * counters, int gate_blocks) {
    const int t = threadIdx.x, lane = t%32, w = t/32, bx = blockIdx.x;
    const int role = bx/8, g = bx%8;
    ik_short_head & p = parts[g];
    const int64_t index = a.kv_idxs[0];
    __shared__ float q[3][128], rel[3][32], sink[3];
    if (role == 0) {
        ik_short_query(a,g,q,rel,sink);
        ik_short_current<taps>(a,g,index,p,q,rel,sink);
    } else if (role <= stripes) {
        __shared__ uint32_t visible[8];
        const float mask = a.mask_f16 ? __half2float(((const half *)a.mask)[t]) : ((const float *)a.mask)[t];
        const unsigned bits = __ballot_sync(0xffffffffu, mask != -INFINITY && t != index);
        if (lane == 0) { visible[w] = bits; }
        __syncthreads();
        constexpr int count = 8/stripes;
        const int original = (role-1)*count+w;
        const bool active = w < count && ((visible[original/2] | visible[4+original/2]) & (0xffffu << (16*(original%2))));
        if (__syncthreads_or(active)) { ik_short_query(a,g,q,rel,sink); }
        if (w < count) {
            if (active) {
                ik_mma_lane<3> st;
                ik_mma_init(st,q,sink);
                a.n_rel = 16; a.n_extent = 1024; a.n_kv = 256;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    if (visible[4*j+original/2] & (0xffffu<<(16*(original%2)))) {
                        const int start = 16*original+128*j;
                        ik_mma_step(a,st,rel,a.k_cache+g*136,1088,a.v_cache+g*136,1088,0,255,start,start+16,index);
                    }
                }
                ik_mma_finish(st,p.warp[original].acc[min(lane&3,2)]);
                if (lane < 3) { p.warp[original].m[lane] = st.mm; p.warp[original].l[lane] = st.ll; }
            } else if (lane < 3) {
                p.warp[original].m[lane] = -INFINITY; p.warp[original].l[lane] = 0.0f;
            }
        }
    } else {
        for (int i = (role-1-stripes)*8+w; i < 384; i += gate_blocks*8) {
            ik_project_row(weights,x,(float *)a.proj,a.off_gate+384*g+i);
        }
    }
    __shared__ int last;
    __syncthreads();
    if (t == 0) { last = ik_atomic_add_acq_rel(counters+g,1) == stripes+gate_blocks; }
    __syncthreads();
    if (last) {
        if (t == 0) { ik_st_relaxed(counters+g,0); }
        ik_short_join(a,g,index,p);
    }
}
