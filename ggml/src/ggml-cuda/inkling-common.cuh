#pragma once
// helpers shared by the fused Inkling kernels (inkling.cu, inkling-fmv.cu)
#include "common.cuh"

// sum reduction over a block of up to 1024 threads, result valid in every thread
template <int block_size>
static __device__ __forceinline__ float ik_block_sum(float v, float * smem) {
    v = warp_reduce_sum(v);
    if constexpr (block_size > WARP_SIZE) {
        const int lane = threadIdx.x % WARP_SIZE;
        const int wid  = threadIdx.x / WARP_SIZE;
        __syncthreads();
        if (lane == 0) {
            smem[wid] = v;
        }
        __syncthreads();
        v = lane < block_size/WARP_SIZE ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
    }
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------
// Q8_1 stores of a row held by the lanes of a warp: one value per lane (32 consecutive values = one block), or one float4
// per lane (8 consecutive lanes = one block). All lanes of the warp take part (v = 0 past the row), only `store` lanes write.

template<bool native>
static __device__ __forceinline__ float ik_absmax(float x) {
    const float a = fabsf(x);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    if constexpr (native) {
        const unsigned bits = __reduce_max_sync(0xffffffffu,__float_as_uint(a));
        if (bits <= 0x7f800000u) { return __uint_as_float(bits); }
    }
#endif
    return warp_reduce_max(a);
}

template<bool native = false>
static __device__ __forceinline__ void ik_warp_store_q8_1(const float v, block_q8_1 * __restrict__ y, const int lane, const bool store) {
    const float amax = ik_absmax<native>(v);
    const float d    = amax/127.0f;
    const int   q    = amax == 0.0f ? 0 : (int) roundf(v/d);
    const int   sum  = warp_reduce_sum(q);
    if (store) {
        y->qs[lane] = (int8_t) q;
        if (lane == 0) {
            y->ds = make_half2(d, d*sum);
        }
    }
}

static __device__ __forceinline__ void ik_store_q8_1_vec4(const float4 v, block_q8_1 * __restrict__ y, const int k, const bool store) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 4));
    const float d = amax/127.0f;
    int q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    if (amax != 0.0f) {
        q0 = (int) roundf(v.x/d);
        q1 = (int) roundf(v.y/d);
        q2 = (int) roundf(v.z/d);
        q3 = (int) roundf(v.w/d);
    }
    int sum = q0 + q1 + q2 + q3;
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    if (store) {
        block_q8_1 * b = y + k/8;
        char4 qs;
        qs.x = (int8_t) q0; qs.y = (int8_t) q1; qs.z = (int8_t) q2; qs.w = (int8_t) q3;
        *(char4 *) (b->qs + (k % 8)*4) = qs;
        if (k % 8 == 0) {
            b->ds = make_half2(d, d*sum);
        }
    }
}

#define IK_MAX_STATES 9

struct ik_state_ptrs {
    const float * p[IK_MAX_STATES];
    int64_t       nb1[IK_MAX_STATES]; // bytes
};


// ---------------------------------------------------------------------------------------------------------------------
// persistent, zero-initialized arrival counters for the last-block-combines patterns: the kernels reset the counters they
// use, so no memset node is needed per launch. Allocated once per context (allowed under relaxed stream capture).

#define IK_N_COUNTERS 1024

static int * ik_counters(ggml_backend_cuda_context & ctx, const int offset, const int n) {
    GGML_ASSERT(offset + n <= IK_N_COUNTERS);
    if (ctx.ik_counters == nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc(&ctx.ik_counters, IK_N_COUNTERS*sizeof(int)));
        CUDA_CHECK(cudaMemset(ctx.ik_counters, 0, IK_N_COUNTERS*sizeof(int)));
    }
    return ctx.ik_counters + offset;
}

// ---------------------------------------------------------------------------------------------------------------------
// device-scope synchronization words

static __device__ __forceinline__ int ik_ld_acquire(const int * p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

static __device__ __forceinline__ void ik_st_release(int * p, const int v) {
    asm volatile("st.release.gpu.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

static __device__ __forceinline__ void ik_st_relaxed(int * p, const int v) {
    asm volatile("st.relaxed.gpu.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// the block's writes, ordered before thread 0 by a __syncthreads(), are released by the (cumulative) acq_rel atomic
static __device__ __forceinline__ int ik_atomic_add_acq_rel(int * p, const int v) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
