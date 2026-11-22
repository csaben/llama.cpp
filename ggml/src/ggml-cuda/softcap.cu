#include "softcap.cuh"

static __global__ void softcap_f32(const float * x, float * dst, const float scale, const float softcap, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = tanhf(scale * x[i]) * softcap;
}

static void softcap_f32_cuda(const float * x, float * dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(softcap_f32, launch_params, x, dst, scale, softcap, k);
}

// fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    softcap_f32_cuda(src0_d, dst_d, scale, softcap, ggml_nelements(src0), stream);
}

// GGML_OP_DSPARK_PICK: the largest softcap(base + bias) over the valid rows (ties: smallest token id) and its margin over the
// second largest. Stage 1: a partial (v1, token, v2) per block; stage 2: one block merges the partials and writes dst.
struct dspark_pick_part {
    float v1;
    int   t1;
    float v2;
};

static __device__ __forceinline__ void dspark_pick_merge(dspark_pick_part & a, const dspark_pick_part & b) {
    if (b.v1 > a.v1 || (b.v1 == a.v1 && b.t1 < a.t1)) {
        a.v2 = fmaxf(a.v1, b.v2);
        a.v1 = b.v1;
        a.t1 = b.t1;
    } else {
        a.v2 = fmaxf(a.v2, b.v1);
    }
}

#define DSPARK_PICK_BLOCK 256
#define DSPARK_PICK_PER_THREAD 16

template <bool last>
static __device__ __forceinline__ dspark_pick_part dspark_pick_block_reduce(dspark_pick_part p, dspark_pick_part * sh) {
    for (int off = 16; off > 0; off >>= 1) { // warp
        dspark_pick_part o;
        o.v1 = __shfl_xor_sync(0xffffffff, p.v1, off);
        o.t1 = __shfl_xor_sync(0xffffffff, p.t1, off);
        o.v2 = __shfl_xor_sync(0xffffffff, p.v2, off);
        dspark_pick_merge(p, o);
    }
    if (threadIdx.x % 32 == 0) {
        sh[threadIdx.x/32] = p;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) blockDim.x/32; ++w) {
            dspark_pick_merge(p, sh[w]);
        }
    }
    return p;
}

static __global__ void dspark_pick_scan(const float * base, const float * bias, const float * ids, dspark_pick_part * parts,
        const int n_valid, const float cap) {
    __shared__ dspark_pick_part sh[DSPARK_PICK_BLOCK/32];
    dspark_pick_part p = { -INFINITY, INT_MAX, -INFINITY };
    const int r0 = blockIdx.x*DSPARK_PICK_BLOCK*DSPARK_PICK_PER_THREAD + threadIdx.x;
    const float scale = cap > 0.0f ? 1.0f/cap : 0.0f;
    for (int k = 0; k < DSPARK_PICK_PER_THREAD; ++k) {
        const int r = r0 + k*DSPARK_PICK_BLOCK;
        if (r >= n_valid) {
            break;
        }
        const int t = ids ? (int) ids[r] : r;
        if (t < 0) {
            continue;
        }
        float v = base[r] + (bias ? bias[r] : 0.0f);
        if (cap > 0.0f) {
            v = tanhf(v*scale)*cap;
        }
        const dspark_pick_part o = { v, t, -INFINITY };
        dspark_pick_merge(p, o);
    }
    p = dspark_pick_block_reduce<false>(p, sh);
    if (threadIdx.x == 0) {
        parts[blockIdx.x] = p;
    }
}

static __global__ void dspark_pick_final(const dspark_pick_part * parts, const int n_parts, const int * prev, const int n_prev, int * dst) {
    __shared__ dspark_pick_part sh[DSPARK_PICK_BLOCK/32];
    dspark_pick_part p = { -INFINITY, INT_MAX, -INFINITY };
    for (int i = threadIdx.x; i < n_parts; i += blockDim.x) {
        dspark_pick_merge(p, parts[i]);
    }
    for (int i = threadIdx.x; i < n_prev; i += blockDim.x) {
        dst[i] = prev[i];
    }
    p = dspark_pick_block_reduce<true>(p, sh);
    if (threadIdx.x == 0) {
        dst[n_prev]     = p.t1 == INT_MAX ? 0 : p.t1;
        dst[n_prev + 1] = __float_as_int(p.v1 - p.v2);
    }
}

void ggml_cuda_op_dspark_pick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * base = dst->src[0];
    const ggml_tensor * bias = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const ggml_tensor * prev = dst->src[3];
    const int   n_valid = ggml_get_op_params_i32(dst, 0);
    const float cap     = ggml_get_op_params_f32(dst, 1);
    cudaStream_t stream = ctx.stream();

    const int n_parts = (n_valid + DSPARK_PICK_BLOCK*DSPARK_PICK_PER_THREAD - 1)/(DSPARK_PICK_BLOCK*DSPARK_PICK_PER_THREAD);
    ggml_cuda_pool_alloc<dspark_pick_part> parts(ctx.pool(), n_parts);

    dspark_pick_scan<<<n_parts, DSPARK_PICK_BLOCK, 0, stream>>>((const float *) base->data, bias ? (const float *) bias->data : nullptr,
            ids ? (const float *) ids->data : nullptr, parts.get(), n_valid, cap);
    dspark_pick_final<<<1, DSPARK_PICK_BLOCK, 0, stream>>>(parts.get(), n_parts, prev ? (const int *) prev->data : nullptr,
            prev ? (int) ggml_nelements(prev) : 0, (int *) dst->data);
}
