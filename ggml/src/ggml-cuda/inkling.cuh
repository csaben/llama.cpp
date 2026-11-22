#include "common.cuh"

// fused ops of the Inkling 2 family (attention residual mixing, short convolutions, loop controllers)

void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_supports_res_mix(const ggml_tensor * node);
void ggml_cuda_op_res_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_shortconv_res(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_loop_ctrl_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_loop_ctrl_apply(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_moe_up(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_moe_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_moe_sum_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_router(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_proj(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_inkling_segmented_projection(ggml_backend_cuda_context & ctx, ggml_tensor * projection, const ggml_tensor * attn);
bool ggml_cuda_inkling_segmented_projection_rows(ggml_backend_cuda_context & ctx, ggml_tensor * projection, int q_rows, int n_tok);
bool ggml_cuda_inkling_q2_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
struct ik_projection_conv {
    const float * past;
    const float * taps;
    const float * residual;
    float * state;
    const int32_t * ids;
    int64_t past_cell;
    float scale;
};
struct ik_dense_visit {
    const block_iq4_nl * wi, * wo;
    const block_q8_1 * x;
    float * out;
    ik_projection_conv conv;
    int * counters;
    int mode, prefetch;
};

bool ggml_cuda_inkling_dense_visit(ggml_backend_cuda_context & ctx, ggml_tensor * attn, ik_dense_visit visit);
bool ggml_cuda_inkling_projection_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * dst, ik_projection_conv * params = nullptr);
