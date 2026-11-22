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
bool ggml_cuda_inkling_project_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_moe_up(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_moe_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_moe_sum_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_router(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * up = nullptr, ggml_tensor * sum = nullptr);
void ggml_cuda_op_inkling_proj(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_inkling_segmented_projection(ggml_backend_cuda_context & ctx, ggml_tensor * projection, const ggml_tensor * attn);
bool ggml_cuda_inkling_segmented_projection_rows(ggml_backend_cuda_context & ctx, ggml_tensor * projection, int q_rows, int n_tok);
bool ggml_cuda_inkling_q2_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_inkling_projection_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * dst);
void ggml_cuda_ctrl_half(cudaStream_t stream, ggml_tensor * down, float2 * terms, bool message);
