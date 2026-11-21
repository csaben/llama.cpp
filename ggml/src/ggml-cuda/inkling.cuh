#include "common.cuh"

// fused ops of the Inkling 2 family (attention residual mixing, short convolutions, loop controllers)

void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_res_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_shortconv_res(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_rel_bias(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_loop_ctrl_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_loop_ctrl_apply(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_moe_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
