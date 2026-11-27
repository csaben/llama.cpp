#include "common.cuh"

bool ggml_cuda_inkling_fast_glue(const ggml_backend_cuda_context & ctx);

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
void ggml_cuda_op_inkling_moe_sum_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool prefix = false);
void ggml_cuda_op_inkling_router(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_inkling_proj(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_inkling_segmented_projection(ggml_backend_cuda_context & ctx, ggml_tensor * projection, const ggml_tensor * attn);
bool ggml_cuda_inkling_segmented_projection_rows(ggml_backend_cuda_context & ctx, ggml_tensor * projection, int q_rows, int n_tok);
bool ggml_cuda_inkling_q2_down(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int slots = 8);
bool ggml_cuda_inkling_projection_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * dst);
bool ggml_cuda_mix_down(ggml_backend_cuda_context & ctx, ggml_tensor * mix, ggml_tensor * down);

bool ggml_cuda_ik_peer_gates_start(ggml_backend_cuda_context & ctx, const ggml_tensor * weights, const ggml_tensor * x, float * gates);
void ggml_cuda_ik_peer_gates_finish(ggml_backend_cuda_context & ctx);

void ggml_cuda_ik_return_q2_prefix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
        const ggml_tensor * ids, const float * route, float * dst);

void ggml_cuda_ik_msg_terms(cudaStream_t stream, const half * weights, const float * msgs, float2 * terms);
const float2 * ggml_cuda_ik_get_msg_terms(ggml_backend_cuda_context & ctx, const ggml_tensor * msg);
