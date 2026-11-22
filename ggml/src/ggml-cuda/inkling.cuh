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
struct ik_layer_args;
void ggml_cuda_op_inkling_router(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * up = nullptr, ggml_tensor * sum = nullptr, ik_layer_args * layer = nullptr);
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

bool ggml_cuda_inkling_projection_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * dst, ik_projection_conv * params = nullptr);
void ggml_cuda_ctrl_half(cudaStream_t stream, ggml_tensor * down, float2 * terms, bool message);
bool ggml_cuda_mix_down(ggml_backend_cuda_context & ctx, ggml_tensor * mix, ggml_tensor * down, int mode, const float2 * prepared = nullptr);
bool ggml_cuda_ctrl_pair(ggml_backend_cuda_context & ctx, ggml_tensor * down, ggml_tensor * apply, int mode);
bool ggml_cuda_inkling_layer(ggml_backend_cuda_context & ctx, ggml_tensor * attn, ggml_tensor * wo, ggml_tensor * conv, ggml_tensor * router, ggml_tensor * up, ggml_tensor * sum);
struct ik_message_pass {
    const half * weights[48];
    const float * inputs[48];
    float eps[48];
};
void ggml_cuda_message_pass(cudaStream_t stream, const ik_message_pass & pass, float2 * terms);
bool ggml_cuda_tail_mix(ggml_backend_cuda_context & ctx, ggml_tensor * tail, ggml_tensor * mix, int mode);
