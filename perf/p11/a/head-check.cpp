// Compare every FP32 row of the peer output head with the ordinary CUDA matvec.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    constexpr int dim = 2304, rows = 4096;
    ggml_backend_load_all();
    auto * main_dev = ggml_backend_dev_by_name("CUDA0");
    auto * peer_dev = ggml_backend_dev_by_name("CUDA1");
    if (!main_dev || !peer_dev) { return 1; }
    auto primary = ggml_backend_dev_init(main_dev, nullptr);
    auto peer = ggml_backend_dev_init(peer_dev, nullptr);
    if (!primary || !peer) { return 1; }

    auto * weights = ggml_init({8*ggml_tensor_overhead(), nullptr, true});
    auto * secondary = ggml_init({8*ggml_tensor_overhead(), nullptr, true});
    auto * w = ggml_new_tensor_2d(weights, GGML_TYPE_IQ4_NL, dim, rows);
    auto * upper = ggml_new_tensor_2d(secondary, GGML_TYPE_IQ4_NL, dim, rows/2);
    ggml_set_name(w, "output.weight");
    auto w_buf = ggml_backend_alloc_ctx_tensors(weights, primary);
    auto upper_buf = ggml_backend_alloc_ctx_tensors(secondary, peer);
    if (!w_buf || !upper_buf) { return 1; }
    uint32_t state = 193;
    auto sample = [&]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (int(state >> 16) - 32768)/8192.0f;
    };
    std::vector<float> raw(dim*rows);
    for (auto & v : raw) { v = sample(); }
    std::vector<unsigned char> packed(ggml_nbytes(w));
    std::vector<float> importance(dim, 1.0f);
    if (ggml_quantize_chunk(w->type, raw.data(), packed.data(), 0, rows, dim, importance.data()) != packed.size()) { return 1; }
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(upper, packed.data() + packed.size()/2, 0, packed.size()/2);

    for (int cols = 1; cols <= 8; ++cols) {
        auto * ctx = ggml_init({32*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(32, false), nullptr, true});
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, cols);
        auto * baseline = ggml_mul_mat(ctx, w, x);
        auto * split = ggml_mul_mat(ctx, w, x);
        ggml_prec_set_acc(baseline, GGML_PREC_F32);
        ggml_prec_set_acc(split, GGML_PREC_F32);
        if (cols > 1) { baseline->op_params[15] = split->op_params[15] = GGML_IK_MR_MAGIC; }
        split->extra = upper;
        auto * old_graph = ggml_new_graph_custom(ctx, 32, false);
        auto * new_graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(old_graph, baseline);
        ggml_build_forward_expand(new_graph, split);
        auto buf = ggml_backend_alloc_ctx_tensors(ctx, primary);
        if (!buf) { return 1; }
        std::vector<float> input(dim*cols), old(rows*cols), current(rows*cols);
        for (int repeat = 0; repeat < 4; ++repeat) {
            for (auto & v : input) { v = sample(); }
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            if (ggml_backend_graph_compute(primary, old_graph) != GGML_STATUS_SUCCESS ||
                    ggml_backend_graph_compute(primary, new_graph) != GGML_STATUS_SUCCESS) { return 1; }
            ggml_backend_synchronize(primary);
            ggml_backend_tensor_get(baseline, old.data(), 0, ggml_nbytes(baseline));
            ggml_backend_tensor_get(split, current.data(), 0, ggml_nbytes(split));
            for (size_t i = 0; i < old.size(); ++i) {
                if (!std::isfinite(old[i]) || memcmp(&old[i], &current[i], sizeof(float))) {
                    std::fprintf(stderr, "width %d repeat %d row %zu differs: %a vs %a\n", cols, repeat, i, old[i], current[i]);
                    return 1;
                }
            }
        }
        std::printf("width %d: %d FP32 values matched over four inputs\n", cols, 4*rows*cols);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(primary);
    ggml_backend_free(peer);
    ggml_backend_buffer_free(upper_buf);
    ggml_backend_buffer_free(w_buf);
    ggml_free(secondary);
    ggml_free(weights);
}
