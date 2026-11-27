// Run one expert projection on the selected backend with repeatable routing and activations.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static int measure(const char * model, const char * projection, int cols, const char * output_path) {
    const bool gate = std::strcmp(projection, "gate") == 0;
    if ((!gate && std::strcmp(projection, "down")) || cols < 1 || cols > 512) {
        return 2;
    }
    const int layer = std::getenv("P11_LAYER") ? std::atoi(std::getenv("P11_LAYER")) : 0;
    const std::string name = "blk." + std::to_string(layer) + (gate ? ".ffn_gate_up_exps.weight" : ".ffn_down_exps.weight");
    gguf_context_ptr meta(gguf_init_from_file(model, {true, nullptr}));
    GGML_ASSERT(meta);
    const int64_t index = gguf_find_tensor(meta.get(), name.c_str());
    GGML_ASSERT(index >= 0);
    const int64_t * ne = gguf_get_tensor_ne(meta.get(), index);
    GGML_ASSERT(ne[2] == 128 && ne[3] == 1);

    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_name("CUDA0", nullptr));
    GGML_ASSERT(backend);
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true}));
    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), gguf_get_tensor_type(meta.get(), index), ne[0], ne[1], ne[2]);
    const bool q8 = std::getenv("P11_EXPERT_Q8") != nullptr;
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), q8 ? GGML_TYPE_Q8_1 : GGML_TYPE_F32, ne[0], gate ? 1 : 8, cols);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 8, cols);
    ggml_tensor * output = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    if (cols <= 7) {
        output->op_params[13] = cols;
        output->op_params[14] = 1;
        output->op_params[15] = GGML_IK_MR_MAGIC;
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    GGML_ASSERT(buf && ggml_nbytes(weights) == gguf_get_tensor_size(meta.get(), index));
    std::ifstream file(model, std::ios::binary);
    file.seekg(gguf_get_data_offset(meta.get()) + gguf_get_tensor_offset(meta.get(), index));
    std::vector<char> block(1 << 20);
    for (size_t offset = 0; offset < ggml_nbytes(weights); offset += block.size()) {
        const size_t size = std::min(block.size(), ggml_nbytes(weights) - offset);
        GGML_ASSERT(file.read(block.data(), size));
        ggml_backend_tensor_set(weights, block.data(), offset, size);
    }

    std::mt19937 rng(731);
    std::vector<float> activations(ggml_nelements(input));
    for (float & value : activations) {
        value = (int(rng() & 65535) - 32768)/32768.0f;
    }
    std::array<int32_t, 128> experts;
    std::iota(experts.begin(), experts.end(), 0);
    std::vector<int32_t> routes(ggml_nelements(ids));
    for (int col = 0; col < cols; ++col) {
        std::shuffle(experts.begin(), experts.end(), rng);
        std::copy_n(experts.begin(), 8, routes.begin() + col*8);
    }
    std::vector<char> quantized;
    const char * directory = std::getenv("P11_EXPERT_INPUT_DIR");
    const char * windows = std::getenv("P11_EXPERT_WINDOW_DIR");
    std::string window_directory;
    if (windows) {
        window_directory = std::string(windows) + "/" + std::to_string(cols);
        directory = window_directory.c_str();
    }
    if (q8) {
        GGML_ASSERT(directory);
        quantized.resize(ggml_nbytes(input));
    }
    if (directory) {
        const std::string prefix = std::string(directory) + "/" + projection;
        std::ifstream x(prefix + ".input", std::ios::binary);
        std::ifstream routed(prefix + ".ids", std::ios::binary);
        GGML_ASSERT(x.read(q8 ? quantized.data() : reinterpret_cast<char *>(activations.data()), ggml_nbytes(input)));
        GGML_ASSERT(routed.read(reinterpret_cast<char *>(routes.data()), ggml_nbytes(ids)));
    }
    ggml_backend_tensor_set(input, q8 ? (const void *) quantized.data() : (const void *) activations.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
    auto run = [&]() {
        GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    };
    for (int i = 0; i < 8; ++i) {
        run();
    }
    int reps = cols > 7 ? 16 : 128;
    if (std::getenv("P11_STEADY")) {
        const int64_t start = ggml_time_us();
        int count = 0;
        do {
            run();
            ++count;
        } while (ggml_time_us()-start < 1000000);
        reps = std::max(1, count/10);
    }
    std::array<double, 9> times;
    for (double & us : times) {
        const int64_t start = ggml_time_us();
        for (int i = 0; i < reps; ++i) {
            run();
        }
        us = double(ggml_time_us() - start)/reps;
    }
    std::vector<float> values(ggml_nelements(output));
    ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
    std::ofstream capture(output_path, std::ios::binary);
    GGML_ASSERT(capture.write(reinterpret_cast<const char *>(values.data()), ggml_nbytes(output)));
    capture.close();
    GGML_ASSERT(capture);
    printf("%s\t%d\t%d", projection, cols, reps);
    for (double us : times) {
        printf("\t%.3f", us);
    }
    printf("\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 5 || (argc - 3) % 2) {
        fprintf(stderr, "usage: expert MODEL gate|down COLUMNS OUTPUT [COLUMNS OUTPUT ...]\n");
        return 2;
    }
    for (int i = 3; i < argc; i += 2) {
        const int status = measure(argv[1], argv[2], std::atoi(argv[i]), argv[i + 1]);
        if (status != 0) { return status; }
    }
    return 0;
}
