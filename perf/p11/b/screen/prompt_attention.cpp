// Measure the direct Q8 prefill attention kernel with a materialized per-head mask.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) { return 2; }
    const int length = std::atoi(argv[1]);
    const int queries = std::atoi(argv[2]);
    if (length < queries || length > 65536 || queries < 1 || queries > 512) { return 2; }
    constexpr int hd = 128, heads = 24, kv_heads = 8, rel = 16, extent = 1024;
    ggml_backend_load_all();
    ggml_backend_ptr gpu(ggml_backend_init_by_name("CUDA0", nullptr));
    GGML_ASSERT(gpu);
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead()*32 + 2*ggml_graph_overhead(), nullptr, true}));
    ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hd, heads, queries);
    ggml_tensor * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q8_0, hd, kv_heads, length);
    ggml_tensor * v = ggml_dup_tensor(ctx.get(), k);
    ggml_tensor * r = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rel, heads, queries);
    ggml_tensor * table = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rel, extent);
    ggml_tensor * sinks = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, heads, queries);
    ggml_tensor * pos = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, length, queries);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, length, queries);
    ggml_tensor * bias = ggml_rel_bias(ctx.get(), r, table, sinks, pos, mask, GGML_TYPE_F16);
    auto permute = [&](ggml_tensor * t) { return ggml_permute(ctx.get(), t, 0, 2, 1, 3); };
    ggml_tensor * result = ggml_flash_attn_ext(ctx.get(), permute(q), permute(k), permute(v), bias, 1.0f/std::sqrt(float(hd)), 0.0f, 0.0f);
    ggml_prec_set_acc(result, GGML_PREC_F32);
    ggml_cgraph * full = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(full, result);
    ggml_cgraph * attention = ggml_new_graph(ctx.get());
    ggml_graph_add_node(attention, result);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), gpu.get()));
    GGML_ASSERT(buf);
    std::mt19937 rng(341);
    auto fill = [&](std::vector<float> & x) {
        for (float & a : x) { a = (int(rng() & 65535) - 32768)/32768.0f; }
    };
    for (ggml_tensor * t : {q, r, table, sinks}) {
        std::vector<float> x(ggml_nelements(t));
        fill(x);
        ggml_backend_tensor_set(t, x.data(), 0, ggml_nbytes(t));
    }
    for (ggml_tensor * t : {k, v}) {
        std::vector<float> x(256*hd*kv_heads);
        std::vector<char> packed(256*t->nb[2]);
        for (int i = 0; i < length; i += 256) {
            const int n = std::min(256, length-i);
            fill(x);
            const size_t bytes = ggml_quantize_chunk(t->type, x.data(), packed.data(), 0, n, hd*kv_heads, nullptr);
            GGML_ASSERT(bytes == n*t->nb[2]);
            ggml_backend_tensor_set(t, packed.data(), i*t->nb[2], bytes);
        }
    }
    std::vector<int32_t> distance(length);
    std::vector<ggml_fp16_t> causal(length);
    for (int j = 0; j < queries; ++j) {
        const int at = length-queries+j;
        for (int i = 0; i < length; ++i) {
            distance[i] = std::clamp(at-i, 0, extent);
            causal[i] = ggml_fp32_to_fp16(i <= at ? 0.0f : -INFINITY);
        }
        ggml_backend_tensor_set(pos, distance.data(), j*pos->nb[1], pos->nb[1]);
        ggml_backend_tensor_set(mask, causal.data(), j*mask->nb[1], mask->nb[1]);
    }
    GGML_ASSERT(ggml_backend_graph_compute(gpu.get(), full) == GGML_STATUS_SUCCESS);
    std::vector<float> values(ggml_nelements(result));
    ggml_backend_tensor_get(result, values.data(), 0, ggml_nbytes(result));
    for (float x : values) { GGML_ASSERT(std::isfinite(x)); }
    std::ofstream output(argv[3], std::ios::binary);
    GGML_ASSERT(output.write(reinterpret_cast<char *>(values.data()), ggml_nbytes(result)));
    output.close();
    GGML_ASSERT(output);
    auto run = [&]() { GGML_ASSERT(ggml_backend_graph_compute(gpu.get(), attention) == GGML_STATUS_SUCCESS); };
    const int64_t warmup = ggml_time_us();
    int count = 0;
    do { run(); ++count; } while (ggml_time_us()-warmup < 1000000);
    const int reps = std::max(1, count/10);
    printf("%d\t%d\t%d", length, queries, reps);
    for (int g = 0; g < 9; ++g) {
        const int64_t start = ggml_time_us();
        for (int rep = 0; rep < reps; ++rep) { run(); }
        printf("\t%.3f", double(ggml_time_us()-start)/reps);
    }
    printf("\n");
}
