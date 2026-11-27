// Isolate the fused Q8 attention scan with the model geometry and a causal verification window.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: attention KV_LENGTH ROWS CAPTURE\n");
        return 2;
    }
    const int length = std::atoi(argv[1]);
    const int rows = std::atoi(argv[2]);
    if (rows < 1 || rows > 7 || length < rows || length > 131072) {
        return 2;
    }
    constexpr int hd = 128, heads = 24, kv_heads = 8, rel = 16, extent = 1024, conv = 4, ch = hd*kv_heads;
    constexpr int projection = hd*heads + rel*heads + heads + hd*heads + 2*(ch + conv*kv_heads);
    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_name("CUDA0", nullptr));
    GGML_ASSERT(backend);
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead()*24 + ggml_graph_overhead(), nullptr, true}));
    ggml_tensor * proj = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, projection, 3*rows);
    ggml_tensor * consts = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, hd*heads + rel*heads + 2*conv*ch + rel*extent);
    ggml_tensor * state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, (conv-1)*2*ch, rows);
    ggml_tensor * rs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_Q8_0, hd, kv_heads, length, 1);
    ggml_tensor * v = ggml_dup_tensor(ctx.get(), k);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, rows);
    ggml_tensor * rp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, length, rows);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, length, rows);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, rows);
    ggml_tensor * before = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 2);
    ggml_tensor * out = ggml_inkling_attn(ctx.get(), proj, consts, state, rs, k, v, idx, rp, mask, pos,
        heads, kv_heads, hd, rel, conv, true, 0, 22.6f, 8.0f, 1e-6f, 0.1f, 8192.0f, 1.0f/std::sqrt(float(hd)), GGML_TYPE_Q8_1);
    if (rows > 1) {
        out->src[10] = before;
        out->op_params[14] = rows;
        out->op_params[15] = 1;
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    GGML_ASSERT(buf);
    std::mt19937 rng(421);
    auto fill = [&](std::vector<float> & values) {
        for (float & value : values) {
            value = (int(rng() & 65535) - 32768)/32768.0f;
        }
    };
    for (auto * t : {proj, consts, state}) {
        std::vector<float> values(ggml_nelements(t));
        fill(values);
        ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
    }
    for (auto * t : {k, v}) {
        const int batch = std::min(256, length);
        std::vector<float> values(batch*ch);
        std::vector<char> quant(batch*t->nb[2]);
        for (int i = 0; i < length; i += batch) {
            const int n = std::min(batch, length-i);
            fill(values);
            size_t bytes = ggml_quantize_chunk(t->type, values.data(), quant.data(), 0, n, ch, nullptr);
            GGML_ASSERT(bytes == n*t->nb[2]);
            ggml_backend_tensor_set(t, quant.data(), i*t->nb[2], bytes);
        }
    }
    std::vector<int64_t> indices(rows);
    std::iota(indices.begin(), indices.end(), length-rows);
    std::vector<int> positions(indices.begin(), indices.end());
    std::vector<int32_t> relative(rows*length);
    std::vector<ggml_fp16_t> causal(rows*length);
    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < length; ++i) {
            relative[row*length + i] = std::clamp(positions[row] - i, 0, extent);
            causal[row*length + i] = ggml_fp32_to_fp16(i <= positions[row] ? 0.0f : -INFINITY);
        }
    }
    const int32_t zero = 0, sizes[] = {length-rows, length-rows};
    ggml_backend_tensor_set(rs, &zero, 0, sizeof(zero));
    ggml_backend_tensor_set(before, sizes, 0, sizeof(sizes));
    ggml_backend_tensor_set(idx, indices.data(), 0, ggml_nbytes(idx));
    ggml_backend_tensor_set(pos, positions.data(), 0, ggml_nbytes(pos));
    ggml_backend_tensor_set(rp, relative.data(), 0, ggml_nbytes(rp));
    ggml_backend_tensor_set(mask, causal.data(), 0, ggml_nbytes(mask));
    auto run = [&]() { GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS); };
    run();
    std::ofstream capture(argv[3], std::ios::binary);
    for (auto * t : {out, state, k, v}) {
        const size_t off = t == k || t == v ? (length-rows)*t->nb[2] : 0;
        std::vector<char> bytes(ggml_nbytes(t)-off);
        ggml_backend_tensor_get(t, bytes.data(), off, bytes.size());
        GGML_ASSERT(capture.write(bytes.data(), bytes.size()));
    }
    capture.close();
    GGML_ASSERT(capture);
    for (int i = 0; i < 8; ++i) { run(); }
    printf("%d\t%d", length, rows);
    for (int group = 0; group < 9; ++group) {
        const int64_t start = ggml_time_us();
        for (int rep = 0; rep < 64; ++rep) { run(); }
        printf("\t%.3f", double(ggml_time_us()-start)/64);
    }
    printf("\n");
}
