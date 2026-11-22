#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    gguf_context * meta = gguf_init_from_file(argv[1], {true, nullptr});
    FILE * f = fopen(argv[1], "rb");
    if (!meta || !f) return 1;
    size_t all = 0, bf = 0, hf = 0;
    for (int64_t id = 0; id < gguf_get_n_tensors(meta); ++id) {
        if (!strstr(gguf_get_tensor_name(meta,id), ".ffn_gate_inp.weight")) continue;
        if (gguf_get_tensor_type(meta,id) != GGML_TYPE_F32) return 2;
        std::vector<float> v(gguf_get_tensor_size(meta,id)/4);
        if (fseek(f,gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,id),SEEK_SET) || fread(v.data(),4,v.size(),f)!=v.size()) return 3;
        size_t b=0,h=0;
        for (float x:v) {
            uint32_t u;
            memcpy(&u,&x,4);
            b += (u & 65535u) == 0;
            h += ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)) == x;
        }
        printf("%s elements=%zu bf16_exact=%zu fp16_exact=%zu\n",gguf_get_tensor_name(meta,id),v.size(),b,h);
        all += v.size(); bf += b; hf += h;
    }
    printf("total %zu bf16_exact %zu fp16_exact %zu\n",all,bf,hf);
    gguf_free(meta); fclose(f);
}
