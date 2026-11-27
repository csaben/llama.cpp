// Write marked expert windows in test-backend-ops --test-file format.
#include "ggml.h"

#include <cstdio>
#include <initializer_list>

int main() {
    for (int cols : {1, 2, 3, 4, 5, 6, 7, 8, 512}) {
        for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q2_0}) {
            const bool gate = type == GGML_TYPE_Q2_K;
            const int k = gate ? 2304 : 1152;
            const int m = cols == 512 ? 2304 : 256;
            const int experts = cols == 512 ? 128 : 32;
            printf("%d %d %d 8 %d 1 16", GGML_OP_MUL_MAT_ID, GGML_TYPE_F32, m, cols);
            for (int i = 0; i < 13; ++i) { printf(" 0"); }
            if (cols == 512) { printf(" 0 0 0"); }
            else { printf(" %d 1 %d", cols, GGML_IK_MR_MAGIC); }
            printf(" 3 %d %d %d %d 1 0 0 0 0", type, k, m, experts);
            printf(" %d %d %d %d 1 0 0 0 0", GGML_TYPE_F32, k, gate ? 1 : 8, cols);
            printf(" %d 8 %d 1 1 0 0 0 0 %s-%d\n", GGML_TYPE_I32, cols, gate ? "gate" : "down", cols);
        }
    }
}
