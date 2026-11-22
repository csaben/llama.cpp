ik_resident_experts                            44.0 calls/tok   34.99 us/call   1.540 ms/tok
inkling_attn2_kernel                           48.0 calls/tok   12.17 us/call   0.584 ms/tok
ik_segmented_projection                        28.0 calls/tok   16.13 us/call   0.452 ms/tok
inkling_router2_kernel                         48.0 calls/tok    7.87 us/call   0.378 ms/tok
mul_mat_vec_q 20 (4300, 1)                     20.0 calls/tok   15.80 us/call   0.316 ms/tok
mul_mat_vec_q 20 (1152, 1)                     48.0 calls/tok    6.24 us/call   0.299 ms/tok
mul_mat_vec_q 20 (100512, 1)                    1.0 calls/tok  298.55 us/call   0.299 ms/tok
res_mix_f32_vec4                               75.0 calls/tok    3.75 us/call   0.281 ms/tok
loop_ctrl_down_f32                             48.0 calls/tok    4.07 us/call   0.196 ms/tok
ik_ctrl_apply8                                 48.0 calls/tok    3.09 us/call   0.148 ms/tok
mul_mat_vec_q 20 (1152, 8)                      4.0 calls/tok   30.28 us/call   0.121 ms/tok
inkling_moe_sum_conv_kernel                    48.0 calls/tok    2.12 us/call   0.102 ms/tok
rms_norm_mul_inplace_f32                       24.0 calls/tok    3.82 us/call   0.092 ms/tok
shortconv_res_f32                              48.0 calls/tok    1.86 us/call   0.089 ms/tok
mul_mat_vec_q 20 (576, 8)                       4.0 calls/tok   15.87 us/call   0.063 ms/tok
situ_q8_1_kernel                                4.0 calls/tok    1.85 us/call   0.007 ms/tok
rms_norm_f32                                    1.0 calls/tok    3.13 us/call   0.003 ms/tok
softcap_f32                                     1.0 calls/tok    2.61 us/call   0.003 ms/tok
k_get_rows_float                                1.0 calls/tok    1.69 us/call   0.002 ms/tok
quantize_q8_1                                   1.0 calls/tok    1.38 us/call   0.001 ms/tok
head-to-head 5.232 ms/tok
