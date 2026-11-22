# Phase 5 common environment (source this): GPU1 only, live kernel flags, the calibrated 2-bit model.
export CUDA_VISIBLE_DEVICES=1 GPU=1 LLAMA_INKLING_CTRL_F16=1 IK_BATCH_MESSAGES=1 IK_SEGMENTED_Q6=1 IK_SEGMENTED_IQ4=1
export IK_CTRL_APPLY8=1 IK_MIX_SPECIALIZE=1
export Q2="$HOME/models/inkling-2-mini/Inkling-2-Mini-experimental-Q2K-Q2_0-fit.gguf"
export IQ4="$HOME/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL-dense-iq4nl.gguf"
export Q6="$HOME/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL-dense-q6k.gguf"
export DRAFT="$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0.gguf"
export DRAFT_FOLD="$HOME/models/inkling-2-mini/dspark-Inkling-2-Mini-Q8_0-fold-Q2K.gguf"
