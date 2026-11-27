"""Build the frozen p9 llama-bench with a per-decode token and full-logit capture."""
import subprocess
from pathlib import Path

work = Path('build/p11/bench-capture')
work.mkdir(exist_ok=True)
source = subprocess.check_output(['git', 'show', '72fd097e:tools/llama-bench/llama-bench.cpp'], text=True)
anchor = 'static bool test_prompt(llama_context * ctx,'
helper = '''static void capture_output(llama_context * ctx, const llama_token * tokens, int count, int vocab) {
    static FILE * output = fopen(getenv("P11_BENCH_DUMP"), "wb");
    GGML_ASSERT(output && fwrite(&count, sizeof(count), 1, output) == 1);
    GGML_ASSERT(fwrite(tokens, sizeof(*tokens), count, output) == (size_t) count);
    const float * logits = llama_get_logits_ith(ctx, -1);
    GGML_ASSERT(logits && fwrite(logits, sizeof(*logits), vocab, output) == (size_t) vocab);
}

'''
assert source.count(anchor) == 1
source = source.replace(anchor, helper + anchor)
for old,new in [('        n_processed += n_tokens;', '        capture_output(ctx, tokens.data(), n_tokens, n_vocab);\n'),
                ('        token = std::rand() % n_vocab;', '        capture_output(ctx, &token, 1, n_vocab);\n')]:
    assert source.count(old) == 1
    source = source.replace(old,new+old)
cpp=work/'llama-bench-capture.cpp'
cpp.write_text(source)
subprocess.run(['g++','-O3','-DNDEBUG','-std=c++17','-DLLAMA_SUBPROCESS','-DGGML_USE_CPU','-DGGML_USE_CUDA',
                '-Itools/llama-bench','-Icommon','-Iinclude','-Iggml/include','-Ivendor',str(cpp),'tools/llama-bench/main.cpp',
                '-Lbuild/bin','-lllama-common','-lllama','-lggml','-lggml-base','build/common/libllama-common-base.a',
                '-pthread','-o',str(work/'llama-bench')],check=True)
