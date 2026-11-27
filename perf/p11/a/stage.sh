#!/usr/bin/env bash
# Preserve exactly the nine runtime files used for the selected verification.
set -euo pipefail
cd "$(dirname "$0")/../../.."
runtime=$(realpath "${1:?verified runtime}")
commit=$(git rev-parse --verify "${2:?source commit}^{commit}")
mode=${3:?one, head, or direct}
case $mode in one|head|direct) ;; *) exit 2;; esac
draft_ub=${4:-512}
case $draft_ub in 128|512) ;; *) exit 2;; esac
short=${commit:0:8}
snapshot="$HOME/inkling-live-bin-p11a-$short"
tmp=$(mktemp -d "$HOME/.inkling-p11a-$short.XXXXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
files=(llama-server libggml.so.1 libggml-base.so.1 libggml-cpu.so.1 libggml-cuda.so.1 libllama-common.so.1 libllama.so.1 libllama-server-impl.so libmtmd.so.1)
mkdir "$tmp/bin"
for f in "${files[@]}"; do cp -pL -- "$runtime/$f" "$tmp/bin/$f"; done
(cd "$tmp/bin" && sha256sum "${files[@]}") >"$tmp/manifest"
if [[ -e $snapshot ]]; then
    (cd "$snapshot" && sha256sum --strict --check "$tmp/manifest")
else
    mv -T -- "$tmp/bin" "$snapshot"
fi
(cd "$snapshot" && sha256sum --strict --check "$tmp/manifest")
LD_LIBRARY_PATH="$snapshot" ldd "$snapshot/llama-server" >"$tmp/ldd"
for f in "${files[@]:1}"; do
    awk -v f="$f" -v path="$snapshot/$f" '$1 == f && $2 == "=>" && $3 == path {ok=1} END {exit !ok}' "$tmp/ldd"
done
cp -- "$tmp/manifest" perf/p11/a/snapshot.sha256
sed -E 's/ \(0x[0-9a-f]+\)$//' "$tmp/ldd" >perf/p11/a/snapshot.ldd
printf '%s %s %s\n' "$short" "$mode" "$draft_ub" >perf/p11/a/release.txt
