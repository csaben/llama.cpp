#!/usr/bin/env python3
"""Fold the DSpark drafter's vocabulary correction into one matrix (after basalt's head fold, MIT,
https://github.com/jesdga95/basalt src/models/inkling2/dspark.hpp; the idea only, no code copied).

Per draft position the drafter adds  target.output @ (hidden_out @ latent + hidden_out_b)  to its logits: a
full-vocabulary matvec over the target head (201,024 x 2,304 IQ4_NL, 261 MB). With
    fold      = target.output @ hidden_out     [n_vocab, rank]  (stored Q8_0, 54 MB)
    fold_bias = target.output @ hidden_out_b   [n_vocab]        (F32)
it is  fold @ latent + fold_bias.  The target still verifies every draft with its full head, so only the drafts
(not the output) can change.

With RANK.bin (token ids by descending frequency, perf/p6/shortlist_rank.py) the file also gets a shortlist head
(round 3; after ingine's shortlist head, Apache-2.0, https://github.com/satellitedown/inkling-2-mini-abliterated-ingine
native/src/model/drafter.h; the idea only, no code or data copied): the K most frequent tokens and the control tokens,
by ascending id, as  dspark_short_head.weight  (their rows of target.output, bytes copied as stored),
dspark_short_fold.weight / .bias  (their rows of the fold) and  dspark_short_ids.weight  (row -> token id as F32, -1 = unused).
The drafter then scores a draft position over these rows only.

usage: .venv/bin/python perf/p6/dspark_fold.py TARGET.gguf DRAFTER.gguf OUT.gguf [RANK.bin [K=32768]]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf  # noqa: E402


def tensor(reader, name):
    for t in reader.tensors:
        if t.name == name:
            return t
    raise SystemExit(f'{name}: not found')


def to_f32(t):
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        return np.asarray(t.data, dtype=np.float32)
    return gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32, copy=False)


def main():
    target_path, draft_path, out_path = sys.argv[1:4]
    rank_path = sys.argv[4] if len(sys.argv) > 4 else None
    short_k = int(sys.argv[5]) if len(sys.argv) > 5 else 32768
    if os.path.exists(out_path):
        raise SystemExit(f'{out_path} exists')
    target = gguf.GGUFReader(target_path)
    draft = gguf.GGUFReader(draft_path)

    out_t = tensor(target, 'output.weight')
    head = to_f32(out_t).reshape(int(out_t.shape[1]), int(out_t.shape[0]))             # [n_vocab, n_embd]
    ho_t = tensor(draft, 'dspark_hidden_out.weight')
    ho = to_f32(ho_t).reshape(int(ho_t.shape[1]), int(ho_t.shape[0]))                 # [n_embd, rank]
    hb = to_f32(tensor(draft, 'dspark_hidden_out.bias')).reshape(-1)                  # [n_embd]
    fold = head @ ho                                                                  # [n_vocab, rank]
    fold_b = head @ hb                                                                # [n_vocab]
    fold_q = gguf.quants.quantize(fold, gguf.GGMLQuantizationType.Q8_0)
    err = np.abs(gguf.quants.dequantize(fold_q, gguf.GGMLQuantizationType.Q8_0) - fold).max()
    print(f'fold {fold.shape}, |fold| max {np.abs(fold).max():.4f}, Q8_0 max error {err:.5f}', file=sys.stderr)

    short = None
    if rank_path:
        n_vocab = int(out_t.shape[1])
        rank = np.fromfile(rank_path, dtype='<u4')
        ids = set(int(t) for t in rank[:short_k] if t < n_vocab)
        types = target.fields['tokenizer.ggml.token_type'].contents()
        control = [i for i, t in enumerate(types) if t == 3 and i < n_vocab]  # LLAMA_TOKEN_TYPE_CONTROL
        ids.update(control)
        ids = np.array(sorted(ids), dtype=np.int64)
        rows = (len(ids) + 255)//256*256
        raw = np.asarray(out_t.data).reshape(n_vocab, -1)                              # the stored bytes of each row
        head_s = np.zeros((rows, raw.shape[1]), dtype=raw.dtype); head_s[:len(ids)] = raw[ids]
        fq = fold_q.reshape(n_vocab, -1)
        fold_s = np.zeros((rows, fq.shape[1]), dtype=fq.dtype); fold_s[:len(ids)] = fq[ids]
        bias_s = np.zeros(rows, dtype=np.float32); bias_s[:len(ids)] = fold_b[ids]
        ids_s = np.full(rows, -1, dtype=np.float32); ids_s[:len(ids)] = ids  # F32: exact below 2^24
        short = (head_s, out_t.tensor_type, fold_s, bias_s, ids_s)
        print(f'shortlist: {len(ids)} tokens ({len(control)} control), {rows} rows, head {head_s.nbytes/1e6:.1f} MB, fold {fold_s.nbytes/1e6:.1f} MB', file=sys.stderr)

    arch = draft.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(out_path, arch=arch, endianess=draft.endianess)
    for field in draft.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    writer.add_string('dspark.head_fold.target', os.path.basename(target_path))
    for t in draft.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.add_tensor_info('dspark_head_fold.weight', fold_q.shape, fold_q.dtype, fold_q.nbytes, gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor_info('dspark_head_fold.bias', fold_b.shape, fold_b.dtype, fold_b.nbytes, gguf.GGMLQuantizationType.F32)
    if short:
        head_s, head_type, fold_s, bias_s, ids_s = short
        writer.add_tensor_info('dspark_short_head.weight', head_s.shape, head_s.dtype, head_s.nbytes, head_type)
        writer.add_tensor_info('dspark_short_fold.weight', fold_s.shape, fold_s.dtype, fold_s.nbytes, gguf.GGMLQuantizationType.Q8_0)
        writer.add_tensor_info('dspark_short_fold.bias', bias_s.shape, bias_s.dtype, bias_s.nbytes, gguf.GGMLQuantizationType.F32)
        writer.add_tensor_info('dspark_short_ids.weight', ids_s.shape, ids_s.dtype, ids_s.nbytes, gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in draft.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=draft.endianess)
    writer.write_tensor_data(fold_q)
    writer.write_tensor_data(fold_b.astype(np.float32))
    if short:
        for a in (head_s, fold_s, bias_s, ids_s):
            writer.write_tensor_data(a)
    writer.close()
    print(f'wrote {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
