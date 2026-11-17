#!/usr/bin/env python3
"""Engine changes that let the ternary adapters be stored as TQ1_P (see repack_tq1p.py):
  - loader: optional row scales for the adapter's down and up layers (ttf_d_r, ttf_u_r)
  - graph:  build_ttf_mm zero-pads its input when the weight is K-padded (the adapter's up layer reads
            the rank-sized vector, whose width is not a multiple of 160)
usage: patch_repack_engine.py ENGINE_DIR"""
import sys
root = sys.argv[1]

p = root + "/src/llama-model.cpp"
s = open(p).read()
if "ttf_d_r" not in s:
    a = '                te.dl.b = ttf_vec(e, "ttf_d_b", bid, r);\n'
    b = '                te.ul.b = ttf_vec(e, "ttf_u_b", bid, N);\n'
    assert s.count(a) == 1 and s.count(b) == 1
    s = s.replace(a, a + '                te.dl.r = ttf_vec(e, "ttf_d_r", bid, r); // row scale of a TQ1_P down layer\n')
    s = s.replace(b, b + '                te.ul.r = ttf_vec(e, "ttf_u_r", bid, N); // row scale of a TQ1_P up layer\n')
    open(p, "w").write(s); print("patched loader")

p = root + "/src/llama-graph.cpp"
s = open(p).read()
if "K-padded up layer" not in s:
    a = "    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);\n    if (l.r) {\n"
    assert s.count(a) == 1
    s = s.replace(a, "    if (w->ne[0] > cur->ne[0]) { // a K-padded up layer of an adapter (TQ1_P): zero-pad its input\n"
                     "        cur = ggml_pad(ctx0, cur, w->ne[0] - cur->ne[0], 0, 0, 0);\n    }\n" + a)
    open(p, "w").write(s); print("patched graph")
