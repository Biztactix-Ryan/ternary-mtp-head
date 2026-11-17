#!/usr/bin/env python3
"""Large batches on TQ1_P weights: plain MUL_MAT with more than 8 tokens went to the dense path (dequantise the whole
weight to F16 + cuBLAS on every call). Route it through the engine's existing TQ1_P -> Q4_0 expand and the Q4_0 MMQ
kernels instead (the route batched MUL_MAT_ID already takes). GGML_CUDA_TQ1P_DENSE=1 restores the old path for A/B.
usage: patch_bigbatch.py path/to/ggml-cuda.cu"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
if "ggml_cuda_mul_mat_tq1_p_big" in s:
    sys.exit("already patched")

decl = """// large batches on TQ1_P: expand to Q4_0 and use MMQ (defined below, next to the MUL_MAT_ID variant)
static bool ggml_cuda_mul_mat_tq1_p_big(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                        ggml_tensor * dst, int cc);

"""
anchor = "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {\n"
assert s.count(anchor) == 1
s = s.replace(anchor, decl + anchor)

old = """        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}
"""
assert s.count(old) == 1
s = s.replace(old, """        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_mul_mat_tq1_p_big(ctx, src0, src1, dst, cc)) {
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}
""")

old = """// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
"""
assert s.count(old) == 1
s = s.replace(old, """static bool ggml_cuda_mul_mat_tq1_p_big(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                        ggml_tensor * dst, const int cc) {
    static const bool dense = getenv("GGML_CUDA_TQ1P_DENSE") != nullptr;
    if (dense || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            !ggml_cuda_tq1_p_via_q4_0(src0, cc, src1->ne[1])) {
        return false;
    }
    ggml_cuda_mul_mat_id_tq1_p_q4_0(ctx, src0, src1, nullptr, dst);
    return true;
}

""" + old)
open(p, "w", encoding="utf-8").write(s)
print("patched")
