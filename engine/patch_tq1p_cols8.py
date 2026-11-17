#!/usr/bin/env python3
"""Table kernel for TQ1_P: 5 to 8 tokens per call (was 1 to 4; 5 to 8 took the generic kernel, about twice the cost
per extra token). The weight stream is still read and decoded once per row; the activations are processed in column
chunks of four so the per-thread state stays in registers. For 1 to 4 tokens the arithmetic is unchanged.
GGML_CUDA_TQ1P_COLS4=1 restores the old limit for A/B.
usage: patch_tq1p_cols8.py path/to/ggml/src/ggml-cuda"""
import os, sys

d = sys.argv[1]


def patch(name, edits):
    p = os.path.join(d, name)
    s = open(p, encoding="utf-8").read()
    if "TQ1P_MAX_COLS" in s:
        sys.exit(f"{name}: already patched")
    for old, new in edits:
        assert s.count(old) == 1, (name, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


OLD_LOOP = """    for (int kb = lane / 4; active && kb < nb; kb += WARP_SIZE / 4) {
        int   u0[ncols_dst][5];
        int   u1[ncols_dst][5];
        float d [ncols_dst][5];
        int   su[ncols_dst][5];   // -sum of the 8 activations this thread holds in q8_1 block n
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int n = 0; n < 5; ++n) {
                const block_q8_1 * yb = y + j*L.scy + kb*5 + n;
                u0[j][n] = get_int_b4(yb->qs, 2*sub);     // q8_1 qs sits at byte 4 of a 36-byte block: 4-byte loads
                u1[j][n] = get_int_b4(yb->qs, 2*sub + 1);
                d [j][n] = __low2float(yb->ds);
                su[j][n] = -ggml_cuda_dp4a(0x01010101, u0[j][n], ggml_cuda_dp4a(0x01010101, u1[j][n], 0));
            }
        }

        // all rows' loads first (several in flight per thread); a row past the end re-reads the last row, unused
        int2 v[TQ1P_ROWS];
#pragma unroll
        for (int r = 0; r < TQ1P_ROWS; ++r) {
            const int row = min(row0 + r, nrows_x - 1);
            v[r] = *((const int2 *) x[(int64_t) row*L.stride_w + kb].qs + sub);
        }
#pragma unroll
        for (int r = 0; r < TQ1P_ROWS; ++r) {
            int wa0, wb0, wa1, wb1;
            tq1p_decode(lut, v[r].x, wa0, wb0);
            tq1p_decode(lut, v[r].y, wa1, wb1);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                float s = 0.0f;
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int c0 = (wa0 >> (2*n)) & 0x03030303;
                    const int c1 = (wa1 >> (2*n)) & 0x03030303;
                    s += d[j][n] * (float) ggml_cuda_dp4a(c0, u0[j][n], ggml_cuda_dp4a(c1, u1[j][n], su[j][n]));
                }
                s += d[j][4] * (float) ggml_cuda_dp4a(wb0, u0[j][4], ggml_cuda_dp4a(wb1, u1[j][4], su[j][4]));
                acc[j][r] += s;
            }
        }
    }
"""

NEW_LOOP = """    for (int kb = lane / 4; active && kb < nb; kb += WARP_SIZE / 4) {
        // all rows' loads first (several in flight per thread); a row past the end re-reads the last row, unused
        int2 v[TQ1P_ROWS];
#pragma unroll
        for (int r = 0; r < TQ1P_ROWS; ++r) {
            const int row = min(row0 + r, nrows_x - 1);
            v[r] = *((const int2 *) x[(int64_t) row*L.stride_w + kb].qs + sub);
        }
        int wa0[TQ1P_ROWS], wb0[TQ1P_ROWS], wa1[TQ1P_ROWS], wb1[TQ1P_ROWS];
#pragma unroll
        for (int r = 0; r < TQ1P_ROWS; ++r) {
            tq1p_decode(lut, v[r].x, wa0[r], wb0[r]);
            tq1p_decode(lut, v[r].y, wa1[r], wb1[r]);
        }

        // the columns in chunks of TQ1P_COL_CHUNK: the decoded weights are reused, the activations of one chunk
        // (its 5 q8_1 blocks per column) are in registers at a time
#pragma unroll
        for (int j0 = 0; j0 < ncols_dst; j0 += TQ1P_COL_CHUNK) {
            int   u0[TQ1P_COL_CHUNK][5];
            int   u1[TQ1P_COL_CHUNK][5];
            float d [TQ1P_COL_CHUNK][5];
            int   su[TQ1P_COL_CHUNK][5];   // -sum of the 8 activations this thread holds in q8_1 block n
#pragma unroll
            for (int jj = 0; jj < TQ1P_COL_CHUNK; ++jj) {
                if (j0 + jj < ncols_dst) {
#pragma unroll
                    for (int n = 0; n < 5; ++n) {
                        const block_q8_1 * yb = y + (j0 + jj)*L.scy + kb*5 + n;
                        u0[jj][n] = get_int_b4(yb->qs, 2*sub);     // q8_1 qs sits at byte 4 of a 36-byte block: 4-byte loads
                        u1[jj][n] = get_int_b4(yb->qs, 2*sub + 1);
                        d [jj][n] = __low2float(yb->ds);
                        su[jj][n] = -ggml_cuda_dp4a(0x01010101, u0[jj][n], ggml_cuda_dp4a(0x01010101, u1[jj][n], 0));
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < TQ1P_ROWS; ++r) {
#pragma unroll
                for (int jj = 0; jj < TQ1P_COL_CHUNK; ++jj) {
                    if (j0 + jj < ncols_dst) {
                        float s = 0.0f;
#pragma unroll
                        for (int n = 0; n < 4; ++n) {
                            const int c0 = (wa0[r] >> (2*n)) & 0x03030303;
                            const int c1 = (wa1[r] >> (2*n)) & 0x03030303;
                            s += d[jj][n] * (float) ggml_cuda_dp4a(c0, u0[jj][n], ggml_cuda_dp4a(c1, u1[jj][n], su[jj][n]));
                        }
                        s += d[jj][4] * (float) ggml_cuda_dp4a(wb0[r], u0[jj][4], ggml_cuda_dp4a(wb1[r], u1[jj][4], su[jj][4]));
                        acc[j0 + jj][r] += s;
                    }
                }
            }
        }
    }
"""

patch("mmvq-tq1p.cuh", [
    ("#define TQ1P_ROWS  4\n", "#define TQ1P_ROWS  4\n#define TQ1P_MAX_COLS  8   // tokens per call\n#define TQ1P_COL_CHUNK 4   // columns whose activations are in registers at a time\n"),
    ("#define TTF_CNT_STRIDE 16\n", "#define TTF_CNT_STRIDE (TQ1P_MAX_GROUP*TQ1P_MAX_COLS)\n"),
    ("        unsigned int * arr = cnt + g*4 + tok;\n", "        unsigned int * arr = cnt + g*TQ1P_MAX_COLS + tok;\n"),
    (OLD_LOOP, NEW_LOOP),
    ("        case 4: mul_mat_vec_tq1_p<4, R, GLU><<<grp.nblocks, block, 0, stream>>>(grp); break;\n",
     "        case 4: mul_mat_vec_tq1_p<4, R, GLU><<<grp.nblocks, block, 0, stream>>>(grp); break;\n" +
     "".join(f"        case {c}: mul_mat_vec_tq1_p<{c}, R, GLU><<<grp.nblocks, block, 0, stream>>>(grp); break;\n" for c in (5, 6, 7, 8))),
    ("    if (ncols_dst < 1 || ncols_dst > 4 || fusion.gate != nullptr",
     "    if (ncols_dst < 1 || ncols_dst > tq1p_max_cols() || fusion.gate != nullptr"),
    ("static int tq1p_blocks(const int N) {",
     "// GGML_CUDA_TQ1P_COLS4=1: the table kernel takes at most 4 tokens, as before (A/B)\n"
     "static int tq1p_max_cols() {\n    static const int v = getenv(\"GGML_CUDA_TQ1P_COLS4\") != nullptr ? 4 : TQ1P_MAX_COLS;\n    return v;\n}\n\n"
     "static int tq1p_blocks(const int N) {"),
])
p = os.path.join(d, "mmvq.cu")
s = open(p, encoding="utf-8").read()
old = "ntok < 1 || ntok > 4 || x->ne[2] != 1"
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, "ntok < 1 || ntok > tq1p_max_cols() || x->ne[2] != 1"))
print("patched")
