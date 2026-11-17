#!/usr/bin/env python3
"""Native MMQ tiles for TQ1_P: the batched quantised kernel reads the ternary weights directly, with no per-call
expansion. A TQ1_P block is five 32-value digit planes, so the tile loader addresses the weight in planes
(8 per 256-value tile) and decodes each plane's trits to int8; from there it is the Q8_0 tile path (scale 1).
Configs are those of Q2_0. GGML_CUDA_TQ1P_NO_MMQ=1 turns it off (then the Q4_0 expand route is used).
usage: patch_mmq_tq1p.py path/to/ggml/src/ggml-cuda"""
import os, sys

d = sys.argv[1]


def edit(name, fn):
    p = os.path.join(d, name)
    s = open(p, encoding="utf-8").read()
    if "TQ1_P" in s and name.startswith("mmq"):
        sys.exit(f"{name}: already patched")
    s = fn(s)
    open(p, "w", encoding="utf-8").write(s)


def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:70], s.count(old))
    return s.replace(old, new)


def mmq_cuh(s):
    # y layout: plain scales
    s = rep(s, """        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
""", """        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_TQ1_P:
            return MMQ_Q8_1_DS_LAYOUT_D4;
""")
    # block unit of the K loop: a 32-value digit plane for TQ1_P
    s = rep(s, "ggml_cuda_type_traits<type>::qk", "ggml_cuda_mmq_qk<type>", 4)
    s = rep(s, "struct tile_x_sizes {", """// the K-direction unit a tile loader counts in: the type's block, or for TQ1_P one 32-value digit plane (5 per block)
template <ggml_type type>
static constexpr int ggml_cuda_mmq_qk = type == GGML_TYPE_TQ1_P ? QK8_1 : ggml_cuda_type_traits<type>::qk;

struct tile_x_sizes {""")
    s = rep(s, "        case GGML_TYPE_Q2_0:    return MMQ_DP4A_TXS_Q8_0;\n",
            "        case GGML_TYPE_Q2_0:    return MMQ_DP4A_TXS_Q8_0;\n        case GGML_TYPE_TQ1_P:   return MMQ_DP4A_TXS_Q8_0;\n")
    # configs: TQ1_P uses the Q2_0 entries of every architecture
    s = rep(s, "    if (type == (type_) && J == (J_) && fallback == (fallback_)) {",
            "    if ((type == (type_) || (type == GGML_TYPE_TQ1_P && (type_) == GGML_TYPE_Q2_0)) && J == (J_) && fallback == (fallback_)) {")
    s = rep(s, "        return ggml_cuda_mmq_config((type_), (nthreads_),", "        return ggml_cuda_mmq_config(type, (nthreads_),")
    s = rep(s, """            case GGML_TYPE_Q4_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q4_0_Q8_1_MMQ,
""", """            case GGML_TYPE_TQ1_P:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q8_0_Q8_1_MMQ,
                    ggml_cuda_mmq_load_tiles_tq1_p<type, J, fallback>,
                    ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a<type, J, fallback>,
                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback, prec_src1>);
            case GGML_TYPE_Q4_0:
                return ggml_cuda_mmq_util_funcs(
                    VDR_Q4_0_Q8_1_MMQ,
""")
    s = rep(s, """        case GGML_TYPE_Q4_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
""", """        case GGML_TYPE_TQ1_P:
            return ggml_cuda_mmq_util_funcs(
                -1,
                ggml_cuda_mmq_load_tiles_tq1_p<type, J, fallback>,
                ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<type, J, fallback, MMQ_Q8_1_DS_LAYOUT_D4>,
                ggml_cuda_mmq_write_back_mma<type, J, fallback, prec_src1, false>);
        case GGML_TYPE_Q4_0:
            return ggml_cuda_mmq_util_funcs(
                -1,
""")
    s = rep(s, "extern DECL_MMQ_CASE(GGML_TYPE_Q2_0);\n", "extern DECL_MMQ_CASE(GGML_TYPE_Q2_0);\nextern DECL_MMQ_CASE(GGML_TYPE_TQ1_P);\n")
    return s


LOADER = r"""
// TQ1_P: kbx0 and stride count 32-value digit planes (5 per 32-byte block; the host passes the row stride in planes).
// Plane p of a row is digit n = p % 5 of the 32 bytes of block p / 5: digit n of byte b = ((b * 3^n mod 256) * 3) >> 8,
// trit = digit - 1. Each thread decodes two ints of code bytes (4 trits each, two 16-bit lanes at a time) per row.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_tq1_p(
        const char * __restrict__ x, const char * __restrict__ x_gate, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    GGML_UNUSED(x_gate);

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int planes_per_iter = MMQ_ITER_K / QK8_1;   // 8
    constexpr int tpp             = warp_size / QI8_1;    // planes a warp covers per pass
    constexpr int npass           = planes_per_iter / tpp;
    static_assert(warp_size % QI8_1 == 0 && planes_per_iter % tpp == 0, "unexpected warp size");

    const int kqsx = threadIdx.x % QI8_1;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

#pragma unroll
        for (int h = 0; h < npass; ++h) {
            const int kl  = threadIdx.x / QI8_1 + h*tpp;   // plane within the tile
            const int p   = kbx0 + i*stride + kl;
            const int blk = p / 5;
            const int n   = p - 5*blk;
            const uint32_t v  = ((const uint32_t *) ((const block_tq1_p *) x + blk)->qs)[kqsx];
            const uint32_t pw = n == 0 ? 1 : n == 1 ? 3 : n == 2 ? 9 : n == 3 ? 27 : 81;
            const uint32_t e  = ((( v       & 0x00FF00FF) * pw) & 0x00FF00FF) * 3;
            const uint32_t o  = ((((v >> 8) & 0x00FF00FF) * pw) & 0x00FF00FF) * 3;
            const uint32_t dg = ((e >> 8) & 0x00030003) | (o & 0x03000300);
            const int q = __vsub4(dg, 0x01010101);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + kl*QI8_0 + kqsx] = q;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kl*QI8_0 + kqsx] = q;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }

    const int ksx = threadIdx.x % planes_per_iter;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                           + ksx] = 1.0f;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + ksx] = 1.0f;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}
"""


def mmq_cu(s):
    s = rep(s, """        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0, GGML_PREC_Q8>(ctx, args, stream);
""", """        case GGML_TYPE_TQ1_P:
            mul_mat_q_case<GGML_TYPE_TQ1_P, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0, GGML_PREC_Q8>(ctx, args, stream);
""")
    s = rep(s, "    const int64_t s01 = src0->nb[1] / ts_src0;\n", """    // TQ1_P tiles address the weight in 32-value digit planes, 5 per block
    const int64_t x_unit = src0->type == GGML_TYPE_TQ1_P ? 5 : 1;
    const int64_t s01 = src0->nb[1] / ts_src0 * x_unit;
""")
    s = rep(s, "    const int64_t s02 = src0->nb[2] / ts_src0;\n", "    const int64_t s02 = src0->nb[2] / ts_src0 * x_unit;\n")
    s = rep(s, "    const int64_t s03 = src0->nb[3] / ts_src0;\n", "    const int64_t s03 = src0->nb[3] / ts_src0 * x_unit;\n")
    s = rep(s, """    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
""", """    bool mmq_supported;

    if (type == GGML_TYPE_TQ1_P) {
        // native tiles; the alternative is a per-call expansion of the whole weight, slower at every batch size
        static const bool off = getenv("GGML_CUDA_TQ1P_NO_MMQ") != nullptr;
        if (off) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_TQ1_P:
        case GGML_TYPE_Q1_0:
""")
    s = rep(s, "        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||", "        if (type == GGML_TYPE_TQ1_P || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||")
    return s


edit("mmq.cuh", mmq_cuh)
edit("mmq-load-tiles.cuh", lambda s: s.rstrip("\n") + "\n" + LOADER)
edit("mmq.cu", mmq_cu)
open(os.path.join(d, "template-instances", "mmq-instance-tq1_p.cu"), "w").write(
    '// TQ1_P MMQ instance (native ternary tiles)\n\n#include "../mmq.cuh"\n\nDECL_MMQ_CASE(GGML_TYPE_TQ1_P);\n')
print("patched")
