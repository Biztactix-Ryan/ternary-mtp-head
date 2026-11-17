#!/usr/bin/env python3
"""Add optional timers to the engine's MTP drafting code (common/speculative.cpp).
With TTF_SPEC_TIMING=1 the server prints, at exit, the time spent in the head's catch-up decode,
its draft decodes and draft sampling. Without the variable nothing changes.
usage: add_spec_timing.py path/to/common/speculative.cpp"""
import sys
p = sys.argv[1]
L = open(p).read().split("\n")
if any("spc_timing" in l for l in L):
    print("already patched"); sys.exit(0)

def find(text, start=0):
    for i in range(start, len(L)):
        if text in L[i]:
            return i
    raise SystemExit("anchor not found: " + text)

timing = '''
// optional timers for the MTP head (TTF_SPEC_TIMING=1)
struct spc_timing {
    bool    on = getenv("TTF_SPEC_TIMING") != nullptr;
    int64_t t_catch = 0, n_catch = 0, tok_catch = 0, t_draft = 0, n_draft = 0, t_smpl = 0, n_smpl = 0;
    ~spc_timing() {
        if (on && n_draft > 0) {
            fprintf(stderr, "spc_timing: catch-up decode %lld calls, %.3f ms each, %.2f tokens each | "
                            "draft decode %lld calls, %.3f ms each | draft sampling %lld calls, %.3f ms each\\n",
                    (long long) n_catch, n_catch ? t_catch / 1000.0 / n_catch : 0.0, n_catch ? (double) tok_catch / n_catch : 0.0,
                    (long long) n_draft, t_draft / 1000.0 / n_draft, (long long) n_smpl, n_smpl ? t_smpl / 1000.0 / n_smpl : 0.0);
        }
    }
};
static spc_timing g_spc_timing;
'''
inc = find('#include "../src/llama-ext.h"')
L.insert(inc + 1, timing)

# catch-up decode in process()
err = find('SPC_ERR("llama_process(ctx_dft) head=%d failed')
i = err - 2
assert "llama_process(ctx_dft" in L[i], L[i]
ind = L[i][:len(L[i]) - len(L[i].lstrip())]
L[i:i + 1] = [ind + "const int64_t t_spc0 = g_spc_timing.on ? ggml_time_us() : 0;", L[i],
              ind + "if (g_spc_timing.on) { llama_synchronize(ctx_dft); g_spc_timing.t_catch += ggml_time_us() - t_spc0; "
                    "g_spc_timing.n_catch++; g_spc_timing.tok_catch += n_tokens; }"]

# draft decode and sampling in draft()
d = find("void draft(common_speculative_draft_params_vec & dparams) override", err)
i = find("int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());", d)
ind = L[i][:len(L[i]) - len(L[i].lstrip())]
L[i:i + 1] = [ind + "const int64_t t_spc1 = g_spc_timing.on ? ggml_time_us() : 0;", L[i],
              ind + "if (g_spc_timing.on) { llama_synchronize(ctx_dft); g_spc_timing.t_draft += ggml_time_us() - t_spc1; g_spc_timing.n_draft++; }"]
i = find("const llama_token id_sampled = common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);", i)
ind = L[i][:len(L[i]) - len(L[i].lstrip())]
j = find("const auto * cur_p = common_sampler_get_candidates(smpl, true);", i)
L[j:j + 1] = [L[j], ind + "if (g_spc_timing.on) { g_spc_timing.t_smpl += ggml_time_us() - t_spc2; g_spc_timing.n_smpl++; }"]
L[i:i + 1] = [ind + "const int64_t t_spc2 = g_spc_timing.on ? ggml_time_us() : 0;", L[i]]
open(p, "w").write("\n".join(L))
print("patched")
