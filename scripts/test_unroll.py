#!/usr/bin/env python3
"""Check MTPHead.unroll against a slow reference built from the ordinary causal layer.

Draft step 2 at position q must equal the ordinary block run on the sequence whose inputs are the
step-1 inputs for positions < q and the step-2 input at q (that is what the engine's KV cache holds).
Step 3 at q: step-1 inputs for positions < q-1, the step-2 input at q-1, the step-3 input at q.
usage: test_unroll.py STOCK_MTP.gguf DUMP_DIR [ckpt] [trunk.gguf]
"""
import sys
import torch
from mtp_head import Dump, load_stock, rms_norm

mtp, dump_dir = sys.argv[1:3]
model = load_stock(mtp, emb_from=sys.argv[4] if len(sys.argv) > 4 else None)
if len(sys.argv) > 3:
    model.load_state_dict(torch.load(sys.argv[3])["model"], strict=False)
model = model.float().eval()
model.tok_embd = model.tok_embd.float()
dump = Dump(dump_dir)
tok, h, *_ = dump.get(dump.seqs[0], 96)
tok, h = tok[None], h[None].float()
T = tok.shape[1]
pos = torch.arange(T, device=tok.device)
shift = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
with torch.no_grad():
    N = 5
    gs = model.unroll(tok, h, N)
    print("step 1 vs forward(): max abs diff %.2e" % (gs[0] - model(tok, h)).abs().max().item())
    e = model.embed(tok)
    head = lambda z: rms_norm(model.layer(z, pos[:z.shape[1]]), model.norms["nextn_shared_head_norm"])
    zs = [model._inp(e, shift(h))] + [model._inp(e, shift(g)) for g in gs[:-1]]      # inputs of steps 1..N
    worst = 0.0
    for s_ in range(2, N + 1):
        dmax = 0.0
        for q in (s_, s_ + 1, 9, 20, 50, T - 1):
            # positions < q-(s-2) from step 1, then one position from each of steps 2..s
            parts = [zs[0][:, :q - s_ + 2]] + [zs[j][:, q - s_ + 1 + j:q - s_ + 2 + j] for j in range(1, s_)]
            ref = head(torch.cat(parts, 1))[0, -1]
            dmax = max(dmax, (ref - gs[s_ - 1][0, q]).abs().max().item())
        print("step %d vs reference: max abs diff %.2e" % (s_, dmax)); worst = max(worst, dmax)
    print("UNROLL_OK" if worst < 1e-2 else "UNROLL_MISMATCH")
