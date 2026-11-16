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
    g1, g2, g3 = model.unroll(tok, h, 3)
    print("step 1 vs forward(): max abs diff %.2e" % (g1 - model(tok, h)).abs().max().item())
    e = model.embed(tok)
    head = lambda z: rms_norm(model.layer(z, pos[:z.shape[1]]), model.norms["nextn_shared_head_norm"])
    zA, zB, zC = model._inp(e, shift(h)), model._inp(e, shift(g1)), model._inp(e, shift(g2))
    d2 = d3 = 0.0
    for q in (2, 3, 7, 20, 50, T - 1):
        ref2 = head(torch.cat([zA[:, :q], zB[:, q:q + 1]], 1))[0, -1]
        ref3 = head(torch.cat([zA[:, :q - 1], zB[:, q - 1:q], zC[:, q:q + 1]], 1))[0, -1]
        d2 = max(d2, (ref2 - g2[0, q]).abs().max().item()); d3 = max(d3, (ref3 - g3[0, q]).abs().max().item())
    print("step 2 vs reference: max abs diff %.2e" % d2)
    print("step 3 vs reference: max abs diff %.2e" % d3)
    print("typical |g|: %.2f" % g1.abs().mean().item())
    print("UNROLL_OK" if max(d2, d3) < 1e-2 else "UNROLL_MISMATCH")
