#!/usr/bin/env python3
"""Feature alignment for the DSpark drafter. The stock drafter projects five full-precision trunk layers through one
linear layer (fc). Given the same text run through the 4-bit control (features the stock drafter understands) and
through the ternary trunk, solve the least-squares map from ternary features to the stock projection:
    fc_new = argmin_W || X_ternary W^T - X_control fc_stock^T ||^2 + ridge
and save a checkpoint with fc replaced. The two dumps must hold the same sequences in the same order.
usage: dspark_align.py TERNARY_DUMP CONTROL_DUMP PREFIX OUT.pt [ridge=1e-3]"""
import glob, json, os, sys
import numpy as np, torch
from dspark import DSpark, D, NAUX
from mtp_head import is_held_out

tdir, cdir, prefix, out = sys.argv[1:5]
ridge = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
m = DSpark()
m.load_stock("ttf-home/models/drafts/dspark-27b/hf/model.safetensors")
Ws = m.w["fc/weight"].detach().cuda().float()                        # [D, 5D]
K = NAUX * D
G = torch.zeros(K, K, dtype=torch.float64, device="cuda")
C = torch.zeros(K, D, dtype=torch.float64, device="cuda")
sse = sst = 0.0; n_fit = n_val = 0
val = []
for cidx in sorted(glob.glob(os.path.join(cdir, prefix + "-*.index.jsonl"))):
    name = os.path.basename(cidx)[:-len(".index.jsonl")]
    tb, cb = os.path.join(tdir, name), os.path.join(cdir, name)
    tt = np.memmap(tb + ".tok.i32", dtype=np.int32, mode="r"); ct = np.memmap(cb + ".tok.i32", dtype=np.int32, mode="r")
    n = min(len(tt), len(ct))
    assert n > 0 and np.array_equal(tt[:n], ct[:n]), f"{name}: the two dumps do not hold the same tokens"
    ta = np.memmap(tb + ".aux.f16", dtype=np.float16, mode="r").reshape(-1, K)
    ca = np.memmap(cb + ".aux.f16", dtype=np.float16, mode="r").reshape(-1, K)
    keep = np.zeros(n, dtype=bool)
    for l in open(cidx):
        e = json.loads(l); e.setdefault("conv", e["id"].rsplit("-t", 1)[0])
        if e["start"] + e["length"] <= n:
            keep[e["start"]:e["start"] + e["length"]] = not is_held_out(e)
            if is_held_out(e) and len(val) < 40:
                val.append((tb, cb, e["start"], e["length"]))
    rows = np.nonzero(keep)[0]
    for i in range(0, len(rows), 4096):
        r = rows[i:i + 4096]
        X = torch.from_numpy(np.asarray(ta[r])).cuda().float()
        Y = torch.from_numpy(np.asarray(ca[r])).cuda().float() @ Ws.T
        G += (X.T @ X).double(); C += (X.T @ Y).double(); n_fit += len(r)
    print(f"{name}: {len(rows)} rows (total {n_fit})", flush=True)
lam = ridge * float(G.diagonal().mean())
G.diagonal().add_(lam)
W = torch.linalg.solve(G, C).T.float()                                # [D, 5D]
del G, C
for tb, cb, s, L in val:
    ta = np.memmap(tb + ".aux.f16", dtype=np.float16, mode="r").reshape(-1, K)
    ca = np.memmap(cb + ".aux.f16", dtype=np.float16, mode="r").reshape(-1, K)
    X = torch.from_numpy(np.asarray(ta[s:s + L])).cuda().float(); Y = torch.from_numpy(np.asarray(ca[s:s + L])).cuda().float() @ Ws.T
    P = X @ W.T
    sse += float((P - Y).pow(2).sum()); sst += float((Y - Y.mean(0)).pow(2).sum()); n_val += L
    cos = float(torch.nn.functional.cosine_similarity(P, Y, dim=-1).mean())
print(f"fit on {n_fit} rows, ridge {lam:.3g}; held-out {n_val} rows: R2 {1 - sse / max(sst, 1e-9):.4f}, cosine (last seq) {cos:.4f}; "
      f"|W| {float(W.norm()):.2f} vs stock {float(Ws.norm()):.2f}")
with torch.no_grad():
    m.w["fc/weight"].copy_(W.cpu())
torch.save({"sd": m.state_dict(), "n_layers": 5, "tern": False, "step": 0, "eval": {}}, out)
print("saved", out)
