#!/usr/bin/env python3
"""Move a trained drafter to another set of trunk layers. The drafter reads its trunk layers through one linear layer
(fc). From a dump that holds both the old layers and the new ones, solve the least-squares fc over the new layers
that reproduces the old projection:
    fc_new = argmin_W || X_new W^T - X_old fc_old^T ||^2 + ridge
usage: dspark_refit.py DUMP10 CKPT.pt COLS OUT.pt [ridge=1e-3] [max_rows=500000]
COLS: comma-separated columns of the ten-layer dump (layers 6,13,20,27,34,41,48,55,62,63)."""
import glob, json, os, sys
import numpy as np, torch
from dspark import D, STOCK_IDS
from mtp_head import is_held_out

DUMP_IDS = [6, 13, 20, 27, 34, 41, 48, 55, 62, 63]
ddir, ckpt, cols, out = sys.argv[1:5]
cols = [int(c) for c in cols.split(",")]
ridge = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
max_rows = int(sys.argv[6]) if len(sys.argv) > 6 else 500000
ck = torch.load(ckpt, map_location="cpu")
assert ck.get("cols") in (None, [0, 2, 4, 6, 8]) and not ck.get("tern", False)
old = [DUMP_IDS.index(i) for i in ck.get("layer_ids", STOCK_IDS)]
Wo = ck["sd"]["w.fc/weight"].cuda().float()                           # [D, 5D]
N, K = len(DUMP_IDS), len(cols) * D
G = torch.zeros(K, K, device="cuda"); C = torch.zeros(K, D, device="cuda")
n_fit, val = 0, []


def rows_of(a, r):
    x = torch.from_numpy(np.asarray(a[r])).cuda().view(-1, N, D)
    return x[:, cols].reshape(len(x), -1).float(), x[:, old].reshape(len(x), -1).float() @ Wo.T


for idx in sorted(glob.glob(os.path.join(ddir, "pub1-*.index.jsonl"))):
    base = idx[:-len(".index.jsonl")]
    a = np.memmap(base + ".aux.f16", dtype=np.float16, mode="r").reshape(-1, N * D)
    keep = np.zeros(len(a), dtype=bool)
    for l in open(idx):
        e = json.loads(l); e.setdefault("conv", e["id"].rsplit("-t", 1)[0])
        keep[e["start"]:e["start"] + e["length"]] = not is_held_out(e)
        if is_held_out(e) and len(val) < 40:
            val.append((base, e["start"], e["length"]))
    rows = np.nonzero(keep)[0]
    for i in range(0, len(rows), 2048):
        X, Y = rows_of(a, rows[i:i + 2048])
        G += X.T @ X; C += X.T @ Y; n_fit += len(X)
    print(f"{os.path.basename(base)}: {len(rows)} rows (total {n_fit})", flush=True)
    if n_fit >= max_rows:
        break
lam = ridge * float(G.diagonal().mean())
G.diagonal().add_(lam)
G = G.cpu().double(); C = C.cpu().double(); torch.cuda.empty_cache()
W = torch.cholesky_solve(C, torch.linalg.cholesky(G)).T.float().cuda()   # [D, n*D]
del G, C
sse = sst = 0.0; n_val = 0
for base, s, L in val:
    a = np.memmap(base + ".aux.f16", dtype=np.float16, mode="r").reshape(-1, N * D)
    X, Y = rows_of(a, slice(s, s + L))
    P = X @ W.T
    sse += float((P - Y).pow(2).sum()); sst += float((Y - Y.mean(0)).pow(2).sum()); n_val += L
print(f"cols {cols}: fit on {n_fit} rows, ridge {lam:.3g}; held-out {n_val} rows: R2 {1 - sse / max(sst, 1e-9):.4f}")
ck["sd"]["w.fc/weight"] = W.cpu()
ck.update(cols=cols, layer_ids=[DUMP_IDS[c] for c in cols], step=0, eval={})
torch.save(ck, out)
print("saved", out)
