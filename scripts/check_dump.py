#!/usr/bin/env python3
"""Sanity checks for llama-dump-nextn output (runbook section 6).
usage: check_dump.py DUMP_DIR_A DUMP_DIR_B INPUT.jsonl [server_url]
"""
import glob, json, os, sys
import numpy as np

a, b, inp = sys.argv[1:4]
server = sys.argv[4] if len(sys.argv) > 4 else None
D, K = 5120, 32

def load(d):
    out = {}
    for idx in sorted(glob.glob(os.path.join(d, "*.index.jsonl"))):
        base = idx[:-len(".index.jsonl")]
        tok = np.fromfile(base + ".tok.i32", dtype=np.int32)
        out[os.path.basename(base)] = dict(
            tok=tok, h=np.memmap(base + ".h.f16", dtype=np.float16, mode="r").reshape(-1, D),
            ti=np.fromfile(base + ".topk.i32", dtype=np.int32).reshape(-1, K),
            tp=np.fromfile(base + ".topk.f16", dtype=np.float16).reshape(-1, K),
            index=[json.loads(l) for l in open(idx)])
    return out

A, B = load(a), load(b)
for name, s in A.items():
    n = len(s["tok"])
    assert len(s["h"]) == n and len(s["ti"]) == n, "row count mismatch"
    t = B[name]
    same_tok = np.array_equal(s["tok"], t["tok"]); same_ti = np.array_equal(s["ti"], t["ti"])
    h = np.asarray(s["h"], dtype=np.float32)
    rms = np.sqrt((h ** 2).mean(axis=1))
    print(f"{name}: {n} positions, {len(s['index'])} sequences")
    print(f"  determinism: tokens identical {same_tok}, top-k ids identical {same_ti}, "
          f"top-1 identical {np.array_equal(s['ti'][:,0], t['ti'][:,0])}, "
          f"max |dh| {np.abs(h - np.asarray(t['h'], dtype=np.float32)).max():.4f}")
    print(f"  hidden: finite {np.isfinite(h).all()}, row RMS min {rms.min():.3f} mean {rms.mean():.3f} max {rms.max():.3f}")
    p1 = np.exp(s["tp"][:, 0].astype(np.float32)); mass = np.exp(s["tp"].astype(np.float32)).sum(axis=1)
    print(f"  top-1 prob mean {p1.mean():.3f}, top-32 mass mean {mass.mean():.4f} min {mass.min():.4f}")
    # teacher-forced: does the trunk's top-1 at p equal the actual next token?
    for e in s["index"]:
        st, ln, npr = e["start"], e["length"], e["n_prompt"]
        gen = slice(st + npr - 1, st + ln - 1)
        agree = (s["ti"][gen, 0] == s["tok"][st + npr: st + ln]).mean() if ln > npr else float("nan")
        print(f"  {e['id']}: length {ln}, n_prompt {npr}, trunk top-1 == sampled token on {agree:.2f} of generated positions")

if server:
    import requests
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(inp) if l.strip()}
    for name, s in A.items():
        for e in s["index"][:5]:
            r = requests.post(server + "/completion", json={"prompt": recs[e["id"]]["prompt"], "n_predict": 1,
                              "temperature": 0, "return_tokens": True, "cache_prompt": False}).json()
            want = r["tokens"][0]; got = int(s["ti"][e["start"] + e["n_prompt"] - 1, 0])
            print(f"  greedy check {e['id']}: server {want}, dump argmax {got} -> {'OK' if want == got else 'MISMATCH'}")
