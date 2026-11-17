#!/usr/bin/env python3
"""Total throughput with K concurrent requests against a llama-server started with -np K.
usage: bench_parallel.py PORT LABEL K"""
import json, os, sys, threading, time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}
exec(open(os.path.join(HERE, "bench12.py"), encoding="utf-8").read().split("\nport = ")[0], ns)
PROMPTS = ns["PROMPTS"]
port, label, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
tot = {"n": 0, "dn": 0, "da": 0, "err": 0}
lock = threading.Lock()


def worker(k):
    for i in range(k, len(PROMPTS), K):
        try:
            r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
                "messages": [{"role": "user", "content": PROMPTS[i]}], "max_tokens": 250, "temperature": 0, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}}, timeout=1800).json()
            t = r["timings"]
            with lock:
                tot["n"] += t.get("predicted_n", 0); tot["dn"] += t.get("draft_n") or 0; tot["da"] += t.get("draft_n_accepted") or 0
        except Exception:
            with lock:
                tot["err"] += 1


t0 = time.time()
th = [threading.Thread(target=worker, args=(k,)) for k in range(K)]
[x.start() for x in th]; [x.join() for x in th]
dt = time.time() - t0
acc = f"{tot['da']}/{tot['dn']}" if tot["dn"] else "n/a"
print(f"{label:<22} slots {K}: {tot['n'] / dt:6.1f} tok/s total over {tot['n']} tokens in {dt:.0f}s, drafts accepted {acc}, errors {tot['err']}", flush=True)
