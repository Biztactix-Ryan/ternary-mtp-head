#!/usr/bin/env python3
"""Next-token distributions after raw prompts of 5 to 8 tokens (the call sizes the extended table kernel now takes),
compared between two servers. usage: compare_small.py REF_PORT PORT"""
import json, sys, urllib.request

BASE = ["The capital of France is", "def fibonacci(n):\n    if n", "Water boils at one hundred degrees", "import numpy as np\nx =",
        "Once upon a time there was a", "The three primary colours are red,", "SELECT name FROM users WHERE id",
        "To restart the nginx service run sudo", "In 1969 the first humans landed on", "The opposite of hot is"]


def post(port, path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                                                    {"Content-Type": "application/json"}), timeout=300))


ref, port = sys.argv[1:3]
rows = []
for text in BASE:
    toks = post(ref, "/tokenize", {"content": text, "add_special": True})["tokens"]
    for n in (5, 6, 7, 8):
        if len(toks) >= n:
            rows.append(toks[:n])
same = cnt = 0; dmax = dsum = 0.0; k = 0
for t in rows:
    out = []
    for p in (ref, port):
        r = post(p, "/completion", {"prompt": t, "n_predict": 1, "temperature": 0, "n_probs": 10, "cache_prompt": False})
        out.append([(c["id"], c["logprob"]) for c in r["completion_probabilities"][0]["top_logprobs"]])
    a, b = out[0], dict(out[1])
    same += a[0][0] == out[1][0][0]; cnt += 1
    for tok, lp in a:
        if tok in b and lp > -6:
            dmax = max(dmax, abs(b[tok] - lp)); dsum += abs(b[tok] - lp); k += 1
print(f"{cnt} prompts of 5-8 tokens: same top token {same}/{cnt}, log-prob difference mean {dsum / max(k, 1):.5f} max {dmax:.5f}")
