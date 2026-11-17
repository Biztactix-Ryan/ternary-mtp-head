#!/usr/bin/env python3
"""Next-token distributions straight after prefill, compared between servers running different batch paths.
For each prompt (short ones and long ones built by joining them) asks for one token with top-10 log-probabilities.
usage: compare_paths.py REF_PORT PORT [PORT...]"""
import json, math, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}
exec(open(os.path.join(HERE, "bench12.py"), encoding="utf-8").read().split("\nport = ")[0], ns)
P = ns["PROMPTS"]
prompts = list(P)
for k in range(len(P)):  # long prompts: several tasks back to back, answer the last
    prompts.append("\n\n".join(P[(k + j) % len(P)] for j in range(6)) + "\n\nAnswer only the last request.")
for k in range(4):
    prompts.append("\n\n".join(P[(k + j) % len(P)] for j in range(12)) * 3 + "\n\nAnswer only the last request.")


def top(port, prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0, "cache_prompt": False,
                       "logprobs": True, "top_logprobs": 10, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                                                 {"Content-Type": "application/json"}), timeout=600))
    c = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return [(t["token"], t["logprob"]) for t in c], r["usage"]["prompt_tokens"]


ref = [top(sys.argv[1], p) for p in prompts]
for port in sys.argv[2:]:
    same, dmax, dsum, n, kl = 0, 0.0, 0.0, 0, 0.0
    for p, (rt, ntok) in zip(prompts, ref):
        t, _ = top(port, p)
        same += t[0][0] == rt[0][0]
        d = dict(t)
        for tok, lp in rt:
            if tok in d and lp > -6:
                diff = abs(d[tok] - lp); dmax = max(dmax, diff); dsum += diff; n += 1
                kl += math.exp(lp) * (lp - d[tok])
    print(f"port {port} vs {sys.argv[1]}: same top token {same}/{len(prompts)}, log-prob difference mean {dsum / max(n, 1):.5f} max {dmax:.5f}, "
          f"KL over shared top-10 {kl / len(prompts):.6f} per prompt; prompt tokens {min(x[1] for x in ref)}-{max(x[1] for x in ref)}", flush=True)
