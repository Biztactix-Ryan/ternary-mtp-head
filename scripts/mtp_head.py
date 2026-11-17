#!/usr/bin/env python3
"""PyTorch replica of the Qwen3.8 MTP draft head (engine: src/models/qwen35.cpp, graph_mtp),
a loader for the stock MTP GGUF, the dump reader, and an agreement evaluation.

  mtp_head.py eval --mtp stock.gguf --dump DIR [--ckpt trained.pt] [--max-seqs N]
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D, N_HEAD, N_KV, HD, FF = 5120, 24, 4, 256, 17408
N_ROT, ROPE_BASE, EPS = 64, 1e7, 1e-6
TOPK = 32
BLK = "blk.64."

MATS = {  # name -> (out, in)
    "nextn.eh_proj": (D, 2 * D), "attn_q": (N_HEAD * HD * 2, D), "attn_k": (N_KV * HD, D), "attn_v": (N_KV * HD, D),
    "attn_output": (N_HEAD * HD, D)[::-1], "ffn_up": (FF, D), "ffn_gate": (FF, D), "ffn_down": (D, FF),
}
NORMS = {"nextn.enorm": D, "nextn.hnorm": D, "nextn.shared_head_norm": D, "attn_norm": D,
         "post_attention_norm": D, "attn_q_norm": HD, "attn_k_norm": HD}


def rms_norm(x, w):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float())


def rope(x, pos):
    """NeoX RoPE on the first N_ROT dims of the last axis. x: [B, T, H, HD], pos: [T]."""
    half = N_ROT // 2
    inv = ROPE_BASE ** (-torch.arange(half, device=x.device, dtype=torch.float32) * 2.0 / N_ROT)
    ang = pos.float()[:, None] * inv[None, :]                    # [T, half]
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    x0, x1, rest = x[..., :half], x[..., half:N_ROT], x[..., N_ROT:]
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos, rest], dim=-1)


class MTPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mats = nn.ParameterDict({k.replace(".", "_"): nn.Parameter(torch.empty(*s)) for k, s in MATS.items()})
        self.norms = nn.ParameterDict({k.replace(".", "_"): nn.Parameter(torch.ones(n)) for k, n in NORMS.items()})
        # frozen, from the MTP file itself (the engine uses the draft file's embedding and output matrix)
        self.register_buffer("tok_embd", torch.empty(0), persistent=False)
        self.register_buffer("out_w", torch.empty(0), persistent=False)

    def lin(self, name, x):
        return F.linear(x, self.mats[name].to(x.dtype))

    def embed(self, tok):
        return self.tok_embd[tok]

    def block(self, e, h_prev, pos, kv_extra=None, return_kv=False):
        """e: token embeddings [B, T, D]; h_prev: hidden state input [B, T, D]; pos: [T].
        Returns g [B, T, D] (post shared_head_norm). Causal attention within the sequence.
        kv_extra / return_kv are for stage 2 (unrolled drafting)."""
        n = self.norms
        z = self.lin("nextn_eh_proj", torch.cat([rms_norm(e, n["nextn_enorm"]), rms_norm(h_prev, n["nextn_hnorm"])], -1).to(e.dtype))
        for sfx in self.layer_suffixes():
            z = self.layer(z, pos, sfx)
        return rms_norm(z, n["nextn_shared_head_norm"])

    def layer_suffixes(self):
        return [""]

    def layer(self, z, pos, sfx=""):
        """one attention + FFN block on z [B, T, D]; sfx selects the weights of a stacked block"""
        n = self.norms
        B, T, _ = z.shape
        a = rms_norm(z, n["attn_norm" + sfx]).to(z.dtype)
        qf = self.lin("attn_q" + sfx, a).view(B, T, N_HEAD, 2 * HD)
        q, gate = qf[..., :HD], qf[..., HD:]
        q = rope(rms_norm(q, n["attn_q_norm" + sfx]), pos)
        k = rope(rms_norm(self.lin("attn_k" + sfx, a).view(B, T, N_KV, HD), n["attn_k_norm" + sfx]), pos)
        v = self.lin("attn_v" + sfx, a).view(B, T, N_KV, HD).float()
        rep = N_HEAD // N_KV
        kk = k.repeat_interleave(rep, dim=2).transpose(1, 2)     # [B, H, T, HD]
        vv = v.repeat_interleave(rep, dim=2).transpose(1, 2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), kk, vv, is_causal=True)   # scale 1/sqrt(HD)
        o = o.transpose(1, 2).reshape(B, T, N_HEAD * HD)
        o = (o * torch.sigmoid(gate.reshape(B, T, N_HEAD * HD).float())).to(z.dtype)
        z = z + self.lin("attn_output" + sfx, o)
        f = rms_norm(z, n["post_attention_norm" + sfx]).to(z.dtype)
        return z + self.lin("ffn_down" + sfx, F.silu(self.lin("ffn_gate" + sfx, f)) * self.lin("ffn_up" + sfx, f))

    # ---- unrolled drafting (stage 2, "training-time test"); single-block heads only ----
    def _inp(self, e, h_in):
        n = self.norms
        return self.lin("nextn_eh_proj", torch.cat([rms_norm(e, n["nextn_enorm"]), rms_norm(h_in, n["nextn_hnorm"])], -1).to(e.dtype))

    def _qkv(self, z, pos):
        """-> q, k, v as [B, H, T, HD] (k, v repeated to all heads) and the gate [B, T, H, HD]"""
        n = self.norms
        B, T, _ = z.shape
        a = rms_norm(z, n["attn_norm"]).to(z.dtype)
        qf = self.lin("attn_q", a).view(B, T, N_HEAD, 2 * HD)
        q = rope(rms_norm(qf[..., :HD], n["attn_q_norm"]), pos)
        k = rope(rms_norm(self.lin("attn_k", a).view(B, T, N_KV, HD), n["attn_k_norm"]), pos)
        v = self.lin("attn_v", a).view(B, T, N_KV, HD).float()
        rep = N_HEAD // N_KV
        return (q.transpose(1, 2), k.repeat_interleave(rep, dim=2).transpose(1, 2),
                v.repeat_interleave(rep, dim=2).transpose(1, 2), qf[..., HD:])

    def _post(self, z, o, gate):
        """attention output o [B, H, T, HD] -> g [B, T, D]"""
        n = self.norms
        B, _, T, _ = o.shape
        o = o.transpose(1, 2).reshape(B, T, N_HEAD * HD)
        o = (o * torch.sigmoid(gate.reshape(B, T, N_HEAD * HD).float())).to(z.dtype)
        z = z + self.lin("attn_output", o)
        f = rms_norm(z, n["post_attention_norm"]).to(z.dtype)
        z = z + self.lin("ffn_down", F.silu(self.lin("ffn_gate", f)) * self.lin("ffn_up", f))
        return rms_norm(z, n["nextn_shared_head_norm"])

    def unroll(self, tok, h, steps=3, pos=None, detach=False):
        """Outputs of draft steps 1..steps at every position, as the engine computes them.
        Step 1 at p: input (tok[p], h[p-1]), attends step-1 keys of positions <= p.
        Step 2 at p: input (tok[p], g1[p-1]), attends step-1 keys of positions < p and its own.
        Step 3 at p: input (tok[p], g2[p-1]), attends step-1 keys < p-1, the step-2 key at p-1, its own.
        Deeper steps follow the same rule. Every output predicts tok[p+1]; step s is meaningful for p >= s."""
        B, T = tok.shape
        pos = torch.arange(T, device=tok.device) if pos is None else pos
        e = self.embed(tok)
        idx = torch.arange(T, device=tok.device)
        sc = HD ** -0.5
        ninf = float("-inf")
        shift = lambda x: torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)          # along T of [B, T, ...]
        shift_h = lambda x: torch.cat([torch.zeros_like(x[:, :, :1]), x[:, :, :-1]], dim=2)   # along T of [B, H, T, HD]

        # step 1: ordinary causal attention over the step-1 keys
        z = self._inp(e, shift(h))
        q, k, v, gate = self._qkv(z, pos)
        ks, vs = [k], [v]                                   # keys / values of steps 1..s
        S = (q @ k.transpose(-1, -2)) * sc
        o = torch.softmax(S.masked_fill(idx[None, :] > idx[:, None], ninf), -1) @ v
        outs = [self._post(z, o, gate)]
        # step s >= 2 at position p: step-1 keys for positions <= p-(s-1), then the step s-m key at p-m
        # for m = s-2 .. 0 (the draft region written by the earlier steps of this cycle, then its own)
        for s_ in range(2, steps + 1):
            g = outs[-1].detach() if detach else outs[-1]
            z = self._inp(e, shift(g).to(e.dtype))
            q, k, v, gate = self._qkv(z, pos)
            ks.append(k); vs.append(v)
            S = ((q @ ks[0].transpose(-1, -2)) * sc).masked_fill(idx[None, :] > idx[:, None] - (s_ - 1), ninf)
            diag, vals = [], []
            for m in range(s_ - 2, -1, -1):
                km, vm = ks[s_ - m - 1], vs[s_ - m - 1]
                for _ in range(m):
                    km, vm = shift_h(km), shift_h(vm)
                d = (q * km).sum(-1, keepdim=True) * sc
                if m:
                    d = d.masked_fill((idx < m)[None, None, :, None], ninf)
                diag.append(d); vals.append(vm)
            P = torch.softmax(torch.cat([S] + diag, -1), -1)
            o = P[..., :T] @ ks[0].new_zeros(()).add(vs[0])
            for i, vm in enumerate(vals):
                o = o + P[..., T + i:T + i + 1] * vm
            outs.append(self._post(z, o, gate))
        return outs

    def prepare(self):      # hooks for heads that cache per-step weights (ternary head)
        pass

    def release(self):
        pass

    def forward(self, tok, h, pos=None):
        """tok: [B, T] token ids; h: [B, T, D] trunk hidden states (row p = h[p]).
        Position p consumes (tok[p], h[p-1]), with zeros at p = 0, as the engine does."""
        B, T = tok.shape
        pos = torch.arange(T, device=tok.device) if pos is None else pos
        h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        return self.block(self.embed(tok), h_prev, pos)

    def logits(self, g):
        return F.linear(g.to(self.out_w.dtype), self.out_w)


def read_gguf_tensors(path, names=None):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    out = {}
    for t in GGUFReader(path).tensors:
        if names is not None and t.name not in names:
            continue
        shape = tuple(int(x) for x in reversed(t.shape))
        arr = dequantize(t.data, t.tensor_type).astype(np.float32, copy=False).reshape(shape)
        out[t.name] = arr
    return out


def load_trunk_io(path, device="cuda"):
    """token_embd / output.weight of a trunk GGUF (what an in-file head uses), as bf16 on `device`."""
    cache = path + ".io.pt"
    if os.path.exists(cache):
        sd = torch.load(cache)
    else:
        raw = read_gguf_tensors(path, {"token_embd.weight", "output.weight"})
        sd = {k: torch.from_numpy(np.ascontiguousarray(v)).to(torch.bfloat16) for k, v in raw.items()}
        torch.save(sd, cache)
    return sd["token_embd.weight"].to(device), sd["output.weight"].to(device)


def load_stock(path, device="cuda", cache=None, big=True, emb_from=None):
    """Build the head from the stock MTP GGUF. `cache` is a .pt path for the dequantised tensors.
    emb_from: take the embedding and output matrices from this trunk GGUF instead (in-file head)."""
    cache = cache or path + ".dequant.pt"
    if os.path.exists(cache):
        sd = torch.load(cache)
    else:
        t0 = time.time()
        raw = read_gguf_tensors(path)
        sd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in raw.items()}
        for k in ("token_embd.weight", "output.weight"):
            sd[k] = sd[k].to(torch.bfloat16)
        torch.save(sd, cache)
        print(f"dequantised {len(sd)} tensors in {time.time() - t0:.0f}s -> {cache}", file=sys.stderr)
    m = MTPHead()
    with torch.no_grad():
        for k, s in MATS.items():
            w = sd[BLK + k + ".weight"]
            assert tuple(w.shape) == tuple(s), (k, w.shape, s)
            m.mats[k.replace(".", "_")].copy_(w)
        for k in NORMS:
            m.norms[k.replace(".", "_")].copy_(sd[BLK + k + ".weight"])
    m = m.to(device)
    if emb_from:
        m.tok_embd, m.out_w = load_trunk_io(emb_from, device)
    elif big:
        m.tok_embd = sd["token_embd.weight"].to(device)
        m.out_w = sd["output.weight"].to(device)
    return m


class Dump:
    """Memory-mapped view of the dumper's shards; .seqs lists every sequence."""
    def __init__(self, d):
        """d: one dump directory, or several separated by commas"""
        self.shards, self.seqs = [], []
        for idx in sorted(p for one in d.split(",") for p in glob.glob(os.path.join(one, "*.index.jsonl"))):
            base = idx[:-len(".index.jsonl")]
            sh = dict(tok=np.memmap(base + ".tok.i32", dtype=np.int32, mode="r"),
                      h=np.memmap(base + ".h.f16", dtype=np.float16, mode="r").reshape(-1, D),
                      ti=np.memmap(base + ".topk.i32", dtype=np.int32, mode="r").reshape(-1, TOPK),
                      tp=np.memmap(base + ".topk.f16", dtype=np.float16, mode="r").reshape(-1, TOPK))
            for l in open(idx):
                e = json.loads(l)
                e["shard"] = len(self.shards)
                e.setdefault("conv", e["id"].rsplit("-t", 1)[0])
                self.seqs.append(e)
            self.shards.append(sh)

    def get(self, e, max_len=None, device="cuda"):
        """Tensors for one sequence (tail-cropped to max_len): tok [T], h [T, D], ti/tp [T, K], gen mask [T]."""
        sh = self.shards[e["shard"]]
        s, L, npr = e["start"], e["length"], e["n_prompt"]
        off = max(0, L - max_len) if max_len else 0
        sl = slice(s + off, s + L)
        tok = torch.from_numpy(np.asarray(sh["tok"][sl]).astype(np.int64)).to(device)
        h = torch.from_numpy(np.asarray(sh["h"][sl])).to(device)
        ti = torch.from_numpy(np.asarray(sh["ti"][sl]).astype(np.int64)).to(device)
        tp = torch.from_numpy(np.asarray(sh["tp"][sl])).to(device).float()
        gen = torch.arange(off, L, device=device) >= npr - 1   # positions whose target is a generated token
        return tok, h, ti, tp, gen


@torch.no_grad()
def evaluate(model, dump, seqs, max_len=4096, chunk=2048, dtype=torch.bfloat16, steps=1):
    """Agreement of the head's first draft with the trunk. Position p: head input (tok[p], h[p-1]),
    target = trunk top-1 at p. Reports generated positions (target token was generated) and all positions.
    steps > 1 adds "gen2", "gen3": the same agreement for draft steps 2 and 3 (teacher-forced tokens)."""
    model.eval()
    tot = {"gen": [0, 0, 0.0], "all": [0, 0, 0.0]}   # n, top-1 hits, sum trunk prob of head's top-1
    for s in range(2, steps + 1):
        tot[f"gen{s}"] = [0, 0, 0.0]
    for e in seqs:
        tok, h, ti, tp, gen = dump.get(e, max_len)
        with torch.autocast("cuda", dtype=dtype):
            if steps > 1:
                gs = [x[0] for x in model.unroll(tok[None], h[None].to(dtype), steps)]
                g = gs[0]
            else:
                g = model(tok[None], h[None].to(dtype))[0]
        for s in range(2, steps + 1):
            ok = gen.clone(); ok[:s] = False
            for i in range(0, len(tok), chunk):
                top = model.logits(gs[s - 1][i:i + chunk]).argmax(-1)
                m = ok[i:i + chunk]
                tot[f"gen{s}"][0] += int(m.sum()); tot[f"gen{s}"][1] += int(((top == ti[i:i + chunk, 0]) & m).sum())
        for i in range(0, len(tok), chunk):
            top = model.logits(g[i:i + chunk]).argmax(-1)
            hit = top == ti[i:i + chunk, 0]
            match = top[:, None] == ti[i:i + chunk]                       # head's top-1 within trunk top-K
            p = (tp[i:i + chunk].exp() * match).sum(-1)
            valid = torch.ones_like(hit);
            if i == 0:
                valid[0] = False                                          # position 0 has no h[p-1]
            for key, m in (("gen", gen[i:i + chunk] & valid), ("all", valid)):
                tot[key][0] += int(m.sum()); tot[key][1] += int((hit & m).sum()); tot[key][2] += float((p * m).sum())
    return {k: {"n": v[0], "top1": v[1] / max(v[0], 1), "trunk_p": v[2] / max(v[0], 1)} for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--mtp", required=True); p.add_argument("--dump", required=True)
    p.add_argument("--ckpt"); p.add_argument("--max-seqs", type=int, default=0); p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--emb-from"); p.add_argument("--held-only", action="store_true", help="only the held-out 5% of conversations")
    p.add_argument("--by-prefix", action="store_true", help="report separately per id prefix (text before the first '-')")
    a = ap.parse_args()
    model = load_stock(a.mtp, emb_from=a.emb_from)
    if a.ckpt:
        sd = torch.load(a.ckpt)
        model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    dump = Dump(a.dump)
    seqs = [e for e in dump.seqs if not a.held_only or is_held_out(e)]
    groups = {"all": seqs}
    if a.by_prefix:
        groups = {}
        for e in seqs:
            groups.setdefault(e["id"].split("-")[0], []).append(e)
    for name, ss in sorted(groups.items()):
        ss = ss[:a.max_seqs] if a.max_seqs else ss
        r = evaluate(model, dump, ss, a.max_len)
        ctx = sum(x["length"] + x.get("skipped", 0) for x in ss) / max(len(ss), 1)
        print(f"{name:<10} seqs {len(ss):>5} avg context {ctx:6.0f}  generated n {r['gen']['n']:>8}  "
              f"agreement {100 * r['gen']['top1']:.1f}%  trunk-prob {r['gen']['trunk_p']:.3f}")


def is_held_out(e):
    """5% of conversations (not turns), stable across runs"""
    import hashlib
    return int(hashlib.md5(e["conv"].encode()).hexdigest(), 16) % 20 == 0


def _unused():
    pass


if __name__ == "__main__":
    main()
