#!/usr/bin/env python3
"""DSpark parallel drafter (DFlash backbone + Markov head) for the ternary trunk: PyTorch replica, training on
dumped trunk features, evaluation, and GGUF export. The drafter reads the inputs of five trunk layers at every
context position and predicts a block of tokens after an anchor token in one pass.

  dspark.py eval   --dump DIR [--ckpt x.pt] [--stock model.safetensors]
  dspark.py train  --dump DIR --out DIR [--init x.pt] [--tern] [--layers 5] [--steps N] ...
  dspark.py export --ckpt x.pt --stock-gguf stock.gguf --out new.gguf [--block 15]

Reference model code: RadixArk/Qwen3.8-27B-DSpark (dflash.py, dspark.py). Dump: llama-dump-nextn --layers 6,20,34,48,62.
"""
import argparse, glob, json, math, os, struct, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mtp_head import load_trunk_io, is_held_out

D, NAUX, NH, NKV, HD, FF, RANK = 5120, 5, 32, 8, 128, 17408, 256
MASK_ID, EPS, TOPK = 248070, 1e-6, 32
ROPE_BASE, YARN_FACTOR, YARN_ORIG, BETA_FAST, BETA_SLOW = 1e7, 32.0, 8192, 32.0, 1.0
MATS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
GG = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
      "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down", "input_layernorm": "attn_norm",
      "post_attention_layernorm": "ffn_norm", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm"}


def rms(x, w):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def inv_freq(yarn=True):
    half = HD // 2
    pf = ROPE_BASE ** (torch.arange(half, dtype=torch.float64) * 2.0 / HD)
    if not yarn:
        return (1.0 / pf).float(), 1.0
    ext, itp = 1.0 / pf, 1.0 / (YARN_FACTOR * pf)
    cd = lambda rot: HD * math.log(YARN_ORIG / (rot * 2 * math.pi)) / (2 * math.log(ROPE_BASE))
    low, high = max(math.floor(cd(BETA_FAST)), 0), min(math.ceil(cd(BETA_SLOW)), HD - 1)
    ramp = ((torch.arange(half, dtype=torch.float64) - low) / max(high - low, 1e-3)).clamp(0, 1)
    ef = 1 - ramp
    return (itp * (1 - ef) + ext * ef).float(), 0.1 * math.log(YARN_FACTOR) + 1.0


def rope(x, pos, inv, af):
    """x [S, H, HD], pos [S]; NeoX (half-split) layout over the whole head dim"""
    ang = pos.float()[:, None] * inv.to(x.device)[None, :]
    cos, sin = (ang.cos() * af)[:, None, :], (ang.sin() * af)[:, None, :]
    x0, x1 = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)


def tq1_0(w):
    """Ternary in the engine's TQ1_0 layout: per row, blocks of 256 inputs share one scale; values scale * {-1,0,1}.
    No gradient here: the trainer applies the gradient of the quantised copy to the latent weight (straight-through).
    Threshold 0.7 * mean|w| of the block, scale = mean |w| of the kept weights."""
    N, K = w.shape
    b = w.view(N, K // 256, 256)
    a = b.abs()
    t = (a > 0.7 * a.mean(-1, keepdim=True)).to(w.dtype)
    d = (a * t).sum(-1, keepdim=True) / t.sum(-1, keepdim=True).clamp(min=1)
    return (t * b.sign() * d).view(N, K)


class DSpark(nn.Module):
    def __init__(self, n_layers=5, tern=False, yarn=True, markov_row0=True):
        super().__init__()
        self.n_layers, self.tern, self.markov_row0 = n_layers, tern, markov_row0
        self.inv, self.af = inv_freq(yarn)
        P = nn.ParameterDict
        shapes = {"self_attn.q_proj": (NH * HD, D), "self_attn.k_proj": (NKV * HD, D), "self_attn.v_proj": (NKV * HD, D),
                  "self_attn.o_proj": (D, NH * HD), "mlp.gate_proj": (FF, D), "mlp.up_proj": (FF, D), "mlp.down_proj": (D, FF)}
        self.w = P()
        for i in range(n_layers):
            for m in MATS:
                self.w[self.key(f"layers.{i}.{m}.weight")] = nn.Parameter(torch.zeros(shapes[m]))
            for m, n in (("input_layernorm", D), ("post_attention_layernorm", D), ("self_attn.q_norm", HD), ("self_attn.k_norm", HD)):
                self.w[self.key(f"layers.{i}.{m}.weight")] = nn.Parameter(torch.ones(n))
        for name, shape in (("fc.weight", (D, NAUX * D)), ("markov_head.markov_w1.weight", (248320, RANK)),
                            ("markov_head.markov_w2.weight", (248320, RANK)), ("confidence_head.proj.weight", (1, D + RANK))):
            self.w[self.key(name)] = nn.Parameter(torch.zeros(shape))
        for name, n in (("hidden_norm.weight", D), ("norm.weight", D), ("confidence_head.proj.bias", 1)):
            self.w[self.key(name)] = nn.Parameter(torch.ones(n))
        self.tok_embd = self.out_w = None
        self._q = {}

    @staticmethod
    def key(name):
        return name.replace(".", "/")

    def is_tern(self, name):
        return self.tern and (name.endswith("_proj.weight") and name.startswith("layers.") or name in ("fc.weight", "markov_head.markov_w2.weight"))

    def mat_names(self):
        return [k.replace("/", ".") for k, p in self.w.items() if p.dim() == 2]

    @torch.no_grad()
    def prepare(self, grad=False):
        """Effective bf16 copy of every matrix (ternarised where the model is ternary). With grad=True the copies are
        leaves: their gradients, kept in bf16, are what the optimiser applies to the float weights."""
        self._q = {}
        for name in self.mat_names():
            w = self.w[self.key(name)]
            q = (tq1_0(w) if self.is_tern(name) else w).to(torch.bfloat16)
            self._q[name] = q.requires_grad_(grad and not name.startswith("confidence_head"))

    def mat(self, name):
        return self._q[name]

    def lin(self, name, x):
        return F.linear(x.to(torch.bfloat16), self._q[name])

    def load_stock(self, path):
        f = open(path, "rb")
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        with torch.no_grad():
            for k, v in hdr.items():
                if k == "__metadata__" or self.key(k) not in self.w:
                    continue
                assert v["dtype"] == "BF16", v
                f.seek(base + v["data_offsets"][0])
                raw = np.frombuffer(f.read(v["data_offsets"][1] - v["data_offsets"][0]), dtype=np.int16)
                t = torch.from_numpy(raw.copy()).view(torch.bfloat16).view(v["shape"]).float()
                self.w[self.key(k)].copy_(t)

    def grow(self, n_layers):
        """append layers: copies of the last layer with zeroed output projections (identity at the start)"""
        with torch.no_grad():
            for i in range(self.n_layers, n_layers):
                for k in [k for k in self.w.keys() if k.startswith(f"layers/{self.n_layers - 1}/")]:
                    p = self.w[k].detach().clone()
                    if "o_proj" in k or "down_proj" in k:
                        p = torch.randn_like(p) * 1e-3 if self.tern else torch.zeros_like(p)
                    self.w[k.replace(f"layers/{self.n_layers - 1}/", f"layers/{i}/")] = nn.Parameter(p)
        self.n_layers = n_layers

    def forward(self, aux, tok, anchors, B):
        """aux [T, 5*D] trunk features, tok [T], anchors [N] -> hidden [N, B, D] (post final norm), prev tokens [N, B]"""
        T, N = aux.shape[0], anchors.numel()
        dev = aux.device
        ctx = rms(self.lin("fc.weight", aux), self.w["hidden_norm/weight"]).to(torch.bfloat16)       # [T, D]
        x = self.tok_embd[MASK_ID].to(dev).expand(N, B, D).clone()
        x[:, 0] = self.tok_embd[tok[anchors].cpu()].to(dev)
        x = x.reshape(N * B, D).float()
        Q = N * B
        pos_q = (anchors[:, None] + torch.arange(B, device=dev)[None, :]).reshape(Q)
        pos_k = torch.cat([torch.arange(T, device=dev), pos_q])
        blk = torch.arange(N, device=dev).repeat_interleave(B)
        allow = torch.cat([torch.arange(T, device=dev)[None, :] < anchors.repeat_interleave(B)[:, None],
                           blk[:, None] == blk[None, :]], dim=1)                                     # [Q, T+Q]
        g = NH // NKV
        for i in range(self.n_layers):
            L = f"layers.{i}."
            h = rms(x, self.w[self.key(L + "input_layernorm.weight")])
            q = rms(self.lin(L + "self_attn.q_proj.weight", h).view(Q, NH, HD), self.w[self.key(L + "self_attn.q_norm.weight")])
            k = torch.cat([self.lin(L + "self_attn.k_proj.weight", ctx), self.lin(L + "self_attn.k_proj.weight", h)])
            v = torch.cat([self.lin(L + "self_attn.v_proj.weight", ctx), self.lin(L + "self_attn.v_proj.weight", h)])
            k = rms(k.view(T + Q, NKV, HD), self.w[self.key(L + "self_attn.k_norm.weight")])
            q = rope(q, pos_q, self.inv, self.af).to(torch.bfloat16).transpose(0, 1)                 # [NH, Q, HD]
            k = rope(k, pos_k, self.inv, self.af).to(torch.bfloat16).transpose(0, 1).repeat_interleave(g, 0)
            v = v.view(T + Q, NKV, HD).transpose(0, 1).repeat_interleave(g, 0)
            o = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=allow[None, None])[0]
            x = x + self.lin(L + "self_attn.o_proj.weight", o.transpose(0, 1).reshape(Q, NH * HD)).float()
            h = rms(x, self.w[self.key(L + "post_attention_layernorm.weight")])
            x = x + self.lin(L + "mlp.down_proj.weight", F.silu(self.lin(L + "mlp.gate_proj.weight", h)) * self.lin(L + "mlp.up_proj.weight", h)).float()
        out = rms(x, self.w["norm/weight"]).view(N, B, D)
        prev = tok[(anchors[:, None] + torch.arange(B, device=dev)[None, :]).clamp(max=T - 1)]      # teacher-forced previous tokens
        return out, prev

    def logits(self, out, prev):
        """[R, D], [R] -> [R, V]: trunk output head plus the Markov bias keyed on the previous token"""
        lg = F.linear(out.to(torch.bfloat16), self.out_w).float()
        e = self._q["markov_head.markov_w1.weight"][prev]
        return lg + F.linear(e, self._q["markov_head.markov_w2.weight"]).float()


class Dump:
    def __init__(self, d):
        self.shards, self.seqs = [], []
        for idx in sorted(p for one in d.split(",") for p in glob.glob(os.path.join(one, "*.index.jsonl"))):
            base = idx[:-len(".index.jsonl")]
            sh = dict(tok=np.memmap(base + ".tok.i32", dtype=np.int32, mode="r"),
                      aux=np.memmap(base + ".aux.f16", dtype=np.float16, mode="r").reshape(-1, NAUX * D),
                      ti=np.memmap(base + ".topk.i32", dtype=np.int32, mode="r").reshape(-1, TOPK),
                      tp=np.memmap(base + ".topk.f16", dtype=np.float16, mode="r").reshape(-1, TOPK))
            for l in open(idx):
                e = json.loads(l)
                e["shard"] = len(self.shards)
                e.setdefault("conv", e["id"].rsplit("-t", 1)[0])
                self.seqs.append(e)
            self.shards.append(sh)

    def get(self, e, max_len, device="cuda"):
        sh = self.shards[e["shard"]]
        s, L, npr = e["start"], e["length"], e["n_prompt"]
        off = max(0, L - max_len)
        sl = slice(s + off, s + L)
        tok = torch.from_numpy(np.asarray(sh["tok"][sl]).astype(np.int64)).to(device)
        aux = torch.from_numpy(np.asarray(sh["aux"][sl])).to(device).to(torch.bfloat16)
        ti = torch.from_numpy(np.asarray(sh["ti"][sl]).astype(np.int64)).to(device)
        tp = torch.from_numpy(np.asarray(sh["tp"][sl])).to(device).float()
        return tok, aux, ti, tp, max(npr - off, 1)


def block_stats(model, tok, aux, ti, tp, first, anchors, B, decay=None, chunk=512):
    """loss (if decay) and per-row agreement with the trunk's top-1. Row j of anchor s predicts the token at s+j+1,
    i.e. the trunk's choice at position s+j."""
    T = tok.numel()
    out, prev = model(aux, tok, anchors, B)
    N = anchors.numel()
    p = anchors[:, None] + torch.arange(B, device=tok.device)[None, :]
    valid = (p <= T - 1)
    pc = p.clamp(max=T - 1)
    out, prev, pc, validf = out.reshape(N * B, D), prev.reshape(-1), pc.reshape(-1), valid.reshape(-1)
    hit = torch.zeros(N * B, dtype=torch.bool, device=tok.device)
    loss = 0.0
    w = (decay ** torch.arange(B, device=tok.device).float())[None, :].expand(N, B).reshape(-1) * validf if decay else None
    # training: the vocabulary projection is back-propagated chunk by chunk, so only one chunk of logits is alive
    head = out.detach().requires_grad_() if decay else out
    for i in range(0, N * B, chunk):
        lg = model.logits(head[i:i + chunk], prev[i:i + chunk])
        hit[i:i + chunk] = lg.argmax(-1) == ti[pc[i:i + chunk], 0]
        if decay:
            lp = F.log_softmax(lg, -1).gather(1, ti[pc[i:i + chunk]])
            tgt = F.softmax(tp[pc[i:i + chunk]], -1)
            l = (-(tgt * lp).sum(-1) * w[i:i + chunk]).sum() / w.sum().clamp(min=1)
            l.backward()
            loss += float(l)
        del lg
    if decay:
        out.backward(head.grad)
    hit = (hit & validf).view(N, B)
    return (loss if decay else None), hit, valid


def pick_anchors(T, first, n, gen):
    lo = max(first - 1, 1)          # row 0 predicts the token at s+1: the first generated token has s = n_prompt - 1
    if T - 1 - lo < 1:
        return None
    cand = torch.arange(lo, T - 1)
    if cand.numel() > n:
        cand = cand[torch.randperm(cand.numel(), generator=gen)[:n]]
    return cand.sort().values.cuda()


@torch.no_grad()
def evaluate(model, dump, seqs, B=16, n_anchor=96, max_len=3072):
    model.eval()
    model.prepare()
    gen = torch.Generator().manual_seed(1)
    hits = torch.zeros(B, dtype=torch.float64); cnt = torch.zeros(B, dtype=torch.float64)
    acc = {7: [0.0, 0], 15: [0.0, 0]}
    for e in seqs:
        tok, aux, ti, tp, first = dump.get(e, max_len)
        an = pick_anchors(tok.numel(), first, n_anchor, gen)
        if an is None:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, hit, valid = block_stats(model, tok, aux, ti, tp, first, an, B)
        hits += hit.double().sum(0).cpu(); cnt += valid.double().sum(0).cpu()
        for n in acc:
            full = valid[:, :n].all(1)
            acc[n][0] += float(hit[full][:, :n].double().cumprod(1).sum()); acc[n][1] += int(full.sum())
    model.train()
    rows = (hits / cnt.clamp(min=1)).tolist()
    return {"row": [round(x, 4) for x in rows], "accept7": acc[7][0] / max(acc[7][1], 1), "accept15": acc[15][0] / max(acc[15][1], 1),
            "blocks": acc[7][1]}


class AdamHalf:
    """AdamW with both moments stored in bf16."""
    def __init__(self, groups, lr, betas=(0.9, 0.99), eps=1e-8):
        self.param_groups = [dict(g, lr=lr) for g in groups]
        self.b1, self.b2, self.eps, self.t = betas[0], betas[1], eps, 0
        self.m = {p: torch.zeros_like(p, dtype=torch.bfloat16) for g in groups for p in g["params"]}
        self.v = {p: torch.zeros_like(p, dtype=torch.bfloat16) for g in groups for p in g["params"]}

    @torch.no_grad()
    def step(self, grads, max_norm=1.0):
        """grads: parameter -> gradient tensor (any float dtype); clipped to max_norm globally"""
        self.t += 1
        c = math.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        tot = math.sqrt(sum(float(g.float().pow(2).sum()) for g in grads.values()))
        sc = min(1.0, max_norm / (tot + 1e-6)) if math.isfinite(tot) else 0.0
        for g in self.param_groups:
            for p in g["params"]:
                if p not in grads:
                    continue
                gr = grads.pop(p).to(torch.bfloat16) * sc
                m, v = self.m[p], self.v[p]
                m.mul_(self.b1).add_(gr, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(gr, gr, value=1 - self.b2)
                del gr
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.addcdiv_(m.float(), v.float().sqrt_().add_(self.eps), value=-g["lr"] * c)


def build(a):
    model = DSpark(n_layers=5, tern=False, yarn=not a.no_yarn, markov_row0=True)
    if a.stock:
        model.load_stock(a.stock)
    if getattr(a, "init", None) or getattr(a, "ckpt", None):
        ck = torch.load(a.init if getattr(a, "init", None) else a.ckpt, map_location="cpu")
        if ck["n_layers"] > model.n_layers:
            model.grow(ck["n_layers"])
        model.tern = ck.get("tern", False)
        model.load_state_dict(ck["sd"])
    n_layers = getattr(a, "layers", 0) or model.n_layers
    if getattr(a, "tern", False):
        model.tern = True
    if n_layers > model.n_layers:
        model.grow(n_layers)
    model = model.cuda()
    emb, model.out_w = load_trunk_io(a.trunk)
    model.tok_embd = emb.cpu()
    del emb
    tio = getattr(a, "tern_io", "") or ""
    if tio in ("out", "both"):   # ternary (TQ1_0) output head; fixed, the layers train against it
        w = model.out_w.cpu(); model.out_w = None; torch.cuda.empty_cache(); model.out_w = tq1_0(w.float()).to(w.dtype).cuda()
    if tio in ("emb", "both"):   # ternary token embeddings
        model.tok_embd = tq1_0(model.tok_embd.float()).to(model.tok_embd.dtype)
    torch.cuda.empty_cache()
    return model


def cmd_eval(a):
    model = build(a)
    dump = Dump(a.dump)
    seqs = [e for e in dump.seqs if is_held_out(e)] if a.held_only else dump.seqs
    seqs = seqs[:a.max_seqs] if a.max_seqs else seqs
    r = evaluate(model, dump, seqs, B=a.block)
    print(json.dumps(r))


def cmd_train(a):
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    model = build(a)
    dump = Dump(a.dump)
    held = [e for e in dump.seqs if is_held_out(e)][:a.eval_seqs]
    train = [e for e in dump.seqs if not is_held_out(e)]
    print(f"{len(train)} train / {len(held)} eval sequences; layers {model.n_layers}, ternary {model.tern}, "
          f"params {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B", flush=True)
    frozen = ("confidence_head",)
    mats = [p for k, p in model.w.items() if p.dim() == 2 and not k.startswith(frozen)]
    norms = [p for k, p in model.w.items() if p.dim() == 1 and not k.startswith(frozen)]
    for k, p in model.w.items():
        if k.startswith(frozen):
            p.requires_grad_(False)
    opt = AdamHalf([{"params": mats, "weight_decay": a.wd}, {"params": norms, "weight_decay": 0.0}], a.lr)
    gen = torch.Generator().manual_seed(0)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    r = evaluate(model, dump, held, B=a.block)
    print("step 0", json.dumps(r), flush=True); log.write(json.dumps({"step": 0, **r}) + "\n"); log.flush()
    best, t0, run = r["accept7"], time.time(), [0.0, 0.0, 0]
    deadline = t0 + a.hours * 3600 if a.hours else None
    step = 0
    while step < a.steps and not (deadline and time.time() > deadline):
        e = train[int(torch.randint(len(train), (1,), generator=gen))]
        tok, aux, ti, tp, first = dump.get(e, a.max_len)
        an = pick_anchors(tok.numel(), first, a.anchors, gen)
        if an is None:
            continue
        step += 1
        frac = step / a.steps if not deadline else (time.time() - t0) / (a.hours * 3600)
        lr = a.lr * min(1.0, step / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))
        for g in opt.param_groups:
            g["lr"] = lr
        model.prepare(grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, hit, valid = block_stats(model, tok, aux, ti, tp, first, an, a.block, decay=a.decay)
        grads = {model.w[model.key(n)]: q.grad for n, q in model._q.items() if q.grad is not None}
        for p in norms:
            if p.grad is not None:
                grads[p] = p.grad
                p.grad = None
        model._q = {}
        opt.step(grads)
        del grads
        run[0] += float(loss); run[1] += float(hit[:, 0].float().mean()); run[2] += 1
        if step % a.log_every == 0:
            print(f"step {step} loss {run[0] / run[2]:.4f} row0 {run[1] / run[2]:.3f} lr {lr:.2e} {(time.time() - t0) / step:.2f}s/step", flush=True)
            run = [0.0, 0.0, 0]
        if step % a.eval_every == 0:
            r = evaluate(model, dump, held, B=a.block)
            print(f"step {step}", json.dumps(r), flush=True); log.write(json.dumps({"step": step, **r}) + "\n"); log.flush()
            sd = {"sd": model.state_dict(), "n_layers": model.n_layers, "tern": model.tern, "step": step, "eval": r}
            torch.save(sd, os.path.join(a.out, "last.pt"))
            if r["accept7"] >= best:
                best = r["accept7"]
                os.replace(os.path.join(a.out, "last.pt"), os.path.join(a.out, "best.pt"))
    r = evaluate(model, dump, held, B=a.block)
    print(f"final step {step}", json.dumps(r), flush=True); log.write(json.dumps({"step": step, "final": True, **r}) + "\n")
    torch.save({"sd": model.state_dict(), "n_layers": model.n_layers, "tern": model.tern, "step": step, "eval": r}, os.path.join(a.out, "last.pt"))
    if r["accept7"] >= best:
        os.replace(os.path.join(a.out, "last.pt"), os.path.join(a.out, "best.pt"))
    print("TRAIN_DONE", flush=True)


def cmd_export(a):
    import gguf
    ck = torch.load(a.ckpt, map_location="cpu")
    model = DSpark(n_layers=5)
    if ck["n_layers"] > 5:
        model.grow(ck["n_layers"])
    model.tern = ck.get("tern", False)
    model.load_state_dict(ck["sd"])
    reader = gguf.GGUFReader(a.stock_gguf)
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(a.out, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name == f"{arch}.block_count":
            val = model.n_layers
        if field.name == f"{arch}.block_size" and a.block:
            val = a.block
        writer.add_key_value(field.name, val, vt, sub_type=st)
    QT = gguf.GGMLQuantizationType
    name_of = {"fc.weight": "fc.weight", "hidden_norm.weight": "enc.output_norm.weight", "norm.weight": "output_norm.weight",
               "markov_head.markov_w1.weight": "markov_w1.weight", "markov_head.markov_w2.weight": "markov_w2.weight",
               "confidence_head.proj.weight": "conf_proj.weight", "confidence_head.proj.bias": "conf_proj.bias"}
    for i in range(model.n_layers):
        for k, g in GG.items():
            name_of[f"layers.{i}.{k}.weight"] = f"blk.{i}.{g}.weight"
    n_t = n_f = 0
    with torch.no_grad():
        for name, gname in name_of.items():
            p = model.w[model.key(name)].detach()
            if p.dim() == 1:
                writer.add_tensor(gname, p.float().numpy(), raw_dtype=QT.F32)
            elif model.is_tern(name):
                q = tq1_0(p.cuda().float())
                # one scale per block as the engine stores it (fp16), so re-quantising is exact
                d16 = q.abs().view(q.shape[0], -1, 256).amax(-1).half().float().repeat_interleave(256, 1)
                q = (q.sign() * d16).cpu().numpy().astype(np.float32)
                data = gguf.quants.quantize(q, QT.TQ1_0)
                writer.add_tensor(gname, data, raw_dtype=QT.TQ1_0); n_t += p.numel()
            else:
                writer.add_tensor(gname, p.to(torch.float16).numpy(), raw_dtype=QT.F16); n_f += p.numel()
    if a.io_from:
        # the drafter's own token embeddings and output head (the ternary trunk's are in a different basis)
        tern_names = {"out": ["output.weight"], "emb": ["token_embd.weight"], "both": ["output.weight", "token_embd.weight"]}.get(a.tern_io, [])
        if tern_names:
            emb, out_w = load_trunk_io(a.io_from, "cpu")
            for nm, w in (("token_embd.weight", emb), ("output.weight", out_w)):
                if nm in tern_names:
                    q = tq1_0(w.float())
                    d16 = q.abs().view(q.shape[0], -1, 256).amax(-1).half().float().repeat_interleave(256, 1)
                    q = (q.sign() * d16).numpy().astype(np.float32)
                    data = gguf.quants.quantize(q, QT.TQ1_0)
                    writer.add_tensor(nm, data, raw_dtype=QT.TQ1_0)
                    print(f"ternarised {nm} ({data.nbytes / 2**20:.0f} MiB)")
            del emb, out_w
        for t in gguf.GGUFReader(a.io_from).tensors:
            if t.name in ("token_embd.weight", "output.weight") and t.name not in tern_names:
                writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
                print(f"copied {t.name} ({t.tensor_type.name}, {t.data.nbytes / 2**20:.0f} MiB) from {a.io_from}")
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    print(f"wrote {a.out}: {n_t / 1e6:.0f}M ternary weights, {n_f / 1e6:.0f}M float weights, {os.path.getsize(a.out) / 2**20:.0f} MiB")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "train", "export"):
        p = sub.add_parser(name)
        p.add_argument("--stock", default="ttf-home/models/drafts/dspark-27b/hf/model.safetensors")
        p.add_argument("--trunk", default="out/trunk-tq1p.gguf")
        p.add_argument("--ckpt"); p.add_argument("--dump"); p.add_argument("--no-yarn", action="store_true")
        p.add_argument("--block", type=int, default=16)
    p = sub.choices["eval"]
    p.add_argument("--held-only", action="store_true"); p.add_argument("--max-seqs", type=int, default=0); p.add_argument("--tern-io", default="")
    p = sub.choices["train"]
    p.add_argument("--out", required=True); p.add_argument("--init"); p.add_argument("--tern", action="store_true"); p.add_argument("--tern-io", default="")
    p.add_argument("--layers", type=int, default=0); p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--hours", type=float, default=0.0); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0); p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--anchors", type=int, default=96); p.add_argument("--max-len", type=int, default=3072)
    p.add_argument("--decay", type=float, default=0.85); p.add_argument("--eval-seqs", type=int, default=60)
    p.add_argument("--eval-every", type=int, default=500); p.add_argument("--log-every", type=int, default=50)
    p = sub.choices["export"]
    p.add_argument("--stock-gguf", default="ttf-home/models/drafts/dspark-27b/gguf/Qwen3.8-27B-DSpark-bf16.gguf")
    p.add_argument("--out", required=True); p.add_argument("--io-from"); p.add_argument("--tern-io", default="")
    a = ap.parse_args()
    {"eval": cmd_eval, "train": cmd_train, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
