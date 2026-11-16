#!/usr/bin/env python3
"""Fine-tune the MTP head on dumped trunk hidden states (runbook section 9).

  train_mtp.py --mtp stock.gguf --dump DIR[,DIR2] --out OUTDIR [--steps 250] [--lr 1e-4]
               [--emb-from TRUNK.gguf] [--init ckpt.pt] [--ttf --rank 4096 --rank-kv 1024]
               [--unroll 3] [--prompt-weight 0.2] [--only input]

Each step accumulates gradients over --seqs sequences (cropped to --seq-len), one at a time.
Loss: cross-entropy of the head's full-vocabulary distribution against the trunk's renormalised
top-32, weight 1.0 on generated positions and --prompt-weight on prompt positions.
--unroll 2|3 adds draft steps 2 and 3 computed on the head's own outputs (stage 2), weighted 0.7 and 0.5.
"""
import argparse, json, math, os, random, time
import torch
from mtp_head import Dump, evaluate, is_held_out, load_stock

ap = argparse.ArgumentParser()
ap.add_argument("--mtp", required=True); ap.add_argument("--dump", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=250); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=50); ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--seqs", type=int, default=8); ap.add_argument("--seq-len", type=int, default=2048)
ap.add_argument("--eval-every", type=int, default=25); ap.add_argument("--eval-seqs", type=int, default=60)
ap.add_argument("--prompt-weight", type=float, default=0.2); ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--only", choices=["all", "input"], default="all",
                help="'input' trains only eh_proj, enorm and hnorm (cheap input re-alignment)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--emb-from", help="trunk GGUF whose token_embd / output.weight the head uses (in-file head)")
ap.add_argument("--init", help="checkpoint to start from instead of the stock weights (float head)")
ap.add_argument("--init-ttf", help="ternary checkpoint to continue from (needs --ttf and the same ranks)")
ap.add_argument("--ttf", action="store_true", help="convert to a ternary (TTF) head after loading and train it quantisation-aware")
ap.add_argument("--rank", type=int, default=1024, help="ternary adapter rank (multiple of 256)")
ap.add_argument("--rank-kv", type=int, default=256)
ap.add_argument("--ffn-mult", type=int, default=1, help="ternary head only: widen the FFN by this factor")
ap.add_argument("--blocks", type=int, default=1, help="ternary head only: number of stacked attention + FFN blocks")
ap.add_argument("--train-frac", type=float, default=1.0, help="use only this fraction of the training conversations")
ap.add_argument("--gen-weight", type=float, default=1.0, help="loss weight on generated positions (0 = train on history only)")
ap.add_argument("--unroll", type=int, default=1, help="draft steps trained (1 = first draft only)")
ap.add_argument("--unroll-weights", default="1,0.7,0.5")
ap.add_argument("--half-state", action="store_true", help="keep the optimiser moments in bf16 (12 instead of 16 bytes per parameter)")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
random.seed(a.seed); torch.manual_seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True
log = open(os.path.join(a.out, "train.log"), "a")
def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()

dump = Dump(a.dump)
train = [e for e in dump.seqs if not is_held_out(e)]
if a.train_frac < 1.0:                               # data-scaling ablation: a stable subset of conversations
    import hashlib
    train = [e for e in train if int(hashlib.sha1(e["conv"].encode()).hexdigest(), 16) % 1000 < a.train_frac * 1000]
val = [e for e in dump.seqs if is_held_out(e)]
random.Random(1).shuffle(val)
val = val[:a.eval_seqs]
say(f"args {vars(a)}")
say(f"sequences: train {len(train)} ({sum(e['length'] for e in train)} positions, "
    f"{sum(e['length'] - e['n_prompt'] for e in train)} generated), held-out {len(val)} "
    f"({len({e['conv'] for e in val})} conversations)")

model = load_stock(a.mtp, emb_from=a.emb_from)
if a.init:
    model.load_state_dict(torch.load(a.init)["model"], strict=False)
if a.only == "input":
    for k, p in list(model.mats.items()) + list(model.norms.items()):
        p.requires_grad_(k in ("nextn_eh_proj", "nextn_enorm", "nextn_hnorm"))
if a.ttf:
    from ttf_head import TTFHead
    r0 = evaluate(model, dump, val)["gen"]["top1"]
    model = TTFHead.from_float(model, a.rank, a.rank_kv, a.ffn_mult, a.blocks)
    if a.init_ttf:
        model.load_state_dict(torch.load(a.init_ttf)["model"], strict=False)
    torch.cuda.empty_cache()
    trits = sum(l.n_trits() for l in model.ttf.values())
    say(f"ternary head: float init agreement {100 * r0:.1f}%, {trits / 1e6:.0f}M trits "
        f"(main {sum(l.W.numel() for l in model.ttf.values()) / 1e6:.0f}M + adapters rank {a.rank}/{a.rank_kv}), "
        f"about {trits * 1.6 / 8 / 2**20:.0f} MiB at 1.6 bpw")
    a.wd = 0.0                                   # decay would pull latent weights to zero trits
mats = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
norms = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
say(f"trainable parameters: {sum(p.numel() for p in mats + norms) / 1e6:.1f}M")


class AdamHalf:
    """AdamW with both moments stored in bf16."""
    def __init__(self, groups, lr, betas=(0.9, 0.99), eps=1e-8):
        self.param_groups = [dict(g, lr=lr) for g in groups]
        self.b1, self.b2, self.eps, self.t = betas[0], betas[1], eps, 0
        self.m = {p: torch.zeros_like(p, dtype=torch.bfloat16) for g in groups for p in g["params"]}
        self.v = {p: torch.zeros_like(p, dtype=torch.bfloat16) for g in groups for p in g["params"]}

    def zero_grad(self, set_to_none=True):
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        c = math.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                gr = p.grad.to(torch.bfloat16)
                m, v = self.m[p], self.v[p]
                m.mul_(self.b1).add_(gr, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(gr, gr, value=1 - self.b2)
                del gr
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.addcdiv_(m.float(), v.float().sqrt_().add_(self.eps), value=-g["lr"] * c)


groups = [{"params": mats, "weight_decay": a.wd}, {"params": norms, "weight_decay": 0.0}]
opt = AdamHalf(groups, a.lr) if a.half_state else torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.99))
lr_at = lambda s: a.lr * (s + 1) / a.warmup if s < a.warmup else \
    a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup))))
UW = [float(x) for x in a.unroll_weights.split(",")][:a.unroll]


def run_eval(step):
    r = evaluate(model, dump, val, steps=a.unroll)
    extra = "".join(f" | draft {s}: {100 * r[f'gen{s}']['top1']:.1f}%" for s in range(2, a.unroll + 1))
    say(f"eval step {step}: generated positions n {r['gen']['n']} top-1 agreement {100 * r['gen']['top1']:.1f}% "
        f"trunk-prob {r['gen']['trunk_p']:.3f} | all positions {100 * r['all']['top1']:.1f}% {r['all']['trunk_p']:.3f}{extra}")
    model.train()
    return r["gen"]["top1"] if a.unroll == 1 else sum(r["gen" if s == 1 else f"gen{s}"]["top1"] for s in range(1, a.unroll + 1)) / a.unroll


def seq_loss_backward(e, denom):
    """Forward one sequence, backward the weighted loss / denom. Returns (weighted loss sum of draft 1, weight sum)."""
    L = e["length"]
    if L > a.seq_len and random.random() < 0.3:      # sometimes a random window instead of the tail
        off = random.randint(0, L - a.seq_len)
        e = dict(e, start=e["start"] + off, length=a.seq_len, n_prompt=max(0, e["n_prompt"] - off))
    tok, h, ti, tp, gen = dump.get(e, a.seq_len)
    model.prepare()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gs = model.unroll(tok[None], h[None].to(torch.bfloat16), a.unroll) if a.unroll > 1 \
            else [model(tok[None], h[None].to(torch.bfloat16))]
    w = torch.where(gen, a.gen_weight, a.prompt_weight)
    target = torch.softmax(tp, dim=-1)                                # renormalised trunk top-32
    first, grads = 0.0, []
    for s, g in enumerate(gs):                                        # draft step s + 1 is defined from position s + 1
        g = g[0]
        ws = w.clone(); ws[:s + 1] = 0.0
        gd = g.detach().requires_grad_(True)
        for i in range(0, len(tok), a.chunk):                         # full-vocab logits in chunks of positions
            lp = torch.log_softmax(model.logits(gd[i:i + a.chunk]).float(), dim=-1)
            ce = -(target[i:i + a.chunk] * lp.gather(1, ti[i:i + a.chunk])).sum(-1)
            loss = (ce * ws[i:i + a.chunk]).sum()
            (loss * UW[s] / denom).backward()
            if s == 0:
                first += float(loss)
        grads.append(gd.grad[None])
    torch.autograd.backward(gs, grads)
    model.release()
    wf = w.clone(); wf[0] = 0.0
    return first, float(wf.sum())


best = run_eval(0)
say(f"step 0 held-out score {100 * best:.1f}%")
torch.save({"model": model.state_dict(), "step": 0, "score": best, "args": vars(a)}, os.path.join(a.out, "best.pt"))
t0 = time.time()
model.train()
for step in range(a.steps):
    for gr in opt.param_groups:
        gr["lr"] = lr_at(step)
    batch = random.sample(train, a.seqs)
    # normalise by the expected weight so the loss scale does not depend on the sample
    denom = sum(min(e["length"], a.seq_len) for e in batch) * 0.5
    opt.zero_grad(set_to_none=True)
    tot = wsum = 0.0
    for e in batch:
        l, w_ = seq_loss_backward(e, denom)
        tot += l; wsum += w_
    gn = torch.nn.utils.clip_grad_norm_(mats + norms, 1.0)
    opt.step()
    if step % 10 == 0 or step == a.steps - 1:
        say(f"step {step + 1}/{a.steps} loss {tot / max(wsum, 1):.4f} grad {float(gn):.2f} lr {lr_at(step):.2e} "
            f"{time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
    if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
        acc = run_eval(step + 1)
        if acc >= best:
            best = acc
            torch.save({"model": model.state_dict(), "step": step + 1, "score": acc, "args": vars(a)}, os.path.join(a.out, "best.pt"))
say(f"done. best held-out score {100 * best:.1f}%")
