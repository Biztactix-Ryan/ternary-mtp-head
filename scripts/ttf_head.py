#!/usr/bin/env python3
"""Ternary (TTF format 1) version of the MTP head, trained quantisation-aware.

Every matrix of the head becomes the same kind of TTF linear the ternary trunk uses
(engine: llm_graph_context::build_lora_mm / build_ttf_mm):

    y = r * (T (c x + m)) + b                      main ternary weight, row scale r, column scale c
      + TU (uc * (TD (dc x + dm) + db))            ternary low-rank adapter, rank R (trunk: 1024, 256 for k/v)
      + FU (FD x)                                  float (F16) low-rank adapter, rank 16

T, TD, TU are {-1, 0, +1}. In training they are the thresholded sign of latent float weights with a
straight-through gradient. The trits and vectors export to blk.N.<w>.weight / .ttf_* tensors.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf

from mtp_head import MATS, NORMS, MTPHead

QT = gguf.GGMLQuantizationType
THR = 0.7          # ternary threshold as a fraction of mean |w| (TWN)
F_RANK = 16


def tern(w, dim):
    """{-1, 0, 1} by thresholding at THR * mean|w| along `dim` (kept per row or per column)."""
    t = THR * w.abs().mean(dim=dim, keepdim=True)
    return torch.sign(w) * (w.abs() > t)


def ste(w, dim):
    return w + (tern(w, dim) - w).detach()


def scale_of(w, t, dim):
    """least-squares scale s for w ~ s * t along dim"""
    return (w * t).sum(dim) / t.abs().sum(dim).clamp(min=1)


class TTFLinear(nn.Module):
    def __init__(self, w, rank, silent=False):
        """w: trained float weight [N, K]; initialises the trits from it and the adapter from the residual.
        silent: start as a zero map that can still learn (for the output matrix of a newly added block):
        the trits are kept but the row scale and the adapter scale start at zero, as LoRA does."""
        super().__init__()
        w = w.detach().float()
        N, K = w.shape
        self.N, self.K, self.R = N, K, rank
        T = tern(w, 1)
        r = scale_of(w, T, 1)
        self.W = nn.Parameter(w.clone())
        self.r = nn.Parameter(torch.zeros_like(r) if silent else r.clone())
        self.c = nn.Parameter(torch.ones(K, device=w.device)); self.m = nn.Parameter(torch.zeros(K, device=w.device))
        self.b = nn.Parameter(torch.zeros(N, device=w.device))
        # ternary low-rank adapter from the SVD of what ternarising lost
        E = w - r[:, None] * T
        q = min(rank, N, K)
        if q == min(N, K):
            U, S, Vh = torch.linalg.svd(E, full_matrices=False)
            V = Vh.T
        else:
            U, S, V = torch.svd_lowrank(E, q=q, niter=2)
        A, B = U * S.sqrt(), (V * S.sqrt()).T                      # [N, q], [q, K]
        TA, TB = tern(A, 0), tern(B, 1)
        uc = scale_of(A, TA, 0) * scale_of(B, TB, 1) * (0.0 if silent else 1.0)
        if rank > q:                                               # wider than the matrix: extra components start silent
            extra = rank - q
            A = torch.cat([A, torch.randn(N, extra, device=w.device) * A.std()], 1)
            B = torch.cat([B, torch.randn(extra, K, device=w.device) * B.std()], 0)
            uc = torch.cat([uc, torch.zeros(extra, device=w.device)])
        del U, S, V, E
        self.U = nn.Parameter(A.contiguous()); self.D = nn.Parameter(B.contiguous())
        self.uc = nn.Parameter(uc)
        self._q = None
        self.dc = nn.Parameter(torch.ones(K, device=w.device)); self.dm = nn.Parameter(torch.zeros(K, device=w.device))
        self.db = nn.Parameter(torch.zeros(rank, device=w.device))
        # float low-rank adapter, starts at zero
        self.FD = nn.Parameter(torch.randn(F_RANK, K, device=w.device) / K ** 0.5)
        self.FU = nn.Parameter(torch.zeros(N, F_RANK, device=w.device))

    def quantised(self, dt):
        return ste(self.W, 1).to(dt), ste(self.D, 1).to(dt), ste(self.U, 0).to(dt)

    def forward(self, x):
        dt = x.dtype
        # the trits are computed once per training step (prepare) and shared by every pass of that step
        Wq, Dq, Uq = self._q if self._q is not None and self._q[0].dtype == dt else self.quantised(dt)
        y = F.linear(x * self.c.to(dt) + self.m.to(dt), Wq) * self.r.to(dt) + self.b.to(dt)
        h = F.linear(x * self.dc.to(dt) + self.dm.to(dt), Dq) + self.db.to(dt)
        y = y + F.linear(h * self.uc.to(dt), Uq)
        return y + F.linear(F.linear(x, self.FD.to(dt)), self.FU.to(dt))

    def n_trits(self):
        return self.W.numel() + self.U.numel() + self.D.numel()


class TTFHead(MTPHead):
    def __init__(self):
        super().__init__()
        self.mats = nn.ParameterDict()
        self.ttf = nn.ModuleDict()
        self.n_blocks, self.ffn_mult = 1, 1

    LAYER_MATS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_up", "ffn_gate", "ffn_down")
    LAYER_NORMS = ("attn_norm", "post_attention_norm", "attn_q_norm", "attn_k_norm")

    @classmethod
    def from_float(cls, fm, rank=1024, rank_kv=256, ffn_mult=1, blocks=1):
        """ffn_mult > 1: widen the FFN to ffn_mult x 17408 hidden units (new units start silent).
        blocks > 1: stack extra attention + FFN blocks after the first (copies of it whose outputs start silent).
        Both start as the same function as the float head, apart from ternarisation. Training-only for now:
        the engine's MTP graph takes one block with the trunk's FFN width."""
        m = cls()
        m.n_blocks, m.ffn_mult = blocks, ffn_mult
        w = {k.replace(".", "_"): fm.mats[k.replace(".", "_")].data.float() for k in MATS}
        rk = lambda k: rank_kv if k in ("attn_k", "attn_v") else rank
        if ffn_mult > 1:
            F0 = w["ffn_up"].shape[0]; extra = F0 * (ffn_mult - 1)
            g = torch.Generator(device="cpu").manual_seed(0)
            for k in ("ffn_up", "ffn_gate"):        # new hidden units: random rows at the existing scale
                new = torch.randn(extra, w[k].shape[1], generator=g).to(w[k].device) * w[k].std()
                w[k] = torch.cat([w[k], new], 0)
            # their output columns start just under the ternary threshold: zero trits, one step from flipping
            d = w["ffn_down"]
            sub = torch.sign(torch.randn(d.shape[0], extra, generator=g)).to(d.device) * 0.5 * THR * d.abs().mean(1, keepdim=True)
            w["ffn_down"] = torch.cat([d, sub], 1)
        for key, v in w.items():
            m.ttf[key] = TTFLinear(v, rk(key))
        for k in NORMS:
            key = k.replace(".", "_")
            m.norms[key] = nn.Parameter(fm.norms[key].data.clone())
        for b in range(1, blocks):
            for k in cls.LAYER_MATS:
                m.ttf[f"{k}_b{b}"] = TTFLinear(w[k], rk(k), silent=k in ("attn_output", "ffn_down"))
            for k in cls.LAYER_NORMS:
                m.norms[f"{k}_b{b}"] = nn.Parameter(fm.norms[k].data.clone())
        m.tok_embd, m.out_w = fm.tok_embd, fm.out_w
        return m

    def layer_suffixes(self):
        return [""] + [f"_b{b}" for b in range(1, self.n_blocks)]

    def lin(self, name, x):
        return self.ttf[name](x)

    def prepare(self, dt=torch.bfloat16):
        for l in self.ttf.values():
            l._q = l.quantised(dt)

    def release(self):
        for l in self.ttf.values():
            l._q = None


def enc_tq1p(T):
    """int trits [rows, K] (K % 160 == 0) -> TQ1_P bytes [rows, K / 160 * 32] (as ttfhome/q38_export.py)"""
    rows, K = T.shape
    x = T.reshape(rows, K // 160, 5, 32).to(torch.int16) + 1
    q = x[..., 0, :] * 81 + x[..., 1, :] * 27 + x[..., 2, :] * 9 + x[..., 3, :] * 3 + x[..., 4, :]
    q = (q.to(torch.int32) * 256 + 242) // 243
    return q.to(torch.uint8).reshape(rows, K // 160 * 32).numpy()


def enc(T, scale=None):
    """trits [rows, K] -> (bytes, type, scale_baked). TQ1_P when K % 160 == 0, else TQ1_0 whose per-block
    scale carries the row scale when one is given."""
    K = T.shape[1]
    if K % 160 == 0:
        return enc_tq1p(T), QT.TQ1_P, False
    assert K % 256 == 0, K
    w = T.float() if scale is None else T.float() * scale[:, None]
    return gguf.quants.quantize(w.numpy(), QT.TQ1_0), QT.TQ1_0, scale is not None


def ttf_tensors(ckpt, pfx):
    """checkpoint (train_mtp.py --ttf) -> [(gguf name, array, type or None)] for build_infile.py"""
    sd = ckpt["model"]
    out = []
    f32 = lambda v: np.ascontiguousarray(v.float().numpy().astype(np.float32))
    for k in MATS:
        g = lambda p: sd[f"ttf.{k.replace('.', '_')}.{p}"].float()
        stem = pfx + k
        T = tern(g("W"), 1)
        data, qt, baked = enc(T, g("r"))
        out.append((stem + ".weight", data, qt))
        if not baked:
            out.append((stem + ".ttf_r", f32(g("r")), None))
        out += [(stem + ".ttf_c", f32(g("c")), None), (stem + ".ttf_m", f32(g("m")), None), (stem + ".ttf_b", f32(g("b")), None)]
        data, qt, _ = enc(tern(g("D"), 1))
        out += [(stem + ".ttf_d", data, qt), (stem + ".ttf_d_c", f32(g("dc")), None), (stem + ".ttf_d_m", f32(g("dm")), None),
                (stem + ".ttf_d_b", f32(g("db")), None)]
        data, qt, _ = enc(tern(g("U"), 0))
        out += [(stem + ".ttf_u", data, qt), (stem + ".ttf_u_c", f32(g("uc")), None)]
        out += [(stem + ".ttf_fd", g("FD").numpy().astype(np.float16), None), (stem + ".ttf_fu", g("FU").numpy().astype(np.float16), None)]
    for k in NORMS:
        out.append((pfx + k + ".weight", f32(sd["norms." + k.replace(".", "_")]), None))
    return out
