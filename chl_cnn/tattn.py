"""Temporal attention stem (train_pinn.py --model temporal3d --t3d-stem attn; io2 track item 7, 2026-10-21).

Replaces TemporalStem3D's Conv3d stack and learned (T,1,1) day collapse with per-pixel attention over the days of the window:
  1. per-day embedding: Conv2d(c_day, w, 3) -> GroupNorm -> GELU -> Conv2d(w, w, 3), shared across days, applied to each day's
     [features, CHL value (0 where not visible), visible mask];
  2. + a learned day-offset embedding (one vector per window position);
  3. `layers` x pre-norm self-attention + MLP blocks over the T day tokens of each pixel; missing days (not in the store) are masked
     out of the keys;
  4. attention pooling: one query per pixel = learned vector + day t's token (+ a linear map of the --memory channels with
     --attn-query-memory), keys / values = the day tokens -> (B, w, H, W).
The static 2D channels are concatenated after the stem and fed to the usual backbone, as in Temporal3DNet.
Attention is computed explicitly (einsum + masked softmax) over B*H*W sequences of length T.

Eval-time hooks (io3 track, 2026-10-23; analyses/io3_attn/attn_checks.py). Module-level EVAL_HOOKS, all off by default (the default
path is unchanged, bit for bit):
  pool_max_offset K   pooling softmax only over window positions with |offset from day t| <= K (others masked, renormalised);
  bypass_self         skip the day-to-day self-attention block(s) (identity), pooling on the embedded tokens;
  recorder f          f(a, x, t_index) after the pooling softmax: a (B*H*W, heads, 1, T) pooling weights, x the stem input.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .models import build_model

EVAL_HOOKS = dict(pool_max_offset=None, bypass_self=False, recorder=None)


class _Block(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(w), nn.LayerNorm(w)
        self.qkv, self.proj = nn.Linear(w, 3 * w), nn.Linear(w, w)
        self.ff = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Linear(2 * w, w))

    def forward(self, x, neg):                                 # x (N, T, w); neg (N, 1, 1, T) additive mask (0 / -inf)
        N, T, w = x.shape
        q, k, v = self.qkv(self.n1(x)).view(N, T, 3, self.h, w // self.h).permute(2, 0, 3, 1, 4)   # (N, h, T, d) each
        a = torch.softmax(torch.einsum("nhtd,nhsd->nhts", q, k) / math.sqrt(w // self.h) + neg, -1)
        x = x + self.proj(torch.einsum("nhts,nhsd->nhtd", a, v).transpose(1, 2).reshape(N, T, w))
        return x + self.ff(self.n2(x))


class TemporalAttnStem(nn.Module):
    def __init__(self, c_day, T, t_index, width=32, heads=4, layers=1, q_extra=0):
        super().__init__()
        assert width % heads == 0
        self.T, self.t_index, self.w, self.h = T, t_index, width, heads
        groups = 8 if width % 8 == 0 else 1
        self.embed = nn.Sequential(nn.Conv2d(c_day, width, 3, padding=1), nn.GroupNorm(groups, width), nn.GELU(),
                                   nn.Conv2d(width, width, 3, padding=1))
        self.day_emb = nn.Parameter(0.02 * torch.randn(T, width))
        self.blocks = nn.ModuleList([_Block(width, heads) for _ in range(layers)])
        self.q0 = nn.Parameter(torch.zeros(width))
        self.q_extra = nn.Linear(q_extra, width) if q_extra else None
        self.nq, self.nk = nn.LayerNorm(width), nn.LayerNorm(width)
        self.wq, self.wk, self.wv = nn.Linear(width, width), nn.Linear(width, width), nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, x, day_ok, q_extra=None):
        """x (B, C, T, H, W); day_ok (B, T) bool (day in the store); q_extra (B, Cq, H, W) or None -> (B, w, H, W)."""
        B, C, T, H, W = x.shape
        assert T == self.T, f"expected {self.T} days, got {T}"
        e = self.embed(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))                    # (B*T, w, H, W)
        tok = e.view(B, T, self.w, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, T, self.w) + self.day_emb.to(e.dtype)
        neg = torch.zeros((B, T), device=x.device, dtype=tok.dtype).masked_fill(~day_ok, float("-inf"))
        neg = neg.view(B, 1, 1, 1, T).expand(B, H, W, 1, T).reshape(B * H * W, 1, 1, T)
        if not EVAL_HOOKS["bypass_self"]:
            for blk in self.blocks:
                tok = blk(tok, neg)
        q = self.q0.to(tok.dtype) + tok[:, self.t_index]
        if self.q_extra is not None:
            q = q + self.q_extra(q_extra.permute(0, 2, 3, 1).reshape(B * H * W, -1).to(tok.dtype))
        d = self.w // self.h
        qh = self.wq(self.nq(q)).view(-1, self.h, 1, d)
        kv = self.nk(tok)
        kh = self.wk(kv).view(-1, T, self.h, d).transpose(1, 2); vh = self.wv(kv).view(-1, T, self.h, d).transpose(1, 2)
        negp = neg
        if EVAL_HOOKS["pool_max_offset"] is not None:
            far = torch.tensor([abs(i - self.t_index) > EVAL_HOOKS["pool_max_offset"] for i in range(T)], device=x.device)
            negp = neg.masked_fill(far.view(1, 1, 1, T), float("-inf"))
        a = torch.softmax(torch.einsum("nhqd,nhsd->nhqs", qh, kh) / math.sqrt(d) + negp, -1)
        if EVAL_HOOKS["recorder"] is not None:
            EVAL_HOOKS["recorder"](a, x, self.t_index)
        o = self.out(torch.einsum("nhqs,nhsd->nhqd", a, vh).reshape(-1, self.w))            # (B*H*W, w)
        return o.view(B, H, W, self.w).permute(0, 3, 1, 2).contiguous()


class TemporalAttnNet(nn.Module):
    """TemporalAttnStem + static channels + 2D backbone. forward(x3d, x_static, day_ok, q_extra=None)."""

    def __init__(self, c_day, n_static, T, t_index, backbone="dilated", stem_width=32, heads=4, layers=1, q_extra=0, **backbone_kw):
        super().__init__()
        self.stem = TemporalAttnStem(c_day, T, t_index, width=stem_width, heads=heads, layers=layers, q_extra=q_extra)
        self.backbone = build_model(backbone, stem_width + n_static, **backbone_kw)
        self.out_channels = self.backbone.out_channels

    def forward(self, x3d, x_static=None, day_ok=None, q_extra=None):
        h = self.stem(x3d, day_ok, q_extra)
        if x_static is not None:
            h = torch.cat([h, x_static.to(h.dtype)], 1)
        return self.backbone(h)
