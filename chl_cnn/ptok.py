"""Gridded patch-token stem (train_pinn.py --model temporal3d --t3d-stem patch; io3 track, 2026-10-23; spec: bridge Log #3/#4/#6).

Tokens: one per ocean patch (P x P px, --ptok-patch, default 4) per window day (t-K..t+K, 17 days at K = 8). A token is embedded from
mask-aware pooling over the patch: mean and std of the z-scored ln-CHL anomaly (CHL z minus that day's DOY climatology z) over the
patch's visible pixels, the visible fraction (visible / ocean px), and the ocean-pixel means of the 9 physics fields of that day;
a 2-layer MLP, + a learned day-offset embedding + a learned patch-position embedding. A fully cloudy patch gives a physics-only token
(visible fraction 0, CHL statistics 0). No full-resolution per-day encoder.

Attention: --ptok-layers pre-norm blocks of self-attention over the T day tokens of each ocean patch (batched over patches, fused
F.scaled_dot_product_attention; days missing from the store are masked out of the keys); no attention across patches.

Per-pixel day weights (--ptok-arm):
  interp  score(p, d) = q(patch, t) . k(patch, d) / sqrt(D) on the patch grid, bilinearly interpolated between ocean-patch centres
          (normalised: land-only patches do not contribute), + a pixel term: a 1x1-conv MLP on cheap per-pixel features of day d
          (visible on d, |d - t| / K, (d - t) / K, |SST_d - SST_t| in z units).
  pixq    each pixel's query (small conv on day t's per-pixel inputs) attends to the day tokens of its own patch and the 8 neighbouring
          patches (learned bias per relative patch position, 3 x 3); the score of day d is the log-sum-exp of its 9 patch scores.
  Either way: softmax over the context days d != t on which pixel p is visible, plus one learned "no day" logit.

Backbone input at full resolution: [blend = sum_d w_d(p) x anomaly_d(p), weight mass on real days (1 - w_none), weight entropy (over the
T - 1 + 1 slots)], day t's own per-day channels (features, visible CHL z, visible mask), the static channels; the dilated backbone's
first block input is FiLM-modulated by the day-t patch token (linear -> (gamma, beta), zero-initialised, bilinearly upsampled as above).
Fine detail enters only through observed pixels: the tokens are patch statistics.

Context-day values (--ptok-value; 2026-10-23, bridge UI #8 / user decision #9):
  own   (default, "arm A") value_d(p) = the pixel's own visible anomaly on day d; the softmax runs over days visible at p.
  lavg  ("arm B") value_d(p) = conv(m c) / conv(m), a normalised convolution of the visible (donor-masked) anomaly c with the visible
        mask m, fixed Gaussian kernel sigma 1.5 px truncated at radius 5 px (normalised to sum 1); the softmax runs over days with
        conv(m) > 0 at p; conv(m) (local confidence) and the own-pixel-visible flag enter the pixel term (interp arm: 5 features).

HOOKS["recorder"] (default None): called as f(dict) with tokens / score terms / weights of each forward (leak test, weight maps).
HOOKS["no_blend"] (default False; io4 track Job 1, 2026-10-23, eval-only): the backbone gets blend 0, mass 0 and entropy 0, the values
of all weight on the "no day" slot; FiLM and day t's channels unchanged; the recorder still sees the computed weights.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .models import build_model

HOOKS = dict(recorder=None, no_blend=False)
N_STAT = 3                                                     # anomaly mean, anomaly std, visible fraction


class _Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, keep):                                # x (N, T, d); keep (N, 1, 1, T) bool (True = attend)
        N, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(N, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        x = x + self.proj(o.transpose(1, 2).reshape(N, T, d))
        return x + self.ff(self.n2(x))


def _patch_sum(x, P):
    """x (..., Hp, Wp) -> (..., Hp/P, Wp/P) sums over P x P patches."""
    *lead, Hp, Wp = x.shape
    return x.reshape(*lead, Hp // P, P, Wp // P, P).sum((-3, -1))


class PatchTokenNet(nn.Module):
    """forward(x3d, x_static, day_ok, clim3d) -> backbone output (B, 3, H, W).
    x3d (B, c_day, T, H, W) per-day [features (n_feat), CHL z (0 where not visible), visible mask]; clim3d (B, T, H, W) DOY climatology
    z of each window day; day_ok (B, T) bool."""

    def __init__(self, c_day, n_static, T, t_index, ocean, P=4, dim=64, heads=4, layers=2, arm="interp", sst_index=0,
                 backbone="dilated", width=64, value="own", **backbone_kw):
        super().__init__()
        assert backbone == "dilated", "patch stem: FiLM is wired into the dilated backbone"
        assert arm in ("interp", "pixq") and value in ("own", "lavg")
        self.value = value
        if value == "lavg":
            r = torch.arange(-5, 6, dtype=torch.float32)
            d2 = r.view(-1, 1) ** 2 + r.view(1, -1) ** 2
            k = torch.exp(-d2 / (2 * 1.5 ** 2)) * (d2 <= 25)
            self.register_buffer("gk", (k / k.sum()).view(1, 1, 11, 11), persistent=False)
        self.c_day, self.T, self.ti, self.P, self.D, self.arm, self.sst = c_day, T, t_index, P, dim, arm, sst_index
        self.n_feat = c_day - 2
        H, W = ocean.shape
        self.H, self.W = H, W
        self.Hp, self.Wp = -(-H // P) * P, -(-W // P) * P
        oc = F.pad(ocean.float(), (0, self.Wp - W, 0, self.Hp - H))
        self.register_buffer("ocean_p", oc, persistent=False)                       # (Hp, Wp) padded ocean mask
        n_oc = _patch_sum(oc, P)
        self.register_buffer("n_oc", n_oc, persistent=False)                        # ocean px per patch
        pm = n_oc > 0
        self.register_buffer("patch_ok", pm, persistent=False)                      # ocean patches
        self.register_buffer("patch_idx", pm.flatten().nonzero().squeeze(1), persistent=False)
        self.nPy, self.nPx = n_oc.shape
        self.Np = int(pm.sum())
        self.embed = nn.Sequential(nn.Linear(N_STAT + self.n_feat, dim), nn.GELU(), nn.Linear(dim, dim))
        self.day_emb = nn.Parameter(0.02 * torch.randn(T, dim))
        self.pos_emb = nn.Parameter(0.02 * torch.randn(self.nPy * self.nPx, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(layers)])
        self.nt = nn.LayerNorm(dim)
        if arm == "interp":
            self.wq, self.wk = nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.n_pf = 5 if value == "lavg" else 4
            self.pix = nn.Sequential(nn.Conv2d(self.n_pf, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1))
        else:
            self.wk = nn.Linear(dim, dim)
            self.qconv = nn.Sequential(nn.Conv2d(c_day, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, dim, 1))
            self.nbias = nn.Parameter(torch.zeros(9))
        self.none_logit = nn.Parameter(torch.zeros(()))
        self.film = nn.Linear(dim, 2 * width)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.backbone = build_model(backbone, 3 + c_day + n_static, width=width, **backbone_kw)
        self.out_channels = self.backbone.out_channels

    # ---- helpers
    def _pad(self, x):
        return F.pad(x, (0, self.Wp - self.W, 0, self.Hp - self.H))

    def _to_grid(self, v):
        """v (B, Np, C) ocean-patch values -> (B, C, nPy, nPx) grid (0 on land patches)."""
        B, _, C = v.shape
        g = v.new_zeros((B, self.nPy * self.nPx, C))
        g[:, self.patch_idx] = v
        return g.transpose(1, 2).reshape(B, C, self.nPy, self.nPx)

    def _upsample(self, g):
        """(B, C, nPy, nPx) patch-centre values -> (B, C, H, W): bilinear between ocean-patch centres (normalised)."""
        m = self.patch_ok.to(g.dtype).view(1, 1, self.nPy, self.nPx)
        num = F.interpolate(g * m, scale_factor=self.P, mode="bilinear", align_corners=False)
        den = F.interpolate(m, scale_factor=self.P, mode="bilinear", align_corners=False)
        return (num / den.clamp(min=1e-4))[..., :self.H, :self.W]

    def tokens(self, x3d, day_ok, clim3d):
        """-> tok (B, Np, T, D) after self-attention, keep (B, T)."""
        B, C, T, H, W = x3d.shape
        P = self.P
        xf = x3d.float()
        vis = self._pad(xf[:, -1])                                                  # (B, T, Hp, Wp)
        anom = self._pad((xf[:, -2] - clim3d.float()) * xf[:, -1])
        nv = _patch_sum(vis, P)
        s1, s2 = _patch_sum(anom, P), _patch_sum(anom * anom, P)
        mu = s1 / nv.clamp(min=1)
        sd = torch.sqrt((s2 / nv.clamp(min=1) - mu * mu).clamp(min=0)) * (nv > 1)
        vf = nv / self.n_oc.clamp(min=1)
        feats = self._pad(xf[:, :self.n_feat])
        fm = _patch_sum(feats * self.ocean_p, P) / self.n_oc.clamp(min=1)            # (B, F, T, nPy, nPx)
        stats = torch.cat([mu.unsqueeze(1), sd.unsqueeze(1), vf.unsqueeze(1), fm], 1)  # (B, 3 + F, T, nPy, nPx)
        stats = stats.flatten(3)[..., self.patch_idx].permute(0, 3, 2, 1)            # (B, Np, T, 3 + F)
        tok = self.embed(stats.to(self.day_emb.dtype)) + self.day_emb + self.pos_emb[self.patch_idx].unsqueeze(1)
        keep = day_ok.view(B, 1, 1, 1, T).expand(B, self.Np, 1, 1, T).reshape(B * self.Np, 1, 1, T)
        t = tok.reshape(B * self.Np, T, self.D)
        for blk in self.blocks:
            t = blk(t, keep)
        return self.nt(t).view(B, self.Np, T, self.D), stats

    def forward(self, x3d, x_static, day_ok, clim3d):
        B, C, T, H, W = x3d.shape
        assert T == self.T and (H, W) == (self.H, self.W)
        tok, stats = self.tokens(x3d, day_ok, clim3d)
        vis = x3d[:, -1].float() > 0.5                                               # (B, T, H, W)
        anom = (x3d[:, -2].float() - clim3d.float()) * vis
        rec = {}
        if self.value == "lavg":                                                     # local masked average of each day's visible anomaly
            with torch.autocast("cuda", enabled=False):
                mv = vis.float().view(B * T, 1, H, W)
                num = F.conv2d(anom.view(B * T, 1, H, W), self.gk, padding=5).view(B, T, H, W)
                conf = F.conv2d(mv, self.gk, padding=5).view(B, T, H, W)
            avail_d = conf > 1e-5
            val = torch.where(avail_d, num / conf.clamp(min=1e-5), torch.zeros_like(num))
            rec.update(conf=conf, value=val)
        else:
            avail_d, val = vis, anom
        if self.arm == "interp":
            q = self.wq(tok[:, :, self.ti])                                          # (B, Np, D)
            k = self.wk(tok)                                                         # (B, Np, T, D)
            s = torch.einsum("bpd,bptd->bpt", q, k) / math.sqrt(self.D)              # (B, Np, T)
            s_pix = self._upsample(self._to_grid(s.float()))                         # (B, T, H, W)
            offs = torch.arange(T, device=x3d.device, dtype=torch.float32) - self.ti
            K = max(self.ti, T - 1 - self.ti)
            sst = x3d[:, self.sst].float()                                           # (B, T, H, W)
            extra = [conf] if self.value == "lavg" else []
            pf = torch.stack([vis.float()] + extra + [(offs.abs() / K).view(1, T, 1, 1).expand(B, T, H, W),
                              (offs / K).view(1, T, 1, 1).expand(B, T, H, W), (sst - sst[:, self.ti:self.ti + 1]).abs()], 2)
            p_term = self.pix(pf.view(B * T, self.n_pf, H, W).to(self.pix[0].weight.dtype)).view(B, T, H, W).float()
            logit = s_pix + p_term
            rec.update(s_patch=s_pix, s_pixel=p_term)
        else:
            qp = self.qconv(x3d[:, :, self.ti].to(self.qconv[0].weight.dtype)).float()   # (B, D, H, W)
            k = self.wk(tok).float()                                                 # (B, Np, T, D)
            kg = self._to_grid(k.reshape(B, self.Np, T * self.D)).view(B, T, self.D, self.nPy, self.nPx)
            okg = self.patch_ok.view(1, 1, self.nPy, self.nPx)
            kg = F.pad(kg.flatten(1, 2), (1, 1, 1, 1)).view(B, T, self.D, self.nPy + 2, self.nPx + 2)
            okg = F.pad(okg.float(), (1, 1, 1, 1)) > 0.5
            qp = self._pad(qp).view(B, self.D, self.nPy, self.P, self.nPx, self.P)
            scores = []
            for j, (dy, dx) in enumerate([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]):
                kn = kg[..., 1 + dy:1 + dy + self.nPy, 1 + dx:1 + dx + self.nPx]       # (B, T, D, nPy, nPx)
                sc = torch.einsum("bdypxq,btdyx->btypxq", qp, kn) / math.sqrt(self.D) + self.nbias[j]
                okn = okg[..., 1 + dy:1 + dy + self.nPy, 1 + dx:1 + dx + self.nPx].view(1, 1, self.nPy, 1, self.nPx, 1)
                scores.append(sc.masked_fill(~okn, float("-inf")))
            logit = torch.logsumexp(torch.stack(scores, 0), 0).reshape(B, T, self.Hp, self.Wp)[..., :H, :W]
            rec.update(s_patch=logit)
        ctx = torch.ones(T, dtype=torch.bool, device=x3d.device); ctx[self.ti] = False
        avail = avail_d & ctx.view(1, T, 1, 1)
        logit = logit.masked_fill(~avail, float("-inf"))
        logit = torch.cat([logit, self.none_logit.float().view(1, 1, 1, 1).expand(B, 1, H, W)], 1)
        w = torch.softmax(logit, 1)                                                  # (B, T + 1, H, W)
        wd = w[:, :T]
        blend = (wd * val).sum(1, keepdim=True)
        mass = 1 - w[:, T:]
        ent = -(w * torch.log(w.clamp(min=1e-12))).sum(1, keepdim=True)
        day_t = x3d[:, :, self.ti].float()
        if HOOKS["no_blend"]:                                                        # eval-only: no-blend values (io4 Job 1)
            h_in = torch.cat([torch.zeros_like(blend), torch.zeros_like(mass), torch.zeros_like(ent), day_t, x_static.float()], 1)
        else:
            h_in = torch.cat([blend, mass, ent, day_t, x_static.float()], 1)
        gb = self._upsample(self._to_grid(self.film(tok[:, :, self.ti]).float()))     # (B, 2 width, H, W)
        g, b = gb.chunk(2, 1)
        bb = self.backbone
        dt = bb.inp.weight.dtype
        h = bb.inp(h_in.to(dt))
        h = h * (1 + g.to(h.dtype)) + b.to(h.dtype)
        y = bb.out(bb.blocks(h))
        if HOOKS["recorder"] is not None:
            HOOKS["recorder"](dict(tok=tok, stats=stats, w=w, blend=blend, mass=mass, ent=ent, x3d=x3d, **rec))
        return y
