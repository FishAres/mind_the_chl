"""Token-decoder stem (train_pinn.py --model temporal3d --t3d-stem tokdec; io4 track, 2026-10-23; runs io4_tokdec, io4_tokdec_blend).

Context reaches the backbone only through patch tokens; day t also enters at full resolution through its own channels.
  1. Per-day input (12 channels; days t-K..t+K after donor masking): CHL z-anomaly vs that day's DOY climatology x the visible mask,
     the visible mask, the 9 physics fields (z-scored, x the ocean mask), the ocean mask. Day t's hidden pixels are not visible
     (zero). Days missing from the store are masked out of the attention keys.
  2. ViT-style patch embedding shared across days: the grid is zero-padded to a multiple of P (as chl_cnn/ptok.py), then
     Conv2d(12, D, kernel 2P, stride P, padding P/2): overlapping 2P x 2P windows on a P-px stride, centred on ptok's P x P patches
     (P 4 -> 8 x 8 windows, 38 x 46 grid). LayerNorm, + learned day-offset and patch-position embeddings. Only ocean patches
     (ptok's patch_ok: >= 1 ocean px in the P x P patch) become tokens.
  3. --tokdec-layers pre-norm self-attention blocks over the T day tokens of each ocean patch (ptok._Block: fused
     F.scaled_dot_product_attention, batched over patches); no attention across patches. Final LayerNorm.
  4. Pooling: day t's output token is the query (--tokdec-heads heads), keys / values are the T output tokens with day t included;
     -> one context token per patch. Its concat with day t's own output token (2D) is decoded.
  5. Decoder: Linear(2D -> P^2 C) per patch, pixel-shuffled to full resolution (land patches 0), cropped to H x W, then
     Conv 3x3 (C -> C) + GELU.
  6. Backbone input: [C decoded channels, day t's own channels (features, visible CHL z, visible mask) as the conv runs give them,
     the static channels] -> the dilated backbone (no FiLM, no per-day blend).
  --tokdec-blend (run io4_tokdec_blend): ptok.py's arm-A blend as a skip, from the same output tokens: patch scores
     q(patch, t) . k(patch, d) / sqrt(D) (own W_q, W_k), bilinearly interpolated between ocean-patch centres, + the 1x1-conv
     pixel term on [visible_d, |d - t| / K, (d - t) / K, |SST_d - SST_t|], softmax over the context days visible at the pixel + a
     learned "no day" logit; blend of the own visible anomalies, weight mass and entropy -> 3 extra backbone channels.

--tokdec-spatial N (io6 track, 2026-10-24; run io6_tokdec_blend_spatial; default 0 = off, unchanged): after the temporal pooling (one pooled
     token per ocean patch), N pre-norm self-attention blocks over all ocean-patch tokens of the sample (global, fused SDPA, the pooling's
     heads, width D) with a learned 2D relative-position bias per (patch offset, head) (zero init; the patch-position embedding is already in
     the tokens); output projection and MLP output zero-initialised, so each block starts as the identity. Then the decoder as before.
extra_in (default 0; io6, chl_cnn/krig.py KrigTokDecNet): extra backbone input channels supplied by a subclass's extra_parts().

HOOKS["recorder"] (default None): called as f(dict) after every forward with emb (B, Np, T, D: patch embedding + embeddings, before
attention), tok (B, Np, T, D: output tokens), pool_w (B, Np, T: pooling weights, mean over heads), dec (B, C, H, W), x3d, and with
--tokdec-blend w (B, T + 1, H, W), blend, mass, ent; net = the module.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from .models import build_model
from .ptok import _Block, _patch_sum

HOOKS = dict(recorder=None)
N_IN = 12                                                      # anomaly, visible, 9 physics, ocean (with 9 features)


class TokDecNet(nn.Module):
    """forward(x3d, x_static, day_ok, clim3d) -> backbone output (B, 3, H, W); inputs as ptok.PatchTokenNet."""

    def __init__(self, c_day, n_static, T, t_index, ocean, P=4, dim=128, ch=32, heads=4, layers=2, blend=False, sst_index=0,
                 backbone="dilated", width=64, spatial=0, extra_in=0, **backbone_kw):
        super().__init__()
        assert backbone == "dilated" and P % 2 == 0 and dim % heads == 0
        self.c_day, self.T, self.ti, self.P, self.D, self.C, self.h = c_day, T, t_index, P, dim, ch, heads
        self.n_feat, self.sst, self.use_blend = c_day - 2, sst_index, blend
        H, W = ocean.shape
        self.H, self.W = H, W
        self.Hp, self.Wp = -(-H // P) * P, -(-W // P) * P
        oc = F.pad(ocean.float(), (0, self.Wp - W, 0, self.Hp - H))
        self.register_buffer("ocean_p", oc, persistent=False)
        pm = _patch_sum(oc, P) > 0
        self.register_buffer("patch_ok", pm, persistent=False)
        self.register_buffer("patch_idx", pm.flatten().nonzero().squeeze(1), persistent=False)
        self.nPy, self.nPx = pm.shape
        self.Np = int(pm.sum())
        n_in = self.n_feat + 3
        self.patch = nn.Conv2d(n_in, dim, 2 * P, stride=P, padding=P // 2)
        self.ne = nn.LayerNorm(dim)
        self.day_emb = nn.Parameter(0.02 * torch.randn(T, dim))
        self.pos_emb = nn.Parameter(0.02 * torch.randn(self.nPy * self.nPx, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(layers)])
        self.nt = nn.LayerNorm(dim)
        self.pq, self.pk, self.pv, self.po = (nn.Linear(dim, dim) for _ in range(4))
        self.dec = nn.Linear(2 * dim, P * P * ch)
        self.smooth = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())
        if blend:
            self.wq, self.wk = nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.pix = nn.Sequential(nn.Conv2d(4, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1))
            self.none_logit = nn.Parameter(torch.zeros(()))
        self.backbone = build_model(backbone, ch + c_day + n_static + (3 if blend else 0) + extra_in, width=width, **backbone_kw)
        self.out_channels = self.backbone.out_channels
        self.n_spatial = spatial
        if spatial:                                            # io6: spatial self-attention over the pooled patch tokens (opt-in)
            py, px = self.patch_idx // self.nPx, self.patch_idx % self.nPx
            rel = (py.view(-1, 1) - py.view(1, -1) + self.nPy - 1) * (2 * self.nPx - 1) + (px.view(-1, 1) - px.view(1, -1) + self.nPx - 1)
            self.register_buffer("rel_idx", rel, persistent=False)                    # (Np, Np) index into the relative-offset table
            self.sp_blocks = nn.ModuleList([_SpatialBlock(dim, heads, (2 * self.nPy - 1) * (2 * self.nPx - 1)) for _ in range(spatial)])

    def _pad(self, x):
        return F.pad(x, (0, self.Wp - self.W, 0, self.Hp - self.H))

    def _to_grid(self, v):
        """v (B, Np, C) ocean-patch values -> (B, C, nPy, nPx) grid (0 on land patches)."""
        B, _, C = v.shape
        g = v.new_zeros((B, self.nPy * self.nPx, C))
        g[:, self.patch_idx] = v
        return g.transpose(1, 2).reshape(B, C, self.nPy, self.nPx)

    def _upsample(self, g):
        """as ptok.PatchTokenNet._upsample: bilinear between ocean-patch centres, normalised, cropped to H x W."""
        m = self.patch_ok.to(g.dtype).view(1, 1, self.nPy, self.nPx)
        num = F.interpolate(g * m, scale_factor=self.P, mode="bilinear", align_corners=False)
        den = F.interpolate(m, scale_factor=self.P, mode="bilinear", align_corners=False)
        return (num / den.clamp(min=1e-4))[..., :self.H, :self.W]

    def tokens(self, x3d, day_ok, clim3d):
        """-> emb (B, Np, T, D) before attention, tok (B, Np, T, D) output tokens."""
        B, C, T, H, W = x3d.shape
        xf = x3d.float()
        vis = xf[:, -1]                                                              # (B, T, H, W)
        anom = (xf[:, -2] - clim3d.float()) * vis
        oc = self.ocean_p[:H, :W]
        x = torch.cat([anom.unsqueeze(1), vis.unsqueeze(1), xf[:, :self.n_feat] * oc, oc.expand(B, 1, T, H, W)], 1)  # (B, 12, T, H, W)
        x = self._pad(x.transpose(1, 2).reshape(B * T, -1, H, W))                   # (B T, 12, Hp, Wp)
        e = self.patch(x.to(self.patch.weight.dtype))                                # (B T, D, nPy, nPx)
        e = e.flatten(2)[..., self.patch_idx].view(B, T, self.D, self.Np).permute(0, 3, 1, 2)   # (B, Np, T, D)
        emb = self.ne(e) + self.day_emb + self.pos_emb[self.patch_idx].unsqueeze(1)
        keep = day_ok.view(B, 1, 1, 1, T).expand(B, self.Np, 1, 1, T).reshape(B * self.Np, 1, 1, T)
        t = emb.reshape(B * self.Np, T, self.D)
        for blk in self.blocks:
            t = blk(t, keep)
        return emb, self.nt(t).view(B, self.Np, T, self.D), keep

    def forward(self, x3d, x_static, day_ok, clim3d):
        B, C, T, H, W = x3d.shape
        assert T == self.T and (H, W) == (self.H, self.W)
        emb, tok, keep = self.tokens(x3d, day_ok, clim3d)
        N, d = B * self.Np, self.D // self.h
        tf = tok.reshape(N, T, self.D)
        q = self.pq(tf[:, self.ti]).view(N, self.h, 1, d)
        k = self.pk(tf).view(N, T, self.h, d).transpose(1, 2)
        v = self.pv(tf).view(N, T, self.h, d).transpose(1, 2)
        att = torch.einsum("nhqd,nhsd->nhqs", q.float(), k.float()) / math.sqrt(d)
        att = torch.softmax(att.masked_fill(~keep, float("-inf")), -1)              # (N, h, 1, T)
        pooled = self.po(torch.einsum("nhqs,nhsd->nhqd", att.to(v.dtype), v).reshape(N, self.D))
        if self.n_spatial:
            ps = pooled.view(B, self.Np, self.D)
            for blk in self.sp_blocks:
                ps = blk(ps, self.rel_idx)
            pooled = ps.reshape(N, self.D)
        z = torch.cat([pooled, tf[:, self.ti]], -1)                                  # (N, 2D)
        P = self.P
        y = self.dec(z).float().view(B, self.Np, self.C * P * P)
        g = self._to_grid(y).view(B, self.C, P, P, self.nPy, self.nPx)              # land patches 0
        g = g.permute(0, 1, 4, 2, 5, 3).reshape(B, self.C, self.nPy * P, self.nPx * P)[..., :H, :W]
        dec = self.smooth(g.to(self.smooth[0].weight.dtype))
        day_t = x3d[:, :, self.ti].float()
        parts = [dec.float()]
        rec = {}
        if self.use_blend:
            vis = x3d[:, -1].float() > 0.5
            anom = (x3d[:, -2].float() - clim3d.float()) * vis
            s = torch.einsum("bpd,bptd->bpt", self.wq(tok[:, :, self.ti]), self.wk(tok)) / math.sqrt(self.D)
            s_pix = self._upsample(self._to_grid(s.float()))                         # (B, T, H, W)
            offs = torch.arange(T, device=x3d.device, dtype=torch.float32) - self.ti
            K = max(self.ti, T - 1 - self.ti)
            sst = x3d[:, self.sst].float()
            pf = torch.stack([vis.float(), (offs.abs() / K).view(1, T, 1, 1).expand(B, T, H, W),
                              (offs / K).view(1, T, 1, 1).expand(B, T, H, W), (sst - sst[:, self.ti:self.ti + 1]).abs()], 2)
            p_term = self.pix(pf.view(B * T, 4, H, W).to(self.pix[0].weight.dtype)).view(B, T, H, W).float()
            ctx = torch.ones(T, dtype=torch.bool, device=x3d.device); ctx[self.ti] = False
            logit = (s_pix + p_term).masked_fill(~(vis & ctx.view(1, T, 1, 1)), float("-inf"))
            logit = torch.cat([logit, self.none_logit.float().view(1, 1, 1, 1).expand(B, 1, H, W)], 1)
            w = torch.softmax(logit, 1)
            blend = (w[:, :T] * anom).sum(1, keepdim=True)
            mass = 1 - w[:, T:]
            ent = -(w * torch.log(w.clamp(min=1e-12))).sum(1, keepdim=True)
            parts += [blend, mass, ent]
            rec.update(w=w, blend=blend, mass=mass, ent=ent)
        parts += self.extra_parts(x3d, x_static, day_ok, clim3d, rec)
        h_in = torch.cat(parts + [day_t, x_static.float()], 1)
        bb = self.backbone
        y = bb(h_in.to(bb.inp.weight.dtype))
        if HOOKS["recorder"] is not None:
            if self.n_spatial:
                rec["pooled"] = pooled.view(B, self.Np, self.D)
            HOOKS["recorder"](dict(net=self, emb=emb, tok=tok, pool_w=att.mean(1)[:, 0].view(B, self.Np, T), dec=dec, x3d=x3d, **rec))
        return y

    def extra_parts(self, x3d, x_static, day_ok, clim3d, rec):
        """Extra full-resolution backbone input channels (list of (B, c, H, W)); none in the base class."""
        return []


class _SpatialBlock(nn.Module):
    """Pre-norm self-attention over the ocean-patch tokens of a sample, with a learned relative-position bias per (offset, head)."""

    def __init__(self, d, heads, n_rel):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.rel = nn.Parameter(torch.zeros(n_rel, heads))
        for lin in (self.proj, self.ff[2]):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, x, rel_idx):                             # x (B, Np, d); rel_idx (Np, Np)
        B, Np, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, Np, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        bias = self.rel[rel_idx].permute(2, 0, 1).unsqueeze(0).to(q.dtype)          # (1, heads, Np, Np)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.expand(B, -1, -1, -1))
        x = x + self.proj(o.transpose(1, 2).reshape(B, Np, d))
        return x + self.ff(self.n2(x))
