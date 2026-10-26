"""Learned-kriging stem (io6 track, 2026-10-24; analyses/io6_kriging/DESIGN.md, "Revision 2026-10-24"; train_pinn.py --t3d-stem tokdec
--tokdec-krig [--krig-signed] [--krig-advect]).

Each ocean pixel p on day t attends over individual visible observations (q, d) in a fixed space-time stencil (|dt| <= 4):
  dt = 0       disc of radius 3 px, excluding p itself                         28 offsets
  |dt| = 1, 2  full disc of radius 5 px                                         81 each
  |dt| = 3, 4  5 x 5 block + 8 points on a radius-6 ring                         33 each      total 484
A candidate is a key only where it is visible after donor masking (the stem input's visible mask x the ocean mask x day_ok); day t's
hidden pixels are never visible, the grid is zero-padded (padding never visible), so the stencil never reads across the border or land.
Softmax over the visible candidates + one learned "no key" logit per head.

Features (all in fp32, autocast off):
  query at (p, t)  [9 physics fields of day t, the static channels, Conv3x3([anomaly x visible, visible] of day t) -> 8] -> 1x1 MLP
                   (-> dim -> GELU -> heads x dim / heads);
  key at (q, d)    [offset (dx / 6, dy / 6, dt / 4); advected offset ((dx - s u(p,t) dt) / 6, (dy - s v(p,t) dt) / 6), s learned, init
                   --krig-s0 0.25, u, v the day-t currents at p in px / day; the 9 physics fields at (q, d); |SST(q,d) - SST(p,t)|,
                   |SSS(q,d) - SSS(p,t)| (z units); local visible density at (q, d) = visible fraction of its 3 x 3 box] -> Linear(17, dim)
                   -> GELU -> Linear(dim, heads x dim / heads);
  score            q_h . k_h / sqrt(dim / heads) + a learned relative-position bias per (offset, head) (zero init);
  value            [CHL z-anomaly at (q, d), offset (3), advected offset (2)] -> Linear(6, dv) -> GELU -> Linear(dv, dv) per candidate,
                   weighted sum per head, heads concatenated -> Linear(heads x dv, dv) = the attention output.
  The key MLP's second layer and the value MLP's second layer are applied after folding (q_h . (W k + b) = (W^T q_h) . k + q_h . b;
  sum_c w_c (W g_c + b) = W sum_c w_c g_c + b sum_c w_c): exact rewrites, not approximations.
Outputs (per ocean pixel, 0 on land), concatenated into the backbone input:
  att (dv), estimate = sum_c wbar_c a_c (wbar = mean of the heads' weights; the no-key slot has value 0), spread =
  sqrt(sum_c wbar_c (a_c - estimate)^2 + 1e-6), mass = 1 - mean_h w_h(no key), entropy = mean over heads of the weight entropy (no-key slot
  included).  --krig-signed adds a signed head pair (heads + 1, heads + 2; same features, own query / key / no-key logit):
  w = softmax(s+) - lambda softmax(s-), lambda = sigmoid(theta) learned (init 0.1); weights incl. the no-key slots sum to 1 - lambda.
  Extra channels: signed estimate sum_c w_c a_c, sum_c softmax(s+)_c (real keys), lambda sum_c softmax(s-)_c.  dv + 4 (+ 3) channels.

Implementations (same model, both verified: analyses/io6_kriging/test_krig.py on CPU, gpu_check.py on GPU):
  triton (default on CUDA; chl_cnn/krig_triton.py)  fused kernels: one program per (16 ocean px, sample) streams all 484 offsets with an
          online softmax per head; flash-style backward that recomputes the scores; no per-candidate tensor in memory; offsets without a visible
          key in the 16-px block are skipped (exact). tl.dot precision --krig-prec (tf32 default).
  torch   (CPU, --krig-impl torch, --krig-advect) the candidates are streamed in chunks of --krig-chunk offsets with an online (flash-style)
          softmax per head: running max m, normaliser l, running sums of e a, e a^2, e s, e g (value features); each chunk is a gather from a
          padded (B, 12, 9 days x Hp x Wp) volume [9 physics, density, visible, anomaly] at precomputed flat indices (ocean pixels only), wrapped
          in activation checkpointing and, on CUDA, torch.compile (the stencil is padded to a multiple of the chunk with never-visible dummy
          offsets). The running max is detached (the normalised outputs do not depend on it).

--krig-advect (opt-in, not run in io6): each lag's stencil is centred at p + round(s0 u(p,t) dt, s0 v(p,t) dt) (nearest-neighbour,
s0 = --krig-s0 fixed, shifts clipped at +-4 px), i.e. where a parcel at p on day t would be on day d; observed values and the visible mask
stay exact. The offset features are then the actual offsets from p.

Fewer than 9 physics features (blk7, 2026-10-26: the Black Sea stores have 8, no air_temp): the gather volume's physics block is
zero-padded to 9 channels, so the key MLP's padded input column is always 0 (no contribution, zero gradient) and the Triton / PyTorch
paths are unchanged; the query MLP takes the n_feat real fields. With 9 features (IO) nothing changes.

HOOKS["keep_ctx"] (default False): keep the last forward's context (gather volume, query terms, final m / l) for weights_chunks().
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HOOKS = dict(keep_ctx=False)
N_PHYS = 9
N_KEYF = 17                                                   # 3 offset + 2 advected + 9 physics + 2 |diff| + 1 density
N_VALF = 6                                                    # anomaly + 3 offset + 2 advected
MAX_LAG = 4
SHIFT_MAX = 4                                                 # --krig-advect: centring shift clip (px)


def stencil_k1():
    """List of (dt, dy, dx) offsets of the revised K1 stencil (DESIGN.md "Revised stencil for K1"), sorted by dt then dy, dx."""
    out = []
    ring6 = [(6, 0), (-6, 0), (0, 6), (0, -6), (4, 4), (4, -4), (-4, 4), (-4, -4)]
    for dt in range(-MAX_LAG, MAX_LAG + 1):
        if dt == 0:
            pts = [(y, x) for y in range(-3, 4) for x in range(-3, 4) if y * y + x * x <= 9 and (y, x) != (0, 0)]
        elif abs(dt) <= 2:
            pts = [(y, x) for y in range(-5, 6) for x in range(-5, 6) if y * y + x * x <= 25]
        else:
            pts = [(y, x) for y in range(-2, 3) for x in range(-2, 3)] + ring6
        out += [(dt, y, x) for y, x in sorted(pts)]
    return out


def _gelu(x):
    return F.gelu(x, approximate="tanh")


def _chunk_core(m, l, Sa, Sa2, Ss, Sg, V, idx, offc, relb, vk, qq, qb, up, vp, sstp, sssp, s_adv, W1, b1, Wv1, bv1,
                sst_i: int, sss_i: int, n_vh: int):
    """One chunk of Kc offsets. Shapes: m, l, Sa, Sa2, Ss (B, nh, N); Sg (B, n_vh, dv, N); V (B, 12, L); idx (Bi, Kc, N) long (Bi = 1 or B);
    offc (Bi, Kc, N, 3) actual (dx, dy, dt) offsets in px / days; relb (Kc, nh); vk (Kc,) 1 = real offset; qq (B, nh, dim, N); qb (B, nh, N);
    up, vp (B, N) day-t currents in px / day at p; sstp, sssp (B, N); s_adv scalar; W1 (dim, 17), b1 (dim); Wv1 (dv, 6), bv1 (dv).
    Returns the updated (m, l, Sa, Sa2, Ss, Sg) and the number of visible keys in the chunk (B, N)."""
    B, C, _ = V.shape
    Bi, Kc, N = idx.shape
    Vg = torch.gather(V, 2, idx.reshape(Bi, 1, Kc * N).expand(B, C, Kc * N)).view(B, C, Kc, N)
    vis = Vg[:, 10]
    a = Vg[:, 11]
    valid = (vis > 0.5) & (vk.view(1, Kc, 1) > 0.5)                               # (B, Kc, N)
    ox, oy, ot = offc[..., 0], offc[..., 1], offc[..., 2]                       # (Bi, Kc, N)
    advx = (ox - s_adv * up.unsqueeze(1) * ot) / 6.0                            # (B, Kc, N)
    advy = (oy - s_adv * vp.unsqueeze(1) * ot) / 6.0
    feats = [ox / 6.0, oy / 6.0, ot / 4.0, advx, advy] + [Vg[:, i] for i in range(N_PHYS)] + \
            [(Vg[:, sst_i] - sstp.unsqueeze(1)).abs(), (Vg[:, sss_i] - sssp.unsqueeze(1)).abs(), Vg[:, 9]]
    pre = b1.view(1, -1, 1, 1)
    for f, x in enumerate(feats):
        pre = pre + W1[:, f].view(1, -1, 1, 1) * x.unsqueeze(1)                 # (B, dim, Kc, N)
    hk = _gelu(pre)
    s = torch.einsum("bhjn,bjkn->bhkn", qq, hk) + qb.unsqueeze(2) + relb.t().unsqueeze(0).unsqueeze(-1)              # (B, nh, Kc, N)
    v3 = valid.unsqueeze(1)
    s = torch.where(v3, s, torch.full_like(s, float("-inf")))
    m_new = torch.maximum(m, s.amax(2)).detach()
    alpha = torch.exp(m - m_new)
    e = torch.exp(s - m_new.unsqueeze(2))                                        # 0 where not valid
    a3 = a.unsqueeze(1)
    l = l * alpha + e.sum(2)
    Sa = Sa * alpha + (e * a3).sum(2)
    Sa2 = Sa2 * alpha + (e * a3 * a3).sum(2)
    Ss = Ss * alpha + (e * torch.where(v3, s, torch.zeros_like(s))).sum(2)
    vf = [a, ox / 6.0, oy / 6.0, ot / 4.0, advx, advy]
    pv = bv1.view(1, -1, 1, 1)
    for f, x in enumerate(vf):
        pv = pv + Wv1[:, f].view(1, -1, 1, 1) * x.unsqueeze(1)                  # (B, dv, Kc, N)
    hv = _gelu(pv)
    Sg = Sg * alpha[:, :n_vh].unsqueeze(2) + torch.einsum("bhkn,bdkn->bhdn", e[:, :n_vh], hv)
    return m_new, l, Sa, Sa2, Ss, Sg, valid.sum(1).to(m.dtype)


_COMPILED = {}
# Triton's runtime C helpers need cuda.h; this conda triton ships without its include dir (2026-10-24), so gcc is pointed (CPATH) at the
# first CUDA 12 header directory found here unless CPATH is already set.
CUDA_H_DIRS = ["/opt/conda/lib/python3.12/site-packages/tensorflow/include/external/cuda_cudart/include", "/usr/local/cuda/include",
               "/opt/conda/include"]


def _ensure_cuda_headers():
    import os
    import triton
    inc = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "include", "cuda.h")
    if os.path.exists(inc) or os.environ.get("CPATH"):
        return
    for d in CUDA_H_DIRS:
        if os.path.exists(os.path.join(d, "cuda.h")):
            os.environ["CPATH"] = d
            print(f"krig: CPATH={d} (cuda.h for Triton's runtime helpers)", flush=True)
            return


def _compiled_core():
    if "f" not in _COMPILED:
        _ensure_cuda_headers()
        import torch._dynamo
        import torch._inductor.config
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
        torch._inductor.config.compile_threads = 1                 # no worker pool (host RAM is shared with other jobs)
        _COMPILED["f"] = torch.compile(_chunk_core, dynamic=False)
    return _COMPILED["f"]


class KrigStem(nn.Module):
    """forward(x3d, x_static, day_ok, clim3d) -> (out (B, n_out, H, W), rec dict).
    x3d (B, c_day, T, H, W) per-day [features (n_feat), CHL z (0 where not visible), visible mask]; clim3d (B, T, H, W)."""

    def __init__(self, n_feat, n_static, T, t_index, ocean, sst_index, sss_index, u_index, v_index, vel, dim=32, heads=4, dv=16, chunk=32,
                 signed=False, advect=False, s0=0.25, compile=True, impl="triton"):
        super().__init__()
        assert n_feat <= N_PHYS and dim % heads == 0
        self.n_feat = n_feat                                                       # < 9: physics block zero-padded to 9 in the volume
        assert t_index - MAX_LAG >= 0 and t_index + MAX_LAG < T, "the kriging stencil needs |dt| <= 4 inside the window"
        self.ti, self.dim, self.heads, self.dv, self.chunk = t_index, dim, heads, dv, chunk
        self.signed, self.advect, self.s0, self.use_compile = signed, advect, float(s0), compile
        assert impl in ("triton", "torch")
        self.impl = "torch" if advect else impl                                   # --krig-advect: PyTorch path only
        self.nh = heads + (2 if signed else 0)
        self.dh = dim // heads
        self.sst, self.sss, self.iu, self.iv = sst_index, sss_index, u_index, v_index
        self.n_out = dv + 4 + (3 if signed else 0)
        H, W = ocean.shape
        self.H, self.W = H, W
        R = 6 + (SHIFT_MAX if advect else 0)
        self.R, self.Hp, self.Wp = R, H + 2 * R, W + 2 * R
        self.Lp = self.Hp * self.Wp
        ocean = ocean.bool().cpu()                                                 # buffers built on the CPU; module.to(dev) moves them
        self.register_buffer("ocean", ocean, persistent=False)
        oc_idx = ocean.flatten().nonzero().squeeze(1)
        self.register_buffer("oc_idx", oc_idx, persistent=False)                  # flat index of ocean px in H x W
        self.N = int(oc_idx.numel())
        yy, xx = oc_idx // W, oc_idx % W
        self.register_buffer("pyx", torch.stack([yy, xx]), persistent=False)
        pflat = (yy + R) * self.Wp + (xx + R)
        st = stencil_k1()
        self.K = len(st)
        Kp = -(-self.K // chunk) * chunk
        self.Kp = Kp
        o = torch.tensor(st + [(0, 0, 0)] * (Kp - self.K), dtype=torch.long)       # (Kp, 3) dt, dy, dx; dummy offsets never valid
        self.register_buffer("offs", o, persistent=False)
        self.register_buffer("vk", (torch.arange(Kp) < self.K).float(), persistent=False)
        dti = o[:, 0] + MAX_LAG
        idx = dti.view(-1, 1) * self.Lp + pflat.view(1, -1) + (o[:, 1] * self.Wp + o[:, 2]).view(-1, 1)
        self.register_buffer("idx", idx, persistent=False)                         # (Kp, N) flat index into the (9, Hp, Wp) volume
        offf = torch.stack([o[:, 2], o[:, 1], o[:, 0]], -1).float()               # (Kp, 3) dx, dy, dt
        self.register_buffer("offf", offf, persistent=False)
        # Triton path: int32 flat bases / offsets of the real stencil entries, dt per entry, first-layer column order of the location features
        self.register_buffer("base32", pflat.int(), persistent=False)
        self.register_buffer("delta32", (dti * self.Lp + o[:, 1] * self.Wp + o[:, 2])[:self.K].int(), persistent=False)
        self.register_buffer("otv", o[:self.K, 0].float(), persistent=False)
        self.wl_cols = list(range(5, 14)) + [16, 14, 15]                          # 9 physics, density, |dSST|, |dSSS| (k1 input order)
        # velocity conversion: px / day = (z sd + mu) * 86400 / d{x,y}(y, x) on ocean px (dy < 0 when latitude descends)
        self.register_buffer("u_a", (vel["sd_u"] * 86400.0 / vel["dx"]).float(), persistent=False)
        self.register_buffer("u_b", (vel["mu_u"] * 86400.0 / vel["dx"]).float(), persistent=False)
        self.register_buffer("v_a", (vel["sd_v"] * 86400.0 / vel["dy"]).float(), persistent=False)
        self.register_buffer("v_b", (vel["mu_v"] * 86400.0 / vel["dy"]).float(), persistent=False)
        # parameters
        self.qconv = nn.Conv2d(2, 8, 3, padding=1)
        self.qmlp = nn.Sequential(nn.Conv2d(n_feat + n_static + 8, dim, 1), nn.GELU(), nn.Conv2d(dim, self.nh * self.dh, 1))
        self.k1 = nn.Linear(N_KEYF, dim)
        self.k2 = nn.Linear(dim, self.nh * self.dh)
        self.v1 = nn.Linear(N_VALF, dv)
        self.v2 = nn.Linear(dv, dv)
        self.merge = nn.Linear(heads * dv, dv)
        self.relbias = nn.Parameter(torch.zeros(Kp, self.nh))
        self.none_logit = nn.Parameter(torch.zeros(self.nh))
        self.s_adv = nn.Parameter(torch.tensor(float(s0)))
        if signed:
            self.lam_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def lam(self):
        return torch.sigmoid(self.lam_logit) if self.signed else None

    # ---------------------------------------------------------------- inputs
    def _prep(self, x3d, x_static, day_ok, clim3d):
        B, C, T, H, W = x3d.shape
        ti, L = self.ti, MAX_LAG
        sl = slice(ti - L, ti + L + 1)
        xf = x3d[:, :, sl].float()                                                 # (B, C, 9, H, W)
        oc = self.ocean.float()
        vis = (xf[:, -1] > 0.5).float() * day_ok[:, sl].float().view(B, -1, 1, 1) * oc   # (B, 9, H, W)
        anom = (xf[:, -2] - clim3d[:, sl].float()) * vis
        phys = xf[:, :self.n_feat] * oc                                            # (B, n_feat, 9, H, W)
        dens = F.avg_pool2d(vis.view(B * (2 * L + 1), 1, H, W), 3, stride=1, padding=1, count_include_pad=True).view(B, -1, H, W)
        physv = phys if self.n_feat == N_PHYS else F.pad(phys, (0, 0, 0, 0, 0, 0, 0, N_PHYS - self.n_feat))   # zero channels after the real ones
        vol = torch.cat([physv, dens.unsqueeze(1), vis.unsqueeze(1), anom.unsqueeze(1)], 1)      # (B, 12, 9, H, W)
        R = self.R
        vol = F.pad(vol, (R, R, R, R)).reshape(B, 12, -1)                          # (B, 12, 9 Lp)
        # query side, day t
        pt = phys[:, :, L]                                                         # (B, 9, H, W)
        qc = self.qconv(torch.stack([anom[:, L], vis[:, L]], 1))
        q = self.qmlp(torch.cat([pt, x_static.float(), qc], 1))                     # (B, nh dh, H, W)
        ocx = self.oc_idx
        q = q.flatten(2)[..., ocx].view(B, self.nh, self.dh, self.N)
        sc = 1.0 / math.sqrt(self.dh)
        W2 = self.k2.weight.view(self.nh, self.dh, self.dim)
        qq = torch.einsum("bhdn,hdj->bhjn", q, W2) * sc                             # (B, nh, dim, N)
        qb = torch.einsum("bhdn,hd->bhn", q, self.k2.bias.view(self.nh, self.dh)) * sc
        u = (xf[:, self.iu, L] * self.u_a + self.u_b) * oc                          # (B, H, W) px / day along x
        v = (xf[:, self.iv, L] * self.v_a + self.v_b) * oc                          # px / day along y (row index)
        up, vp = u.flatten(1)[:, ocx], v.flatten(1)[:, ocx]
        sstp, sssp = pt[:, self.sst].flatten(1)[:, ocx], pt[:, self.sss].flatten(1)[:, ocx]
        return dict(vol=vol, qq=qq, qb=qb, q=q * sc, up=up, vp=vp, sstp=sstp, sssp=sssp, B=B, vis_t=vis[:, L])

    def _chunk_index(self, c0, c1, ctx):
        """Flat gather indices (Bi, Kc, N) and actual offsets (Bi, Kc, N, 3) of stencil entries c0:c1."""
        idx = self.idx[c0:c1].unsqueeze(0)
        off = self.offf[c0:c1].view(1, -1, 1, 3).expand(1, c1 - c0, self.N, 3)
        if not self.advect:
            return idx, off
        sh = ctx["shift"]                                                         # (B, 2, 9, N) long: shift x, y per lag
        dti = self.offs[c0:c1, 0] + MAX_LAG
        sx, sy = sh[:, 0][:, dti], sh[:, 1][:, dti]                                 # (B, Kc, N)
        idx = idx + sy * self.Wp + sx
        off = off + torch.stack([sx, sy, torch.zeros_like(sx)], -1).float()
        return idx, off

    def forward(self, x3d, x_static, day_ok, clim3d):
        with torch.autocast("cuda", enabled=False):
            return self._forward(x3d, x_static, day_ok, clim3d)

    def _forward(self, x3d, x_static, day_ok, clim3d):
        ctx = self._prep(x3d, x_static, day_ok, clim3d)
        B, nh, N = ctx["B"], self.nh, self.N
        dev = x3d.device
        nk = self.none_logit.float().view(1, nh, 1).expand(B, nh, N).contiguous()
        if self.impl == "triton" and x3d.is_cuda:
            E, A2, ent, w_nk, G, nkeys, m, logl = self._attend_triton(ctx)
            l = torch.exp(logl)
        else:
            E, A2, ent, w_nk, G, nkeys, m, l = self._attend_torch(ctx, nk)
        mass = 1.0 - w_nk
        return self._outputs(ctx, E, A2, ent, w_nk, mass, G, nkeys, m, l)

    def _attend_triton(self, ctx):
        """Fused Triton kernels (chl_cnn/krig_triton.py): the same model, nothing per candidate materialised."""
        _ensure_cuda_headers()
        from .krig_triton import KrigAttnFn, np2
        K, B, N, nh, dh, dim = self.K, ctx["B"], self.N, self.nh, self.dh, self.dim
        nhp, dhp, dimp = np2(nh), np2(dh), np2(dim)                                 # kernel tiles need powers of 2: zero padding
        W1, b1, Wv1, bv1 = self.k1.weight, self.k1.bias, self.v1.weight, self.v1.bias
        off = self.offf[:K]                                                        # (K, 3) dx, dy, dt
        offn = off / off.new_tensor([6.0, 6.0, 4.0])
        Ck = b1 + offn @ W1[:, 0:3].t() + (off[:, :2] / 6.0) @ W1[:, 3:5].t()       # (K, dim)
        Cvk = bv1 + offn @ Wv1[:, 1:4].t() + (off[:, :2] / 6.0) @ Wv1[:, 4:6].t()   # (K, dv)
        up, vp = ctx["up"], ctx["vp"]
        U = self.s_adv * (W1[:, 3].view(1, -1, 1) * up.unsqueeze(1) + W1[:, 4].view(1, -1, 1) * vp.unsqueeze(1))      # (B, dim, N)
        Uv = self.s_adv * (Wv1[:, 4].view(1, -1, 1) * up.unsqueeze(1) + Wv1[:, 5].view(1, -1, 1) * vp.unsqueeze(1))   # (B, dv, N)
        Wl = F.pad(W1[:, self.wl_cols].t(), (0, dimp - dim, 0, 16 - len(self.wl_cols)))   # (16, dimp): 9 physics, density, |dSST|, |dSSS|
        Ck, U = F.pad(Ck, (0, dimp - dim)), F.pad(U, (0, 0, 0, dimp - dim))         # padded hidden units: pre 0, gelu 0, no key contribution
        W2T = F.pad(self.k2.weight.t().reshape(dim, nh, dh), (0, dhp - dh, 0, nhp - nh, 0, dimp - dim)).reshape(dimp, nhp * dhp)
        b2 = F.pad(self.k2.bias.view(nh, dh), (0, dhp - dh, 0, nhp - nh)).reshape(nhp * dhp)
        q = F.pad(ctx["q"], (0, 0, 0, dhp - dh, 0, nhp - nh)).reshape(B, nhp * dhp, N)          # scaled by 1 / sqrt(dh)
        c = lambda t: t.float().contiguous()
        return KrigAttnFn.apply(c(ctx["vol"]), self.base32, self.delta32, self.otv, c(Ck), c(Cvk), c(self.relbias[:K]), c(Wl), c(W2T), c(b2),
                                c(Wv1[:, 0]), c(q), c(U), c(Uv), c(ctx["sstp"]), c(ctx["sssp"]), c(self.none_logit), self.heads, dhp,
                                self.sst, self.sss)

    def _attend_torch(self, ctx, nk):
        """PyTorch streamed reference (chunks of --krig-chunk offsets, checkpointed, torch.compile on CUDA)."""
        B, nh, N = ctx["B"], self.nh, self.N
        dev = nk.device
        if self.advect:
            dts = torch.arange(-MAX_LAG, MAX_LAG + 1, device=dev, dtype=torch.float32).view(1, -1, 1)
            sx = torch.round(self.s0 * ctx["up"].unsqueeze(1) * dts).clamp(-SHIFT_MAX, SHIFT_MAX).long()
            sy = torch.round(self.s0 * ctx["vp"].unsqueeze(1) * dts).clamp(-SHIFT_MAX, SHIFT_MAX).long()
            ctx["shift"] = torch.stack([sx, sy], 1)                                   # (B, 2, 9, N)
        # initial state = the no-key slot alone: running max m = nk, l = exp(nk - m) = 1, sum e s = nk (value 0: Sa, Sa2, Sg = 0).
        # m and Ss start from the live logit, so the gradient wrt nk flows through the first chunk's rescaling; later maxima are detached.
        m = nk
        l = torch.ones(B, nh, N, device=dev)
        Sa = torch.zeros(B, nh, N, device=dev); Sa2 = torch.zeros_like(Sa)
        Ss = nk
        Sg = torch.zeros(B, self.heads, self.dv, N, device=dev)
        nkeys = torch.zeros(B, N, device=dev)
        core = _compiled_core() if (self.use_compile and dev.type == "cuda") else _chunk_core
        prm = (self.s_adv, self.k1.weight, self.k1.bias, self.v1.weight, self.v1.bias)
        for c0 in range(0, self.Kp, self.chunk):
            c1 = c0 + self.chunk
            idx, off = self._chunk_index(c0, c1, ctx)
            args = (m, l, Sa, Sa2, Ss, Sg, ctx["vol"], idx, off, self.relbias[c0:c1], self.vk[c0:c1], ctx["qq"], ctx["qb"], ctx["up"],
                    ctx["vp"], ctx["sstp"], ctx["sssp"]) + prm
            if torch.is_grad_enabled():
                m, l, Sa, Sa2, Ss, Sg, nkc = checkpoint(core, *args, self.sst, self.sss, self.heads, use_reentrant=False)
            else:
                m, l, Sa, Sa2, Ss, Sg, nkc = core(*args, self.sst, self.sss, self.heads)
            nkeys = nkeys + nkc.detach()
        w_nk = torch.exp(nk - m) / l                                               # (B, nh, N) weight of the no-key slot
        E = Sa / l
        A2 = Sa2 / l
        ent = torch.log(l) + m - Ss / l
        G = Sg / l[:, :self.heads].unsqueeze(2)                                     # (B, heads, dv, N)
        return E, A2, ent, w_nk, G, nkeys, m, l

    def _outputs(self, ctx, E, A2, ent, w_nk, mass, G, nkeys, m, l):
        B, dev, hv = ctx["B"], E.device, self.heads
        o = torch.einsum("bhdn,ed->bhen", G, self.v2.weight) + self.v2.bias.view(1, 1, -1, 1) * mass[:, :hv].unsqueeze(2)
        att = torch.einsum("bkn,ek->ben", o.reshape(B, hv * self.dv, self.N), self.merge.weight) + self.merge.bias.view(1, -1, 1)
        est = E[:, :hv].mean(1)
        S2 = A2[:, :hv].mean(1)
        Mb = mass[:, :hv].mean(1)
        spread = torch.sqrt((S2 - 2 * est * est + est * est * Mb).clamp(min=0) + 1e-6)
        entm = ent[:, :hv].mean(1)
        chans = [att, est.unsqueeze(1), spread.unsqueeze(1), Mb.unsqueeze(1), entm.unsqueeze(1)]
        rec = dict(n_keys=nkeys)
        if self.signed:
            lam = self.lam()
            sest = E[:, hv] - lam * E[:, hv + 1]
            chans += [sest.unsqueeze(1), mass[:, hv].unsqueeze(1), (lam * mass[:, hv + 1]).unsqueeze(1)]
            rec.update(w_nk_pos=w_nk[:, hv], w_nk_neg=w_nk[:, hv + 1], lam=lam)
        out = torch.cat(chans, 1)                                                  # (B, n_out, N)
        grid = out.new_zeros((B, self.n_out, self.H * self.W))
        grid[:, :, self.oc_idx] = out
        grid = grid.view(B, self.n_out, self.H, self.W)
        if HOOKS["keep_ctx"]:
            ctx.update(m=m.detach(), l=l.detach(), E=E.detach())
            self._last = ctx
        nkg = torch.zeros(B, self.H * self.W, device=dev)
        nkg[:, self.oc_idx] = nkeys
        rec["n_keys"] = nkg.view(B, self.H, self.W)
        return grid, rec

    # ---------------------------------------------------------------- diagnostics (eval, no grad)
    @torch.no_grad()
    def weights_chunks(self, ctx=None):
        """Yield (c0, c1, w (B, nh, Kc, N) final weights, valid (B, Kc, N), dsst (B, Kc, N) |SST(q,d) - SST(p,t)| in z units) of the
        last forward (HOOKS["keep_ctx"] must have been on)."""
        ctx = ctx or self._last
        B = ctx["B"]
        for c0 in range(0, self.Kp, self.chunk):
            c1 = c0 + self.chunk
            idx, off = self._chunk_index(c0, c1, ctx)
            V = ctx["vol"]; C = V.shape[1]; Bi, Kc, N = idx.shape
            Vg = torch.gather(V, 2, idx.reshape(Bi, 1, Kc * N).expand(B, C, Kc * N)).view(B, C, Kc, N)
            valid = (Vg[:, 10] > 0.5) & (self.vk[c0:c1].view(1, Kc, 1) > 0.5)
            s = self.scores(ctx, Vg, off, c0, c1)
            w = torch.where(valid.unsqueeze(1), torch.exp(s - ctx["m"].unsqueeze(2)) / ctx["l"].unsqueeze(2), torch.zeros_like(s))
            yield c0, c1, w, valid, (Vg[:, self.sst] - ctx["sstp"].unsqueeze(1)).abs()

    def scores(self, ctx, Vg, off, c0, c1):
        ox, oy, ot = off[..., 0], off[..., 1], off[..., 2]
        s_adv = self.s_adv
        advx = (ox - s_adv * ctx["up"].unsqueeze(1) * ot) / 6.0
        advy = (oy - s_adv * ctx["vp"].unsqueeze(1) * ot) / 6.0
        feats = [ox / 6.0, oy / 6.0, ot / 4.0, advx, advy] + [Vg[:, i] for i in range(N_PHYS)] + \
                [(Vg[:, self.sst] - ctx["sstp"].unsqueeze(1)).abs(), (Vg[:, self.sss] - ctx["sssp"].unsqueeze(1)).abs(), Vg[:, 9]]
        pre = self.k1.bias.view(1, -1, 1, 1)
        for f, x in enumerate(feats):
            pre = pre + self.k1.weight[:, f].view(1, -1, 1, 1) * x.unsqueeze(1)
        return torch.einsum("bhjn,bjkn->bhkn", ctx["qq"], _gelu(pre)) + ctx["qb"].unsqueeze(2) + self.relbias[c0:c1].t().unsqueeze(0).unsqueeze(-1)


def build_krig_tokdec(c_day, n_static, T, t_index, ocean, krig_kw, **tokdec_kw):
    """KrigTokDecNet: chl_cnn/tokdec.py TokDecNet (blend off) + the kriging stem's channels in the backbone input."""
    from .tokdec import TokDecNet

    class KrigTokDecNet(TokDecNet):
        def __init__(self):
            n_feat = c_day - 2
            n_out = krig_kw.get("dv", 16) + 4 + (3 if krig_kw.get("signed") else 0)
            super().__init__(c_day, n_static, T, t_index, ocean, blend=False, extra_in=n_out, **tokdec_kw)
            self.krig = KrigStem(n_feat, n_static, T, t_index, ocean, **krig_kw)

        def extra_parts(self, x3d, x_static, day_ok, clim3d, rec):
            out, krec = self.krig(x3d, x_static, day_ok, clim3d)
            rec.update(krig=out, **krec)
            return [out]

    return KrigTokDecNet()
