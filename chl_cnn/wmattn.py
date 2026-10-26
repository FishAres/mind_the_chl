"""Water-mass attention stem (io7 track, bridge plan #25 Phase 1 run 2, 2026-10-24; analyses/io7/wmattn/DESIGN.md;
train_pinn.py --t3d-stem tokdec --tokdec-blend [--tokdec-spatial N] --wm-attn; run io7_wmattn). Default off: nothing here runs without --wm-attn.

Each ocean pixel p of day t attends over a pool of --wm-keys (1024) visible observations sampled from the whole basin on days t-L..t+L
(L = --wm-lags, 2). The model is TokDecNet with the own-pixel blend kept (chl_cnn/tokdec.py) plus this stem's channels in the backbone input,
as the io6 kriging stem's (chl_cnn/krig.py) are wired.

Key pool (per sample; candidates = ocean px x the 2L + 1 days):
  visible = the stem input's visible mask (after donor masking: day t's fake-cloud hidden px and the donor-hidden px of context days are 0)
            x day_ok x ocean. Only visible candidates can be drawn; CHL values of other px are never read.
  strata  = (day slot, SST bin), --wm-sst-bins (8) equal-width bins between the 0.5 and 99.5 % SST quantiles of the sample's visible
            candidates (equal-count bins would make the draw uniform in SST; equal-width bins over-sample rare water masses).
  draw    = weighted sampling without replacement (Efraimidis-Spirakis: top-K of log(u) x n_stratum), weight 1 / n_stratum: each
            (day, SST band) stratum gets ~K / (5 x 8) keys while it has px left; empty strata hand their share to the others.
            Fewer than K visible candidates: the remaining slots are invalid keys (score -1e4, weight exactly 0).
  u       = torch.rand in train mode; in eval mode a torch.Generator per sample seeded from an integer hash of day t's 9 physics fields
            (quantised x 1024 on ocean px): a fixed seed per sample day, independent of batch composition and of any CHL value.
Scores (per head h, one fused F.scaled_dot_product_attention call, scale 1, fp32, autocast off) = q'_h . k'_h with
  q'_h(p) = [ MLPq(9 physics, clim, static, Conv3x3([anomaly x visible, visible] of day t) -> 8) / sqrt(dh)  |  alpha_h * FF(SST, SSS, clim)(p)
             |  alpha_pos_h * FF(y, x)(p)  |  beta_h  |  1 ]
  k'_h(q) = [ MLPk(SST, SSS, MLD, SLA, clim, visible density 3 x 3) at (q, d)  |  FF(SST, SSS, clim)(q, d)  |  FF(y, x)(q)  |
             onehot(day slot)  |  0 (real) / -1e4 (invalid) ]
  plus one "no key" slot k'_none = [0 ... 0 | none_logit_h].
  FF(x) = [cos(w_f x), sin(w_f x)] at fixed periods (z units 0.5, 1, 2, 4, 8; grid coords in [-1, 1] at periods 4, 2), so the alpha blocks
  add a learned stationary kernel sum_f alpha_hf cos(w_f (x_p - x_q)) in dSST, dSSS, dclim and a weak smooth bias in (dy, dx); beta_h is a
  learned bias per (day offset, head). alpha, alpha_pos, beta, none_logit are zero-initialised.
Values per key (shared by the heads): [MLPv(anomaly, density, onehot slot) (dv) | a | a^2 | 1 | 0 | k'_h] (the no-key slot: [0 | 0 | 0 | 0
  | 1 | k'_none]); a = the exact observed CHL z-anomaly (CHL z - DOY climatology) at (q, d).
Outputs (per ocean px, 0 on land; dv + 4 channels, as the kriging stem): att = Linear(heads x dv -> dv) of the heads' weighted embeddings;
  estimate = mean_h sum w a; spread = sqrt(sum wbar (a - est)^2 + 1e-6) from the head-mean moments; mass = mean_h (1 - w_none); entropy =
  mean_h (log Z_h - sum w s), log Z_h = none_logit_h - log w_none,h and sum w s = q'_h . sum w k'_h (exact rewrites, nothing per pair stored).
Memory: on CUDA the call runs on the memory-efficient SDPA kernel only (no (N x K) matrix). If that kernel refuses the shapes in the
  forward, or with --wm-qchunk C > 0 (the OOM / kernel fallback), queries go in chunks of C (2048) ocean px through the math backend,
  checkpointed per chunk (~1.3 GB transient at B 8, K 1024, C 2048).
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FF_PERIODS = (0.5, 1.0, 2.0, 4.0, 8.0)                        # z units: stationary kernel in dSST, dSSS, dclim
POS_PERIODS = (4.0, 2.0)                                      # grid coords scaled to [-1, 1]: basin, half basin
BIG = 1e4
HOOKS = dict(verbose=True)
RESUME_DEFAULTS = {"wm_attn": False, "wm_keys": 1024, "wm_lags": 2, "wm_dim": 64, "wm_heads": 4, "wm_dv": 16, "wm_sst_bins": 8,
                   "wm_qchunk": 0}


def _r8(n):
    return -(-n // 8) * 8


def _ff(x, periods):
    """x (..., P) -> (..., 2 P F): cos, sin at 2 pi / period, ordered (prop, freq, [cos, sin])."""
    w = x.new_tensor([2 * math.pi / p for p in periods])
    a = x.unsqueeze(-1) * w                                                      # (..., P, F)
    return torch.stack([torch.cos(a), torch.sin(a)], -1).flatten(-3)


class WaterMassAttn(nn.Module):
    """forward(x3d, x_static, day_ok, clim3d) -> (out (B, dv + 4, H, W), rec dict). Inputs as chl_cnn/krig.py KrigStem."""

    def __init__(self, n_feat, n_static, T, t_index, ocean, sst_index, sss_index, mld_index, sla_index, keys=1024, lags=2, dim=64,
                 heads=4, dv=16, sst_bins=8, qchunk=0):
        super().__init__()
        assert dim % heads == 0 and t_index - lags >= 0 and t_index + lags < T
        self.nf, self.ti, self.L, self.S = n_feat, t_index, lags, 2 * lags + 1
        self.K, self.h, self.dh, self.dim, self.dv, self.nb, self.qchunk = keys, heads, dim // heads, dim, dv, sst_bins, qchunk
        self.pidx = [sst_index, sss_index, mld_index, sla_index]
        self.n_out = dv + 4
        H, W = ocean.shape
        self.H, self.W = H, W
        ocean = ocean.bool().cpu()
        self.register_buffer("ocean", ocean, persistent=False)
        oc_idx = ocean.flatten().nonzero().squeeze(1)
        self.register_buffer("oc_idx", oc_idx, persistent=False)
        self.N = int(oc_idx.numel())
        yy, xx = (oc_idx // W).float(), (oc_idx % W).float()
        pos = torch.stack([2 * yy / max(H - 1, 1) - 1, 2 * xx / max(W - 1, 1) - 1], -1)          # (N, 2) in [-1, 1]
        self.register_buffer("posff", _ff(pos, POS_PERIODS), persistent=False)                   # (N, 8)
        g = torch.Generator().manual_seed(20261024)
        self.register_buffer("hash_mult", torch.randint(1, 2 ** 20, (n_feat, self.N), generator=g), persistent=False)
        self.n_ff = 3 * 2 * len(FF_PERIODS)                                                      # SST, SSS, clim
        self.n_pos = 2 * 2 * len(POS_PERIODS)
        self.dqk_raw = self.dh + self.n_ff + self.n_pos + self.S + 1
        self.dqk = _r8(self.dqk_raw)
        self.dv_raw = dv + 4 + self.dqk
        self.dvp = _r8(self.dv_raw)
        # parameters
        self.qconv = nn.Conv2d(2, 8, 3, padding=1)
        self.qmlp = nn.Sequential(nn.Conv2d(n_feat + 1 + n_static + 8, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.kmlp = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.vmlp = nn.Sequential(nn.Linear(2 + self.S, dv), nn.GELU(), nn.Linear(dv, dv))
        self.merge = nn.Linear(heads * dv, dv)
        self.alpha = nn.Parameter(torch.zeros(heads, 3 * len(FF_PERIODS)))
        self.alpha_pos = nn.Parameter(torch.zeros(heads, 2 * len(POS_PERIODS)))
        self.beta = nn.Parameter(torch.zeros(heads, self.S))
        self.none_logit = nn.Parameter(torch.zeros(heads))
        self._backend = None                                                         # CUDA: "fused" or "chunk" after the first call

    # ---------------------------------------------------------------- key pool
    def _seeds(self, x3d):
        """Per-sample integer seeds from day t's physics fields (ocean px, quantised): no CHL value or mask enters."""
        f = x3d[:, :self.nf, self.ti].float().flatten(2)[..., self.oc_idx]                       # (B, nf, N)
        qv = torch.nan_to_num(f * 1024.0, nan=0.0, posinf=0.0, neginf=0.0).round().clamp(-2 ** 20, 2 ** 20).long()
        h = (qv * self.hash_mult).sum((1, 2))                                                    # |.| < 2^40 x 9 N: no int64 overflow
        return [int(v) % (2 ** 61 - 1) for v in h.tolist()]

    def _uniform(self, x3d, n):
        B, dev = x3d.shape[0], x3d.device
        if self.training:
            return torch.rand(B, n, device=dev)
        g = torch.Generator(device=dev)
        out = []
        for s in self._seeds(x3d):
            g.manual_seed(s)
            out.append(torch.rand(n, generator=g, device=dev))
        return torch.stack(out)

    @torch.no_grad()
    def sample_pool(self, valid, sst, u):
        """valid (B, S N) bool, sst (B, S N), u (B, S N) uniform -> idx (B, K) long into S N, kvalid (B, K) bool."""
        B, n = valid.shape
        lo, hi = torch.nanquantile(torch.where(valid, sst, torch.full_like(sst, float("nan"))), sst.new_tensor([0.005, 0.995]), dim=1)
        lo, hi = torch.nan_to_num(lo, nan=0.0), torch.nan_to_num(hi, nan=0.0)                     # (B,) robust SST range of the pool
        fr = torch.linspace(0, 1, self.nb + 1, device=sst.device)[1:-1]
        edges = (lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * fr).contiguous()                      # (B, nb - 1) equal-width bins
        b = torch.searchsorted(edges, sst.contiguous(), right=True).clamp(max=self.nb - 1)
        slot = torch.arange(n, device=sst.device) // self.N
        strat = slot.unsqueeze(0) * self.nb + b                                                    # (B, S N)
        cnt = torch.zeros(B, self.S * self.nb, device=sst.device).scatter_add_(1, strat, valid.float())
        key = torch.log(u.clamp(min=1e-30)) * cnt.gather(1, strat).clamp(min=1)                   # log(u) / weight, weight = 1 / n_stratum
        key = torch.where(valid, key, torch.full_like(key, float("-inf")))
        K = min(self.K, n)
        top, idx = key.topk(K, dim=1)
        return idx, torch.isfinite(top)

    # ---------------------------------------------------------------- forward
    def forward(self, x3d, x_static, day_ok, clim3d):
        with torch.autocast("cuda", enabled=False):
            return self._forward(x3d, x_static, day_ok, clim3d)

    def _forward(self, x3d, x_static, day_ok, clim3d):
        B, C, T, H, W = x3d.shape
        ti, L, S, N, h, dh = self.ti, self.L, self.S, self.N, self.h, self.dh
        sl = slice(ti - L, ti + L + 1)
        oc = self.ocean.float()
        xf = x3d[:, :, sl].float()                                                               # (B, C, S, H, W)
        vis = (xf[:, -1] > 0.5).float() * day_ok[:, sl].float().view(B, S, 1, 1) * oc            # (B, S, H, W)
        clim = clim3d[:, sl].float() * oc
        anom = (xf[:, -2] - clim3d[:, sl].float()) * vis
        dens = F.avg_pool2d(vis.view(B * S, 1, H, W), 3, stride=1, padding=1, count_include_pad=True).view(B, S, H, W)
        props = torch.stack([xf[:, i] * oc for i in self.pidx] + [clim, dens, anom], 1)            # (B, 7, S, H, W)
        cand = props.flatten(3)[..., self.oc_idx].reshape(B, 7, S * N)                             # (B, 7, S N), slot-major
        valid = vis.flatten(2)[..., self.oc_idx].reshape(B, S * N) > 0.5
        idx, kvalid = self.sample_pool(valid, cand[:, 0], self._uniform(x3d, S * N))
        K = idx.shape[1]
        kf = cand.gather(2, idx.unsqueeze(1).expand(B, 7, K))                                      # (B, 7, K)
        kf = torch.where(kvalid.unsqueeze(1), kf, torch.zeros_like(kf))
        slot, j = idx // N, idx % N
        onehot = F.one_hot(slot, S).float()                                                      # (B, K, S)
        a = kf[:, 6]
        # keys
        kp = kf[:, :6].transpose(1, 2)                                                           # SST, SSS, MLD, SLA, clim, density
        kh = self.kmlp(kp).view(B, K, h, dh).transpose(1, 2)                                     # (B, h, K, dh)
        kff = _ff(kf[:, [0, 1, 4]].transpose(1, 2), FF_PERIODS)                                  # (B, K, 30)
        kpos = self.posff[j]                                                                     # (B, K, 8)
        aug = torch.where(kvalid, torch.zeros_like(a), torch.full_like(a, -BIG))
        kr = torch.cat([kh, kff.unsqueeze(1).expand(B, h, K, self.n_ff), kpos.unsqueeze(1).expand(B, h, K, self.n_pos),
                        onehot.unsqueeze(1).expand(B, h, K, S), aug.view(B, 1, K, 1).expand(B, h, K, 1)], -1)
        kn = F.pad(self.none_logit.view(1, h, 1, 1).expand(B, h, 1, 1), (self.dqk_raw - 1, 0))    # no-key slot
        Kp = F.pad(torch.cat([kr, kn], 2), (0, self.dqk - self.dqk_raw))                          # (B, h, K + 1, dqk)
        # values
        emb = self.vmlp(torch.cat([a.unsqueeze(-1), kf[:, 5].unsqueeze(-1), onehot], -1))       # (B, K, dv)
        one = kvalid.float()
        vr = torch.cat([emb.unsqueeze(1).expand(B, h, K, self.dv), torch.stack([a, a * a, one, torch.zeros_like(a)], -1).unsqueeze(1)
                        .expand(B, h, K, 4)], -1)
        vn = torch.zeros(B, h, 1, self.dv + 4, device=x3d.device); vn[..., -1] = 1.0
        V = F.pad(torch.cat([torch.cat([vr, vn], 2), Kp], -1), (0, self.dvp - self.dv_raw))      # (B, h, K + 1, dvp)
        # queries (day t, ocean px)
        pt = xf[:, :self.nf, L] * oc
        qc = self.qconv(torch.stack([anom[:, L], vis[:, L]], 1))
        q = self.qmlp(torch.cat([pt, clim[:, L:L + 1], x_static.float(), qc], 1)).flatten(2)[..., self.oc_idx]   # (B, dim, N)
        q = q.view(B, h, dh, N).transpose(2, 3) / math.sqrt(dh)                                    # (B, h, N, dh)
        qp = torch.stack([pt[:, self.pidx[0]], pt[:, self.pidx[1]], clim[:, L]], -1).flatten(1, 2)[:, self.oc_idx]   # (B, N, 3)
        qff = _ff(qp, FF_PERIODS).view(B, 1, N, 3 * len(FF_PERIODS), 2) * self.alpha.view(1, h, 1, -1, 1)
        qpos = self.posff.view(1, 1, N, 2 * len(POS_PERIODS), 2) * self.alpha_pos.view(1, h, 1, -1, 1)
        Q = torch.cat([q, qff.flatten(-2), qpos.expand(B, h, N, -1, 2).flatten(-2), self.beta.view(1, h, 1, S).expand(B, h, N, S),
                       torch.ones(B, h, N, 1, device=x3d.device)], -1)
        Q = F.pad(Q, (0, self.dqk - self.dqk_raw))                                                # (B, h, N, dqk)
        O = self._attend(Q, Kp, V)                                                               # (B, h, N, dvp)
        dv = self.dv
        E, A2, mass, wn = O[..., dv], O[..., dv + 1], O[..., dv + 2], O[..., dv + 3]
        sws = (Q * O[..., dv + 4:dv + 4 + self.dqk]).sum(-1)                                     # sum_c w_c s_c
        ent = self.none_logit.view(1, h, 1) - torch.log(wn.clamp(min=1e-30)) - sws
        att = self.merge(O[..., :dv].transpose(1, 2).reshape(B, N, h * dv)).transpose(1, 2)     # (B, dv, N)
        est, S2, Mb = E.mean(1), A2.mean(1), mass.mean(1)
        spread = torch.sqrt((S2 - 2 * est * est + est * est * Mb).clamp(min=0) + 1e-6)
        out = torch.cat([att, est.unsqueeze(1), spread.unsqueeze(1), Mb.unsqueeze(1), ent.mean(1).unsqueeze(1)], 1)   # (B, dv + 4, N)
        grid = out.new_zeros((B, self.n_out, H * W))
        grid[:, :, self.oc_idx] = out
        pool = torch.where(kvalid, idx, torch.full_like(idx, -1)).float()                         # -1 = invalid slot
        rec = dict(wm=grid.view(B, self.n_out, H, W), wm_pool=pool, wm_nvalid=kvalid.sum(1).float())
        return rec["wm"], rec

    def _sdpa(self, Q, Kp, V):
        return F.scaled_dot_product_attention(Q, Kp, V, scale=1.0)

    def _attend(self, Q, Kp, V):
        if self.qchunk > 0 or self._backend == "chunk":
            return self._attend_chunked(Q, Kp, V, self.qchunk or 2048)
        if not Q.is_cuda:
            return self._sdpa(Q, Kp, V)
        from torch.nn.attention import sdpa_kernel, SDPBackend
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                O = self._sdpa(Q, Kp, V)
            if self._backend is None and HOOKS["verbose"]:
                print(f"wmattn: memory-efficient SDPA, q/k dim {self.dqk}, v dim {self.dvp}, keys {Kp.shape[2]}", flush=True)
            self._backend = "fused"
            return O
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise
            print(f"wmattn: memory-efficient SDPA unavailable ({str(e).splitlines()[0][:200]})", flush=True)
            return self._attend_chunked(Q, Kp, V, self.qchunk or 2048)

    def _sdpa_math(self, Q, Kp, V):
        if not Q.is_cuda:
            return self._sdpa(Q, Kp, V)
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.MATH]):                                     # the fallback must not depend on the fused kernels
            return self._sdpa(Q, Kp, V)

    def _attend_chunked(self, Q, Kp, V, c):
        if self._backend != "chunk" and HOOKS["verbose"]:
            print(f"wmattn: query chunks of {c} ocean px (math SDPA, checkpointed), q/k dim {self.dqk}, v dim {self.dvp}", flush=True)
            self._backend = "chunk"
        outs = []
        for s in range(0, Q.shape[2], c):
            q = Q[:, :, s:s + c]
            outs.append(checkpoint(self._sdpa_math, q, Kp, V, use_reentrant=False) if torch.is_grad_enabled() else self._sdpa_math(q, Kp, V))
        return torch.cat(outs, 2)


def build_wm_tokdec(c_day, n_static, T, t_index, ocean, wm_kw, **tokdec_kw):
    """WmTokDecNet: chl_cnn/tokdec.py TokDecNet (own-pixel blend on, as passed) + the water-mass attention channels in the backbone input."""
    from .tokdec import TokDecNet

    class WmTokDecNet(TokDecNet):
        def __init__(self):
            super().__init__(c_day, n_static, T, t_index, ocean, extra_in=wm_kw.get("dv", 16) + 4, **tokdec_kw)
            self.wm = WaterMassAttn(c_day - 2, n_static, T, t_index, ocean, **wm_kw)

        def extra_parts(self, x3d, x_static, day_ok, clim3d, rec):
            out, wrec = self.wm(x3d, x_static, day_ok, clim3d)
            rec.update(wrec)
            return [out]

    return WmTokDecNet()


# ---------------------------------------------------------------- train_pinn.py hooks (each a one-line call there; all no-ops without --wm-attn)
def add_args(p):
    g = p.add_argument_group("io7 water-mass attention (chl_cnn/wmattn.py; default off = unchanged)")
    g.add_argument("--wm-attn", action="store_true", help="--t3d-stem tokdec: water-mass attention stem channels (keep --tokdec-blend)")
    g.add_argument("--wm-keys", type=int, default=1024, help="--wm-attn: key pool size per sample (visible px, days t-L..t+L)")
    g.add_argument("--wm-lags", type=int, default=2, help="--wm-attn: key pool days t-L..t+L")
    g.add_argument("--wm-dim", type=int, default=64, help="--wm-attn: query / key MLP width (split over heads)")
    g.add_argument("--wm-heads", type=int, default=4, help="--wm-attn: heads")
    g.add_argument("--wm-dv", type=int, default=16, help="--wm-attn: value embedding / attention output channels")
    g.add_argument("--wm-sst-bins", type=int, default=8, help="--wm-attn: SST strata per day for the key draw")
    g.add_argument("--wm-qchunk", type=int, default=0, help="--wm-attn: query chunk (ocean px), checkpointed; 0 = one fused call (OOM fallback)")


def check_args(p, a):
    if not getattr(a, "wm_attn", False):
        return
    if a.t3d_stem != "tokdec" or getattr(a, "tokdec_krig", False):
        p.error("--wm-attn needs --t3d-stem tokdec and not --tokdec-krig")
    if a.context_before < a.wm_lags or a.context_after < a.wm_lags or a.wm_dim % a.wm_heads:
        p.error("--wm-attn needs a window of at least t-L..t+L (--wm-lags) and --wm-dim divisible by --wm-heads")


def n_extra(a):
    return a.wm_dv + 4 if getattr(a, "wm_attn", False) else 0


def build(a, c_day, n_static, n_days, t_index, ocean, **tokdec_kw):
    fi = a.features.index
    wm_kw = dict(sst_index=fi("sst"), sss_index=fi("so"), mld_index=fi("mlotst"), sla_index=fi("sla"), keys=a.wm_keys, lags=a.wm_lags,
                 dim=a.wm_dim, heads=a.wm_heads, dv=a.wm_dv, sst_bins=a.wm_sst_bins, qchunk=a.wm_qchunk)
    model = build_wm_tokdec(c_day, n_static, n_days, t_index, ocean, wm_kw, **tokdec_kw)
    w = model.wm
    print(f"water-mass attention: {a.wm_keys} keys from days t-{a.wm_lags}..t+{a.wm_lags} (strata day x {a.wm_sst_bins} SST bins), "
          f"dim {a.wm_dim}, heads {a.wm_heads}, dv {a.wm_dv}, q/k dim {w.dqk}, v dim {w.dvp}, qchunk {a.wm_qchunk}, {w.N} ocean px; "
          f"params {sum(p.numel() for p in w.parameters())}", flush=True)
    return model
