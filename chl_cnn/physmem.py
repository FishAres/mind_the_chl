"""Physics memory + longer token window (io7 track, bridge plan #25 Phase 1 run 1; run io7_physmem; train_pinn.py --physmem).

TokDecNet (chl_cnn/tokdec.py) with
  1. a token window of any day offsets (e.g. t-15..t+15, or with --physmem-stride2 every day to +-K_b and even offsets beyond), while the
     own-pixel blend keeps the days |d - t| <= K_b (--physmem-blend-k, default 8; offset features scaled by K_b as in io4/io6);
  2. past-only physics summaries of target day t added to every day token of each ocean patch (before the temporal self-attention):
     12 channels on ptok's P x P patch grid -> Conv2d(12, D, 3, padding 1) on the patch grid -> GELU -> Conv2d(D, D, 1) (zero-init, so the
     memory starts as a no-op) -> + emb[:, patch, every day].

Daily fields (full 0.25 deg grid, then ocean-and-finite mean per P x P patch; cached in ~/mind_the_chl_gap/data/io7/):
  curl  wind-stress curl  d(tau_y)/dx - d(tau_x)/dy  (N m^-3; tau from 10 m winds, Large & Pond drag, chl_cnn/physics.wind_stress)
  wE    Ekman pumping  curl(tau) / (rho_w f) + beta tau_x / (rho_w f^2)  (m/s, + up; NaN for |lat| < 5 deg; chl_cnn/physics.ekman)
  tau_s alongshore stress  sign(lat) tau . s_hat, s_hat = z_hat x n_hat (n_hat: offshore unit normal = -grad L / |grad L|, L = land fraction
        (1 - ocean) Gaussian-smoothed with sigma 2 px); + = upwelling-favourable (offshore Ekman transport); ocean px within 4 px (~100 km,
        Chebyshev) of land only, NaN elsewhere
  sst   SST (K) -> SSTA = SST - DOY climatology of the training days (per patch, 31-day circular running mean over the DOY means)
  mld   mlotst (m)
Summaries for target t (calendar days; store days only; per patch, mean of finite values):
  A = [t-30, t-1], B = [t-60, t-31]:  mean curl, mean wE, mean tau_s, mean SSTA, dMLD = mean(second half) - mean(first half)
  (A: mean[t-15, t-1] - mean[t-30, t-16]; B: mean[t-45, t-31] - mean[t-60, t-46])  -> 10 channels, + coverage of A and B (fraction of
  the 30 calendar days missing from the store; raw, 0 = complete) -> 12. A window mean needs >= 15 finite days (a half-window mean >= 8), else it is 0 (z units).
  Start of record / gaps: windows reaching before 1997-10-01 or over store gaps simply have fewer days; coverage says how many.
  The 10 means are z-scored with mean / std over the training days and ocean patches where valid, clipped to +-6, 0 where invalid / land.
Day t itself and later days never enter a summary (past-only); physics has no cloud mask, so CHL masking is unaffected.

Black Sea stores (blk7, 2026-10-26; MemTable(..., wind_is_stress=True), set by train_pinn.py for --store blk / blk4): u_wind / v_wind are
already wind stress (Pa, CMEMS tauuo / tauvo), so tau = (u_wind, v_wind) directly (no drag law) and w_E uses the same formula as
chl_cnn/physics.ekman on that tau. The store variable of "sst" ("sst32") comes from the trainer's read function. Latitude may ascend:
grid_metrics gives signed dy, so every derivative, the offshore normal and the alongshore vector are in physical (east, north) axes.
Pixel constants (COAST_PX 4, COAST_SIGMA 2) stay in pixels: the coastal band is ~18 km wide at 4.5 km (IO ~100 km).
The cache key of the IO default (wind speed) is unchanged; wind_is_stress adds " stress" to it.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage

from . import physics as PH
from .tokdec import TokDecNet, HOOKS

CACHE = Path.home() / "mind_the_chl_gap" / "data" / "io7"
N_MEM = 12
NAMES = ["curl_A", "wE_A", "taus_A", "ssta_A", "dmld_A", "curl_B", "wE_B", "taus_B", "ssta_B", "dmld_B", "miss_A", "miss_B"]
WE_LAT_MIN, COAST_PX, COAST_SIGMA = 5.0, 4, 2.0


def patch_mean(x, ok, P):
    """x (n, H, W) float, ok (n, H, W) bool -> mean of x over ok px per P x P patch (grid zero-padded to a multiple of P), NaN where none."""
    n, H, W = x.shape
    Hp, Wp = -(-H // P) * P, -(-W // P) * P
    pad = lambda a: np.pad(a, ((0, 0), (0, Hp - H), (0, Wp - W)))
    xs = pad(np.where(ok, x, 0.0)).reshape(n, Hp // P, P, Wp // P, P).sum((2, 4))
    cs = pad(ok.astype("float64")).reshape(n, Hp // P, P, Wp // P, P).sum((2, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cs > 0, xs / cs, np.nan)


def coast_geometry(ocean, lat, lon):
    """(s_x, s_y) alongshore unit vector (z_hat x offshore normal) and the coastal-band mask (ocean px within COAST_PX px of land)."""
    land = (~ocean).astype("float64")
    Ls = ndimage.gaussian_filter(land, COAST_SIGMA, mode="nearest")
    dx, dy, _, _ = PH.grid_metrics(lat, lon)
    gx, gy = PH.ddx(Ls, dx), PH.ddy(Ls, dy)
    g = np.hypot(gx, gy)
    with np.errstate(invalid="ignore", divide="ignore"):
        nx, ny = np.where(g > 0, -gx / g, 0.0), np.where(g > 0, -gy / g, 0.0)
    band = ocean & ndimage.binary_dilation(~ocean, np.ones((3, 3), bool), iterations=COAST_PX) & (g > 0)
    return -ny, nx, band


def ekman_from_stress(tx, ty, lat, lon, lat_min):
    """w_E (m/s, + up) from wind stress (Pa), as chl_cnn/physics.ekman (NaN for |lat| < lat_min, beta term included)."""
    dx, dy, f, beta = PH.grid_metrics(lat, lon)
    f = np.where(np.abs(lat)[:, None] < lat_min, np.nan, f)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (PH.ddx(ty, dx) - PH.ddy(tx, dy)) / (PH.RHO_W * f) + beta * tx / (PH.RHO_W * f ** 2)


def daily_fields(rd, n, ocean, lat, lon, P, chunk=500, log=print, wind_is_stress=False):
    """(n, 5, nPy, nPx) float32 patch means of curl, wE, tau_s, SST, MLD per store day (cached).
    wind_is_stress: u_wind / v_wind are wind stress in Pa (Black Sea stores), else 10 m wind (m/s, IO)."""
    key = hashlib.md5(np.ascontiguousarray(ocean).tobytes() + lat.tobytes() + lon.tobytes() + (f"{n} {P} {WE_LAT_MIN} {COAST_PX} {COAST_SIGMA}"
                      + (" stress" if wind_is_stress else "")).encode()).hexdigest()[:10]
    f = CACHE / f"physmem_daily_P{P}_{key}.npy"
    if f.exists():
        return np.load(f)
    sx, sy, band = coast_geometry(ocean, lat, lon)
    sgn = np.sign(lat)[:, None]
    out = []
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(b0 + chunk, n))
        u, v = rd("u_wind", sl).astype("float64"), rd("v_wind", sl).astype("float64")
        if wind_is_stress:
            tx, ty = u, v
            wE = ekman_from_stress(tx, ty, lat, lon, WE_LAT_MIN)
        else:
            _, _, wE, tx, ty = PH.ekman(u, v, lat, lon, lat_min=WE_LAT_MIN)
        dx, dy, _, _ = PH.grid_metrics(lat, lon)
        curl = PH.ddx(ty, dx) - PH.ddy(tx, dy)
        taus = np.where(band, sgn * (tx * sx + ty * sy), np.nan)
        sst, mld = rd("sst", sl).astype("float64"), rd("mlotst", sl).astype("float64")
        fl = [curl, wE, taus, sst, mld]
        out.append(np.stack([patch_mean(x, np.isfinite(x) & ocean, P) for x in fl], 1).astype("float32"))
        log(f"physmem: daily fields {sl.stop}/{n}")
    out = np.concatenate(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f.with_suffix(".tmp.npy"), out); f.with_suffix(".tmp.npy").rename(f)
    return out


def _doy(times):
    return (np.asarray(times) - np.asarray(times).astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)


def summaries(daily, times, tr, patch_ok):
    """(n, 12, nPy, nPx) float16 z-scored summaries (see the module doc) and a stats dict. daily (n, 5, nPy, nPx) from daily_fields."""
    n = len(times)
    daily = daily.astype("float64").copy()
    doy = _doy(times)                                          # SST -> SSTA: DOY climatology of the training days, 31-day circular mean
    s, c = np.zeros((366,) + daily.shape[2:]), np.zeros((366,) + daily.shape[2:])
    ok = np.isfinite(daily[tr, 3])
    np.add.at(s, doy[tr], np.where(ok, daily[tr, 3], 0.0)); np.add.at(c, doy[tr], ok)
    ker = lambda a: sum(np.roll(a, k, 0) for k in range(-15, 16))
    clim = ker(s) / np.maximum(ker(c), 1)
    daily[:, 3] = daily[:, 3] - clim[doy]
    cal = (np.asarray(times) - np.asarray(times)[0]).astype("timedelta64[D]").astype(int)  # calendar index of each store day
    Nc = cal[-1] + 1
    V = np.full((Nc,) + daily.shape[1:], np.nan); V[cal] = daily
    fin = np.isfinite(V)
    CS = np.concatenate([np.zeros((1,) + V.shape[1:]), np.cumsum(np.where(fin, V, 0.0), 0)])    # CS[k] = sum over calendar days < k
    CN = np.concatenate([np.zeros((1,) + V.shape[1:], np.int32), np.cumsum(fin, 0, dtype=np.int32)])
    present = np.zeros(Nc, bool); present[cal] = True
    CP = np.concatenate([[0], np.cumsum(present)])

    def wmean(lo, hi, ch, need):                               # mean over calendar days [cal - lo, cal - hi] (lo > hi >= 1)
        a_, b_ = np.clip(cal - lo, 0, None), np.clip(cal - hi + 1, 0, None)
        sm, cn = CS[b_, ch] - CS[a_, ch], CN[b_, ch] - CN[a_, ch]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cn >= need, sm / np.maximum(cn, 1), np.nan)

    out = np.full((n, N_MEM) + daily.shape[2:], np.nan, dtype=np.float32)
    for w, (lo, hi) in enumerate([(30, 1), (60, 31)]):
        for j in range(4):                                     # curl, wE, tau_s, SSTA
            out[:, 5 * w + j] = wmean(lo, hi, j, 15)
        out[:, 5 * w + 4] = wmean(hi + 14, hi, 4, 8) - wmean(lo, lo - 14, 4, 8)   # dMLD: second half - first half
        a_, b_ = np.clip(cal - lo, 0, None), np.clip(cal - hi + 1, 0, None)
        out[:, 10 + w] = (1.0 - (CP[b_] - CP[a_]) / 30.0)[:, None, None]       # missing fraction (not z-scored)
    pm = np.asarray(patch_ok, bool)
    stats = {}
    for k in range(10):
        x = out[tr, k][:, pm]; x = x[np.isfinite(x)]
        mu, sd = float(x.mean()), float(x.std()) + 1e-12
        stats[NAMES[k]] = dict(mean=mu, std=sd, valid_frac_train=float(np.isfinite(out[tr, k][:, pm]).mean()))
        out[:, k] = np.clip((out[:, k] - mu) / sd, -6, 6)
    out = np.where(np.isfinite(out) & pm, out, 0.0).astype("float16")
    return out, stats


class MemTable:
    """Per-day summary table on the CPU; batch(bi) -> (B, 12, nPy, nPx) float32 on bi's device."""

    def __init__(self, rd, times, tr, ocean, lat, lon, P, log=print, wind_is_stress=False):
        self.P = P
        oc = torch.as_tensor(np.asarray(ocean)).float()
        H, W = oc.shape
        oc = F.pad(oc, (0, -(-W // P) * P - W, 0, -(-H // P) * P - H))
        self.patch_ok = (F.avg_pool2d(oc[None, None], P)[0, 0] > 0).numpy()
        daily = daily_fields(rd, len(times), np.asarray(ocean, bool), np.asarray(lat, "float64"), np.asarray(lon, "float64"), P, log=log,
                            wind_is_stress=wind_is_stress)
        tab, self.stats = summaries(daily, times, np.asarray(tr), self.patch_ok)
        self.tab = torch.from_numpy(tab)
        log(f"physmem: {N_MEM} summary channels on a {tab.shape[2]} x {tab.shape[3]} patch grid, {len(times)} days; "
            f"valid frac (train) " + ", ".join(f"{k} {v['valid_frac_train']:.3f}" for k, v in self.stats.items()))

    def batch(self, bi):
        return self.tab[bi.cpu()].to(bi.device).float()


class PhysMemTokDecNet(TokDecNet):
    """TokDecNet over the token-window day offsets `offsets`, blend restricted to |offset| <= blend_k, + the physics-memory token term.
    forward(x3d, x_static, day_ok, clim3d) as TokDecNet; the memory input is set per batch in self.mem_in ((B, 12, nPy, nPx) or None = 0)."""

    def __init__(self, c_day, n_static, offsets, ocean, blend_k=8, mem_ch=N_MEM, **kw):
        offsets = list(offsets)
        super().__init__(c_day, n_static, len(offsets), offsets.index(0), ocean, blend=True, **kw)
        self.offs = offsets
        self.bsel = [i for i, o in enumerate(offsets) if abs(o) <= blend_k]
        self.bti, self.blend_k = self.bsel.index(self.ti), blend_k
        D = self.D
        self.mem = nn.Sequential(nn.Conv2d(mem_ch, D, 3, padding=1), nn.GELU(), nn.Conv2d(D, D, 1))
        nn.init.zeros_(self.mem[2].weight); nn.init.zeros_(self.mem[2].bias)
        self.mem_in = None

    def mem_tokens(self, B, dtype, device):
        """(B, Np, D) memory term (zeros when no memory input is set)."""
        if self.mem_in is None:
            return torch.zeros((B, self.Np, self.D), dtype=dtype, device=device)
        m = self.mem_in.float() * self.patch_ok.to(torch.float32)
        y = self.mem(m.to(self.mem[0].weight.dtype))                      # (B, D, nPy, nPx)
        return y.flatten(2)[..., self.patch_idx].transpose(1, 2).to(dtype)

    def tokens(self, x3d, day_ok, clim3d):
        """as TokDecNet.tokens, + the memory term on every day token before the self-attention blocks."""
        B, C, T, H, W = x3d.shape
        xf = x3d.float()
        vis = xf[:, -1]
        anom = (xf[:, -2] - clim3d.float()) * vis
        oc = self.ocean_p[:H, :W]
        x = torch.cat([anom.unsqueeze(1), vis.unsqueeze(1), xf[:, :self.n_feat] * oc, oc.expand(B, 1, T, H, W)], 1)
        x = self._pad(x.transpose(1, 2).reshape(B * T, -1, H, W))
        e = self.patch(x.to(self.patch.weight.dtype))
        e = e.flatten(2)[..., self.patch_idx].view(B, T, self.D, self.Np).permute(0, 3, 1, 2)
        emb = self.ne(e) + self.day_emb + self.pos_emb[self.patch_idx].unsqueeze(1)
        emb = emb + self.mem_tokens(B, emb.dtype, emb.device).unsqueeze(2)
        keep = day_ok.view(B, 1, 1, 1, T).expand(B, self.Np, 1, 1, T).reshape(B * self.Np, 1, 1, T)
        t = emb.reshape(B * self.Np, T, self.D)
        for blk in self.blocks:
            t = blk(t, keep)
        return emb, self.nt(t).view(B, self.Np, T, self.D), keep

    def forward(self, x3d, x_static, day_ok, clim3d):
        """as TokDecNet.forward with blend; the blend uses the days self.bsel (|offset| <= blend_k) only."""
        B, C, T, H, W = x3d.shape
        assert T == self.T and (H, W) == (self.H, self.W)
        emb, tok, keep = self.tokens(x3d, day_ok, clim3d)
        N, d = B * self.Np, self.D // self.h
        tf = tok.reshape(N, T, self.D)
        q = self.pq(tf[:, self.ti]).view(N, self.h, 1, d)
        k = self.pk(tf).view(N, T, self.h, d).transpose(1, 2)
        v = self.pv(tf).view(N, T, self.h, d).transpose(1, 2)
        att = torch.einsum("nhqd,nhsd->nhqs", q.float(), k.float()) / d ** 0.5
        att = torch.softmax(att.masked_fill(~keep, float("-inf")), -1)
        pooled = self.po(torch.einsum("nhqs,nhsd->nhqd", att.to(v.dtype), v).reshape(N, self.D))
        if self.n_spatial:
            ps = pooled.view(B, self.Np, self.D)
            for blk in self.sp_blocks:
                ps = blk(ps, self.rel_idx)
            pooled = ps.reshape(N, self.D)
        z = torch.cat([pooled, tf[:, self.ti]], -1)
        P = self.P
        y = self.dec(z).float().view(B, self.Np, self.C * P * P)
        g = self._to_grid(y).view(B, self.C, P, P, self.nPy, self.nPx)
        g = g.permute(0, 1, 4, 2, 5, 3).reshape(B, self.C, self.nPy * P, self.nPx * P)[..., :H, :W]
        dec = self.smooth(g.to(self.smooth[0].weight.dtype))
        day_t = x3d[:, :, self.ti].float()
        parts = [dec.float()]
        bs, Tb, bti = self.bsel, len(self.bsel), self.bti
        xb, cb = x3d[:, :, bs], clim3d[:, bs]
        vis = xb[:, -1].float() > 0.5
        anom = (xb[:, -2].float() - cb.float()) * vis
        s = torch.einsum("bpd,bptd->bpt", self.wq(tok[:, :, self.ti]), self.wk(tok[:, :, bs])) / self.D ** 0.5
        s_pix = self._upsample(self._to_grid(s.float()))
        offs = torch.tensor([self.offs[i] for i in bs], device=x3d.device, dtype=torch.float32)
        K = float(max(abs(o) for o in offs.tolist()))
        sst = xb[:, self.sst].float()
        pf = torch.stack([vis.float(), (offs.abs() / K).view(1, Tb, 1, 1).expand(B, Tb, H, W),
                          (offs / K).view(1, Tb, 1, 1).expand(B, Tb, H, W), (sst - sst[:, bti:bti + 1]).abs()], 2)
        p_term = self.pix(pf.view(B * Tb, 4, H, W).to(self.pix[0].weight.dtype)).view(B, Tb, H, W).float()
        ctx = torch.ones(Tb, dtype=torch.bool, device=x3d.device); ctx[bti] = False
        logit = (s_pix + p_term).masked_fill(~(vis & ctx.view(1, Tb, 1, 1)), float("-inf"))
        logit = torch.cat([logit, self.none_logit.float().view(1, 1, 1, 1).expand(B, 1, H, W)], 1)
        w = torch.softmax(logit, 1)
        blend = (w[:, :Tb] * anom).sum(1, keepdim=True)
        mass = 1 - w[:, Tb:]
        ent = -(w * torch.log(w.clamp(min=1e-12))).sum(1, keepdim=True)
        parts += [blend, mass, ent]
        rec = dict(w=w, blend=blend, mass=mass, ent=ent)
        parts += self.extra_parts(x3d, x_static, day_ok, clim3d, rec)
        h_in = torch.cat(parts + [day_t, x_static.float()], 1)
        bb = self.backbone
        y = bb(h_in.to(bb.inp.weight.dtype))
        if HOOKS["recorder"] is not None:
            if self.n_spatial:
                rec["pooled"] = pooled.view(B, self.Np, self.D)
            HOOKS["recorder"](dict(net=self, emb=emb, tok=tok, pool_w=att.mean(1)[:, 0].view(B, self.Np, T), dec=dec, x3d=x3d, **rec))
        return y


def window_offsets(before, after, blend_k, stride2):
    """Token-window day offsets: every day of t-before..t+after, or with stride2 every day to +-blend_k and even offsets beyond."""
    o = list(range(-before, after + 1))
    return [k for k in o if abs(k) <= blend_k or k % 2 == 0] if stride2 else o


def add_args(p):
    """train_pinn.py flags (default off = unchanged)."""
    p.add_argument("--physmem", action="store_true", help="io7: physics-memory tokens + token window --context-before/after with the blend on "
                   "+-(--physmem-blend-k) (chl_cnn/physmem.py; needs --t3d-stem tokdec --tokdec-blend)")
    p.add_argument("--physmem-blend-k", type=int, default=8, help="--physmem: own-pixel blend over days |d - t| <= K")
    p.add_argument("--physmem-stride2", action="store_true", help="--physmem: token window keeps every day to +-K and even offsets beyond (memory fallback)")


def check_args(p, a):
    if (a.physmem_stride2 or a.physmem_blend_k != 8) and not a.physmem:
        p.error("--physmem-stride2 / --physmem-blend-k need --physmem")
    if a.physmem and (a.model != "temporal3d" or a.t3d_stem != "tokdec" or not a.tokdec_blend or a.tokdec_krig or getattr(a, "wm_attn", False)
                      or getattr(a, "analog", False) or a.ablate or min(a.context_before, a.context_after) < a.physmem_blend_k):
        p.error("--physmem needs --model temporal3d --t3d-stem tokdec --tokdec-blend (no --tokdec-krig / --wm-attn / --analog / --ablate) and "
                "a window of at least +-(--physmem-blend-k)")
