"""Black Sea physics memory (blk8 track, bridge #29; run blk8_spatial_bsmem; train_pinn.py --bsmem, default off; --store blk4 only).

Model: chl_cnn/physmem.PhysMemTokDecNet with mem_ch = 12 and the ordinary token window (t-8..t+8 for blk4_tokdec_blend_spatial, so the
blend covers every window day and the net equals TokDecNet(blend) apart from the memory term): 12 past-only summary channels on the P x P
patch grid -> Conv2d(12, D, 3) -> GELU -> Conv2d(D, D, 1) (zero-init) -> + every day token of the patch before the temporal self-attention.

Daily fields (store days; cached in ~/mind_the_chl_gap/data/blk8/):
  Q_r     river discharge (kg s-1) of r in RIVERS = Danube, Dniester, Dnieper-Bug: sum over the river's mouth px of friver x px area
          (store friver = CMEMS river freshwater flux, mapped conservatively onto 4 km mouth px, chl_cnn/store_blk4_extra.py). Mouth px of r:
          px with friver > 0 on any store day whose nearest of the 8 store mouths (river_dist, km) is r and within MOUTH_KM (40 km).
          NaN on a day when every mouth px of r is NaN.
  mld     mlotst (m), patch mean over ocean px with finite values
  swr     rsntds (W m-2, net downward shortwave at the surface), patch mean; -> anomaly vs the DOY climatology of the training days
          (per patch, 31-day circular running mean over the DOY means), as physmem's SSTA
Summaries for target day t (calendar days, store days only; day t and later never enter):
  windows W30 = [t-30, t-1], W60 = [t-60, t-1]
  river_r_W   z(log mean Q_r over W) x prox_r; z over the training days; prox_r = patch mean over ocean px of exp(-river_dist_r / PROX_KM),
              PROX_KM = 100 km. Needs >= half the window's days with Q_r; else 0.                                    6 channels
  dmld_W      MLD deepening: W30: mean[t-15, t-1] - mean[t-30, t-16]; W60: mean[t-30, t-1] - mean[t-60, t-31] (+ = deepening);
              a half-window mean needs >= half its days finite.                                                      2 channels
  swr_W       mean SW anomaly over W (>= half the days finite).                                                      2 channels
  miss_W      fraction of W's calendar days missing from the store (raw, 0 = complete).                             2 channels
  dmld / swr are z-scored over the training days and ocean patches where valid; river / dmld / swr clipped to +-6; invalid / land 0.
Not available on disk (not used): GloFAS discharge; net surface heat flux (hfds) and its longwave / latent / sensible parts (the store has
only rsntds, shortwave).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import physics as PH
from .physmem import patch_mean, _doy

CACHE = Path.home() / "mind_the_chl_gap" / "data" / "blk8"
RIVERS = ("Danube", "Dniester", "Dnieper-Bug")
MOUTH_KM, PROX_KM = 40.0, 100.0
NAMES = ["danube_30", "dniester_30", "dnieper_30", "danube_60", "dniester_60", "dnieper_60", "dmld_30", "dmld_60", "swr_30", "swr_60",
         "miss_30", "miss_60"]
N_MEM = len(NAMES)
RESUME_DEFAULTS = {"bsmem": False}


def add_args(p):
    p.add_argument("--bsmem", action="store_true", help="blk8: Black Sea physics-memory tokens (river discharge x mouth proximity, MLD deepening, "
                   "shortwave anomaly; 30- / 60-day past means; chl_cnn/bsmem.py); needs --store blk4 --t3d-stem tokdec --tokdec-blend")


def check_args(p, a):
    if a.bsmem and (a.store != "blk4" or a.model != "temporal3d" or a.t3d_stem != "tokdec" or not a.tokdec_blend or a.tokdec_krig or a.physmem
                    or getattr(a, "wm_attn", False) or getattr(a, "analog", False) or a.ablate):
        p.error("--bsmem needs --store blk4 --model temporal3d --t3d-stem tokdec --tokdec-blend (no --physmem / --tokdec-krig / --wm-attn / "
                "--analog / --ablate)")


def mouths(fr_any, river_dist, names):
    """{river: (rows, cols)} mouth px per RIVERS entry and a report. fr_any (H, W) bool: friver > 0 on any day."""
    rd = np.asarray(river_dist, "float64")
    near = np.argmin(np.where(np.isfinite(rd), rd, np.inf), 0)
    dmin = np.nanmin(rd, 0)
    out, rep = {}, {}
    for r in RIVERS:
        k = names.index(r)
        m = fr_any & (near == k) & (dmin <= MOUTH_KM)
        out[r] = np.nonzero(m)
        rep[r] = dict(n_px=int(m.sum()), max_dist_km=float(dmin[m].max()) if m.any() else None)
    return out, rep


def daily_fields(rd, n, ocean, lat, lon, P, river_dist, names, chunk=250, log=print):
    """-> Q (n, 3) kg s-1, mld (n, nPy, nPx), swr (n, nPy, nPx) float32, prox (3, nPy, nPx), report (cached)."""
    key = hashlib.md5(np.ascontiguousarray(ocean).tobytes() + lat.tobytes() + lon.tobytes() + np.ascontiguousarray(river_dist).tobytes()
                      + f"{n} {P} {MOUTH_KM} {PROX_KM} {names}".encode()).hexdigest()[:10]
    f = CACHE / f"bsmem_daily_P{P}_{key}.npz"
    if f.exists():
        z = np.load(f)
        return z["Q"], z["mld"], z["swr"], z["prox"], json.loads(str(z["report"]))
    dx, dy, _, _ = PH.grid_metrics(lat, lon)
    area = np.abs(np.broadcast_to(dx, ocean.shape) * np.broadcast_to(dy, ocean.shape))
    fr_any = np.zeros(ocean.shape, bool)
    for b0 in range(0, n, chunk):
        fr_any |= np.nan_to_num(rd("friver", slice(b0, min(b0 + chunk, n))).astype("float64")).max(0) > 0
    mo, rep = mouths(fr_any, river_dist, names)
    Q, mld, swr = np.full((n, len(RIVERS)), np.nan), [], []
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(b0 + chunk, n))
        fr = rd("friver", sl).astype("float64")
        for i, r in enumerate(RIVERS):
            v = fr[:, mo[r][0], mo[r][1]] * area[mo[r]]
            Q[sl, i] = np.where(np.isfinite(v).any(1), np.nansum(v, 1), np.nan)
        del fr
        for name, acc in (("mlotst", mld), ("rsntds", swr)):
            x = rd(name, sl).astype("float32")
            acc.append(patch_mean(x, np.isfinite(x) & ocean, P).astype("float32"))
        log(f"bsmem: daily fields {sl.stop}/{n}")
    mld, swr = np.concatenate(mld), np.concatenate(swr)
    oc = ocean.astype("float64")
    prox = np.stack([patch_mean(np.exp(-np.asarray(river_dist[names.index(r)], "float64") / PROX_KM)[None], ocean[None], P)[0] for r in RIVERS])
    prox = np.nan_to_num(prox).astype("float32")
    rep.update({f"Q_mean_kg_s_{r}": float(np.nanmean(Q[:, i])) for i, r in enumerate(RIVERS)})
    rep.update({f"Q_nan_days_{r}": int(np.isnan(Q[:, i]).sum()) for i, r in enumerate(RIVERS)})
    rep.update(swr_all_nan_days=int(np.isnan(swr).all((1, 2)).sum()), mld_all_nan_days=int(np.isnan(mld).all((1, 2)).sum()), ocean_px=int(oc.sum()))
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez(tmp, Q=Q.astype("float32"), mld=mld, swr=swr, prox=prox, report=json.dumps(rep)); tmp.rename(f)
    return Q.astype("float32"), mld, swr, prox, rep


class _Windows:
    """Means over calendar-day windows ending before each store day (past-only), from cumulative sums on the calendar axis."""

    def __init__(self, times):
        self.cal = (np.asarray(times) - np.asarray(times)[0]).astype("timedelta64[D]").astype(int)
        self.Nc = self.cal[-1] + 1
        present = np.zeros(self.Nc, bool); present[self.cal] = True
        self.CP = np.concatenate([[0], np.cumsum(present)])

    def prep(self, x):
        V = np.full((self.Nc,) + x.shape[1:], np.nan); V[self.cal] = x
        fin = np.isfinite(V)
        CS = np.concatenate([np.zeros((1,) + V.shape[1:]), np.cumsum(np.where(fin, V, 0.0), 0)])
        CN = np.concatenate([np.zeros((1,) + V.shape[1:], np.int32), np.cumsum(fin, 0, dtype=np.int32)])
        return CS, CN

    def mean(self, cs_cn, lo, hi, need):
        """mean over calendar days [cal - lo, cal - hi] (lo >= hi >= 1) with >= need finite days, else NaN."""
        CS, CN = cs_cn
        a_, b_ = np.clip(self.cal - lo, 0, None), np.clip(self.cal - hi + 1, 0, None)
        sm, cn = CS[b_] - CS[a_], CN[b_] - CN[a_]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cn >= need, sm / np.maximum(cn, 1), np.nan)

    def missing(self, lo, hi):
        a_, b_ = np.clip(self.cal - lo, 0, None), np.clip(self.cal - hi + 1, 0, None)
        return 1.0 - (self.CP[b_] - self.CP[a_]) / float(lo - hi + 1)


def summaries(Q, mld, swr, prox, times, tr, patch_ok):
    """(n, 12, nPy, nPx) float16 summary table (module doc) and a stats dict."""
    n = len(times)
    pm = np.asarray(patch_ok, bool)
    win = _Windows(times)
    swr = swr.astype("float64").copy()                         # SW anomaly: DOY climatology of the training days, 31-day circular mean
    doy = _doy(times)
    s, c = np.zeros((366,) + swr.shape[1:]), np.zeros((366,) + swr.shape[1:])
    ok = np.isfinite(swr[tr])
    np.add.at(s, doy[tr], np.where(ok, swr[tr], 0.0)); np.add.at(c, doy[tr], ok)
    ker = lambda a_: sum(np.roll(a_, k, 0) for k in range(-15, 16))
    swr = swr - (ker(s) / np.maximum(ker(c), 1))[doy]
    out = np.zeros((n, N_MEM) + pm.shape, dtype=np.float32)
    stats = {}
    cq, cm, cw = win.prep(Q.astype("float64")), win.prep(mld.astype("float64")), win.prep(swr)
    for w, L in enumerate((30, 60)):
        lq = np.log(np.maximum(win.mean(cq, L, 1, L // 2), 1e-12))                     # (n, 3) log mean discharge; NaN where too few days
        lq = np.where(np.isfinite(win.mean(cq, L, 1, L // 2)), lq, np.nan)
        for i, r in enumerate(RIVERS):
            x = lq[tr, i]; x = x[np.isfinite(x)]
            mu, sd = float(x.mean()), float(x.std()) + 1e-12
            z = np.clip((lq[:, i] - mu) / sd, -6, 6)
            k = 3 * w + i
            out[:, k] = np.where(np.isfinite(z), z, 0.0)[:, None, None] * prox[i][None]
            stats[NAMES[k]] = dict(log_q_mean=mu, log_q_std=sd, valid_frac_train=float(np.isfinite(lq[tr, i]).mean()))
        h = L // 2
        dm = win.mean(cm, h, 1, h // 2 + (h % 2)) - win.mean(cm, L, h + 1, h // 2 + (h % 2))  # second half - first half
        sw = win.mean(cw, L, 1, L // 2)
        for k, x in ((6 + w, dm), (8 + w, sw)):
            v = x[tr][:, pm]; v = v[np.isfinite(v)]
            mu, sd = float(v.mean()), float(v.std()) + 1e-12
            stats[NAMES[k]] = dict(mean=mu, std=sd, valid_frac_train=float(np.isfinite(x[tr][:, pm]).mean()))
            out[:, k] = np.where(np.isfinite(x), np.clip((x - mu) / sd, -6, 6), 0.0)
        out[:, 10 + w] = win.missing(L, 1)[:, None, None]
    out = np.where(pm, out, 0.0).astype("float16")
    return out, stats


class BSMemTable:
    """Per-day summary table on the CPU; batch(bi) -> (B, 12, nPy, nPx) float32 on bi's device."""

    def __init__(self, rd, times, tr, ocean, lat, lon, P, river_dist, names, log=print):
        oc = torch.as_tensor(np.asarray(ocean)).float()
        H, W = oc.shape
        oc = F.pad(oc, (0, -(-W // P) * P - W, 0, -(-H // P) * P - H))
        self.patch_ok = (F.avg_pool2d(oc[None, None], P)[0, 0] > 0).numpy()
        Q, mld, swr, prox, self.report = daily_fields(rd, len(times), np.asarray(ocean, bool), np.asarray(lat, "float64"),
                                                      np.asarray(lon, "float64"), P, np.asarray(river_dist), list(names), log=log)
        tab, self.stats = summaries(Q, mld, swr, prox, times, np.asarray(tr), self.patch_ok)
        self.tab = torch.from_numpy(tab)
        log(f"bsmem: {N_MEM} summary channels on a {tab.shape[2]} x {tab.shape[3]} patch grid, {len(times)} days; mouths "
            + ", ".join(f"{r} {self.report[r]['n_px']} px (Q mean {self.report[f'Q_mean_kg_s_{r}']:.3g} kg/s)" for r in RIVERS)
            + "; valid frac (train) " + ", ".join(f"{k} {v['valid_frac_train']:.3f}" for k, v in self.stats.items()))

    def batch(self, bi):
        return self.tab[bi.cpu()].to(bi.device).float()
