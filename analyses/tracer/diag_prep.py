"""Inputs for the tracer diagnostics (CPU): observed ln CHL ring spectrum, climatology ring spectrum, noise nugget,
lead-0 coherence intercepts and per-wavelength e-folding times -> results/diag_prep.json.

- Observed ring power P(m), m = 1..16 (wavelength 32 / m px ~ 145..9 km): mean over spectral_motion cache rows (k = 1,
  data pairs, both days >= 90 % observed) of u_xx = ring sum of |FFT((x - Hann-weighted mean) Hann)|^2 of the day-t patch.
  Re-computed here on a 3,000-row subsample from the store as a check of the replication.
- Climatology ring power P_clim(m): same prep on the train-years doy climatology (train2.doy_climatology, 8-day bins,
  ln CHL > ln 100 removed as train_tracer) at the same patches / days, masked with the day-t observed mask.
- rho0(m) = min(exp(intercept_m), 1): lead-0 intercepts of the per-ring aligned decay fits (spectral_motion.json).
  Correlated-noise spectrum N(m) = (1 - rho0) P; signal spectrum S(m) = max(rho0 P - P_clim, 0.02 P).
- Nugget: semivariance g(h) = 0.5 mean (z(x + h) - z(x))^2 of ln CHL over ocean px pairs observed at both ends, lags
  h = 1..4 px along rows and columns pooled, every 2nd val start day (val years 2017, 2021); OLS line on h = 1..3,
  intercept = nugget variance; sigma_iid = sqrt(nugget).
Run: python -m analyses.tracer.diag_prep
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from chl_cnn import data as D, train2 as T2, store_blk4 as S4

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; RES.mkdir(exist_ok=True)
CACHE = ROOT / "analyses/blk4/results/spectral_motion_patches.npz"
SPEC = ROOT / "analyses/blk4/results/spectral_motion.json"
P, NR = 32, 16
WIN = np.outer(np.hanning(P + 2)[1:-1], np.hanning(P + 2)[1:-1])
FY, FX = np.meshgrid(np.fft.fftfreq(P), np.fft.fftfreq(P), indexing="ij")
RING = np.rint(np.hypot(FY, FX) * P).astype(int)


def prep_clear(X):
    m = np.isfinite(X)
    w = WIN * m
    mu = (w * np.where(m, X, 0)).sum((1, 2), keepdims=True) / w.sum((1, 2), keepdims=True)
    return (np.where(m, X, mu) - mu) * WIN


def ring_power(X):
    Z = np.abs(np.fft.fft2(prep_clear(X))) ** 2
    return np.stack([Z[:, RING == m].sum(1) for m in range(1, NR + 1)], 1)


def main():
    sp = S4.store_path()
    times = np.load(sp / "time.npy")
    ocean = np.load(sp / "ocean.npy").astype(bool)
    chl = np.load(sp / "chl_l3.npy", mmap_mode="r")
    out = {}
    z = np.load(CACHE)
    sel = (z["k"] == 1) & (z["null"] == 0) & z["clear"] & np.isfinite(z["u_xx"]).all(1)
    uxx = z["u_xx"][sel].astype("float64")
    out["obs_ring_power_mean"] = uxx.mean(0).tolist(); out["obs_ring_power_median"] = np.median(uxx, 0).tolist()
    out["n_patch_days"] = int(sel.sum())
    print("patch-days", sel.sum(), flush=True)
    # climatology
    y = D.log_chl(np.asarray(chl, dtype="float32"))
    chl_gt = int(np.sum(np.exp(np.nan_to_num(y, nan=-50)) > 100.0))
    y[np.exp(np.nan_to_num(y, nan=-50)) > 100.0] = np.nan
    obs = np.isfinite(y) & ocean[None]
    T2.TIMES = times
    tr, va, te = D.split_by_years(times, (2017, 2021), (2019, 2023))
    clim, dbin = T2.doy_climatology(np.nan_to_num(y, nan=0.0), obs, tr)
    print("clim done", flush=True)
    rng = np.random.default_rng(0)
    rows = np.flatnonzero(sel)
    sub = rng.choice(rows, size=min(3000, len(rows)), replace=False)
    pos = z["pos"]; col = np.searchsorted(z["pidx"], z["p"][sub]); days = z["i"][sub]
    PA = np.stack([y[t, r:r + P, c:c + P] for t, (r, c) in zip(days, pos[col])])
    PC = np.stack([clim[dbin[t], r:r + P, c:c + P] for t, (r, c) in zip(days, pos[col])])
    PC = np.where(np.isfinite(PA), PC, np.nan)
    pa, pc = ring_power(PA), ring_power(PC)
    out["check_recomputed_obs_power_subsample"] = pa.mean(0).tolist()
    out["check_cache_obs_power_subsample"] = z["u_xx"][sub].astype("float64").mean(0).tolist()
    out["check_max_rel_diff"] = float(np.max(np.abs(pa.mean(0) / z["u_xx"][sub].mean(0) - 1)))
    out["clim_ring_power_mean"] = pc.mean(0).tolist()
    out["n_subsample"] = int(len(sub))
    # coherence intercepts and e-folding
    fits = json.load(open(SPEC))["coherence"]["decay_fits"]["aligned"]["per_ring"]
    wl = [f["wavelength_km"] for f in fits]
    rho0 = np.minimum(np.exp([f["intercept"] for f in fits]), 1.0)
    efold = np.array([f["efold_days"] for f in fits])
    out.update(wavelength_km=wl, rho0=rho0.tolist(), efold_days_raw=efold.tolist())
    # e-folding used: rings >= 14.4 km as fitted; shorter wavelengths (rho there near the null) set to the 14.4 km value
    ef = efold.copy(); i14 = wl.index(14.4); ef[i14 + 1:] = ef[i14]
    out["efold_days_used"] = ef.tolist()
    Pm = np.array(out["obs_ring_power_mean"]); Pc = np.array(out["clim_ring_power_mean"])
    sig = np.maximum(rho0 * Pm - Pc, 0.02 * Pm)
    out["signal_ring_power"] = sig.tolist(); out["noise_ring_power"] = ((1 - rho0) * Pm).tolist()
    out["signal_floor_used_rings"] = [int(m + 1) for m in np.flatnonzero(rho0 * Pm - Pc < 0.02 * Pm)]
    # nugget
    vs = va[::2]
    g = np.zeros((4, 2))
    for t in vs:
        zt = np.where(obs[t], y[t], np.nan)
        for h in range(1, 5):
            for dd in (zt[:, h:] - zt[:, :-h], zt[h:, :] - zt[:-h, :]):
                f = np.isfinite(dd)
                g[h - 1] += [float((dd[f] ** 2).sum()), float(f.sum())]
    gam = 0.5 * g[:, 0] / g[:, 1]
    c = np.polyfit([1, 2, 3], gam[:3], 1)
    out["semivariance_lag1to4"] = gam.tolist(); out["semivariance_pairs"] = g[:, 1].astype(int).tolist()
    out["nugget_var"] = float(c[1]); out["sigma_iid"] = float(np.sqrt(max(c[1], 0.0))); out["nugget_days"] = int(len(vs))
    out["anomaly_var_observed_val"] = float(np.nanvar(np.where(obs[va], y[va] - clim[dbin[va]], np.nan)))
    out["chl_gt_100_removed"] = chl_gt
    (RES / "diag_prep.json").write_text(json.dumps(out, indent=1))
    for k in ("check_max_rel_diff", "sigma_iid", "nugget_var", "semivariance_lag1to4", "anomaly_var_observed_val"):
        print(k, out[k])
    print("P", np.round(Pm, 2)); print("Pclim", np.round(Pc, 2)); print("signal", np.round(sig, 2)); print("noise", np.round((1 - rho0) * Pm, 2))


if __name__ == "__main__":
    main()
