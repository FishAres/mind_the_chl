"""Static-base + advected-anomaly tracer (hypothesis test; nothing in tracer.py / train_tracer.py depends on this file).

    ln C = B + A;  B = a slowly varying base, A = ln C - B the anomaly carried by the currents.
    Forecast from day t to day t+k:  ln C^(x, t+k) = B(x, t+k) + A(X_k(x), t),  A(., t) = ln C_obs(., t) - B(., t)
    X_k(x) = departure point at day t of the backward trajectory from x at day t+k (tracer.Integrator, unchanged).

Bases (all on the cropped 4 km grid of train_tracer.load; y = ln CHL with 0 where unobserved, obs bool):
  doy   day-of-year climatology over TRAIN days: per pixel and doy bin (366 bins) sum and count of observed ln C,
        circular Gaussian smoothing along doy (sigma_days, default 15 d) of both, mean = smoothed sum / smoothed count
        (NaN where smoothed count < 1e-3). B(x, t+k) = clim(doy(t+k)).
  harm  per-pixel least squares on TRAIN observed ln C: m + c1 sin w + c2 cos w + c3 sin 2w + c4 cos 2w (w = 2 pi doy /
        365.25); NaN where a pixel has < min_obs train observations. B(x, t+k) = fit at doy(t+k).
  run   causal running mean of observed ln C over the previous `window` calendar days [t - window, t - 1] (day t itself
        excluded); where fewer than min_obs observations, the doy base is used instead (fallback flag kept). The base is
        held fixed over the lead: B(., t+k) = B(., t) (no information after day t).
  lp    spatial low-pass of day t: normalised convolution (Gaussian, sigma = 0.1874 lam_km / 4.5 km px, i.e. 0.5 amplitude
        response at lam_km = 150 km; zero padding; masked px never enter): G*(m y) / G*m; where G*m < min_w the doy base is
        used instead (fallback flag kept). Held fixed over the lead as `run`.
"""
import math

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .tracer import SIGMA_PER_WAVELENGTH

BASES = ["doy", "harm", "run", "lp"]
HELD = {"doy": False, "harm": False, "run": True, "lp": True}        # True: B(t+k) = B(t)


def doy_of(times):
    t = np.asarray(times)
    return (t - t.astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)


def base_doy(y, obs, train_idx, doy, sigma_days=15.0, chunk=200):
    """(366, H, W) float32 doy climatology over train days, and the raw per-bin count (366, H, W)."""
    T, H, W = y.shape
    s = np.zeros((366, H, W)); n = np.zeros((366, H, W))
    tr = np.asarray(train_idx)
    for b0 in range(0, len(tr), chunk):
        ii = tr[b0:b0 + chunk]
        o = np.asarray(obs[ii]); v = np.where(o, np.asarray(y[ii], "float64"), 0.0)
        np.add.at(s, doy[ii], v); np.add.at(n, doy[ii], o.astype("float64"))
    ss = gaussian_filter1d(s, sigma_days, axis=0, mode="wrap"); ns = gaussian_filter1d(n, sigma_days, axis=0, mode="wrap")
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(ns > 1e-3, ss / ns, np.nan)
    return clim.astype("float32"), n.astype("int32")


def base_harm(y, obs, train_idx, doy, min_obs=30, chunk=200):
    """Per-pixel annual + semi-annual harmonic fit on train obs. Returns (366, H, W) float32 evaluated at doy 0..365,
    the coefficients (5, H, W) and the per-pixel train obs count."""
    T, H, W = y.shape
    G = np.zeros((5, 5, H, W)); r = np.zeros((5, H, W)); cnt = np.zeros((H, W))
    tr = np.asarray(train_idx)
    for b0 in range(0, len(tr), chunk):
        ii = tr[b0:b0 + chunk]
        w = 2 * np.pi * doy[ii] / 365.25
        X = np.stack([np.ones_like(w), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)], 1)        # (n, 5)
        o = np.asarray(obs[ii]).astype("float64"); v = np.where(o > 0, np.asarray(y[ii], "float64"), 0.0)
        XX = (X[:, :, None] * X[:, None, :]).reshape(len(ii), 25)
        G += np.tensordot(XX, o, axes=(0, 0)).reshape(5, 5, H, W); r += np.tensordot(X, v, axes=(0, 0)); cnt += o.sum(0)
    ok = cnt >= min_obs
    coef = np.full((5, H, W), np.nan)
    Gm = np.moveaxis(G[:, :, ok], -1, 0); rm = np.moveaxis(r[:, ok], -1, 0)
    coef[:, ok] = np.linalg.solve(Gm, rm[..., None])[..., 0].T
    w = 2 * np.pi * np.arange(366) / 365.25
    X = np.stack([np.ones_like(w), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)], 1)
    return np.einsum("di,ihw->dhw", X, coef).astype("float32"), coef.astype("float32"), cnt.astype("int32")


def base_run(y, obs, times, window=20, min_obs=3):
    """Causal running mean over calendar days [t - window, t - 1]. Returns (T, H, W) float32 (NaN where < min_obs obs)
    and the obs count (T, H, W) int16."""
    T, H, W = y.shape
    dn = np.asarray(times).astype("datetime64[D]").astype(int)
    lo = np.searchsorted(dn, dn - window, "left"); hi = np.searchsorted(dn, dn - 1, "right")      # store idx [lo, hi)
    out = np.full((T, H, W), np.nan, "float32"); cnt = np.zeros((T, H, W), "int16")
    pre_s = np.zeros((T + 1, H, W), "float64"); pre_n = np.zeros((T + 1, H, W), "int32")
    for i in range(T):
        o = np.asarray(obs[i]); pre_s[i + 1] = pre_s[i] + np.where(o, np.asarray(y[i], "float64"), 0.0); pre_n[i + 1] = pre_n[i] + o
    for i in range(T):
        n = pre_n[hi[i]] - pre_n[lo[i]]; s = pre_s[hi[i]] - pre_s[lo[i]]
        cnt[i] = n
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(n >= min_obs, s / np.maximum(n, 1), np.nan)
    return out, cnt


def base_lp(y, obs, lam_km=150.0, px_km=4.5, min_w=0.05):
    """Normalised-convolution Gaussian low-pass of each day. Returns (T, H, W) float32 (NaN where G*m < min_w) and the
    sigma in px."""
    T, H, W = y.shape
    sig = SIGMA_PER_WAVELENGTH * lam_km / px_km
    out = np.full((T, H, W), np.nan, "float32")
    for i in range(T):
        m = np.asarray(obs[i]).astype("float64")
        num = gaussian_filter(np.where(m > 0, np.asarray(y[i], "float64"), 0.0), sig, mode="constant", truncate=4.0)
        den = gaussian_filter(m, sig, mode="constant", truncate=4.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(den >= min_w, num / den, np.nan)
    return out, sig


def with_fallback(B, Bdoy, doy):
    """B (T, H, W) with NaN -> doy base of the same day. Returns (filled, fallback bool)."""
    fb = ~np.isfinite(B)
    out = B.copy()
    for i in np.flatnonzero(fb.reshape(len(B), -1).any(1)):
        out[i][fb[i]] = Bdoy[doy[i]][fb[i]]
    return out, fb


def predict(it, pos, y_t, obs_t, B_t, B_tk):
    """Base + advected anomaly. y_t, obs_t (N, H, W) day-t ln C / observed; B_t (N, H, W) base at day t (NaN allowed:
    such px are treated as unobserved for the anomaly); B_tk (N, H, W) base at the arrival px and day t+k; pos departure
    points from Integrator.departure. Returns (prediction, valid) with valid = departure observed fraction >= min_obs and
    B_tk finite."""
    ok = obs_t & torch.isfinite(B_t)
    A = torch.where(ok, y_t - torch.nan_to_num(B_t), torch.zeros_like(y_t))
    As, v = it.depart_sample(A, ok, pos)
    return torch.nan_to_num(B_tk) + As, v & torch.isfinite(B_tk)


BASE_DIR_NAME = "static_bases"


def bases_cached(y, obs, times, train_idx, out_dir, window=20, lam_km=150.0, sigma_days=15.0, rebuild=False, log=print):
    """Build (or load) the four bases on the given (cropped) grid. y (T, H, W) float32 ln C (0 where unobserved), obs bool.
    Files in out_dir: doy.npy, harm.npy (366, H, W); run.npy, lp.npy (T, H, W) with the doy fallback applied;
    run_fb.npy, lp_fb.npy (T, H, W) bool fallback flags; info.json. Returns dict of memmaps + info."""
    import json, time
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    names = ["doy", "harm", "run", "lp", "run_fb", "lp_fb"]
    if rebuild or not all((out_dir / f"{n}.npy").exists() for n in names) or not (out_dir / "info.json").exists():
        t0 = time.time(); info = {}
        doy = doy_of(times)
        Bd, cnt = base_doy(y, obs, train_idx, doy, sigma_days)
        np.save(out_dir / "doy.npy", Bd); log(f"doy {time.time() - t0:.0f}s")
        Bh, coef, hc = base_harm(y, obs, train_idx, doy)
        np.save(out_dir / "harm.npy", Bh); log(f"harm {time.time() - t0:.0f}s")
        Br, rc = base_run(y, obs, times, window)
        Br, fbr = with_fallback(Br, Bd, doy)
        np.save(out_dir / "run.npy", Br); np.save(out_dir / "run_fb.npy", fbr); del Br, rc; log(f"run {time.time() - t0:.0f}s")
        Bl, sig = base_lp(y, obs, lam_km)
        Bl, fbl = with_fallback(Bl, Bd, doy)
        np.save(out_dir / "lp.npy", Bl); np.save(out_dir / "lp_fb.npy", fbl); del Bl; log(f"lp {time.time() - t0:.0f}s")
        info.update(window_days=window, run_min_obs=3, lam_km=lam_km, lp_sigma_px=sig, lp_min_w=0.05, doy_sigma_days=sigma_days,
                    harm_min_obs=30, n_train_days=int(len(train_idx)),
                    doy_nan_px_any_bin=int((~np.isfinite(Bd)).any(0).sum()), harm_nan_px=int((~np.isfinite(Bh[0])).sum()),
                    build_s=round(time.time() - t0))
        (out_dir / "info.json").write_text(json.dumps(info, indent=1))
    out = {n: np.load(out_dir / f"{n}.npy", mmap_mode="r") for n in names}
    out["info"] = json.loads((out_dir / "info.json").read_text())
    return out
