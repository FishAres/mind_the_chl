"""Precomputed prior-knowledge fields for the PINN physics loss (train_pinn.py --src-* / --phys-weight).

ekman_lag_channels   w_E+ (upward Ekman pumping from the 10 m winds, m/day, clipped at 0) averaged over
                     lag windows ending at day t: [0-3], [4-10], [11-20], [21-30] days back. Calendar-aware
                     (the store skips days without CHL). Cached as <store>/wE_pos_lags.npy, float16 (n, 4, H, W).
equator_taper        weight 0 for |lat| < lat0, rising linearly to 1 at lat1 (Coriolis -> 0 at the equator).
courant_weight       1 / max(1, Co_clim) with Co_clim the training-day mean Courant number of the surface
                     currents: full weight in the interior, downweighted in the boundary currents.
"""
import numpy as np

from . import store as S
from .physics import ekman, courant

LAG_WINDOWS = [(0, 3), (4, 10), (11, 20), (21, 30)]


def ekman_lag_channels(d, windows=LAG_WINDOWS, lat_min=3.0, verbose=True):
    out = S.store_path() / "wE_pos_lags.npy"
    if out.exists():
        return np.load(out, mmap_mode="r")
    times = np.asarray(d["time"]).astype("datetime64[D]")
    lat, lon = d["lat"].astype("float64"), d["lon"].astype("float64")
    n, H, W = d["u_wind"].shape
    day0 = times[0]; ncal = int((times[-1] - day0).astype(int)) + 1
    pos = (times - day0).astype(int)
    # cumulative sum of w_E+ over calendar days (missing store days count as 0 with a presence counter)
    csum = np.zeros((ncal + 1, H, W), dtype="float64"); cnt = np.zeros(ncal + 1, dtype="float64")
    wpos = np.zeros((ncal, H, W), dtype="float32"); have = np.zeros(ncal, dtype=bool)
    for b0 in range(0, n, 500):
        u = np.asarray(d["u_wind"][b0:b0 + 500]).astype("float32"); v = np.asarray(d["v_wind"][b0:b0 + 500]).astype("float32")
        for i in range(len(u)):
            _, _, w, _, _ = ekman(u[i], v[i], lat, lon, lat_min=lat_min)
            wpos[pos[b0 + i]] = np.nan_to_num(np.maximum(w, 0) * 86400.0, nan=0.0)   # m/day, equator band -> 0
            have[pos[b0 + i]] = True
        if verbose and (b0 // 500) % 4 == 0:
            print(f"  ekman {str(times[b0])[:10]}", flush=True)
    csum[1:] = np.cumsum(wpos, axis=0); cnt[1:] = np.cumsum(have)
    res = np.zeros((n, len(windows), H, W), dtype="float16")
    for k, (l0, l1) in enumerate(windows):
        hi = np.clip(pos - l0 + 1, 0, ncal); lo = np.clip(pos - l1, 0, ncal)
        num = csum[hi] - csum[lo]; den = np.maximum(cnt[hi] - cnt[lo], 1)
        res[:, k] = (num / den[:, None, None]).astype("float16")
    np.save(out, res)
    if verbose:
        print("wrote", out, res.shape)
    return np.load(out, mmap_mode="r")


def equator_taper(lat, lat0=3.0, lat1=5.0):
    return np.clip((np.abs(lat) - lat0) / (lat1 - lat0), 0, 1).astype("float32")


def courant_weight(d, train_idx, step=10):
    lat, lon = d["lat"].astype("float64"), d["lon"].astype("float64")
    idx = train_idx[::step]
    acc = np.zeros(d["ocean"].shape); cnt = np.zeros_like(acc)
    for b0 in range(0, len(idx), 200):
        sel = idx[b0:b0 + 200]
        u = np.asarray(d["u_curr"][sel]).astype("float32"); v = np.asarray(d["v_curr"][sel]).astype("float32")
        co = courant(u, v, lat, lon)
        ok = np.isfinite(co)
        acc += np.where(ok, co, 0).sum(0); cnt += ok.sum(0)
    co_clim = acc / np.maximum(cnt, 1)
    co_clim[cnt == 0] = np.nan
    w = 1.0 / np.maximum(1.0, np.nan_to_num(co_clim, nan=1.0))
    return w.astype("float32"), co_clim.astype("float32")
