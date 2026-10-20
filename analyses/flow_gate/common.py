"""Shared helpers for the flow-gate checks (CPU only). Store access, crop, regions, seasons, x0.25 advection, bootstrap."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from chl_cnn import data as D, tracer as TR, store_blk4 as S4
from chl_cnn.physics import grid_metrics

torch.set_num_threads(int(os.environ.get("FLOW_GATE_THREADS", "4")))
HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
VAL_YEARS, TEST_YEARS = (2017, 2021), (2019, 2023)
REG_NAMES = {1: "NW shelf", 2: "W open", 3: "E open", 4: "Anatolian", 5: "Crimea/Caucasus", 6: "Azov", 7: "Marmara"}
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEAS = ["DJF", "MAM", "JJA", "SON"]
MIN_N = 5
A_CUR = 0.25          # x0.25 reference
N_SUB, VMAX = 4, 10.0  # as train_tracer / band_train


def load_store():
    """Memory-mapped blk4 store + the train_pinn --crop slices."""
    d = S4.load_store()
    oc = np.asarray(d["ocean"]).astype(bool)
    rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
    ys = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0]))
    xs = slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
    ocean = oc[ys, xs]
    lat = np.asarray(d["lat"])[ys].astype("float64"); lon = np.asarray(d["lon"])[xs].astype("float64")
    assert lat[1] > lat[0] and ocean.shape == (177, 273), ocean.shape
    dx, dy, _, _ = grid_metrics(lat, lon)
    times = np.asarray(d["time"]).astype("datetime64[D]")
    tr, va, te = D.split_by_years(times, VAL_YEARS, TEST_YEARS)
    return dict(d=d, ys=ys, xs=xs, ocean=ocean, lat=lat, lon=lon, dx=dx, dy=dy, times=times, tr=tr, va=va, te=te,
                tday=times.astype(int))


def regions(ocean):
    from analyses.blk4.error_breakdown import regions_4km
    reg, ties = regions_4km(ocean)
    return reg, ties


def season_of(times):
    m = np.asarray(times).astype("datetime64[M]").astype(int) % 12 + 1
    return np.array([SEASONS[int(x)] for x in m])


def neighbour_index(tday, k):
    """Store index of the calendar day k days after each store day, -1 if missing (= train2.neighbour_index)."""
    lut = {int(t): i for i, t in enumerate(tday)}
    return np.array([lut.get(int(t) + k, -1) for t in tday], dtype=np.int64)


def currents_px(S, i):
    """x0.25 velocity (px/day, x = col, y = row) of store day i on the crop, and validity (finite & ocean)."""
    d, ys, xs = S["d"], S["ys"], S["xs"]
    u = np.asarray(d["u_curr"][i, ys, xs], "float32"); v = np.asarray(d["v_curr"][i, ys, xs], "float32")
    ok = np.isfinite(u) & np.isfinite(v) & S["ocean"]
    U = np.stack([np.where(ok, u, 0) * 86400.0 / S["dx"], np.where(ok, v, 0) * 86400.0 / S["dy"]]).astype("float32") * A_CUR
    return U, ok


def advect(S, it, days, y_src, obs_src, cache=None):
    """Semi-Lagrangian x0.25 prediction for the target day days[-1] from the source day days[0].
    days: store indices of consecutive calendar days ordered from SOURCE to TARGET (either time direction).
    If the source is later than the target (days decreasing), the velocity is negated: the backward-in-tau integration
    then follows the flow forward in real time (parcel at x on the target day -> its position on the source day).
    Returns (pred, valid) on the crop; valid = trajectory valid (Integrator min_valid 0.99) & departure observed frac >= 0.99."""
    sign = 1.0 if days[-1] > days[0] else -1.0
    Us, oks = [], []
    for i in days:
        if cache is not None and i in cache:
            U, ok = cache[i]
        else:
            U, ok = currents_px(S, i)
            if cache is not None:
                cache[i] = (U, ok)
        Us.append(sign * U); oks.append(ok)
    U = torch.as_tensor(np.stack(Us)[None]); ok = torch.as_tensor(np.stack(oks)[None].astype("float32"))
    k = len(days) - 1
    with torch.no_grad():
        pos, val = it.departure(U, ok, k)
        pred, v2 = it.depart_sample(torch.as_tensor(np.nan_to_num(y_src)[None].astype("float32")),
                                    torch.as_tensor(obs_src[None]), pos)
    return pred[0].numpy(), (val & v2)[0].numpy()


def integrator(S):
    H, W = S["ocean"].shape
    return TR.Integrator(H, W, torch.device("cpu"), n_sub=N_SUB, vmax=VMAX)


def boot_ratio_diff(day_ids, sa, sb, n, nb=2000, seed=0):
    """Day-bootstrap 95 % CI of (sum sa - sum sb) / sum n; arrays per (day, ...) row, resampled by day."""
    uday, inv = np.unique(day_ids, return_inverse=True)
    A = np.bincount(inv, sa, len(uday)); B = np.bincount(inv, sb, len(uday)); N = np.bincount(inv, n, len(uday))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uday), (nb, len(uday)))
    w = np.stack([np.bincount(r, minlength=len(uday)) for r in idx]).astype("float64")
    est = (w @ (A - B)) / np.maximum(w @ N, 1)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"
