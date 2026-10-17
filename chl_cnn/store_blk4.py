"""Black Sea 4 km store: 2x2 block average of the 2 km store (store_blk.py), same layout and dtypes, so
train_pinn.py can run on it with --store blk4.

Layout ~/mind_the_chl_gap/data/blk4_store_<y0>_<y1>/ (local disk, as the 2 km store). If H or W is odd, the last
row / column of the 2 km grid is dropped (395 x 550 -> 197 x 275).
  ocean      bool: at least 2 of the 4 fine pixels are ocean
  chl_l3     float32 mg m-3: observed if at least 2 of the 4 fine pixels are observed (finite, > 0, on ocean); value
             exp(mean ln CHL over the observed fine pixels), so the trainer's ln is the mean ln; NaN otherwise
  sst32 (float32), so, mlotst, sla, u_curr, v_curr, u_wind, v_wind (float16): nanmean over the finite fine pixels
             (computed in float32; NaN if none)
  lat, lon   mean of the pairs; topo, depth, river_dist: nanmean; river_names.json copied; time unchanged (so a day
             can have no coarse CHL observation; the count is printed)
Built block by block in time from the memory-mapped 2 km arrays; each array is written to <name>.npy.part and renamed.
Run: python -m chl_cnn.store_blk4 [y0 y1]
"""
import shutil
import sys
import time
import warnings

import numpy as np

from . import data as D
from . import store_blk as SB

PERIOD = SB.PERIOD
FEATS = list(SB.VARS)          # sst32 u_curr v_curr u_wind v_wind so mlotst sla


def store_path(years=PERIOD):
    return D.DATA / f"blk4_store_{years[0]}_{years[1]}"


def _blocks(x, H2, W2):
    """(..., H, W) -> (..., H2, W2, 4) with the 2x2 fine pixels in the last axis (odd last row / column dropped)."""
    x = x[..., :2 * H2, :2 * W2]
    s = x.shape[:-2]
    return x.reshape(*s, H2, 2, W2, 2).swapaxes(-3, -2).reshape(*s, H2, W2, 4)


def coarse_nanmean(x, H2, W2):
    b = _blocks(np.asarray(x, dtype="float32"), H2, W2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, -1)


def coarse_chl(chl, ocean_f, H2, W2):
    c = np.asarray(chl, dtype="float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        l = np.log(c)
    ok = np.isfinite(l) & ocean_f[None]
    l = np.where(ok, l, 0.0)
    nb, sb = _blocks(ok, H2, W2).sum(-1), _blocks(l, H2, W2).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(sb / nb)
    out[nb < 2] = np.nan
    return out.astype("float32")


def build(years=PERIOD, block=50, verbose=True):
    t0 = time.time()
    src, out = SB.store_path(years), store_path(years)
    out.mkdir(parents=True, exist_ok=True)
    d = SB.load_store(years)
    ocean_f = d["ocean"].astype(bool)
    H, W = ocean_f.shape; H2, W2 = H // 2, W // 2
    n = len(d["time"])
    ocean4 = _blocks(ocean_f, H2, W2).sum(-1) >= 2

    def save(name, arr):
        with open(out / f"{name}.npy.part", "wb") as f:
            np.save(f, arr)
        (out / f"{name}.npy.part").rename(out / f"{name}.npy")

    save("time", d["time"]); save("ocean", ocean4)
    save("lat", d["lat"][:2 * H2].reshape(H2, 2).mean(1).astype(d["lat"].dtype))
    save("lon", d["lon"][:2 * W2].reshape(W2, 2).mean(1).astype(d["lon"].dtype))
    for k in ["topo", "depth", "river_dist"]:
        if (src / f"{k}.npy").exists():
            a = np.load(src / f"{k}.npy")
            save(k, coarse_nanmean(a, H2, W2).astype(a.dtype))
    if (src / "river_names.json").exists():
        shutil.copy(src / "river_names.json", out / "river_names.json")

    todo = [v for v in ["chl_l3"] + FEATS if not (out / f"{v}.npy").exists()]
    if verbose:
        print(f"grid {H} x {W} -> {H2} x {W2}; {n} days; ocean px {int(ocean_f.sum())} -> {int(ocean4.sum())}; building {todo}", flush=True)
    for v in todo:
        dt = d[v].dtype
        mm = np.lib.format.open_memmap(out / f"{v}.npy.part", mode="w+", dtype=dt, shape=(n, H2, W2))
        for b0 in range(0, n, block):
            x = d[v][b0:b0 + block]
            mm[b0:b0 + block] = coarse_chl(x, ocean_f, H2, W2) if v == "chl_l3" else coarse_nanmean(x, H2, W2).astype(dt)
        mm.flush(); del mm
        (out / f"{v}.npy.part").rename(out / f"{v}.npy")
        if verbose:
            print(f"  {v} done {time.time() - t0:.0f} s", flush=True)
    if verbose:
        print(f"wrote {out} in {time.time() - t0:.0f} s", flush=True)


def load_store(years=PERIOD):
    return SB.load_store(years, path=store_path(years))


if __name__ == "__main__":
    y = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else PERIOD
    build(y)
