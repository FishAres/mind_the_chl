"""Data subset, caching, splits and normalisation.

Mirrors the geo-smart "Simple CNN" baseline (book/notebooks/Data_Prep.ipynb):
  region  lat 35 -> -5 (the grid starts at 32N, so 32N..5S), lon 45 -> 90   -> 149 x 181
  time    2020, days where CHL L3 has at least one observed pixel
  target  ln(CHL_cmes-level3)
  split   chronological 70 / 20 / 10

The subset is cached as one npz on local disk (~/mind_the_chl_gap/data/) so
training never touches the 66 GB zarr again.
"""
from pathlib import Path

import numpy as np
import xarray as xr

DATA = Path.home() / "mind_the_chl_gap" / "data"
SRC = DATA / "IO.zarr"

REGION = dict(lat=slice(35, -5), lon=slice(45, 90))
YEARS = ("2020-01-01", "2020-12-31")
LAND_VALUE = 2
CHL_L3 = "CHL_cmes-level3"
CHL_L4 = "CHL_cmes-gapfree"
FEATURE_VARS = ["sst", "so", "air_temp", "u_wind", "v_wind", "sla", "mlotst", "topo"]
SPLIT = (0.7, 0.2, 0.1)


def cache_path(years=YEARS, variables=None):
    tag = "" if variables is None else "_" + "-".join(variables)
    return DATA / f"cnn_subset_{years[0][:4]}_{years[1][:4]}{tag}.npz"


def build_subset(years=YEARS, region=REGION, variables=None, verbose=True, block=400):
    """Load the baseline subset from IO.zarr and cache it. Returns the dict of arrays.

    variables: feature variables to load (default FEATURE_VARS + CHL L4). Loads in blocks of
    `block` days so peak memory stays near the size of the output (~1 GB per variable for 1997-2020).
    """
    out = cache_path(years, variables)
    if out.exists():
        if verbose:
            print(f"loading cached subset {out}")
        return dict(np.load(out, allow_pickle=False))
    feats = FEATURE_VARS if variables is None else list(variables)
    extra = [CHL_L4] if variables is None else []
    ds = xr.open_zarr(SRC, consolidated=True).sel(**region).sel(time=slice(*years))
    has_obs = ds[CHL_L3].notnull().any(["lat", "lon"]).compute().values
    keep = np.flatnonzero(has_obs)
    if verbose:
        print(f"{len(keep)} / {len(has_obs)} days with CHL observations")
    n, ny, nx = len(keep), ds.sizes["lat"], ds.sizes["lon"]
    arrays = {"time": ds.time.values[keep], "lat": ds.lat.values.astype("float32"), "lon": ds.lon.values.astype("float32"),
              "ocean": ((ds["CHL_cmes-land"] != LAND_VALUE) & (ds["topo"] <= 0)).values}
    names = {"chl_l3": CHL_L3, "chl_l4": CHL_L4}
    todo = ["chl_l3"] + (["chl_l4"] if extra else []) + [f for f in feats if f != "topo"]
    for k in todo:
        arrays[k] = np.empty((n, ny, nx), dtype="float32")
    if "topo" in feats:
        arrays["topo"] = ds["topo"].values.astype("float32")
    for b0 in range(0, len(has_obs), block):
        sel = keep[(keep >= b0) & (keep < b0 + block)]
        if len(sel) == 0:
            continue
        sub = ds.isel(time=sel)
        pos = np.searchsorted(keep, sel)
        for k in todo:
            arrays[k][pos] = sub[names.get(k, k)].values
        if verbose and (b0 // block) % 5 == 0:
            print(f"  {str(ds.time.values[b0])[:10]}", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    if verbose:
        print(f"wrote {out}")
    return arrays


def obs_fraction(d):
    """Per-day fraction of ocean pixels with an L3 CHL observation."""
    return np.isfinite(d["chl_l3"][:, d["ocean"]]).mean(1)


def quality_report(d):
    """NaN fractions over ocean pixels, all-zero days, and CHL range. Returns a dict of tables."""
    ocean = d["ocean"]
    n_day = len(d["time"])
    rows = {}
    for v in ["chl_l3", "chl_l4"] + [f for f in FEATURE_VARS if f != "topo"]:
        if v not in d:
            continue
        x = d[v][:, ocean]
        nan_frac = np.isnan(x).mean()
        allzero = int(((np.nan_to_num(x) == 0).all(1) & ~np.isnan(x).all(1)).sum())
        allnan = int(np.isnan(x).all(1).sum())
        rows[v] = dict(nan_frac_ocean=float(nan_frac), days_all_nan=allnan, days_all_zero=allzero,
                       min=float(np.nanmin(x)), max=float(np.nanmax(x)))
    obs_frac = (~np.isnan(d["chl_l3"][:, ocean])).mean(1)
    return dict(n_days=n_day, n_ocean=int(ocean.sum()), per_var=rows,
                l3_obs_frac=dict(mean=float(obs_frac.mean()), min=float(obs_frac.min()), max=float(obs_frac.max())))


def split_indices(n, split=SPLIT):
    """Chronological train / val / test index arrays (baseline: 70/20/10)."""
    a = int(n * split[0])
    b = int(n * (split[0] + split[1]))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


VAL_YEARS = (2001, 2007, 2013, 2018)
TEST_YEARS = (2004, 2010, 2016, 2020)


def split_by_years(times, val_years=VAL_YEARS, test_years=TEST_YEARS):
    """Year-interleaved train / val / test index arrays (used for the 1997-2020 runs, where a
    chronological split would put every clear day into the test set)."""
    yr = np.asarray(times).astype("datetime64[Y]").astype(int) + 1970
    va = np.flatnonzero(np.isin(yr, val_years))
    te = np.flatnonzero(np.isin(yr, test_years))
    tr = np.flatnonzero(~np.isin(yr, val_years + test_years))
    return tr, va, te


class Normaliser:
    """Per-channel mean/std from the training days, NaN-aware. Missing values -> 0 after scaling."""

    def __init__(self, x_train):
        # x_train: (N, C, H, W)
        self.mean = np.nanmean(x_train, axis=(0, 2, 3), keepdims=True)
        self.std = np.nanstd(x_train, axis=(0, 2, 3), keepdims=True) + 1e-6

    def __call__(self, x):
        z = (x - self.mean) / self.std
        return np.nan_to_num(z, nan=0.0).astype("float32")

    def state(self):
        return dict(mean=self.mean.squeeze().tolist(), std=self.std.squeeze().tolist())


def log_chl(chl):
    """ln(CHL); non-positive or missing -> NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(chl)
    y[~np.isfinite(y)] = np.nan
    return y.astype("float32")


def feature_stack(d, features):
    """(N, C, H, W) float32 with NaNs. 'topo' is broadcast over time."""
    n = len(d["time"])
    chans = []
    for f in features:
        x = d[f]
        if x.ndim == 2:
            x = np.broadcast_to(x, (n,) + x.shape)
        chans.append(x.astype("float32"))
    return np.stack(chans, axis=1)


if __name__ == "__main__":
    import json
    d = build_subset()
    print(json.dumps(quality_report(d), indent=1))
