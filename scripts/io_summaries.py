"""One-pass summaries of IO.zarr, cached to a small local zarr.

The cube is processed in chunk-aligned blocks of days. Each block is reduced
lazily (xarray + dask) and computed with one `dask.compute`, so every chunk is
read once and peak memory stays ~1-2 GB. (A first single-pass version was
OOM-killed because the store was opened with chunks=None, i.e. without dask,
so `.where()` loaded all variables eagerly. Blocks also bound memory
regardless of scheduler behaviour.)

Outputs (~22 MB), used by IO_EDA.ipynb:
  daily/  per day x variable: ocean-pixel mean, valid (non-NaN) fraction, all-zero-map flag
  clim/   per calendar month: climatology maps of CLIM_VARS + CHL L3 observed fraction

Run:  python scripts/io_summaries.py
"""
from pathlib import Path
import time

import dask
import numpy as np
import xarray as xr

# Data lives on local disk (not s3fs); code lives in ~/shared/ecology/mind_the_chl_gap
DATA = Path.home() / "mind_the_chl_gap" / "data"
SRC = DATA / "IO.zarr"
OUT = DATA / "summaries.zarr"

LAND_VALUE = 2  # CHL_cmes-land: 0 = ocean, 2 = land (checked against topo)
CHL_START = "1997-09-01"  # GlobColour starts Sept 1997; earlier days are all-NaN
BLOCK = 200  # days per block (2 chunks of 100)

# Variables with a monthly climatology map (others only get daily domain stats)
CLIM_VARS = [
    "CHL_cmes-gapfree", "CHL_cmes-level3", "sst", "air_temp",
    "u_wind", "v_wind", "wind_speed", "u_curr", "v_curr", "curr_speed",
    "sla", "adt", "mlotst", "so",
]


def ocean_mask(ds):
    """Ocean = not flagged land AND at/below sea level (drops 43 inland lake pixels, e.g. Tibet)."""
    return (ds["CHL_cmes-land"] != LAND_VALUE) & (ds["topo"] <= 0)


def open_io(chunks={}):
    """Lazy, dask-backed open using the on-disk chunks; nothing is read beyond metadata.

    Note: chunks=None would NOT use dask - variables become lazily-indexed numpy
    arrays that load in full on the first operation (e.g. np.log10 -> 2.7 GB).
    """
    return xr.open_zarr(SRC, consolidated=True, chunks=chunks)


def _block_reductions(sub, ocean, n_ocean, vars3d):
    """Lazy reductions for one block of days."""
    cube = sub[vars3d].where(ocean)
    daily = xr.Dataset()
    for v in vars3d:
        daily[f"{v}__mean"] = cube[v].mean(["lat", "lon"])
        daily[f"{v}__valid"] = cube[v].notnull().sum(["lat", "lon"]) / n_ocean
        # all-zero maps are a known failure mode -> flag them (only on days with data)
        daily[f"{v}__allzero"] = (cube[v].fillna(0) == 0).all(["lat", "lon"]) & cube[v].notnull().any(["lat", "lon"])
    month = sub.time.dt.month
    c = sub[CLIM_VARS]
    msum = c.fillna(0).groupby(month).sum("time")
    mcnt = c.notnull().groupby(month).sum("time")
    l3 = sub["CHL_cmes-level3"].where(sub.time >= np.datetime64(CHL_START))
    obs = l3.notnull().groupby(month).sum("time")
    ndays = (l3.time >= np.datetime64(CHL_START)).groupby(month).sum("time")
    return daily, msum, mcnt, obs, ndays


def build_summaries(ds, block=BLOCK, verbose=True):
    ocean = ocean_mask(ds).compute()
    n_ocean = int(ocean.sum())
    vars3d = [v for v in ds.data_vars if ds[v].dims == ("time", "lat", "lon")]

    dailies = []
    shape = (12, ds.sizes["lat"], ds.sizes["lon"])
    msum = {v: np.zeros(shape) for v in CLIM_VARS}
    mcnt = {v: np.zeros(shape) for v in CLIM_VARS}
    obs, ndays = np.zeros(shape), np.zeros(12)

    t0 = time.time()
    for i0 in range(0, ds.sizes["time"], block):
        sub = ds.isel(time=slice(i0, i0 + block))
        d, s, c, o, n = dask.compute(*_block_reductions(sub, ocean, n_ocean, vars3d))
        dailies.append(d)
        idx = s.month.values - 1
        for v in CLIM_VARS:
            msum[v][idx] += s[v].values
            mcnt[v][idx] += c[v].values
        obs[idx] += o.values
        ndays[idx] += n.values
        if verbose and (i0 // block) % 10 == 0:
            print(f"  {str(sub.time.values[0])[:10]}  ({time.time() - t0:.0f} s)", flush=True)

    daily = xr.concat(dailies, "time")
    coords = dict(month=np.arange(1, 13), lat=ds.lat, lon=ds.lon)
    clim = xr.Dataset({v: (("month", "lat", "lon"), np.where(mcnt[v] > 0, msum[v] / np.maximum(mcnt[v], 1), np.nan)
                           .astype("float32")) for v in CLIM_VARS}, coords=coords)
    for v in CLIM_VARS:
        clim[v].attrs = ds[v].attrs
        clim[f"{v}__n_days"] = (("month", "lat", "lon"), mcnt[v].astype("int32"))
    clim["CHL_l3_obs_frac"] = (("month", "lat", "lon"),
                               (obs / np.maximum(ndays, 1)[:, None, None]).astype("float32"))
    clim["CHL_l3_obs_frac"] = clim["CHL_l3_obs_frac"].where(ocean)
    return daily, clim


def main():
    ds = open_io()
    print("computing summaries block by block...")
    with dask.config.set(scheduler="threads"):
        daily, clim = build_summaries(ds)
    for obj in (daily, clim):  # drop zarr-v2 codecs inherited from the source store
        for v in obj.variables.values():
            v.encoding = {}
    daily.to_zarr(OUT, group="daily", mode="w")
    clim.to_zarr(OUT, group="clim", mode="w")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
