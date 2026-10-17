"""Memory-mapped store of the 1997-2020 subset: one .npy per variable, features in float16.

The npz cache in data.py keeps everything in RAM (about 1 GB per variable for 1997-2020),
which does not fit for 9 features on a 15 GB machine. Here each variable is written block by
block to its own .npy (features as float16, ~460 MB each) and read back with mmap, so training
copies them straight to the GPU chunk by chunk.

Layout  ~/mind_the_chl_gap/data/cnn_store_1997_2020/{time,lat,lon,ocean,chl_l3,<var>...}.npy
"""
from pathlib import Path

import numpy as np
import xarray as xr

from . import data as D

STORE_VARS = ["sst", "so", "air_temp", "u_wind", "v_wind", "sla", "mlotst", "u_curr", "v_curr"]
PERIOD = ("1997-09-01", "2020-12-31")


def store_path(years=PERIOD):
    return D.DATA / f"cnn_store_{years[0][:4]}_{years[1][:4]}"


def build_store(years=PERIOD, region=D.REGION, variables=STORE_VARS, block=300, verbose=True):
    out = store_path(years)
    out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(D.SRC, consolidated=True).sel(**region).sel(time=slice(*years))
    has_obs = ds[D.CHL_L3].notnull().any(["lat", "lon"]).compute().values
    keep = np.flatnonzero(has_obs)
    n, ny, nx = len(keep), ds.sizes["lat"], ds.sizes["lon"]
    np.save(out / "time.npy", ds.time.values[keep])
    np.save(out / "lat.npy", ds.lat.values.astype("float32"))
    np.save(out / "lon.npy", ds.lon.values.astype("float32"))
    np.save(out / "ocean.npy", ((ds["CHL_cmes-land"] != D.LAND_VALUE) & (ds["topo"] <= 0)).values)
    np.save(out / "topo.npy", ds["topo"].values.astype("float32"))
    todo = [v for v in ["chl_l3"] + list(variables) if not (out / f"{v}.npy").exists()]
    if verbose:
        print(f"{n} days; building {todo}", flush=True)
    mm = {v: np.lib.format.open_memmap(out / f"{v}.npy.part", mode="w+",
                                       dtype="float32" if v == "chl_l3" else "float16", shape=(n, ny, nx)) for v in todo}
    for b0 in range(0, len(has_obs), block):
        sel = keep[(keep >= b0) & (keep < b0 + block)]
        if len(sel) == 0:
            continue
        sub = ds.isel(time=sel)
        pos = np.searchsorted(keep, sel)
        for v in todo:
            mm[v][pos] = sub[D.CHL_L3 if v == "chl_l3" else v].values
        if verbose and (b0 // block) % 5 == 0:
            print(f"  {str(ds.time.values[b0])[:10]}", flush=True)
    for v in todo:
        mm[v].flush(); del mm[v]
        (out / f"{v}.npy.part").rename(out / f"{v}.npy")
    if verbose:
        print("wrote", out)


def load_store(years=PERIOD):
    """Dict of memory-mapped arrays (features float16, chl_l3 float32) plus small arrays in RAM."""
    p = store_path(years)
    d = {k: np.load(p / f"{k}.npy") for k in ["time", "lat", "lon", "ocean", "topo"]}
    for f in p.glob("*.npy"):
        if f.stem not in d:
            d[f.stem] = np.load(f, mmap_mode="r")
    return d


if __name__ == "__main__":
    build_store()
