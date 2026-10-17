"""Memory-mapped store of the Black Sea 2 km data (fetch_cmems.open_arranged("blk", 2)) in the layout of store.py,
so train_spatial.py can run on it with --store blk.

Layout ~/mind_the_chl_gap/data/blk_store_<y0>_<y1>/{time,lat,lon,ocean,topo,chl_l3,sst32,u_curr,v_curr,u_wind,v_wind,so,mlotst,sla}.npy
  chl_l3  float32 mg m-3 (linear; the trainer takes the log)          sst32   float32 degC (thetao)
  u_curr, v_curr float16 m/s (uo, vo)                                  u_wind, v_wind float16 Pa (wind stress tauuo, tauvo; the
  trainer z-scores them)                                               so, mlotst, sla float16
  topo    float32: minus the distance to the coast in km (no bathymetry in the store; the trainer's log depth channel
          then becomes log10(1 + distance to coast in km))
Only days with at least one CHL observation are kept (as store.py). Run: python -m chl_cnn.store_blk [y0 y1]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import data as D

VARS = {"sst32": ("thetao", "float32"), "u_curr": ("uo", "float16"), "v_curr": ("vo", "float16"), "u_wind": ("tauuo", "float16"),
        "v_wind": ("tauvo", "float16"), "so": ("so", "float16"), "mlotst": ("mlotst", "float16"), "sla": ("zos", "float16")}
PERIOD = (2014, 2024)


def store_path(years=PERIOD):
    return D.DATA / f"blk_store_{years[0]}_{years[1]}"


def build(years=PERIOD, block=100, verbose=True):
    import xarray as xr
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import fetch_cmems as F
    out = store_path(years); out.mkdir(parents=True, exist_ok=True)
    ds = F.open_arranged("blk", 2).sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
    ocean = ds["ocean"].values
    has_obs = ds["chl_l3"].notnull().any(["lat", "lon"]).compute().values
    keep = np.flatnonzero(has_obs)
    n, ny, nx = len(keep), ds.sizes["lat"], ds.sizes["lon"]
    lat, lon = ds.lat.values.astype("float32"), ds.lon.values.astype("float32")
    np.save(out / "time.npy", ds.time.values[keep]); np.save(out / "lat.npy", lat); np.save(out / "lon.npy", lon); np.save(out / "ocean.npy", ocean)
    # distance to coast (km) from the ocean mask; stored as negative "topo" so log10(max(-topo, 0) + 1) is log distance
    dy_km = abs(float(lat[1] - lat[0])) * 111.2; dx_km = abs(float(lon[1] - lon[0])) * 111.2 * np.cos(np.deg2rad(lat.mean()))
    dist = distance_transform_edt(ocean, sampling=(dy_km, dx_km))
    np.save(out / "topo.npy", (-dist).astype("float32"))
    todo = [v for v in ["chl_l3"] + list(VARS) if not (out / f"{v}.npy").exists()]
    if verbose:
        print(f"{n} / {len(has_obs)} days with CHL; {int(ocean.sum())} ocean px; building {todo}", flush=True)
    mm = {v: np.lib.format.open_memmap(out / f"{v}.npy.part", mode="w+", dtype="float32" if v == "chl_l3" else VARS[v][1], shape=(n, ny, nx)) for v in todo}
    src = {v: ("chl_l3" if v == "chl_l3" else VARS[v][0]) for v in todo}
    for b0 in range(0, len(has_obs), block):
        sel = keep[(keep >= b0) & (keep < b0 + block)]
        if len(sel) == 0:
            continue
        sub = ds.isel(time=sel)[list(set(src.values()))].load()
        pos = np.searchsorted(keep, sel)
        for v in todo:
            mm[v][pos] = sub[src[v]].values
        if verbose and (b0 // block) % 10 == 0:
            print(f"  {str(ds.time.values[b0])[:10]}", flush=True)
    for v in todo:
        mm[v].flush(); del mm[v]
        (out / f"{v}.npy.part").rename(out / f"{v}.npy")
    if verbose:
        print("wrote", out)


def load_store(years=PERIOD, path=None):
    p = store_path(years) if path is None else path
    d = {k: np.load(p / f"{k}.npy") for k in ["time", "lat", "lon", "ocean", "topo"]}
    for f in p.glob("*.npy"):
        if f.stem not in d:
            d[f.stem] = np.load(f, mmap_mode="r")
    return d


if __name__ == "__main__":
    y = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else PERIOD
    build(y)
