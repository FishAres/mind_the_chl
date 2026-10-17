"""Download Copernicus Marine CHL (1 km L3 + gap-free L4) and the regional physics reanalysis for the
Black Sea or the Mediterranean, then arrange them into one zarr on a common grid, like IO.zarr.

Needs the copernicusmarine toolbox with credentials already stored (`copernicusmarine login`).

  python scripts/fetch_cmems.py download --basin blk --what chl phys --start 2016-01-01 --end 2020-12-31
  python scripts/fetch_cmems.py arrange  --basin blk --coarsen 2          # -> ~/mind_the_chl_gap/data/cmems/blk_2km/<year>.zarr
  python scripts/fetch_cmems.py arrange  --basin blk --coarsen 1          # native 1 km (Med at 1 km is ~5 M px/day: large)
  scripts/fetch_cmems_all.sh blk 1998 2024 2                             # tmux: download + arrange + delete raw, year by year

download   one compressed NetCDF per dataset and year under ~/mind_the_chl_gap/data/cmems/<basin>/raw/ (skips existing files)
arrange    one zarr per year (a failed year leaves nothing behind; re-run to retry): CHL L3/L4 block-averaged by
           --coarsen (geometric mean of valid pixels, count kept), physics fields (surface layer) interpolated onto
           that grid, all float32, chunked 100 days x full map. --delete-raw removes a year's NetCDFs once written.
           open_arranged(basin, coarsen) returns all years as one lazy xarray Dataset.
           Variables: chl_l3, chl_l3_n (valid 1 km px per block), chl_l4, qi_chl, thetao, so, uo, vo, mlotst, zos,
           tauuo, tauvo, ocean (L4 ever finite).
Rough sizes (uncompressed float32, from the toolbox dry run): Black Sea L3 CHL ~1.2 GB/yr, L4 ~1.2 GB/yr, physics
~1.5 GB/yr for 7 fields at 2.5 km; Mediterranean L3 CHL ~7 GB/yr, L4 ~7 GB/yr, physics ~2.5 GB/yr at 4.2 km.
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path.home() / "mind_the_chl_gap" / "data" / "cmems"

BASINS = {
    "blk": dict(
        chl_l3=("cmems_obs-oc_blk_bgc-plankton_my_l3-multi-1km_P1D", ["CHL", "QI_CHL"]),
        chl_l4=("cmems_obs-oc_blk_bgc-plankton_my_l4-gapfree-multi-1km_P1D", ["CHL"]),
        phys=[("cmems_mod_blk_phy-temp_my_2.5km_P1D-m", ["thetao"]), ("cmems_mod_blk_phy-sal_my_2.5km_P1D-m", ["so"]),
              ("cmems_mod_blk_phy-cur_my_2.5km_P1D-m", ["uo", "vo"]), ("cmems_mod_blk_phy-mld_my_2.5km_P1D-m", ["mlotst"]),
              ("cmems_mod_blk_phy-ssh_my_2.5km_P1D-m", ["zos"]), ("cmems_mod_blk_phy-mflux_my_2.5km_P1D-m", ["tauuo", "tauvo"])],
        box=dict(minimum_longitude=26.5, maximum_longitude=42.0, minimum_latitude=40.0, maximum_latitude=48.0)),
    "med": dict(
        chl_l3=("cmems_obs-oc_med_bgc-plankton_my_l3-multi-1km_P1D", ["CHL", "QI_CHL"]),
        chl_l4=("cmems_obs-oc_med_bgc-plankton_my_l4-gapfree-multi-1km_P1D", ["CHL"]),
        phys=[("cmems_mod_med_phy-temp_my_4.2km_P1D-m", ["thetao"]), ("cmems_mod_med_phy-sal_my_4.2km_P1D-m", ["so"]),
              ("cmems_mod_med_phy-cur_my_4.2km_P1D-m", ["uo", "vo"]), ("cmems_mod_med_phy-mld_my_4.2km_P1D-m", ["mlotst"]),
              ("cmems_mod_med_phy-ssh_my_4.2km_P1D-m", ["zos"]), ("cmems_mod_med_phy-mflux_my_4.2km_P1D-m", ["tauuo", "tauvo"])],
        box=dict(minimum_longitude=-6.0, maximum_longitude=36.5, minimum_latitude=30.0, maximum_latitude=46.0)),
}


def download(a):
    import copernicusmarine as cm
    logging.getLogger("copernicusmarine").setLevel(logging.WARNING)
    cfg = BASINS[a.basin]
    out = ROOT / a.basin / "raw"; out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if "chl" in a.what:
        jobs += [cfg["chl_l3"], cfg["chl_l4"]]
    if "phys" in a.what:
        jobs += cfg["phys"]
    years = range(int(a.start[:4]), int(a.end[:4]) + 1)
    for did, vs in jobs:
        for y in years:
            if a.skip_arranged and year_store(a.basin, a.skip_arranged, y).exists():
                print(f"  {y} already arranged at {a.skip_arranged} km, skipping download"); continue
            t0, t1 = max(a.start, f"{y}-01-01"), min(a.end, f"{y}-12-31")
            f = out / f"{did}_{y}.nc"
            if f.exists():
                print(f"  have {f.name}"); continue
            t = time.time()
            kw = dict(dataset_id=did, variables=vs, start_datetime=t0, end_datetime=t1, output_directory=str(out), output_filename=f.stem + ".part",
                      file_format="netcdf", netcdf_compression_level=4, disable_progress_bar=True, overwrite=True, **cfg["box"])
            if did.startswith("cmems_mod"):
                kw.update(minimum_depth=0, maximum_depth=1.5)              # surface layer only
            for attempt in range(3):
                try:
                    r = cm.subset(**kw); break
                except Exception as e:
                    print(f"  retry {attempt + 1} {f.name}: {type(e).__name__}: {str(e)[:100]}", flush=True); time.sleep(30)
            else:
                raise RuntimeError(f"download failed: {f.name}")
            Path(r.file_path).rename(f)                                   # the toolbox appends .nc itself
            print(f"  {f.name}  {f.stat().st_size / 1e6:.0f} MB  {time.time() - t:.0f}s", flush=True)


def _open(files, chunks=None):
    return xr.open_mfdataset(sorted(files), combine="by_coords", chunks=chunks or {"time": 50}, parallel=False)


def _coarsen_chl(da, n):
    """Geometric-mean block average of a linear CHL field over n x n pixels; returns (mean, valid count)."""
    if n == 1:
        return da, xr.ones_like(da).where(np.isfinite(da), 0)
    ln = np.log(da)
    c = ln.coarsen(latitude=n, longitude=n, boundary="trim")
    cnt = ln.notnull().coarsen(latitude=n, longitude=n, boundary="trim").sum()
    mean = np.exp(c.mean())
    return mean.where(cnt > 0), cnt


def year_store(basin, coarsen, year):
    return ROOT / f"{basin}_{coarsen}km" / f"{year}.zarr"


def open_arranged(basin, coarsen=2):
    """All arranged years of a basin as one lazy dataset (time-sorted)."""
    files = sorted((ROOT / f"{basin}_{coarsen}km").glob("*.zarr"))
    if not files:
        raise FileNotFoundError(f"no arranged years under {ROOT / f'{basin}_{coarsen}km'}")
    return xr.open_mfdataset(files, engine="zarr", combine="by_coords", consolidated=True, data_vars="minimal", coords="minimal", compat="override")


def arrange(a):
    """One zarr per year under <basin>_<coarsen>km/ (a failed year leaves no half-written store; re-run to retry).
    With --delete-raw the year's NetCDFs are removed after the year's store is complete."""
    cfg = BASINS[a.basin]
    raw = ROOT / a.basin / "raw"
    years = sorted({int(f.stem.rsplit("_", 1)[1]) for f in raw.glob(cfg["chl_l4"][0] + "_*.nc")})
    if a.years:
        years = [y for y in years if a.years[0] <= y <= a.years[1]]
    for y in years:
        out = year_store(a.basin, a.coarsen, y)
        if out.exists():
            print(f"  {y} already arranged"); continue
        f3, f4 = raw / f"{cfg['chl_l3'][0]}_{y}.nc", raw / f"{cfg['chl_l4'][0]}_{y}.nc"
        if not (f3.exists() and f4.exists()):
            print(f"  {y}: missing L3 or L4 file, skipping"); continue
        t = time.time()
        l3, l4 = xr.open_dataset(f3, chunks={"time": 50}), xr.open_dataset(f4, chunks={"time": 50})
        times = np.intersect1d(l3.time.values, l4.time.values)
        l3, l4 = l3.sel(time=times), l4.sel(time=times)
        chl3, n3 = _coarsen_chl(l3["CHL"], a.coarsen)
        chl4, _ = _coarsen_chl(l4["CHL"], a.coarsen)
        qi = l3["QI_CHL"].coarsen(latitude=a.coarsen, longitude=a.coarsen, boundary="trim").mean() if a.coarsen > 1 else l3["QI_CHL"]
        ds = xr.Dataset(dict(chl_l3=chl3.astype("float32"), chl_l3_n=n3.astype("int16"), chl_l4=chl4.astype("float32"), qi_chl=qi.astype("float32")))
        ds = ds.rename(latitude="lat", longitude="lon")
        for did, vs in cfg["phys"]:
            fp = raw / f"{did}_{y}.nc"
            if not fp.exists():
                print(f"  {y}: no {did}, physics channels {vs} left out"); continue
            p = xr.open_dataset(fp, chunks={"time": 50})
            if "depth" in p.dims:
                p = p.isel(depth=0, drop=True)
            p = p.rename(latitude="lat", longitude="lon").reindex(time=times)
            for v in vs:
                ds[v] = p[v].interp(lat=ds.lat, lon=ds.lon, method="linear").astype("float32")
        ds["ocean"] = ds["chl_l4"].notnull().any("time")
        tmp = out.with_suffix(".zarr.part")
        if tmp.exists():
            import shutil; shutil.rmtree(tmp)
        ds.chunk({"time": 100, "lat": -1, "lon": -1}).to_zarr(tmp, mode="w", consolidated=True)
        tmp.rename(out)
        print(f"  {y}: {len(times)} days -> {out.relative_to(ROOT)}  {time.time() - t:.0f}s", flush=True)
        if a.delete_raw:
            for fp in [f3, f4] + [raw / f"{did}_{y}.nc" for did, _ in cfg["phys"]]:
                fp.unlink(missing_ok=True)
    d = ROOT / f"{a.basin}_{a.coarsen}km"
    if d.exists():
        print(f"{d}: {len(list(d.glob('*.zarr')))} years, {sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / 1e9:.2f} GB on disk")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download"); d.add_argument("--basin", choices=BASINS, required=True); d.add_argument("--what", nargs="+", choices=["chl", "phys"], default=["chl", "phys"])
    d.add_argument("--start", default="1997-09-01"); d.add_argument("--end", default="2024-12-31")
    d.add_argument("--skip-arranged", type=int, default=0, metavar="COARSEN", help="skip years whose <basin>_<COARSEN>km/<year>.zarr exists"); d.set_defaults(fn=download)
    r = sub.add_parser("arrange"); r.add_argument("--basin", choices=BASINS, required=True); r.add_argument("--coarsen", type=int, default=2)
    r.add_argument("--years", nargs=2, type=int, default=None, metavar=("Y0", "Y1")); r.add_argument("--delete-raw", action="store_true"); r.set_defaults(fn=arrange)
    a = ap.parse_args(); a.fn(a)
