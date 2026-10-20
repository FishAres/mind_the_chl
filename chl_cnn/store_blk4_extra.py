"""Add the 2-D surface fluxes of fetch_cmems.BASINS["blk"]["extra"] to the Black Sea 4 km store (store_blk4.py):

  rsntds.npy  float16 W m-2, net downward shortwave at the surface (cmems_mod_blk_phy-hflux_my_2.5km_P1D-m)
  friver.npy  float32 kg m-2 s-1, river freshwater flux (cmems_mod_blk_phy-wflux_my_2.5km_P1D-m); float32 because the
              mouth values span ~1e-7 .. 1e-1 (below float16's normal range)

Both (time, H, W) on the store's time.npy days and 197 x 275 grid, written next to the existing arrays.

rsntds: exactly the chain of the other physics channels: fetch_cmems.phys_on_grid (linear interpolation of the 2.5 km
  model onto the 2 km CHL grid, days reindexed to the store days) -> store_blk4.coarse_nanmean (2x2 nanmean, last
  2 km row dropped). Days missing in the source come out all-NaN (counted and printed).
friver: a point-source field (nonzero only at the model's river-mouth cells), which linear interpolation dilutes and,
  at the coast, loses (any NaN neighbour -> NaN). Instead it is mapped conservatively: each source cell's mass flux
  friver * cell area (kg s-1) is added to the nearest 4 km pixel that is ocean (store ocean.npy) and has finite rsntds
  on the first day (model sea), then divided by that pixel's area. Every other such pixel is 0; pixels without model sea
  are NaN. The basin total in kg s-1 is therefore preserved; the per-day totals of source, conservative and the
  linear (rsntds-style) chain are saved to --diag (npz) for checking, with the source-cell -> pixel mapping.
Areas: spherical cell areas, R = 6371.0088 km, from the 0.025 deg model spacing and the 4 km lat/lon spacing.

Year by year: download the year's raw NetCDFs (fetch_cmems download --what extra), regrid, delete them. Arrays are built
as <name>.npy.part and renamed when all years are done; re-running restarts any unfinished array.
Run:  python -m chl_cnn.store_blk4_extra [--diag analyses/blk4/results/extra_vars_friver_totals.npz] [--keep-raw]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import store_blk4 as S4

R_EARTH = 6371.0088e3          # m
SRC_STEP = 0.025               # deg, model grid spacing (checked against the file)


def cell_area(lat, dlat, dlon):
    """Spherical area (m2) of lat-lon cells centred at lat (deg) with spacings dlat, dlon (deg)."""
    return R_EARTH ** 2 * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(np.deg2rad(lat))


def _fetch():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import fetch_cmems as F
    return F


def friver_map(src_lat, src_lon, src_mask, lat4, lon4, target):
    """For each source cell in src_mask (bool H x W) the index of the nearest target 4 km pixel (flat index) and the
    distance (km). Distance: equirectangular on the sphere, good to << 1 km at these scales."""
    si, sj = np.nonzero(src_mask)
    ti, tj = np.nonzero(target)
    la, lo = src_lat[si][:, None], src_lon[sj][:, None]
    dy = np.deg2rad(la - lat4[ti][None]); dx = np.deg2rad(lo - lon4[tj][None]) * np.cos(np.deg2rad(la))
    d = R_EARTH / 1e3 * np.hypot(dy, dx)
    k = d.argmin(1)
    return si, sj, np.ravel_multi_index((ti[k], tj[k]), target.shape), d[np.arange(len(k)), k]


def build(years=S4.PERIOD, diag=None, keep_raw=False, verbose=True):
    import xarray as xr
    F = _fetch()
    t0 = time.time()
    out = S4.store_path(years)
    days = pd.DatetimeIndex(np.load(out / "time.npy"))
    ocean4 = np.load(out / "ocean.npy"); lat4 = np.load(out / "lat.npy"); lon4 = np.load(out / "lon.npy")
    H2, W2 = ocean4.shape; n = len(days)
    g = xr.open_zarr(F.year_store("blk", 2, years[0]))           # the 2 km CHL grid arrange interpolates onto
    lat2, lon2 = g.lat, g.lon
    assert (H2, W2) == (g.sizes["lat"] // 2, g.sizes["lon"] // 2)
    assert np.allclose(lat2.values[:2 * H2].reshape(H2, 2).mean(1), lat4) and np.allclose(lon2.values[:2 * W2].reshape(W2, 2).mean(1), lon4)
    a4 = cell_area(lat4, abs(float(np.diff(lat4).mean())), abs(float(np.diff(lon4).mean())))[:, None] * np.ones((1, W2))
    (did_r, _), (did_f, _) = F.BASINS["blk"]["extra"]
    raw = F.ROOT / "blk" / "raw"
    if (out / "rsntds.npy").exists() and (out / "friver.npy").exists():
        print("rsntds.npy and friver.npy exist; nothing to do"); return
    mm_r = np.lib.format.open_memmap(out / "rsntds.npy.part", mode="w+", dtype="float16", shape=(n, H2, W2))
    mm_f = np.lib.format.open_memmap(out / "friver.npy.part", mode="w+", dtype="float32", shape=(n, H2, W2))
    tot_src, tot_cons, tot_lin = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    missing_r, missing_f = [], []
    mp = None
    for y in range(years[0], years[1] + 1):
        pos = np.flatnonzero(days.year == y)
        if len(pos) == 0:
            continue
        F.download(argparse.Namespace(basin="blk", what=["extra"], start=f"{y}-01-01", end=f"{y}-12-31", skip_arranged=0))
        fr, ff = raw / f"{did_r}_{y}.nc", raw / f"{did_f}_{y}.nc"
        times = days[pos].values
        # rsntds: arrange chain then the store_blk4 chain
        r2 = F.phys_on_grid(fr, ["rsntds"], lat2, lon2, times)["rsntds"].values
        r4 = S4.coarse_nanmean(r2, H2, W2)
        mm_r[pos] = r4.astype("float16")
        missing_r += [str(t)[:10] for t, ok in zip(times, np.isfinite(r2).any((1, 2))) if not ok]
        # friver: source on its own grid, store days
        src = xr.open_dataset(ff)["friver"].reindex(time=times)
        slat, slon = src.latitude.values.astype("float64"), src.longitude.values.astype("float64")
        assert np.allclose(np.diff(slat), SRC_STEP, atol=1e-4) and np.allclose(np.diff(slon), SRC_STEP, atol=1e-4)
        fv = src.values                                           # (t, h, w) kg m-2 s-1
        missing_f += [str(t)[:10] for t, ok in zip(times, np.isfinite(fv).any((1, 2))) if not ok]
        if mp is None:
            target = ocean4 & np.isfinite(r4[0])
            sea4 = np.isfinite(r4[0])
            src_area = cell_area(slat, SRC_STEP, SRC_STEP)[:, None] * np.ones((1, len(slon)))
            mp = dict(target=target, sea4=sea4, src_area=src_area, slat=slat, slon=slon)
        assert np.array_equal(mp["slat"], slat) and np.array_equal(mp["slon"], slon)
        nz = (np.nan_to_num(fv) > 0).any(0)
        if "si" not in mp or not np.isin(np.ravel_multi_index(np.nonzero(nz), nz.shape), mp["src_flat"]).all():
            # (re)build the mapping on the union of nonzero source cells seen so far
            prev = np.zeros_like(nz) if "si" not in mp else mp["nz"]
            nzu = prev | nz
            si, sj, tk, dk = friver_map(slat, slon, nzu, lat4, lon4, mp["target"])
            mp.update(nz=nzu, si=si, sj=sj, tk=tk, dk=dk, src_flat=np.ravel_multi_index((si, sj), nzu.shape))
        mass = np.nan_to_num(fv[:, mp["si"], mp["sj"]]) * mp["src_area"][mp["si"], mp["sj"]]     # (t, k) kg s-1
        f4 = np.zeros((len(pos), H2 * W2))
        for k in range(mass.shape[1]):
            f4[:, mp["tk"][k]] += mass[:, k]
        f4 = f4.reshape(len(pos), H2, W2) / a4
        f4[:, ~mp["sea4"]] = np.nan
        f4[~np.isfinite(fv).any((1, 2))] = np.nan                 # day missing in the source
        mm_f[pos] = f4.astype("float32")
        tot_src[pos] = np.nansum(fv * mp["src_area"], (1, 2))
        tot_cons[pos] = np.nansum(mm_f[pos].astype("float64") * a4, (1, 2))
        lin4 = S4.coarse_nanmean(F.phys_on_grid(ff, ["friver"], lat2, lon2, times)["friver"].values, H2, W2)
        tot_lin[pos] = np.nansum(lin4 * a4, (1, 2))
        if not keep_raw:
            fr.unlink(missing_ok=True); ff.unlink(missing_ok=True)
        if verbose:
            print(f"  {y}: {len(pos)} days; friver total src {np.nanmean(tot_src[pos]):.4g} / conservative {np.nanmean(tot_cons[pos]):.4g} / "
                  f"linear {np.nanmean(tot_lin[pos]):.4g} kg s-1; {time.time() - t0:.0f} s", flush=True)
    mm_r.flush(); mm_f.flush(); del mm_r, mm_f
    for v in ["rsntds", "friver"]:
        (out / f"{v}.npy.part").rename(out / f"{v}.npy")
    if diag:
        Path(diag).parent.mkdir(parents=True, exist_ok=True)
        np.savez(diag, time=days.values, total_src=tot_src, total_conservative=tot_cons, total_linear=tot_lin,
                 src_lat=mp["slat"][mp["si"]], src_lon=mp["slon"][mp["sj"]], target_flat=mp["tk"], snap_km=mp["dk"],
                 missing_rsntds=np.array(missing_r), missing_friver=np.array(missing_f))
    if verbose:
        print(f"wrote rsntds, friver to {out} in {time.time() - t0:.0f} s; store days missing in source: rsntds {missing_r}, friver {missing_f}; "
              f"{len(mp['si'])} nonzero source cells -> {len(np.unique(mp['tk']))} pixels, max snap {mp['dk'].max():.1f} km", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs=2, type=int, default=list(S4.PERIOD))
    ap.add_argument("--diag", default=None, help="npz with per-day friver totals and the source -> pixel mapping")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    build(tuple(a.years), diag=a.diag, keep_raw=a.keep_raw)
