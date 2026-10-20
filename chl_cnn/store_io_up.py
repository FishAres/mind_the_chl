"""Store for the 2024 GeoSMART hackweek ("upstream") gap-fill protocol (2026-10-20; train_pinn.py --store io_up).

Their protocol (geo-smart/mind-the-chl-gap U-Net_Tutorial.ipynb cells 6, 9, 12; audit in docs/upstream_pipeline_diff.md):
  box      lat 32 -> -11.75, lon 42 -> 101.75 (176 x 240) of IO.zarr
  records  the days with >= 1 finite CHL_cmes-level3 pixel in the box, 1979-2022 (9193 days, 1997-10-01 .. 2022-12-31)
  clouds   hidden(t) = finite L3(t) & CHL_cmes-cloud(record t+10, wrapping at the end) == 0, one fixed mask per day
  label    ln L3, z-scored with nanmean / nanstd of ln L3 over all records (their CHL_mean / CHL_stdev; used for their
           all-pixel z metric only, see analyses/upstream/compare.py; the trainer keeps its own train-day z-score)
  split    xarray slices with inclusive ends: train 2015-01-01..2018-01-01, val 2018-01-01..2019-01-01, test 2019-01-01..
           2020-01-01. split_up() keeps their val / test day sets and drops 2018-01-01 from train (it is a val day).

The store holds only the records of SPAN (the split days plus a buffer for the +-4 day window), with the record index in
the full 1997-2022 list so the +10 shift matches theirs. Features float16 and chl_l3 float32 as in store.py.

Layout  ~/mind_the_chl_gap/data/io_up_store_2015_2019/
  {time, lat, lon, ocean, topo, chl_l3, <var>...}.npy   as store.py (time = store days)
  record_idx.npy    position of each store day in the full record list
  hidden_up.npy     (n, H, W) bool, their fake-cloud mask (finite L3 & cloud code 0 at record t+10)
  record_time.npy   the full record list (9193 dates)
  up_norm.json      their label mean / std of ln L3 over all records, counts, and data checks
"""
import json

import numpy as np
import xarray as xr

from . import data as D

BOX = dict(lat=slice(32, -11.75), lon=slice(42, 101.75))
SPAN = ("2014-12-20", "2020-01-15")
SHIFT = 10
STORE_VARS = ["sst", "so", "air_temp", "u_wind", "v_wind", "sla", "mlotst", "u_curr", "v_curr"]
CLOUD = "CHL_cmes-cloud"


def store_path():
    return D.DATA / "io_up_store_2015_2019"


def split_up(times):
    """(train, val, test) store indices of their split. val / test are their inclusive slices (they share 2019-01-01);
    train is 2015-01-01..2017-12-31 (their slice also takes 2018-01-01, a val day)."""
    t = np.asarray(times).astype("datetime64[D]")
    between = lambda a, b: np.flatnonzero((t >= np.datetime64(a)) & (t <= np.datetime64(b)))
    return between("2015-01-01", "2017-12-31"), between("2018-01-01", "2019-01-01"), between("2019-01-01", "2020-01-01")


def build_store(block=300, verbose=True):
    out = store_path(); out.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(D.SRC, consolidated=True).sel(**BOX)
    nt = ds.sizes["time"]
    # pass 1: record list and their label normalisation (ln L3 over every record day; all-NaN days add nothing)
    has_obs = np.zeros(nt, bool); s = ss = 0.0; cnt = 0; cmin = np.inf; n_nonpos = 0
    for b0 in range(0, nt, block):
        x = ds[D.CHL_L3].isel(time=slice(b0, b0 + block)).values.astype("float64")
        fin = np.isfinite(x)
        has_obs[b0:b0 + block] = fin.any((1, 2))
        v = x[fin]; n_nonpos += int((v <= 0).sum()); v = v[v > 0]
        if v.size:
            cmin = min(cmin, float(v.min())); lv = np.log(v); s += lv.sum(); ss += (lv ** 2).sum(); cnt += lv.size
        if verbose and (b0 // block) % 10 == 0:
            print(f"  pass 1 {str(ds.time.values[b0])[:10]}", flush=True)
    rec = np.flatnonzero(has_obs); rec_time = ds.time.values[rec]; nrec = len(rec)
    mu = s / cnt; sd = (ss / cnt - mu ** 2) ** 0.5
    t = rec_time.astype("datetime64[D]")
    keep = np.flatnonzero((t >= np.datetime64(SPAN[0])) & (t <= np.datetime64(SPAN[1])))
    n, ny, nx = len(keep), ds.sizes["lat"], ds.sizes["lon"]
    np.save(out / "record_time.npy", rec_time)
    np.save(out / "record_idx.npy", keep)
    np.save(out / "time.npy", rec_time[keep])
    np.save(out / "lat.npy", ds.lat.values.astype("float32"))
    np.save(out / "lon.npy", ds.lon.values.astype("float32"))
    ocean = ((ds["CHL_cmes-land"] != D.LAND_VALUE) & (ds["topo"] <= 0)).values
    np.save(out / "ocean.npy", ocean)
    np.save(out / "topo.npy", ds["topo"].values.astype("float32"))
    # calendar gaps inside the span (their t+-1 and t+10 are records, the trainer's neighbours are calendar days)
    full = np.arange(t[keep][0], t[keep][-1] + 1)
    missing = sorted(set(full.tolist()) - set(t[keep].tolist()))
    if verbose:
        print(f"{nrec} records {str(t[0])}..{str(t[-1])}; store {n} days {str(t[keep][0])}..{str(t[keep][-1])}; "
              f"calendar days missing in span: {len(missing)}", flush=True)
    # pass 2: store days
    mm = {v: np.lib.format.open_memmap(out / f"{v}.npy.part", mode="w+", dtype="float32" if v == "chl_l3" else "float16",
                                       shape=(n, ny, nx)) for v in ["chl_l3"] + STORE_VARS}
    hid = np.lib.format.open_memmap(out / "hidden_up.npy.part", mode="w+", dtype=bool, shape=(n, ny, nx))
    chk = dict(cloud0_not_finite=0, finite_not_cloud0=0, hidden_px=0, hidden_px_outside_ocean=0, finite_px_outside_ocean=0)
    for b0 in range(0, n, block):
        sel = slice(b0, min(b0 + block, n))
        days = rec[keep[sel]]; shifted = rec[(keep[sel] + SHIFT) % nrec]
        sub = ds.isel(time=days)
        for v in ["chl_l3"] + STORE_VARS:
            mm[v][sel] = sub[D.CHL_L3 if v == "chl_l3" else v].values
        fin = np.isfinite(mm["chl_l3"][sel])
        cl_t = sub[CLOUD].values
        cl_s = ds[CLOUD].isel(time=shifted).values
        h = fin & (cl_s == 0)
        hid[sel] = h
        chk["cloud0_not_finite"] += int(((cl_t == 0) & ~fin).sum()); chk["finite_not_cloud0"] += int((fin & (cl_t != 0)).sum())
        chk["hidden_px"] += int(h.sum()); chk["hidden_px_outside_ocean"] += int((h & ~ocean).sum())
        chk["finite_px_outside_ocean"] += int((fin & ~ocean).sum())
        if verbose:
            print(f"  pass 2 {str(t[keep][b0])}", flush=True)
    for v in list(mm):
        mm[v].flush(); del mm[v]
        (out / f"{v}.npy.part").rename(out / f"{v}.npy")
    hid.flush(); del hid
    (out / "hidden_up.npy.part").rename(out / "hidden_up.npy")
    meta = dict(box=dict(lat=[32, -11.75], lon=[42, 101.75]), shift=SHIFT, n_records=int(nrec),
                record_first=str(t[0]), record_last=str(t[-1]), span=list(SPAN), n_store=int(n),
                calendar_days_missing_in_span=[str(np.datetime64(int(m), "D")) for m in missing],
                label_mean=float(mu), label_std=float(sd), label_n=int(cnt), chl_min=cmin, chl_nonpositive=n_nonpos,
                checks=chk)
    (out / "up_norm.json").write_text(json.dumps(meta, indent=1))
    if verbose:
        print(json.dumps(meta, indent=1))


def load_store():
    """Dict like store.load_store(): memory-mapped arrays plus small arrays in RAM, and 'up' (up_norm.json)."""
    p = store_path()
    d = {k: np.load(p / f"{k}.npy") for k in ["time", "lat", "lon", "ocean", "topo", "record_idx", "record_time"]}
    for f in p.glob("*.npy"):
        if f.stem not in d:
            d[f.stem] = np.load(f, mmap_mode="r")
    d["up"] = json.loads((p / "up_norm.json").read_text())
    return d


if __name__ == "__main__":
    build_store()
