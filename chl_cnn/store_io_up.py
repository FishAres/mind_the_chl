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

2024 recipe extras (2026-10-20; train_pinn.py --recipe up2024; python -m chl_cnn.store_io_up --recipe-extras)
  up24_<var>.npy    float32 (n, H, W) air_temp, sst, u_wind, v_wind read again from IO.zarr (the store copies are float16:
                    0.25 K steps at 300 K), as their notebook uses the float32 zarr values
  up24_cloud.npy    uint8 (n, H, W) CHL_cmes-cloud code at each store day (0 clear, 1 cloud, 2 land)
  up24_cloud_first.npy  uint8 (H, W) the code on the first record (1997-10-01): their land_flag source
  recipe_up2024(d) builds their 13 input channels and z-scored target from these (U-Net_Tutorial cells 9, 12).
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


# ---------------------------------------------------------------- 2024 recipe (train_pinn.py --recipe up2024)
UP24_FEATURES = ["u_wind", "v_wind", "sst", "air_temp"]            # their cell 10 order
UP24_CHANNELS = UP24_FEATURES + ["sin_time", "cos_time", "masked_CHL", "prev_day_CHL", "next_day_CHL",
                                 "land_flag", "real_cloud_flag", "valid_CHL_flag", "fake_cloud_flag"]
UP24_TRAIN = ("2015-01-01", "2018-01-01")      # their xarray slice (inclusive); stats use [start, end) (cell 9 index range)


def split_up2024(times):
    """(train, val, test) store indices with their inclusive slices for all three: train 2015-01-01..2018-01-01 (1097 days,
    2018-01-01 also a val day), val 2018-01-01..2019-01-01, test 2019-01-01..2020-01-01."""
    t = np.asarray(times).astype("datetime64[D]")
    between = lambda a, b: np.flatnonzero((t >= np.datetime64(a)) & (t <= np.datetime64(b)))
    return between(*UP24_TRAIN), between("2018-01-01", "2019-01-01"), between("2019-01-01", "2020-01-01")


def build_recipe_extras(verbose=True):
    """Write up24_<var>.npy (float32 features), up24_cloud.npy and up24_cloud_first.npy for the store days."""
    out = store_path()
    times = np.load(out / "time.npy"); rec_time = np.load(out / "record_time.npy")
    ds = xr.open_zarr(D.SRC, consolidated=True).sel(**BOX)
    tpos = np.searchsorted(ds.time.values, times)
    assert np.array_equal(ds.time.values[tpos], times)
    first = int(np.searchsorted(ds.time.values, rec_time[0])); assert ds.time.values[first] == rec_time[0]
    np.save(out / "up24_cloud_first.npy", ds[CLOUD].isel(time=first).values.astype("uint8"))
    for v, dt in [(CLOUD, "uint8")] + [(f, "float32") for f in UP24_FEATURES]:
        x = ds[v].isel(time=tpos).values.astype(dt)
        np.save(out / ("up24_cloud.npy" if v == CLOUD else f"up24_{v}.npy"), x)
        if verbose:
            print(f"  {v}: {x.shape} {x.dtype}" + (f" codes {np.unique(x).tolist()}" if v == CLOUD else
                                                   f" nan frac {np.isnan(x).mean():.4f}"), flush=True)


def recipe_up2024(d, verbose=True):
    """Their 13 input channels and target for every store day (U-Net_Tutorial cells 9, 12; records = store days).

    Returns dict(X (n, 13, H, W) float32 [UP24_CHANNELS order; z-scored numeric channels, NaN -> 0; 0/1 flags],
                 Y (n, H, W) float32 target z = (ln L3 - their all-record mean) / std, NaN -> 0,
                 masked (n, H, W) bool = their hidden px (finite L3 & cloud code 0 at record t+10; no ocean cut),
                 stats {channel: (mean, std)}, checks {...}).
      numeric  u_wind, v_wind, sst, air_temp (float32 IO.zarr values); sin / cos of float32((days since 1900-01-01) / 365 * 2 pi);
               masked_CHL = ln L3 with their hidden px NaN; prev_day / next_day CHL = ln L3 of records t-1 / t+1, unmasked, 0 past
               the store ends (their 0 padding at the record ends; never used for 2015-2019 days).
               mean / std = nanmean / nanstd over the train records [2015-01-01, 2018-01-01) (prev / next over their own values
               on those records), computed in float64.
      flags    land = cloud code 2 on the first record; real cloud = code 1 at t; valid = finite masked_CHL;
               fake cloud = (land + real cloud + valid) == 0.
    """
    p = store_path()
    times = np.asarray(d["time"]).astype("datetime64[D]")
    n = len(times)
    rel = np.asarray(d["record_idx"]); assert np.array_equal(np.diff(rel), np.ones(n - 1, int)), "store days must be consecutive records"
    y = D.log_chl(np.asarray(d["chl_l3"]))
    hid = np.asarray(d["hidden_up"])
    cloud = np.load(p / "up24_cloud.npy"); cloud0 = np.load(p / "up24_cloud_first.npy")
    i0, i1 = np.searchsorted(times, np.datetime64(UP24_TRAIN[0])), np.searchsorted(times, np.datetime64(UP24_TRAIN[1]))
    st = slice(i0, i1)                                                   # train records [2015-01-01, 2018-01-01)
    H, W = y.shape[1:]
    X = np.zeros((n, len(UP24_CHANNELS), H, W), dtype="float32")
    stats = {}

    def put(c, arr):
        name = UP24_CHANNELS[c]
        tr_ = arr[st].astype("float64")
        mu, sd = float(np.nanmean(tr_)), float(np.nanstd(tr_))
        stats[name] = (mu, sd)
        X[:, c] = np.nan_to_num((arr - mu) / sd, nan=0.0)

    for c, f in enumerate(UP24_FEATURES):
        put(c, np.load(p / f"up24_{f}.npy", mmap_mode="r"))
    day_rad = ((times - np.datetime64("1900-01-01")).astype("float64") / 365 * 2 * np.pi).astype("float32")
    for c, fn in [(4, np.sin), (5, np.cos)]:
        put(c, np.broadcast_to(fn(day_rad)[:, None, None], (n, H, W)))
    masked = np.where(hid, np.nan, y).astype("float32")
    put(6, masked)
    prev = np.concatenate([np.zeros((1, H, W), "float32"), y[:-1]]); nxt = np.concatenate([y[1:], np.zeros((1, H, W), "float32")])
    put(7, prev); put(8, nxt)
    land = np.broadcast_to(cloud0 == 2, (n, H, W)); real = cloud == 1; valid = np.isfinite(masked)
    fake = (land.astype(np.int8) + real + valid) == 0
    for c, f in [(9, land), (10, real), (11, valid), (12, fake)]:
        X[:, c] = f
    mu, sd = d["up"]["label_mean"], d["up"]["label_std"]
    Y = np.nan_to_num((y - mu) / sd, nan=0.0).astype("float32")
    checks = dict(n_train_stat_records=int(i1 - i0), train_stat_first=str(times[i0]), train_stat_last=str(times[i1 - 1]),
                  fake_ne_hidden_px=int((fake != hid).sum()), fake_px=int(fake.sum()), hidden_px=int(hid.sum()),
                  cloud_codes=np.unique(cloud).tolist(), cloud0_codes=np.unique(cloud0).tolist(),
                  land_px=int((cloud0 == 2).sum()), finite_L3_on_land_flag_px=int((np.isfinite(y) & land).sum()),
                  real_and_finite_px=int((real & np.isfinite(y)).sum()), label_mean=mu, label_std=sd)
    if verbose:
        print("recipe up2024 channel stats (train records):", json.dumps({k: [round(v[0], 5), round(v[1], 5)] for k, v in stats.items()}), flush=True)
        print("recipe up2024 checks:", json.dumps(checks), flush=True)
    return dict(X=X, Y=Y, masked=hid, stats=stats, checks=checks)


if __name__ == "__main__":
    import sys
    if "--recipe-extras" in sys.argv:
        build_recipe_extras()
    else:
        build_store()
