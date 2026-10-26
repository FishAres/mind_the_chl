"""Store context for CPU post-processing / leak tests outside the trainer (blk7, 2026-10-26).

load(store, crop, val_years, test_years) mirrors chl_cnn/train_pinn.py's data section: the store (io, blk, blk4), the optional --crop
(ocean bounding box + 1 px), ln CHL, observed mask (finite & ocean), the year split (blk / blk4 default val 2017 2021, test 2019 2023;
io: chl_cnn.data VAL_YEARS / TEST_YEARS) and the fixed fake-cloud hidden masks / donors (eval seed 0 val, 1 test; donor_exclude 0).
from_args(argv) reads --store / --crop / --val-years / --test-years from a trainer argument list (e.g. a leak test's --base-args).
"""
import argparse
import shlex

import numpy as np

from chl_cnn import data as D, clouds as C


def load_raw(store):
    if store == "blk":
        from chl_cnn import store_blk as SB
        return SB.load_store()
    if store == "blk4":
        from chl_cnn import store_blk4 as SB
        return SB.load_store()
    if store == "io":
        from chl_cnn import store as S
        return S.load_store()
    raise ValueError(f"store {store} not supported")


def load(store="io", crop=False, val_years=None, test_years=None):
    d = load_raw(store)
    times = d["time"]
    if crop:
        oc = d["ocean"]; rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
        ys, xs = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0])), slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
    else:
        ys, xs = slice(None), slice(None)
    if store in ("blk", "blk4"):
        vy, ty = tuple(val_years or (2017, 2021)), tuple(test_years or (2019, 2023))
    else:
        vy, ty = tuple(val_years or D.VAL_YEARS), tuple(test_years or D.TEST_YEARS)
    ocean = np.asarray(d["ocean"][ys, xs])
    y = D.log_chl(np.asarray(d["chl_l3"][:, ys, xs]))
    obs = np.isfinite(y) & ocean
    tr, va, te = D.split_by_years(times, vy, ty)
    hv, dv = C.fixed_hidden(obs, va, seed=0, times=times)
    ht, dt_ = C.fixed_hidden(obs, te, seed=1, times=times)
    return dict(store=store, times=times, y=y, obs=obs, ocean=ocean, lat=np.asarray(d["lat"][ys]), lon=np.asarray(d["lon"][xs]),
                tr=tr, va=va, te=te, hid_va=hv, don_va=dv, hid_te=ht, don_te=dt_, val_years=vy, test_years=ty, crop=(ys, xs))


def parse(argv):
    if isinstance(argv, str):
        argv = shlex.split(argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--store", default="io")
    p.add_argument("--crop", action="store_true")
    p.add_argument("--val-years", type=int, nargs="+", default=None)
    p.add_argument("--test-years", type=int, nargs="+", default=None)
    p.add_argument("--val-stride", type=int, default=1)
    a, _ = p.parse_known_args(argv)
    return a


def from_args(argv):
    a = parse(argv)
    return load(a.store, a.crop, a.val_years, a.test_years)


def score_context(store="blk4", crop=True, stride=4, cache_dir=None, val_years=None, test_years=None):
    """As analyses/io_improve/score.context for a strided Black Sea run: y (all store days), obs, ocean, times, va / te = the scored days
    (va[::stride], te[::stride]), hidden / dist / age for va then te, and the donors (for analyses/io7/keys.py). dist / age are cached."""
    from pathlib import Path
    from scipy.ndimage import distance_transform_edt
    from chl_cnn import train2 as T2
    from analyses.forecast.forecast_eval import persistence
    c = load(store, crop, val_years, test_years)
    T2.TIMES = c["times"]
    va, te = c["va"][::stride], c["te"][::stride]
    hidden = np.concatenate([c["hid_va"][::stride], c["hid_te"][::stride]])
    donors = (c["don_va"][::stride], c["don_te"][::stride])
    cache_dir = Path(cache_dir or Path.home() / "mind_the_chl_gap" / "data" / "blk7")
    f = cache_dir / f"score_ctx_{store}{'_crop' if crop else ''}_s{stride}.npz"
    if f.exists() and np.array_equal(np.load(f)["hidden"], hidden):
        z = np.load(f); dist, age = z["dist"], z["age"]
    else:
        vis = c["obs"][np.r_[va, te]] & ~hidden
        dist = np.stack([distance_transform_edt(~v) for v in vis]).astype("float16")
        age = np.concatenate([persistence(c["y"], c["obs"], c["times"], ix, 8)[1] for ix in (va, te)])
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(f, hidden=hidden, dist=dist, age=age)
    return dict(y=c["y"], obs=c["obs"], va=va, te=te, times=c["times"], ocean=c["ocean"], hidden=hidden, dist=dist, age=age,
                donors=donors, hid_split=(c["hid_va"][::stride], c["hid_te"][::stride]), stride=stride, store=store)
