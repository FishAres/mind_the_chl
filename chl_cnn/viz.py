"""Per-day input/output figures for a finished run, drawn from its saved predictions (preds.npz) on a fixed
sample of validation days, so different models can be compared on the same days.

Works for every store (IO, blk 2 km, blk4 4 km): the store, crop, split years, donor exclusion and context-cloud
mode are read from the run's config.json, the same way train_pinn.py sets them up.

  ctx = context_for(run_dir)                 # store fields for that run's store / crop (cached per store)
  days = sample_days(ctx, run_dir, n=32)     # val days with saved predictions, evenly spaced over observed fraction
  fig = draw_day(run_dir, days[5], ctx)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import data as D, clouds as C
from .plotting import OKABE_ITO, MAP_CMAP, ERR_CMAP

_CTX = {}
BLK_SPLIT = ((2017, 2021), (2019, 2023))                       # train_pinn.py defaults for --store blk / blk4


def _cfg(run_dir):
    return json.load(open(Path(run_dir) / "config.json"))


def load_context(store="io", crop=False, val_years=None, test_years=None):
    """Store fields as the run saw them (cropped like train_pinn.py --crop). Large arrays stay memory-mapped;
    ln CHL is read one day at a time (ctx['y'](i)), so the 2 km store does not need ~3 GB of RAM."""
    key = (store, bool(crop), tuple(val_years or ()), tuple(test_years or ()))
    if key in _CTX:
        return _CTX[key]
    if store in ("blk", "blk4"):
        from . import store_blk as SB, store_blk4 as SB4
        d = (SB if store == "blk" else SB4).load_store()
        vy, ty = BLK_SPLIT
    else:
        from . import store as S
        d = S.load_store()
        vy, ty = D.VAL_YEARS, D.TEST_YEARS
    vy, ty = tuple(val_years or vy), tuple(test_years or ty)
    oc = np.asarray(d["ocean"])
    if crop:
        rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
        ys, xs = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0])), slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
    else:
        ys, xs = slice(None), slice(None)
    ocean = oc[ys, xs]
    times = np.asarray(d["time"])
    n = len(times)
    chl = d["chl_l3"]
    y = lambda i: D.log_chl(np.asarray(chl[i, ys, xs]))
    obs_day = lambda i: np.isfinite(y(i)) & ocean
    obs_frac = np.empty(n)
    for i0 in range(0, n, 256):                                 # chunked pass: observed fraction of every day
        blk = np.isfinite(D.log_chl(np.asarray(chl[i0:i0 + 256, ys, xs]))) & ocean
        obs_frac[i0:i0 + 256] = blk.reshape(len(blk), -1).sum(1) / ocean.sum()
    tr, va, te = D.split_by_years(times, vy, ty)
    sst_name = "sst32" if store in ("blk", "blk4") else "sst"
    sst = lambda i: np.asarray(d[sst_name][i, ys, xs]).astype("float32")
    lat, lon = np.asarray(d["lat"])[ys].astype("float64"), np.asarray(d["lon"])[xs].astype("float64")
    _CTX[key] = dict(store=store, d=d, y=y, obs=obs_day, sst=sst, ocean=ocean, lat=lat, lon=lon, times=times,
                     tr=tr, va=va, te=te, obs_frac=obs_frac, _donors={})
    return _CTX[key]


def context_for(run_dir):
    cfg = _cfg(run_dir); sp = cfg.get("split") or {}
    return load_context(cfg.get("store", "io"), cfg.get("crop", False), sp.get("val_years"), sp.get("test_years"))


_PRED = {}


def load_preds(run_dir):
    run_dir = Path(run_dir)
    if run_dir not in _PRED:
        z = np.load(run_dir / "preds.npz")
        _PRED[run_dir] = dict(pred=z["pred"], hidden=z["hidden"], pos={int(i): k for k, i in enumerate(z["idx"])}, cfg=_cfg(run_dir))
    return _PRED[run_dir]


def sample_days(ctx, run_dir=None, n=32, split="va"):
    """n days of the split (only days with saved predictions of run_dir, if given), evenly spaced in rank of
    observed fraction. Runs on the same store with the same --val-stride get the same days."""
    idx = ctx[split]
    if run_dir is not None:
        idx = np.array(sorted(set(idx.tolist()) & set(load_preds(run_dir)["pos"])))
    order = idx[np.argsort(ctx["obs_frac"][idx], kind="stable")]
    return order[np.linspace(0, len(order) - 1, min(n, len(order))).round().astype(int)]


def _donor(ctx, cfg, day):
    """Donor day of the fixed val/test hidden mask (same draw as C.fixed_hidden in train_pinn.py)."""
    split = "va" if day in set(ctx["va"].tolist()) else "te"
    seed = (cfg.get("eval_seed") or 0) + (0 if split == "va" else 1)
    ex = cfg.get("donor_exclude") or 0
    key = (split, seed, ex)
    if key not in ctx["_donors"]:
        idx = ctx[split]
        local = C.donor_indices(len(idx), np.random.default_rng(seed), times=(ctx["times"][idx] if ex > 0 else None), exclude=ex)
        ctx["_donors"][key] = dict(zip(idx.tolist(), idx[local].tolist()))
    return split, ctx["_donors"][key][int(day)]


def _neighbours(ctx, k):
    """Store index of the calendar day k days after each day (-1 if that date is not in the store), cached per context."""
    nb = ctx.setdefault("_nb", {})
    if k not in nb:
        from .train2 import neighbour_index
        nb[k] = neighbour_index(ctx["times"], k)
    return nb[k]


def draw_day(run_dir, day, ctx=None, fig=None):
    """2x4 panel for one day: visible input, prediction, truth, hidden-pixel error (top); input at t-1 and t+1
    as the model sees it (calendar neighbours, as the trainer; donor clouds persisted unless --context-clouds none), SST, and the
    error histogram (bottom)."""
    run_dir = Path(run_dir); pr = load_preds(run_dir); cfg = pr["cfg"]
    ctx = ctx or context_for(run_dir)
    ocean, lat, lon, times = ctx["ocean"], ctx["lat"], ctx["lon"], ctx["times"]
    k = pr["pos"][int(day)]
    pred, hid = pr["pred"][k].astype("float32"), pr["hidden"][k]
    pred[~ocean] = np.nan
    truth = ctx["y"](day); obs_t = np.isfinite(truth) & ocean
    inp = np.where(obs_t & ~hid, truth, np.nan)
    err = np.where(hid, pred - truth, np.nan)
    split, donor = _donor(ctx, cfg, day)
    org = "lower" if lat[0] < lat[-1] else "upper"              # Black Sea stores run south -> north
    ext = [lon[0], lon[-1], min(lat[0], lat[-1]), max(lat[0], lat[-1])]
    mae = float(np.nanmean(np.abs(err))) if np.isfinite(err).any() else np.nan
    before = cfg.get("context_before") if cfg.get("context_before") is not None else (cfg.get("context") or 1)
    after = cfg.get("context_after") if cfg.get("context_after") is not None else (cfg.get("context") or 1)
    persist = cfg.get("context_clouds", "persist") == "persist"
    if fig is None:
        fig = plt.figure(figsize=(20, 8.2))
    fig.clf(); ax = fig.subplots(2, 4)
    vmin, vmax = np.nanpercentile(truth, 1), np.nanpercentile(truth, 99)
    panels = [("Input ln CHL, day t (visible pixels)", inp), ("Predicted ln CHL, day t", pred), ("True ln CHL, day t (observed pixels)", truth)]
    for j, (ttl, img) in enumerate(panels):
        im = ax[0, j].imshow(img, extent=ext, cmap=MAP_CMAP, vmin=vmin, vmax=vmax, aspect="auto", origin=org)
        ax[0, j].set(title=ttl, xlabel="lon (°E)", ylabel="lat (°N)"); fig.colorbar(im, ax=ax[0, j], label="ln CHL", shrink=.85)
    lim = np.nanpercentile(np.abs(err), 99) if np.isfinite(err).any() else 1
    im = ax[0, 3].imshow(err, extent=ext, cmap=ERR_CMAP, vmin=-lim, vmax=lim, aspect="auto", origin=org)
    ax[0, 3].set(title=f"Error on fake-cloud hidden pixels (MAE {mae:.3f})", xlabel="lon (°E)", ylabel="lat (°N)")
    fig.colorbar(im, ax=ax[0, 3], label="pred − true, ln CHL", shrink=.85)
    for j, off in enumerate((-1, 1)):
        nb = _neighbours(ctx, off)                            # calendar neighbours, as the trainer (train2.neighbour_index)
        i, di = int(nb[day]), int(nb[donor])
        di = di if di >= 0 else donor                         # trainer: donor's neighbour missing -> the donor day's own clouds
        in_window = before >= 1 if off < 0 else after >= 1
        if in_window and i >= 0:
            yi = ctx["y"](i); m = np.isfinite(yi) & ocean
            if persist:
                m &= ctx["obs"](di)
            img, ttl = np.where(m, yi, np.nan), f"Input ln CHL, day t{off:+d} (visible to the model)"
        elif in_window:
            img, ttl = np.full_like(truth, np.nan), f"Day t{off:+d}: date not in the store"
        else:
            img, ttl = np.full_like(truth, np.nan), f"Day t{off:+d}: not in the model's input window"
        im = ax[1, j].imshow(img, extent=ext, cmap=MAP_CMAP, vmin=vmin, vmax=vmax, aspect="auto", origin=org)
        ax[1, j].set(title=ttl, xlabel="lon (°E)", ylabel="lat (°N)")
        fig.colorbar(im, ax=ax[1, j], label="ln CHL", shrink=.85)
    sst = ctx["sst"](day); sst[~ocean] = np.nan
    im = ax[1, 2].imshow(sst, extent=ext, cmap="viridis", aspect="auto", origin=org)
    ax[1, 2].set(title="SST, day t", xlabel="lon (°E)", ylabel="lat (°N)"); fig.colorbar(im, ax=ax[1, 2], label="SST (store units)", shrink=.85)
    e = err[np.isfinite(err)]
    if e.size:
        ax[1, 3].hist(e, bins=np.linspace(-1.5, 1.5, 61), color=OKABE_ITO[0])
    ax[1, 3].axvline(0, color="k", lw=0.8)
    ax[1, 3].set(title=f"Hidden-pixel error distribution, day t (n={e.size:,})", xlabel="pred − true, ln CHL", ylabel="pixels"); ax[1, 3].grid(alpha=.3)
    fig.suptitle(f"{run_dir.name}: {split} day {str(times[day])[:10]}, observed {ctx['obs_frac'][day]:.0%} of ocean, "
                 f"hidden {hid[ocean].mean():.0%}, donor day {str(times[donor])[:10]}")
    fig.tight_layout()
    return fig
