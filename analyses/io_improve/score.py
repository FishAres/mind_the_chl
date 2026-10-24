"""Per-run metrics for the io2_* track (IO store, fixed val/test hidden pixels; plan: analyses/io_improve/PLAN.md section 2).

For each run (preds.npz; float32 with --preds-f32, float16 otherwise) and each split (val, test):
  hidden-pixel MAE, RMSE, bias (pred - truth), n hidden px, n days, n non-finite predictions on hidden px;
  gradient ratio sum|d pred| / sum|d truth| and uncentred gradient correlation sum(dp dt) / sqrt(sum dp^2 sum dt^2) at lags 1, 2, 4 px
    along both axes, on pairs with truth at both ends and >= 1 hidden end (lag 1 = the trainer's val_grad_ratio / val_grad_corr);
  MAE by Euclidean distance of the hidden px to the nearest visible px of day t (visible = observed and not hidden; land counts
    as not visible): 1, 2-3, 4-7, 8-15, 16+ px;
  MAE by age of the last real observation within t-8..t-1 (analyses/forecast/forecast_eval.persistence): 1 d, 2-3 d, 4-8 d, none.
Ensembles: --ensemble NAME=run1,run2,run3 scores the per-pixel mean (NAME_mean) and median (NAME_median) of the runs' predictions.

Usage (project root): python -m analyses.io_improve.score --runs io2_r_plat io2_seed1 [--ensemble ens3=a,b,c]
Writes analyses/io_improve/results/<run>.json. The distance / age arrays are cached in ~/mind_the_chl_gap/data/io2_score_cache.npz.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from chl_cnn import data as D, store as S, train2 as T2
from chl_cnn.runs import find_run
from analyses.forecast.forecast_eval import persistence, BINS as AGE_BINS

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CACHE = Path.home() / "mind_the_chl_gap" / "data" / "io2_score_cache.npz"
DIST_BINS = [("1", 0.5, 1.5), ("2-3", 1.5, 3.5), ("4-7", 3.5, 7.5), ("8-15", 7.5, 15.5), ("16+", 15.5, np.inf)]
LAGS = (1, 2, 4)
_CTX = {}


def context(hidden_ref=None):
    """Store truth, observed mask, val/test indices, and (cached) distance-to-visible and age-of-last-observation arrays."""
    if _CTX:
        return _CTX
    d = S.load_store(); times = d["time"]; T2.TIMES = times
    y = D.log_chl(np.asarray(d["chl_l3"])); obs = np.isfinite(y) & d["ocean"]
    tr, va, te = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    _CTX.update(y=y, obs=obs, va=va, te=te, times=times, ocean=d["ocean"])
    if CACHE.exists():
        c = np.load(CACHE)
        if hidden_ref is None or np.array_equal(c["hidden"], hidden_ref):
            _CTX.update(hidden=c["hidden"], dist=c["dist"], age=c["age"])
            return _CTX
    assert hidden_ref is not None, "no cache yet: pass the hidden mask of a run"
    idx = np.r_[va, te]
    vis = obs[idx] & ~hidden_ref
    dist = np.stack([distance_transform_edt(~v) for v in vis]).astype("float16")
    age = np.concatenate([persistence(y, obs, times, ix, 8)[1] for ix in (va, te)])
    np.savez(CACHE, hidden=hidden_ref, dist=dist, age=age)
    _CTX.update(hidden=hidden_ref, dist=dist, age=age)
    return _CTX


def load_preds(run):
    p = find_run(run); z = np.load(p / "preds.npz")
    c = context(z["hidden"])
    assert np.array_equal(z["val_idx"], c["va"]) and np.array_equal(z["test_idx"], c["te"]), f"{run}: not every val/test day"
    assert np.array_equal(z["hidden"], c["hidden"]), f"{run}: different hidden set"
    return z["pred"].astype("float32"), str(z["pred"].dtype)


def grad_stats(pred, y, obs, hid, lag):
    s = dict(p=0.0, t=0.0, pt=0.0, pp=0.0, tt=0.0, n=0)
    for ax in (1, 2):
        n = pred.shape[ax]
        a0, a1 = (slice(None),) * ax + (slice(0, n - lag),), (slice(None),) * ax + (slice(lag, n),)
        m = obs[a0] & obs[a1] & (hid[a0] | hid[a1])
        dp = (pred[a1] - pred[a0])[m].astype("float64"); dt = (y[a1] - y[a0])[m].astype("float64")
        s["p"] += np.abs(dp).sum(); s["t"] += np.abs(dt).sum(); s["pt"] += (dp * dt).sum(); s["pp"] += (dp ** 2).sum(); s["tt"] += (dt ** 2).sum()
        s["n"] += int(m.sum())
    return dict(ratio=s["p"] / s["t"], corr=s["pt"] / np.sqrt(s["pp"] * s["tt"]), n_pairs=s["n"])


def split_metrics(pred, y, obs, hid, dist, age):
    e = (pred - y)[hid].astype("float64")
    fin = np.isfinite(e)
    out = dict(mae=float(np.abs(e[fin]).mean()), rmse=float(np.sqrt((e[fin] ** 2).mean())), bias=float(e[fin].mean()),
               n_hidden=int(hid.sum()), n_days=int(len(pred)), n_nonfinite=int((~fin).sum()))
    out["grad"] = {str(L): grad_stats(pred, y, obs, hid, L) for L in LAGS}
    ae = np.abs(pred - y)
    dv = dist.astype("float32")
    out["by_dist"] = {}
    for name, lo, hi in DIST_BINS:
        m = hid & (dv > lo) & (dv <= hi)
        out["by_dist"][name] = dict(mae=float(ae[m].mean()) if m.any() else None, n=int(m.sum()))
    out["by_age"] = {}
    for name, lo, hi in AGE_BINS:
        m = hid & ((age == 0) if hi == 0 else ((age >= lo) & (age <= hi)))
        out["by_age"][name] = dict(mae=float(ae[m].mean()) if m.any() else None, n=int(m.sum()))
    return out


def score_pred(name, pred, info):
    c = context()
    nv = len(c["va"])
    res = dict(run=name, **info)
    for split, sl, idx in (("val", slice(0, nv), c["va"]), ("test", slice(nv, None), c["te"])):
        res[split] = split_metrics(pred[sl], c["y"][idx], c["obs"][idx], c["hidden"][sl], c["dist"][sl], c["age"][sl])
    RES.mkdir(exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(res, indent=1))
    g = res["val"]["grad"]
    print(f"{name:28s} val {res['val']['mae']:.6f} test {res['test']['mae']:.6f}  bias {res['val']['bias']:+.4f}/{res['test']['bias']:+.4f}  "
          f"grad ratio/corr lag1 {g['1']['ratio']:.3f}/{g['1']['corr']:.3f} lag2 {g['2']['ratio']:.3f}/{g['2']['corr']:.3f} "
          f"lag4 {g['4']['ratio']:.3f}/{g['4']['corr']:.3f}  nonfinite {res['val']['n_nonfinite']}/{res['test']['n_nonfinite']}", flush=True)
    return res


def score_run(run):
    pred, dtype = load_preds(run)
    p = find_run(run)
    info = dict(preds_dtype=dtype)
    if (p / "metrics.json").exists():                          # trainer's own numbers, for the consistency check
        m = json.loads((p / "metrics.json").read_text())["scores"]
        info["trainer"] = {s: dict(mae=m[s]["cnn_hidden"]["mae"], grad_ratio=m[s]["physics"]["grad_ratio"], grad_corr=m[s]["physics"]["grad_corr"])
                           for s in ("val", "test")}
    res = score_pred(run, pred, info)
    if "trainer" in info:
        t = info["trainer"]["val"]
        print(f"   check vs trainer (val): MAE {res['val']['mae'] - t['mae']:+.2e}  grad ratio {res['val']['grad']['1']['ratio'] - t['grad_ratio']:+.2e}  "
              f"corr {res['val']['grad']['1']['corr'] - t['grad_corr']:+.2e}  (preds {dtype})", flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--ensemble", nargs="*", default=[], help="NAME=run1,run2,...")
    a = p.parse_args(argv)
    for r in a.runs:
        try:
            score_run(r)
        except FileNotFoundError as e:
            print(f"{r}: skipped ({e})", flush=True)
    for spec in a.ensemble:
        name, runs = spec.split("=")
        runs = runs.split(",")
        P = np.stack([load_preds(r)[0] for r in runs])
        score_pred(f"{name}_mean", P.mean(0), dict(members=runs, combine="mean"))
        score_pred(f"{name}_median", np.median(P, 0), dict(members=runs, combine="median"))


if __name__ == "__main__":
    main()
