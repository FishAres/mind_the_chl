"""Score io5 runs with analyses/io_improve/score.py metrics, writing analyses/io5_ablation/results/<run>.json (io5 track, 2026-10-24).

Per split: hidden-px MAE, RMSE, bias, n, non-finite count; gradient ratio / corr at lags 1, 2, 4 px; MAE by distance to the nearest visible
px (1, 2-3, 4-7, 8-15, 16+); MAE by past-only age (t-8..t-1; "by_age") and, on val, by two-sided age (nearest real obs within t+-8 excl. t;
"by_age2"; cache ~/mind_the_chl_gap/data/io3/val_age_twosided.npy from analyses/io3_attn/checks.age_twosided).

Usage (project root):
  python -m analyses.io5_ablation.score_abl RUN [RUN ...]              val and test
  python -m analyses.io5_ablation.score_abl --val-only --drop RUN ...   val only; then delete the run's preds.npz and model.pt (ablation evals:
                                                                        model.pt is a copy of the source weights)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyses.io_improve import score as SC
from analyses.forecast.forecast_eval import BINS as AGE_BINS
from chl_cnn.runs import find_run

RES = Path(__file__).resolve().parent / "results"
AGE2 = Path.home() / "mind_the_chl_gap" / "data" / "io3" / "val_age_twosided.npy"


def age2_bins(pred_v, y_v, hid_v, age2):
    ae = np.abs(pred_v - y_v)
    out = {}
    for name, lo, hi in AGE_BINS:
        m = hid_v & ((age2 == 0) if hi == 0 else ((age2 >= lo) & (age2 <= hi)))
        out[name] = dict(mae=float(ae[m].mean()) if m.any() else None, n=int(m.sum()))
    return out


def score_run(run, val_only=False, drop=False):
    p = find_run(run)
    pred, dtype = SC.load_preds(run)
    c = SC.context()
    nv = len(c["va"])
    res = dict(run=run, preds_dtype=dtype)
    cfg = json.loads((p / "config.json").read_text())
    res["ablate_info"] = cfg.get("ablate_info")
    m = json.loads((p / "metrics.json").read_text())["scores"]
    res["trainer_physics"] = {s: {k: m[s]["physics"][k] for k in ("abs_r_cloud", "abs_r_obs", "n_cloud", "n_obs")} for s in ("val", "test")}
    splits = [("val", slice(0, nv), c["va"])] + ([] if val_only else [("test", slice(nv, None), c["te"])])
    for split, sl, idx in splits:
        res[split] = SC.split_metrics(pred[sl], c["y"][idx], c["obs"][idx], c["hidden"][sl], c["dist"][sl], c["age"][sl])
        res[split]["trainer_mae"] = m[split]["cnn_hidden"]["mae"]
    res["val"]["by_age2"] = age2_bins(pred[:nv], c["y"][c["va"]], c["hidden"][:nv], np.load(AGE2))
    RES.mkdir(exist_ok=True)
    (RES / f"{run}.json").write_text(json.dumps(res, indent=1))
    g = res["val"]["grad"]["1"]
    print(f"{run:40s} val {res['val']['mae']:.6f} (trainer {res['val']['trainer_mae']:.6f})  rmse {res['val']['rmse']:.4f}  bias {res['val']['bias']:+.4f}  "
          f"grad1 {g['ratio']:.3f}/{g['corr']:.3f}  nonfinite {res['val']['n_nonfinite']}"
          + ("" if val_only else f"  test {res['test']['mae']:.6f}"), flush=True)
    if drop:
        for f in ("preds.npz", "model.pt"):
            (p / f).unlink(missing_ok=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--drop", action="store_true")
    a = ap.parse_args()
    for r in a.runs:
        score_run(r, a.val_only, a.drop)
