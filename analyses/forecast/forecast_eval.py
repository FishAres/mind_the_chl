"""Forecast baselines and model scoring on the observed pixels of day t (IO store), for the causal 3D-stem run.

Scored pixels: every observed ocean pixel of each val / test day t (the trainer's metrics.json "cnn_observed" set;
all val and test days, --val-stride 1).
  model         preds.npz of the run (float16 ln CHL; metrics.json holds the exact float32 score)
  persistence   most recent observed ln CHL at that pixel within days t-1..t-8 (calendar days; days missing from
                the store count as unobserved); day-of-year climatology where none of the 8 days has an observation
  climatology   day-of-year climatology (train_pinn: train2.doy_climatology over the training days, 8-day bins)
Every score is also split by the age of the last observation: 1, 2-3, 4-8 days, none.

Usage: python -m analyses.forecast.forecast_eval [--run t3d_fc_b8] [--max-lag 8]
(run from the project root; without a finished run only the baselines are scored).
Writes analyses/forecast/results/<run>.json and figures/<run>_by_age.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chl_cnn import data as D, store as S, train2 as T2
from chl_cnn.plotting import OKABE_ITO
from chl_cnn.runs import find_run

HERE = Path(__file__).resolve().parent
BINS = [("1 d", 1, 1), ("2-3 d", 2, 3), ("4-8 d", 4, 8), ("none", 0, 0)]


def persistence(y, obs, times, idx, max_lag):
    """(len(idx), H, W) last observed ln CHL within t-1..t-max_lag (NaN if none) and its age in days (0 = none)."""
    nb = {k: T2.neighbour_index(times, -k) for k in range(1, max_lag + 1)}
    pers = np.full((len(idx),) + y.shape[1:], np.nan, dtype="float32"); age = np.zeros(pers.shape, dtype=np.int8)
    for i, t in enumerate(idx):
        for k in range(1, max_lag + 1):
            j = nb[k][t]
            if j < 0:
                continue
            new = obs[j] & (age[i] == 0)
            pers[i][new] = y[j][new]; age[i][new] = k
    return pers, age


def score(err, age, mask):
    out = {"all": dict(mae=float(np.abs(err[mask]).mean()), n=int(mask.sum()))}
    for name, lo, hi in BINS:
        m = mask & ((age == 0) if hi == 0 else ((age >= lo) & (age <= hi)))
        out[name] = dict(mae=float(np.abs(err[m]).mean()) if m.any() else None, n=int(m.sum()))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="t3d_fc_b8")
    p.add_argument("--max-lag", type=int, default=8)
    a = p.parse_args(argv)

    d = S.load_store(); times = d["time"]; T2.TIMES = times
    ocean = d["ocean"]
    y = D.log_chl(np.asarray(d["chl_l3"])); obs = np.isfinite(y) & ocean
    tr, va, te = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    clim, dbin = T2.doy_climatology(y, obs, tr)

    run = None
    try:
        run = find_run(a.run)
        pz = np.load(run / "preds.npz")
        assert np.array_equal(pz["val_idx"], va) and np.array_equal(pz["test_idx"], te), "run was not scored on every val/test day"
        preds = {"val": pz["pred"][:len(va)].astype("float32"), "test": pz["pred"][len(va):].astype("float32")}
        metr = json.loads((run / "metrics.json").read_text())["scores"]
    except FileNotFoundError:
        print(f"run {a.run} not found: baselines only")

    res = dict(run=a.run, max_lag=a.max_lag, scored="all observed ocean pixels of day t, every val/test day")
    for split, idx in [("val", va), ("test", te)]:
        yt, m = y[idx], obs[idx]
        pers, age = persistence(y, obs, times, idx, a.max_lag)
        cl = clim[dbin[idx]]
        pers = np.where(age > 0, pers, cl)
        r = dict(n_days=len(idx), persistence=score(pers - yt, age, m), climatology=score(cl - yt, age, m))
        if run is not None:
            r["model"] = score(preds[split] - yt, age, m)
            r["model_metrics_json"] = dict(observed=metr[split]["cnn_observed"], hidden=metr[split]["cnn_hidden"])
        res[split] = r
        print(split, json.dumps({k: (v["all"] if isinstance(v, dict) and "all" in v else None) for k, v in r.items()}))

    (HERE / "results").mkdir(exist_ok=True); (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "results" / f"{a.run}.json").write_text(json.dumps(res, indent=1))

    names = [("model", f"model ({a.run})"), ("persistence", "persistence (last obs. within 8 d, else clim.)"), ("climatology", "day-of-year climatology")]
    names = [(k, lab) for k, lab in names if k in res["val"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split in zip(axes, ["val", "test"]):
        cats = [b[0] for b in BINS] + ["all"]
        x = np.arange(len(cats)); w = 0.8 / len(names)
        for i, (k, lab) in enumerate(names):
            v = [res[split][k][c]["mae"] or np.nan for c in cats]
            ax.bar(x + (i - (len(names) - 1) / 2) * w, v, w, color=OKABE_ITO[i], label=lab)
        ns = [res[split]["climatology"][c]["n"] for c in cats]
        ax.set_xticks(x, [f"{c}\nn={n / 1e6:.2f}M" for c, n in zip(cats, ns)])
        ax.set(title=f"{split}: MAE on observed day-t pixels by age of last observation", xlabel="age of the last observation within t-1..t-8 (pixels)",
               ylabel="MAE, ln CHL"); ax.grid(axis="y", alpha=.3)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=len(names), fontsize=9, frameon=False)
    fig.suptitle("IO day-t ln CHL from CHL of days t-8..t-1 (model also gets physical features of t-8..t): model vs persistence vs climatology")
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(HERE / "figures" / f"{a.run}_by_age.png", dpi=120); plt.close(fig)
    print("wrote", HERE / "results" / f"{a.run}.json")


if __name__ == "__main__":
    main()
