"""Hidden-pixel comparison on the donor-exclude-8 fake-cloud set (IO store): nowcast vs gap filler vs forecast vs baselines.

Scored pixels: the fixed val / test hidden pixels drawn with --donor-exclude 8 (eval seed 0 / 1; no donor within 8 calendar
days of its target day), every val / test day. The hidden mask is read from each run's preds.npz and must be identical.
  nowcast      t3d_nc_b8: window t-8..t, day-t visible pixels as input, past days with their real observations
  gap filler   t3d_dil_c4_60ep_gdl100 weights, eval-only on the new set (run t3d_dil_c4_60ep_gdl100_dx8): window t-4..t+4,
               context days with the donor's clouds (--context-clouds persist), as trained
  forecast     t3d_fc_b8 weights, eval-only on the new set (run t3d_fc_b8_dx8): window t-8..t-1 CHL, no day-t CHL
  persistence  most recent observed ln CHL within t-1..t-8 at that pixel, DOY climatology where none (forecast_eval.py)
  climatology  DOY climatology (train2.doy_climatology over the training days)
Split by age of the last past observation (1, 2-3, 4-8 days, none). Model MAEs by bin use the float16 preds.npz;
"all (metrics.json)" is the trainer's float32 score.

Usage (project root): python -m analyses.forecast.nowcast_eval
Writes analyses/forecast/results/nowcast_comparison.json and figures/nowcast_comparison.png.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chl_cnn import data as D, store as S, train2 as T2
from chl_cnn.plotting import OKABE_ITO
from chl_cnn.runs import find_run
from analyses.forecast.forecast_eval import persistence, BINS

HERE = Path(__file__).resolve().parent
RUNS = [("nowcast", "t3d_nc_b8", "nowcast t-8..t (t3d_nc_b8)"),
        ("gapfill", "t3d_dil_c4_60ep_gdl100_dx8", "gap filler t-4..t+4 (t3d_dil_c4_60ep_gdl100)"),
        ("forecast", "t3d_fc_b8_dx8", "forecast t-8..t-1 (t3d_fc_b8)")]
BASE = [("persistence", "persistence (last obs. in t-8..t-1, else clim.)"), ("climatology", "DOY climatology")]
DONOR_EXCLUDE = 8


def score(err, age, mask):
    out = {"all": dict(mae=float(np.abs(err[mask]).mean()), n=int(mask.sum()))}
    for name, lo, hi in BINS:
        m = mask & ((age == 0) if hi == 0 else ((age >= lo) & (age <= hi)))
        out[name] = dict(mae=float(np.abs(err[m]).mean()) if m.any() else None, n=int(m.sum()))
    return out


def main():
    d = S.load_store(); times = d["time"]; T2.TIMES = times
    y = D.log_chl(np.asarray(d["chl_l3"])); obs = np.isfinite(y) & d["ocean"]
    tr, va, te = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    clim, dbin = T2.doy_climatology(y, obs, tr)

    preds, hidden, cfgs, metr = {}, None, {}, {}
    for key, run, _ in RUNS:
        p = find_run(run); pz = np.load(p / "preds.npz")
        cfg = json.loads((p / "config.json").read_text()); cfgs[key] = cfg
        assert cfg.get("donor_exclude") == DONOR_EXCLUDE and cfg["eval_seed"] == 0, (run, cfg.get("donor_exclude"))
        assert np.array_equal(pz["val_idx"], va) and np.array_equal(pz["test_idx"], te), f"{run}: not every val/test day"
        if hidden is None:
            hidden = pz["hidden"]
        assert np.array_equal(hidden, pz["hidden"]), f"{run}: different hidden set"
        preds[key] = {"val": pz["pred"][:len(va)].astype("float32"), "test": pz["pred"][len(va):].astype("float32")}
        metr[key] = json.loads((p / "metrics.json").read_text())["scores"]
    hid = {"val": hidden[:len(va)], "test": hidden[len(va):]}

    res = dict(donor_exclude=DONOR_EXCLUDE, runs={k: r for k, r, _ in RUNS},
               scored="fixed val/test hidden pixels (donor-exclude 8), every val/test day")
    for split, idx in [("val", va), ("test", te)]:
        yt, m = y[idx], hid[split]
        assert not (m & ~obs[idx]).any()
        pers, age = persistence(y, obs, times, idx, 8)
        cl = clim[dbin[idx]]; pers = np.where(age > 0, pers, cl)
        r = dict(n_days=len(idx), n_hidden=int(m.sum()), persistence=score(pers - yt, age, m), climatology=score(cl - yt, age, m))
        for key, run, _ in RUNS:
            r[key] = score(preds[key][split] - yt, age, m)
            r[key]["all_metrics_json"] = metr[key][split]["cnn_hidden"]["mae"]
            nan = ~np.isfinite(preds[key][split][m]); r[key]["n_nan_pred"] = int(nan.sum())
        res[split] = r
        print(split, "n_hidden", r["n_hidden"])
        for key, lab in [(k, l) for k, _, l in RUNS] + BASE:
            print(f"  {lab:50s} " + "  ".join(f"{c} {r[key][c]['mae']:.4f}" for c in ["all"] + [b[0] for b in BINS]))

    (HERE / "results").mkdir(exist_ok=True); (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "results" / "nowcast_comparison.json").write_text(json.dumps(res, indent=1))

    names = [(k, l) for k, _, l in RUNS] + BASE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cats = [b[0] for b in BINS] + ["all"]
    for ax, split in zip(axes, ["val", "test"]):
        x = np.arange(len(cats)); w = 0.8 / len(names)
        for i, (k, lab) in enumerate(names):
            v = [res[split][k][c]["mae"] or np.nan for c in cats]
            ax.bar(x + (i - (len(names) - 1) / 2) * w, v, w, color=OKABE_ITO[i], label=lab)
        ns = [res[split]["climatology"][c]["n"] for c in cats]
        ax.set_xticks(x, [f"{c}\nn={n / 1e6:.2f}M" for c, n in zip(cats, ns)])
        ax.set(title=f"{split} ({res[split]['n_days']} days)", xlabel="age of the last observation within t-1..t-8 (hidden pixels)",
               ylabel="MAE (ln CHL)"); ax.grid(axis="y", alpha=.3)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Hidden-pixel MAE by age of last observation (donor-exclude 8)")
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(HERE / "figures" / "nowcast_comparison.png", dpi=120); plt.close(fig)
    print("wrote", HERE / "results" / "nowcast_comparison.json", HERE / "figures" / "nowcast_comparison.png")


if __name__ == "__main__":
    main()
