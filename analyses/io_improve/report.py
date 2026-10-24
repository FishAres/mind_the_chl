"""Summary tables and figures of the io2_* track (plan: analyses/io_improve/PLAN.md), from results/<run>.json (score.py),
the runs' history.json / config.json, results/decisions.json (decide.py) and results/stoch_eval.json.

Main table: one row per run, effect first (val / test MAE, Δ val vs the baseline at the matched comparison, gradient ratio /
correlation at lag 1 on val), then epochs, then n. Matched comparison (PLAN section 3): 30-epoch runs = best val over epochs 11..30
vs the baseline's 0.132826; full-length runs and warm restarts = best val vs the baseline's 0.130165; ensembles = val vs 0.130165.
The item-4 train+val fits have test only (their val is in-sample).

Usage (project root): python -m analyses.io_improve.report
Writes results/summary.md, results/summary.csv and figures/{val_curves,seed_curves,bins,grad_lags}.png.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chl_cnn.plotting import OKABE_ITO
from chl_cnn.runs import find_run
from analyses.io_improve import decide as DC
from analyses.io_improve.score import DIST_BINS, LAGS
from analyses.forecast.forecast_eval import BINS as AGE_BINS

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
BASE_VAL60, BASE_VAL30 = 0.130165, None
ROWS = [  # (run, item, kind) kind: base | s30 | full | restart | ens | stoch | final
    ("io2_base_eval", "baseline t3d_dil_c4_60ep_gdl100", "base"),
    ("io2_r_plat", "1a restart, plateau lr 1e-3", "restart"), ("io2_r_cos", "1b restart, cosine lr 3e-4 x 20", "restart"),
    ("io2_seed1", "2 seed 1", "full"), ("io2_seed2", "2 seed 2", "full"),
    ("io2_seed1_r", "2 seed 1 + item-1 restart", "restart"), ("io2_seed2_r", "2 seed 2 + item-1 restart", "restart"),
    ("ens3pre_mean", "2 ensemble mean, 3 seeds before restarts", "ens"), ("ens3pre_median", "2 ensemble median, before restarts", "ens"),
    ("ens3_mean", "2 ensemble mean, 3 restarted seeds", "ens"), ("ens3_median", "2 ensemble median, 3 restarted seeds", "ens"),
    ("io2_w96_s30", "3 width 96, 30 ep", "s30"), ("io2_mem_s30", "5 memory channels, 30 ep", "s30"),
    ("io2_anom_s30", "6 memory + anomaly base, 30 ep", "s30"), ("io2_tattn8_s30", "7 attention stem ±8 d, 30 ep", "s30"),
    ("io2_tattn15_s30", "7 attention stem ±15 d, 30 ep", "s30"),
    ("io2_stoch", "8 stochastic (sample median, K=8)", "restart"),
    ("io2_w96", "B width 96, 60 ep", "full"), ("io2_w96_r", "B width 96 + restart", "restart"),
    ("io2_mem", "B memory, 60 ep", "full"), ("io2_mem_r", "B memory + restart", "restart"),
    ("io2_anom", "B memory + anomaly base, 60 ep", "full"), ("io2_anom_r", "B memory + anomaly + restart", "restart"),
    ("io2_tattn8", "B attention ±8, 60 ep", "full"), ("io2_tattn8_r", "B attention ±8 + restart", "restart"),
    ("io2_tattn15", "B attention ±15, 60 ep", "full"), ("io2_tattn15_r", "B attention ±15 + restart", "restart"),
    ("io2_combo", "B combination, 60 ep", "full"), ("io2_combo_r", "B combination + restart", "restart"),
    ("io2_final_s0_r", "4 train+val fit (seed 0)", "final"), ("io2_final_s1_r", "4 train+val fit (seed 1)", "final"),
    ("io2_final_s2_r", "4 train+val fit (seed 2)", "final"), ("ens3final_mean", "4 train+val fit, ensemble mean", "final"),
    ("ens3final_median", "4 train+val fit, ensemble median", "final"),
]


def res(run):
    f = RES / f"{run}.json"
    return json.loads(f.read_text()) if f.exists() else None


def epochs(run, kind):
    if kind in ("ens",) or run.startswith("ens"):
        return "", ""
    if run == "io2_base_eval":
        return 60, 54
    try:
        c = DC.cfg(run)
    except FileNotFoundError:
        return "", ""
    return c.get("stopped_epoch", ""), c.get("best_epoch", "")


def main():
    base30 = DC.val_at(DC.BASELINE, 30)[0]
    rows = []
    for run, item, kind in ROWS:
        r = res(run)
        if r is None:
            continue
        if kind == "s30":
            v30 = DC.val_at(run, 30)[0]
            delta, cmp_ = v30 - base30, f"best ep 11-30 vs {base30:.6f}"
        elif kind == "final":
            delta, cmp_ = None, "train+val fit: test only"
        else:
            delta, cmp_ = r["val"]["mae"] - BASE_VAL60, f"vs {BASE_VAL60:.6f}"
        if run == "io2_base_eval":
            delta, cmp_ = 0.0, "reference"
        ep, be = epochs(run, kind)
        g = r["val"]["grad"]["1"]
        rows.append(dict(run=run, item=item, val=(None if kind == "final" else r["val"]["mae"]), test=r["test"]["mae"], delta=delta, comparison=cmp_,
                         grad_ratio1=(None if kind == "final" else g["ratio"]), grad_corr1=(None if kind == "final" else g["corr"]),
                         test_grad_ratio1=r["test"]["grad"]["1"]["ratio"], test_grad_corr1=r["test"]["grad"]["1"]["corr"],
                         epochs=ep, best_epoch=be, n_val=r["val"]["n_hidden"], days_val=r["val"]["n_days"], n_test=r["test"]["n_hidden"],
                         days_test=r["test"]["n_days"], nonfinite=r["val"]["n_nonfinite"] + r["test"]["n_nonfinite"]))
    lines = ["| run | item | val MAE | test MAE | Δ val vs baseline | grad ratio / corr lag 1 (val) | epochs (best) | n val px / days | n test px / days |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        gr = "–" if r["grad_ratio1"] is None else f"{r['grad_ratio1']:.3f} / {r['grad_corr1']:.3f}"
        dl = "–" if r["delta"] is None else f"{r['delta']:+.4f}"
        vs = "–" if r["val"] is None else f"{r['val']:.4f}"
        lines.append(f"| {r['run']} | {r['item']} | {vs} | {r['test']:.4f} | {dl} | {gr} | "
                     f"{r['epochs']} ({r['best_epoch']}) | {r['n_val']:,} / {r['days_val']} | {r['n_test']:,} / {r['days_test']} |")
    b = DC.band()
    lines += ["", "Seed band (decide.band): " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in b.items()})]
    # bins: baseline vs best model by val (any non-final row)
    cand = [r for r in rows if r["val"] is not None and r["run"] != "io2_base_eval"]
    best = min(cand, key=lambda r: r["val"]) if cand else None
    if best:
        rb, rx = res("io2_base_eval"), res(best["run"])
        lines += ["", f"MAE by distance to nearest visible pixel and by age of last real observation (t-8..t-1): baseline vs {best['run']} (lowest val)", "",
                  "| split | bin | baseline MAE | best MAE | Δ | n px |", "|---|---|---|---|---|---|"]
        for split in ("val", "test"):
            for kind, bins in (("by_dist", [x[0] for x in DIST_BINS]), ("by_age", [x[0] for x in AGE_BINS])):
                for nm in bins:
                    u, v = rb[split][kind][nm], rx[split][kind][nm]
                    lab = (f"{nm} px" if kind == "by_dist" else f"age {nm}")
                    lines.append(f"| {split} | {lab} | {u['mae']:.4f} | {v['mae']:.4f} | {v['mae'] - u['mae']:+.4f} | {u['n']:,} |")
        lines += ["", "Gradient ratio / correlation by lag (val; test in brackets)", "", "| run | lag 1 | lag 2 | lag 4 |", "|---|---|---|---|"]
        for run in ["io2_base_eval", best["run"]] + [x for x in ("ens3_mean", "ens3_median") if res(x) and x != best["run"]]:
            q = res(run)
            lines.append(f"| {run} | " + " | ".join(f"{q['val']['grad'][str(L)]['ratio']:.3f} / {q['val']['grad'][str(L)]['corr']:.3f} "
                                                   f"({q['test']['grad'][str(L)]['ratio']:.3f} / {q['test']['grad'][str(L)]['corr']:.3f})" for L in LAGS) + " |")
    se = RES / "stoch_eval.json"
    if se.exists():
        s = json.loads(se.read_text())
        lines += ["", f"Item 8 ({s['run']}, K = {s['K']} samples; baseline {s['baseline']})", "",
                  "| split | MAE median | MAE mean | MAE baseline | CRPS (fair) | mean spread | single-sample ratio lag 1 / 2 / 4 | single-sample corr lag 1 / 2 / 4 | baseline ratio lag 1 / 2 / 4 | n px / days |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for split in ("val", "test"):
            q = s[split]
            sr = " / ".join(f"{q['single'][str(L)]['ratio_mean']:.3f}" for L in LAGS); sc = " / ".join(f"{q['single'][str(L)]['corr_mean']:.3f}" for L in LAGS)
            br = " / ".join(f"{q['baseline']['grad'][str(L)]['ratio']:.3f}" for L in LAGS)
            lines.append(f"| {split} | {q['median']['mae']:.4f} | {q['mean']['mae']:.4f} | {q['baseline']['mae']:.4f} | {q['crps']:.4f} | {q['mean_spread']:.4f} | "
                         f"{sr} | {sc} | {br} | {q['n_hidden']:,} / {q['n_days']} |")
        lines += ["", "| split | distance bin | spread | RMSE of mean | MAE of median | CRPS | MAE baseline | n px |", "|---|---|---|---|---|---|---|---|"]
        for split in ("val", "test"):
            for nm, _, _ in DIST_BINS:
                q = s[split]["by_dist"][nm]
                lines.append(f"| {split} | {nm} px | {q['spread']:.4f} | {q['rmse_mean']:.4f} | {q['mae_median']:.4f} | {q['crps']:.4f} | {q['mae_baseline']:.4f} | {q['n']:,} |")
    RES.mkdir(exist_ok=True)
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    with open(RES / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n".join(lines))
    figures(best)


def figures(best):
    FIG.mkdir(exist_ok=True)
    hb = DC.hist(DC.BASELINE)["val"]
    # 30-epoch options vs the baseline's first 30 epochs; warm restarts vs the source score
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(range(1, 31), hb[:30], color="k", lw=1.5, label="baseline (60-ep schedule), epochs 1-30")
    for i, run in enumerate(["io2_w96_s30", "io2_mem_s30", "io2_anom_s30", "io2_tattn8_s30", "io2_tattn15_s30"]):
        try:
            v = DC.hist(run)["val"]
        except FileNotFoundError:
            continue
        ax[0].plot(range(1, len(v) + 1), v, color=OKABE_ITO[i], label=run)
    ax[0].set(ylim=(0.128, 0.15), xlabel="epoch (60-epoch cosine schedule, stopped at 30)", ylabel="val hidden-pixel MAE (ln CHL)",
              title="Val MAE by epoch: tier-2 options vs baseline (1458 val days)")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].axhline(BASE_VAL60, color="k", lw=1, ls=":", label=f"baseline best {BASE_VAL60:.4f}")
    for i, run in enumerate(["io2_r_plat", "io2_r_cos", "io2_seed1_r", "io2_seed2_r", "io2_stoch"]):
        try:
            v = DC.hist(run)["val"]
        except FileNotFoundError:
            continue
        ax[1].plot(range(1, len(v) + 1), v, color=OKABE_ITO[i], label=run + (" (sample median)" if run == "io2_stoch" else ""))
    ax[1].set(xlabel="epoch after the warm restart", ylabel="val hidden-pixel MAE (ln CHL)", title="Val MAE by epoch after warm restarts (1458 val days)")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "val_curves.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (s, run) in enumerate(DC.SEEDS.items()):
        try:
            v = DC.hist(run)["val"]
        except FileNotFoundError:
            continue
        ax.plot(range(1, len(v) + 1), v, color=OKABE_ITO[i], label=f"seed {s} ({run})")
    ax.set(ylim=(0.128, 0.16), xlabel="epoch", ylabel="val hidden-pixel MAE (ln CHL)", title="Val MAE by epoch, baseline recipe, 3 seeds (1458 val days)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "seed_curves.png", dpi=120); plt.close(fig)

    if best is None:
        return
    rb, rx = res("io2_base_eval"), res(best["run"])
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for j, split in enumerate(("val", "test")):
        for i, (kind, bins, xl) in enumerate((("by_dist", [x[0] for x in DIST_BINS], "distance of hidden pixel to nearest visible pixel (px)"),
                                              ("by_age", [x[0] for x in AGE_BINS], "age of the last real observation within t-8..t-1"))):
            x = np.arange(len(bins)); a_ = ax[i, j]
            for k, (q, lab) in enumerate(((rb, "baseline"), (rx, best["run"]))):
                a_.bar(x + (k - 0.5) * 0.38, [q[split][kind][b]["mae"] for b in bins], 0.38, color=OKABE_ITO[k], label=lab)
            a_.set_xticks(x, [f"{b}\nn={rb[split][kind][b]['n'] / 1e6:.2f}M" for b in bins])
            a_.set(xlabel=xl, ylabel="hidden-pixel MAE (ln CHL)", title=f"{split}: MAE {'by distance to visible pixels' if i == 0 else 'by age of last observation'}")
            a_.grid(axis="y", alpha=.3); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "bins.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    runs = ["io2_base_eval", best["run"]] + [x for x in ("ens3_mean",) if res(x) and x != best["run"]]
    for k, run in enumerate(runs):
        q = res(run)
        for j, m in enumerate(("ratio", "corr")):
            ax[j].plot(LAGS, [q["val"]["grad"][str(L)][m] for L in LAGS], "o-", color=OKABE_ITO[k], label=run)
    ax[0].set(xlabel="lag (px, 1 px ≈ 28 km)", ylabel="Σ|Δpred| / Σ|Δtruth|", title="Val gradient ratio by lag", xticks=list(LAGS))
    ax[1].set(xlabel="lag (px, 1 px ≈ 28 km)", ylabel="uncentred correlation of Δpred, Δtruth", title="Val gradient correlation by lag", xticks=list(LAGS))
    for a_ in ax:
        a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "grad_lags.png", dpi=120); plt.close(fig)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
