"""Item 8 diagnostics: is the 0.54 gradient ratio MAE shrinkage? (io2_stoch vs the deterministic baseline; PLAN.md section 5)

io2_stoch = the baseline warm-started with zero-initialised noise injection, trained with the fair CRPS (M = 2 samples).
Its samples.npz holds K = 8 samples of day t (ln CHL, fixed noise seed) on every val / test day. On the fixed hidden pixels:
  - single samples: gradient ratio / correlation at lags 1, 2, 4 px for each sample (mean and range over the K samples);
  - sample median and sample mean: MAE, and their gradient ratio / correlation;
  - fair CRPS of the K samples (the deterministic baseline's CRPS equals its MAE);
  - spread-skill by distance to the nearest visible pixel: mean ensemble spread (std over samples, ddof 1) vs RMSE of the sample
    mean and MAE of the sample median; for a calibrated K-member ensemble, RMSE(mean) ~ sqrt((K + 1) / K) x spread.
Baseline: io2_base_eval (the baseline's weights, float32 predictions).

Usage (project root): python -m analyses.io_improve.stoch_eval [--run io2_stoch]
Writes analyses/io_improve/results/stoch_eval.json and figures/stoch_{gradients,spread_skill}.png.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chl_cnn.runs import find_run
from chl_cnn.plotting import OKABE_ITO
from analyses.io_improve.score import context, load_preds, grad_stats, DIST_BINS, LAGS

HERE = Path(__file__).resolve().parent


def fair_crps(S, y):
    K = S.shape[0]
    t1 = np.abs(S - y).mean(0)
    t2 = np.zeros_like(t1)
    for i in range(K):
        for j in range(i + 1, K):
            t2 += np.abs(S[i] - S[j])
    return t1 - t2 / (K * (K - 1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="io2_stoch")
    p.add_argument("--baseline", default="io2_base_eval")
    a = p.parse_args(argv)
    base, _ = load_preds(a.baseline)
    c = context()
    z = np.load(find_run(a.run) / "samples.npz")
    samples = z["samples"]; K = samples.shape[0]
    nv, nvr = len(c["va"]), len(z["val_idx"])                  # the run may score a subset of days (debug runs with --val-stride)
    res = dict(run=a.run, baseline=a.baseline, K=int(K), all_days=bool(nvr == nv))
    for split, ssl, idx, off, ref in (("val", slice(0, nvr), z["val_idx"], 0, c["va"]), ("test", slice(nvr, None), z["test_idx"], nv, c["te"])):
        pos = off + np.searchsorted(ref, idx); assert np.array_equal(np.r_[c["va"], c["te"]][pos], idx)
        S = samples[:, ssl].astype("float32"); y = c["y"][idx]; obs = c["obs"][idx]; hid = c["hidden"][pos]; dist = c["dist"][pos].astype("float32")
        b = base[pos]
        med, mean = np.median(S, 0), S.mean(0)
        r = dict(n_hidden=int(hid.sum()), n_days=len(idx), n_nonfinite_samples=int((~np.isfinite(S[:, hid])).sum()))
        r["single"] = {}
        for L in LAGS:
            g = [grad_stats(S[k], y, obs, hid, L) for k in range(K)]
            r["single"][str(L)] = dict(ratio_mean=float(np.mean([v["ratio"] for v in g])), ratio_min=float(min(v["ratio"] for v in g)),
                                       ratio_max=float(max(v["ratio"] for v in g)), corr_mean=float(np.mean([v["corr"] for v in g])),
                                       corr_min=float(min(v["corr"] for v in g)), corr_max=float(max(v["corr"] for v in g)), n_pairs=g[0]["n_pairs"])
        for nm, f in (("median", med), ("mean", mean), ("baseline", b)):
            r[nm] = dict(mae=float(np.abs(f - y)[hid].mean()), grad={str(L): grad_stats(f, y, obs, hid, L) for L in LAGS})
        Sh, yh = S[:, hid], y[hid]
        crps = fair_crps(Sh, yh); sd = Sh.std(0, ddof=1)
        r["crps"] = float(crps.mean()); r["baseline_crps_eq_mae"] = r["baseline"]["mae"]
        r["mean_spread"] = float(sd.mean())
        dh = dist[hid]; e_mean = (Sh.mean(0) - yh); e_med = (np.median(Sh, 0) - yh); e_b = b[hid] - yh
        r["by_dist"] = {}
        for name, lo, hi in DIST_BINS:
            m = (dh > lo) & (dh <= hi)
            r["by_dist"][name] = dict(n=int(m.sum()), spread=float(sd[m].mean()), rmse_mean=float(np.sqrt((e_mean[m] ** 2).mean())),
                                      mae_median=float(np.abs(e_med[m]).mean()), crps=float(crps[m].mean()), mae_baseline=float(np.abs(e_b[m]).mean()))
        res[split] = r
        print(f"{split}: n {r['n_hidden']} px / {r['n_days']} d; MAE median {r['median']['mae']:.5f} mean {r['mean']['mae']:.5f} baseline "
              f"{r['baseline']['mae']:.5f}; CRPS {r['crps']:.5f}; spread {r['mean_spread']:.4f}", flush=True)
        for L in LAGS:
            s = r["single"][str(L)]
            print(f"   lag {L}: single-sample ratio {s['ratio_mean']:.3f} [{s['ratio_min']:.3f}, {s['ratio_max']:.3f}] corr {s['corr_mean']:.3f}; "
                  f"median {r['median']['grad'][str(L)]['ratio']:.3f}/{r['median']['grad'][str(L)]['corr']:.3f}; "
                  f"baseline {r['baseline']['grad'][str(L)]['ratio']:.3f}/{r['baseline']['grad'][str(L)]['corr']:.3f}", flush=True)
        del S, Sh
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "stoch_eval.json").write_text(json.dumps(res, indent=1))

    (HERE / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (key, lab) in enumerate((("single", f"single sample (mean of {K})"), ("median", f"sample median ({K})"), ("baseline", "baseline (deterministic)"))):
        for j, q in enumerate(("ratio", "corr")):
            if key == "single":
                v = [res["val"]["single"][str(L)][f"{q}_mean"] for L in LAGS]
                lo = [res["val"]["single"][str(L)][f"{q}_min"] for L in LAGS]; hi = [res["val"]["single"][str(L)][f"{q}_max"] for L in LAGS]
                ax[j].fill_between(LAGS, lo, hi, color=OKABE_ITO[i], alpha=.25)
            else:
                v = [res["val"][key]["grad"][str(L)][q] for L in LAGS]
            ax[j].plot(LAGS, v, "o-", color=OKABE_ITO[i], label=lab)
    for j, q in enumerate(("gradient ratio Σ|Δpred| / Σ|Δtruth|", "gradient correlation (uncentred)")):
        ax[j].set(xlabel="lag (px, 1 px ≈ 28 km)", ylabel=q, xticks=list(LAGS), title=f"Val hidden-pixel {q.split(' ')[1] if j == 0 else 'correlation'} by lag")
        ax[j].grid(alpha=.3)
    ax[0].axhline(1, color="k", lw=.8, ls=":"); ax[0].legend(fontsize=8)
    n = res["val"]["single"]["1"]["n_pairs"]
    fig.suptitle(f"Gradient ratio and correlation by lag: stochastic samples vs baseline (val, {res['val']['n_days']} days, {n:,} lag-1 pairs)")
    fig.tight_layout(); fig.savefig(HERE / "figures" / "stoch_gradients.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = [b[0] for b in DIST_BINS]; x = np.arange(len(names))
    for i, (k, lab) in enumerate((("spread", "mean ensemble spread (std)"), ("rmse_mean", "RMSE of sample mean"),
                                  ("mae_median", "MAE of sample median"), ("crps", "CRPS (fair)"), ("mae_baseline", "MAE of baseline"))):
        ax.plot(x, [res["val"]["by_dist"][nm][k] for nm in names], "o-", color=OKABE_ITO[i], label=lab)
    ax.set_xticks(x, [f"{nm}\nn={res['val']['by_dist'][nm]['n'] / 1e6:.2f}M" for nm in names])
    ax.set(xlabel="distance of hidden pixel to nearest visible pixel (px)", ylabel="ln CHL", title="Spread and error by distance to visible pixels (val)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(HERE / "figures" / "stoch_spread_skill.png", dpi=120); plt.close(fig)
    print("wrote", HERE / "results" / "stoch_eval.json")


if __name__ == "__main__":
    main()
