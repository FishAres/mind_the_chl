"""Head-to-head on the 2024 hackweek protocol: our io_up runs, their public U-Net predictions, and a DOY climatology,
scored on the same pixels (2026-10-20).

Pixels and days
  days     their val slice 2018-01-01..2019-01-01 and test slice 2019-01-01..2020-01-01 (366 days each; store_io_up.split_up)
  hidden   their fixed mask: finite L3 at t & CHL_cmes-cloud == 0 at record t+10 (store hidden_up.npy), intersected with our
           ocean mask (land != 2 & topo <= 0). Same pixels for every model; pixels dropped by the ocean mask are counted.
Metrics
  1. hidden-pixel MAE / RMSE / bias in ln CHL (pooled over pixels), n pixels, n days; per-day MAE; days with < 5 hidden px.
     Also visible-pixel MAE: observed ocean px not in their hidden mask (the px each model saw as input on day t).
  2. gradient ratio sum|d pred| / sum|d true| and uncentred correlation sum(dp dt) / sqrt(sum dp^2 sum dt^2) over 1 px pairs
     along lat and lon with truth at both ends (observed, ocean) and at least one hidden end (as train_pinn.residual_stats)
  3. their z metric: mean over ALL pixels of the box x days of |z_pred - z_true|, z = (ln CHL - their all-record mean) / std,
     target NaN -> 0 (Keras model.evaluate MAE, U-Net_Tutorial cells 12, 28-29).
       raw        prediction at every pixel with a finite target; 0 where the target is NaN (land, real cloud), so those pixels
                  add 0 error. For their U-Net, level3_pred is their raw output at every finite-target pixel and NaN elsewhere,
                  so this convention is the only one available and gives a lower bound of their Keras number.
       composite  as raw, but pixels visible in the input (finite target, not hidden) take the observation (error 0)
  4. reproduction: their test MAE 0.0301 (cell 29) recomputed from level3_pred with the raw convention.
Data quality: NaN / inf counts of every prediction inside the scored pixels (reported, then excluded), days whose prediction is
all zero over the ocean.
Usage: python -m analyses.upstream.compare      -> results/summary.json, results/summary.csv, results/per_day_<split>.csv, figures/
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chl_cnn import data as D
from chl_cnn import store_io_up as SU
from chl_cnn import train2 as T2
from chl_cnn.plotting import OKABE_ITO
from chl_cnn.runs import CATEGORIES

HERE = Path(__file__).resolve().parent
PRED = D.DATA / "upstream_unet_pred"
RUNS = {"A ours, our inputs": "up_t3d_lam0p3_ours", "B ours, their inputs": "up_t3d_lam0p3_theirin",
        "A vis-weight 1.0": "up_t3d_lam0p3_ours_vis1", "B vis-weight 1.0": "up_t3d_lam0p3_theirin_vis1"}
UNET, CLIM = "2024 U-Net (level3_pred)", "DOY climatology 2015-17"
PUBLISHED = {"2024 U-Net, test 2019 (their box)": 0.03008805774152279, "OHW25 U-Net, test 2019 (Arabian Sea crop; not comparable)": 0.0354}


# ---------------------------------------------------------------- data
def load():
    d = SU.load_store()
    times = d["time"]; tr, va, te = SU.split_up(times)
    ocean = d["ocean"]
    y = D.log_chl(np.asarray(d["chl_l3"]))
    hid_up = np.asarray(d["hidden_up"])
    T2.TIMES = times
    obs = np.isfinite(y) & ocean
    clim, bins = T2.doy_climatology(y, obs, tr)
    preds = {}
    for name, run in RUNS.items():
        p = CATEGORIES["pinn/upstream"] / run / "preds.npz"
        if not p.exists():
            print(f"missing {p}; skipped"); continue
        z = np.load(p)
        full = np.full(y.shape, np.nan, dtype="float32"); full[z["idx"]] = z["pred"]
        preds[name] = full
    pt = np.load(PRED / "time.npy").astype("datetime64[D]"); l3p = np.load(PRED / "level3_pred.npy")
    pos = np.searchsorted(times.astype("datetime64[D]"), pt)
    assert np.array_equal(times.astype("datetime64[D]")[pos], pt)
    full = np.full(y.shape, np.nan, dtype="float32"); full[pos] = l3p; preds[UNET] = full
    preds[CLIM] = clim[bins]
    return dict(times=times, split=dict(val=va, test=te), ocean=ocean, y=y, hid_up=hid_up, obs=obs, preds=preds,
                mu=d["up"]["label_mean"], sd=d["up"]["label_std"])


# ---------------------------------------------------------------- analysis
def grad_stats(p, t, ob, hid):
    s = dict(gp=0.0, gt=0.0, gpt=0.0, gpp=0.0, gtt=0.0, n=0, n_nonfinite_pairs=0)
    for ax in (-1, -2):
        dp = np.diff(p, axis=ax); dt = np.diff(t, axis=ax)
        a0 = lambda x: np.take(x, np.arange(x.shape[ax] - 1), axis=ax); a1 = lambda x: np.take(x, np.arange(1, x.shape[ax]), axis=ax)
        m = a0(ob) & a1(ob) & (a0(hid) | a1(hid))
        bad = m & ~np.isfinite(dp); s["n_nonfinite_pairs"] += int(bad.sum()); m = m & np.isfinite(dp)
        dp, dt = dp[m].astype("float64"), dt[m].astype("float64")
        s["gp"] += np.abs(dp).sum(); s["gt"] += np.abs(dt).sum(); s["gpt"] += (dp * dt).sum(); s["gpp"] += (dp * dp).sum()
        s["gtt"] += (dt * dt).sum(); s["n"] += int(m.sum())
    return dict(grad_ratio=s["gp"] / s["gt"], grad_corr=s["gpt"] / np.sqrt(s["gpp"] * s["gtt"]), n_pairs=s["n"],
                n_nonfinite_pairs=s["n_nonfinite_pairs"])


def z_metric(p, t, hid_up, mu, sd):
    """Their all-pixel z MAE (raw and composite) and MSE (raw). p, t: (n, H, W) ln CHL. Non-finite predictions at finite-target
    pixels are counted and add 0 error (excluded from the numerator, kept in the denominator like every other pixel)."""
    fin = np.isfinite(t); n_all = t.size
    zt = np.where(fin, (t - mu) / sd, 0.0); zp = np.where(fin, (p - mu) / sd, 0.0)
    bad = fin & ~np.isfinite(zp); zp = np.where(bad, zt, zp)
    e = np.abs(zp - zt).astype("float64")
    comp = np.where(fin & ~hid_up, 0.0, e)
    return dict(z_mae_raw=e.sum() / n_all, z_mse_raw=(e ** 2).sum() / n_all, z_mae_composite=comp.sum() / n_all,
                n_pixels_all=int(n_all), n_target_finite=int(fin.sum()), n_hidden_up=int((hid_up & fin).sum()),
                n_pred_nonfinite_at_finite_target=int(bad.sum()))


def score(D_):
    y, ocean, hid_up, obs = D_["y"], D_["ocean"], D_["hid_up"], D_["obs"]
    out, per_day = {}, {}
    for split, idx in D_["split"].items():
        t = y[idx]; hu = hid_up[idx]; hid = hu & ocean
        ob = obs[idx]
        nonfin = {m: int((hid & ~np.isfinite(p[idx])).sum()) for m, p in D_["preds"].items()}
        common = hid.copy()
        for p in D_["preds"].values():
            common &= np.isfinite(p[idx])
        nh_day = common.reshape(len(idx), -1).sum(1)
        res = dict(n_days=len(idx), n_days_with_hidden=int((nh_day > 0).sum()), n_days_lt5_hidden=int((nh_day < 5).sum()),
                   days_lt5_hidden=[str(D_["times"][i])[:10] for i in np.asarray(idx)[nh_day < 5]],
                   n_hidden_their_mask=int((hu & np.isfinite(t)).sum()), n_hidden_dropped_outside_ocean=int((hu & ~ocean).sum()),
                   n_hidden_scored=int(common.sum()), nonfinite_pred_in_hidden=nonfin, models={})
        pdays = {}
        for m, P in D_["preds"].items():
            p = P[idx]; e = (p - t)[common].astype("float64")
            r = dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()), n_px=int(common.sum()))
            r.update(grad_stats(p, t, ob, common))
            vis = ob & ~hu & np.isfinite(p)                      # observed (ocean) px visible in the input
            ev = (p - t)[vis].astype("float64")
            r.update(vis_mae=float(np.abs(ev).mean()), vis_n_px=int(vis.sum()), vis_nonfinite=int((ob & ~hu & ~np.isfinite(p)).sum()))
            r.update(z_metric(p, t, hu, D_["mu"], D_["sd"]))
            oc_pred = np.where(ocean, p, np.nan)
            r["n_days_all_zero_pred"] = int((np.nan_to_num(oc_pred.reshape(len(idx), -1), nan=1.0) == 0).all(1).sum())
            ae = np.where(common, np.abs(p - t), 0.0).reshape(len(idx), -1).sum(1)
            pdays[m] = np.where(nh_day > 0, ae / np.maximum(nh_day, 1), np.nan)
            res["models"][m] = r
        out[split] = res
        per_day[split] = (D_["times"][idx], nh_day, pdays)
    return out, per_day


# ---------------------------------------------------------------- reporting
def write_tables(out, per_day):
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(dict(scores=out, published=PUBLISHED), indent=1))
    rows = ["model,val_mae,test_mae,val_rmse,test_rmse,val_bias,test_bias,val_grad_ratio,test_grad_ratio,val_grad_corr,test_grad_corr,"
            "val_z_mae_raw,test_z_mae_raw,val_z_mae_composite,test_z_mae_composite,val_vis_mae,test_vis_mae,val_n_px,test_n_px,val_vis_n_px,test_vis_n_px,val_n_days,test_n_days"]
    for m in out["val"]["models"]:
        v, t = out["val"]["models"][m], out["test"]["models"][m]
        f = lambda k: f"{v[k]:.4f},{t[k]:.4f}"
        rows.append(f"{m},{f('mae')},{f('rmse')},{f('bias')},{f('grad_ratio')},{f('grad_corr')},{f('z_mae_raw')},{f('z_mae_composite')},{f('vis_mae')},"
                    f"{v['n_px']},{t['n_px']},{v['vis_n_px']},{t['vis_n_px']},{out['val']['n_days_with_hidden']},{out['test']['n_days_with_hidden']}")
    (HERE / "results" / "summary.csv").write_text("\n".join(rows) + "\n")
    for split, (tt, nh, pdays) in per_day.items():
        names = list(pdays)
        lines = ["date,n_hidden," + ",".join(names)]
        for i in range(len(tt)):
            lines.append(f"{str(tt[i])[:10]},{nh[i]}," + ",".join(f"{pdays[m][i]:.5f}" for m in names))
        (HERE / "results" / f"per_day_{split}.csv").write_text("\n".join(lines) + "\n")


def plots(out, per_day):
    (HERE / "figures").mkdir(exist_ok=True)
    tt, nh, pdays = per_day["test"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    show = [m for m in list(RUNS) + [UNET] if m in pdays]
    for k, m in enumerate(show):
        ax[0].plot(tt, pdays[m], lw=1.0, color=OKABE_ITO[k], label=f"{m} (pooled {out['test']['models'][m]['mae']:.4f})")
    ax[0].set(ylabel="hidden-pixel MAE, ln CHL", title="Per-day hidden-pixel MAE, test days 2019-01-01..2020-01-01 (their t+10 fake-cloud mask)")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].plot(tt, nh, color=OKABE_ITO[7], lw=0.8)
    ax[1].set(ylabel="hidden px", xlabel="date", yscale="log", title="Hidden pixels scored per test day")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE / "figures" / "per_day_mae_test.png", dpi=120); plt.close(fig)

    models = list(out["val"]["models"])
    panels = [("mae", "hidden-pixel MAE (ln CHL)"), ("vis_mae", "visible-pixel MAE (ln CHL)"), ("grad_ratio", "gradient ratio, hidden pairs"), ("grad_corr", "gradient correlation, hidden pairs"),
              ("z_mae_raw", "all-pixel MAE, z units (raw)"), ("z_mae_composite", "all-pixel MAE, z units (composite)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5)); axes = axes.ravel()
    x = np.arange(len(models)); w = 0.38
    for ax, (k, lab) in zip(axes, panels):
        for j, split in enumerate(["val", "test"]):
            vals = [out[split]["models"][m][k] for m in models]
            ax.bar(x + (j - 0.5) * w, vals, w, color=OKABE_ITO[j], label=f"{split} (n px {out[split]['n_hidden_scored']:,})" if k == "mae" else split)
            for xi, v in zip(x + (j - 0.5) * w, vals):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        if k == "z_mae_raw":
            ax.axhline(PUBLISHED["2024 U-Net, test 2019 (their box)"], color=OKABE_ITO[3], ls="--", lw=1, label="published 2024 test 0.0301")
        ax.set_xticks(x); ax.set_xticklabels([m.replace(" (level3_pred)", "").replace(", ", ",\n") for m in models], fontsize=7, rotation=30, ha="right")
        top = max(out[s]["models"][m][k] for s in ("val", "test") for m in models)
        ax.set_ylim(0, top * 1.22)
        ax.set(title=lab, ylabel=lab); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Gap-fill scores on the 2024 hackweek protocol: val 2018, test 2019, identical hidden pixels")
    fig.tight_layout(); fig.savefig(HERE / "figures" / "summary_metrics.png", dpi=120); plt.close(fig)


def main():
    D_ = load()
    out, per_day = score(D_)
    write_tables(out, per_day)
    plots(out, per_day)
    for split in out:
        print(f"== {split}: days {out[split]['n_days']} hidden px scored {out[split]['n_hidden_scored']} "
              f"(their mask {out[split]['n_hidden_their_mask']}, outside ocean {out[split]['n_hidden_dropped_outside_ocean']}) "
              f"days <5 hidden {out[split]['n_days_lt5_hidden']}; non-finite in hidden {out[split]['nonfinite_pred_in_hidden']}")
        for m, r in out[split]["models"].items():
            print(f"  {m:28s} MAE {r['mae']:.4f} RMSE {r['rmse']:.4f} bias {r['bias']:+.4f} grad ratio {r['grad_ratio']:.3f} corr {r['grad_corr']:.3f} "
                  f"z raw {r['z_mae_raw']:.4f} (mse {r['z_mse_raw']:.5f}) z comp {r['z_mae_composite']:.4f} vis {r['vis_mae']:.4f} (n {r['vis_n_px']}) nonfinite@target {r['n_pred_nonfinite_at_finite_target']} "
                  f"all-zero days {r['n_days_all_zero_pred']}")


if __name__ == "__main__":
    main()
