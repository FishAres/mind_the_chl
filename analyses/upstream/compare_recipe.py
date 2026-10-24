"""Scores of the 2024-recipe runs (train_pinn.py --recipe up2024) next to the 10-20 io_up runs and the 2024 U-Net (2026-10-20).

Days      their val slice 2018-01-01..2019-01-01 and test slice 2019-01-01..2020-01-01 (366 days each; store_io_up.split_up)
Metrics   per split, every model on the same pixels:
  1. their all-pixel z metric (Keras model.evaluate, U-Net_Tutorial cells 12, 28-29): MSE and MAE of z_pred - z_true over all
     366 x 176 x 240 pixels, z = (ln CHL - their all-record mean) / std, target NaN -> 0.
       everywhere   prediction = the raw model output at every pixel (land / real cloud px add |z_pred|): the Keras number.
                    Pooled over pixels; Keras weights each batch mean by its sample count, so with 366 = 91 x 4 + 2 days the pooled
                    mean is what model.evaluate returns.
       lower bound  prediction replaced by the target (error 0) where the target is NaN. The only convention available for
                    level3_pred (no output at land / cloud px), so it is a lower bound of their Keras number.
  2. hidden-pixel ln MAE / RMSE / bias on their hidden set (finite L3 at t & cloud code 0 at record t+10):
       full         no ocean cut (the recipe's training mask)
       ocean        cut to our ocean mask (land != 2 & topo <= 0), the pixel set of the 10-20 table (analyses/upstream/compare.py)
  3. visible-pixel ln MAE: observed ocean px not in their hidden mask (as compare.py)
  4. gradient ratio sum|d pred| / sum|d true| and uncentred correlation over 1 px pairs along lat / lon with truth at both ends and
     >= 1 hidden end (as compare.py; ocean set, and full set in the JSON)
  5. n px for each; non-finite prediction counts.
Best / stopped epoch from each run's config.json / history.json. Published 2024 values: test MSE 0.00815, MAE 0.0301.
Usage: python -m analyses.upstream.compare_recipe -> results/recipe_{summary.json,summary.csv,per_day_val.csv,per_day_test.csv},
       figures/recipe_summary.png
       python -m analyses.upstream.compare_recipe --set recipe2   (2026-10-21, stage 1: length / lr schedule / physics weight)
           the 4 upr_* runs of 10-21, the 4 stage-1 runs (RUNS2) and the U-Net -> results/recipe2_{summary.json,summary.csv,
           per_day_val.csv,per_day_test.csv}, figures/recipe2_summary.png and figures/recipe2_curves.png (val all-pixel z MSE and lr
           vs epoch of each stage-1 run and its source / reference run; warm restarts drawn from the source's best epoch on).
           Run info adds final lr, number of lr drops (plateau) and, for warm restarts, the source run's best epoch and val z MSE.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chl_cnn import data as D
from chl_cnn import store_io_up as SU
from chl_cnn.plotting import OKABE_ITO
from chl_cnn.runs import CATEGORIES

HERE = Path(__file__).resolve().parent
PRED = D.DATA / "upstream_unet_pred"
RUNS = {"upr_t3d_lam0": "recipe, 3D stem, no physics", "upr_t3d_lam0p3": "recipe, 3D stem, upwind lam 0.3",
        "upr_dil96_lam0": "recipe, dilated w96, no physics", "upr_dil96_lam0p3": "recipe, dilated w96, upwind lam 0.3",
        "up_t3d_lam0p3_theirin": "10-20 B: 3D stem, their inputs, our loss", "up_t3d_lam0p3_ours": "10-20 A: 3D stem, our inputs, our loss"}
RUNS_UPR = {k: v for k, v in RUNS.items() if k.startswith("upr_")}
# stage 1 (2026-10-21): run -> (description, source / reference run for the curves, warm restart?)
RUNS2 = {"upr_dil96_lam0_plat": ("warm restart of upr_dil96_lam0, plateau lr, <= 200 ep", "upr_dil96_lam0", True),
         "upr_t3d_lam0_plat": ("warm restart of upr_t3d_lam0, plateau lr, <= 200 ep", "upr_t3d_lam0", True),
         "upr_t3d_lam0p03": ("recipe, 3D stem, upwind lam 0.03", "upr_t3d_lam0", False),
         "upr_dil96_lam0_cos150": ("recipe, dilated w96, cosine lr 150 ep", "upr_dil96_lam0", False)}
UNET = "level3_pred"
UNET_LABEL = "2024 U-Net (level3_pred)"
PUBLISHED = dict(test_z_mse=0.00815, test_z_mae=0.0301, source="U-Net_Tutorial cell 29 output (test 2019, their box)")


# ---------------------------------------------------------------- data
def load(runs=None):
    runs = RUNS if runs is None else runs
    d = SU.load_store()
    times = d["time"]; _, va, te = SU.split_up(times)
    y = D.log_chl(np.asarray(d["chl_l3"]))
    preds, runinfo = {}, {}
    for run in runs:
        rdir = CATEGORIES["pinn/upstream"] / run
        if not (rdir / "preds.npz").exists():
            print(f"missing {rdir / 'preds.npz'}; skipped"); continue
        z = np.load(rdir / "preds.npz")
        full = np.full(y.shape, np.nan, dtype="float32"); full[z["idx"]] = z["pred"]
        preds[run] = full
        cfg = json.loads((rdir / "config.json").read_text()); hist = json.loads((rdir / "history.json").read_text())
        runinfo[run] = dict(best_epoch=cfg.get("best_epoch"), stopped_epoch=cfg.get("stopped_epoch", len(hist["val"])),
                            max_epochs=cfg.get("epochs"), recipe=cfg.get("recipe", "none"), wall_min=cfg.get("wall_s", np.nan) / 60,
                            model=cfg.get("model"), width=cfg.get("width"), lam=cfg.get("lam"), phys=cfg.get("phys"), n_params=cfg.get("n_params"),
                            sched=cfg.get("sched"), final_lr=(hist["lr"][-1] if hist.get("lr") else None), n_lr_drops=cfg.get("n_lr_drops"),
                            best_val_z_mse=(min(hist["val_up_mse"]) if hist.get("val_up_mse") else None), source=hist.get("source"))
    pt = np.load(PRED / "time.npy").astype("datetime64[D]"); l3p = np.load(PRED / "level3_pred.npy")
    pos = np.searchsorted(times.astype("datetime64[D]"), pt)
    assert np.array_equal(times.astype("datetime64[D]")[pos], pt)
    full = np.full(y.shape, np.nan, dtype="float32"); full[pos] = l3p; preds[UNET] = full
    return dict(times=times, split=dict(val=va, test=te), ocean=d["ocean"], y=y, hid_up=np.asarray(d["hidden_up"]),
                preds=preds, runinfo=runinfo, mu=d["up"]["label_mean"], sd=d["up"]["label_std"])


# ---------------------------------------------------------------- metrics
def z_allpx(p, t, mu, sd):
    """Their all-pixel z MSE / MAE: 'everywhere' (raw output at every px) and 'lower bound' (0 error at NaN-target px)."""
    fin = np.isfinite(t); n = t.size
    zt = np.where(fin, (t - mu) / sd, 0.0)
    zp = ((p - mu) / sd).astype("float64")
    pfin = np.isfinite(zp)
    out = dict(n_px=int(n), n_target_finite=int(fin.sum()), n_pred_nonfinite=int((~pfin).sum()),
               n_pred_nonfinite_at_finite_target=int((fin & ~pfin).sum()))
    e_lb = np.where(fin & pfin, zp - zt, 0.0)
    out.update(z_mse_lb=float((e_lb ** 2).sum() / n), z_mae_lb=float(np.abs(e_lb).sum() / n))
    if pfin.all():
        e = zp - zt
        out.update(z_mse=float((e ** 2).sum() / n), z_mae=float(np.abs(e).sum() / n))
    else:                       # no output at some px (level3_pred): the 'everywhere' metric is undefined
        out.update(z_mse=np.nan, z_mae=np.nan)
    return out


def grad_stats(p, t, ob, hid):
    s = dict(gp=0.0, gt=0.0, gpt=0.0, gpp=0.0, gtt=0.0, n=0)
    for ax in (-1, -2):
        dp = np.diff(p, axis=ax); dt = np.diff(t, axis=ax)
        a0 = lambda x: np.take(x, np.arange(x.shape[ax] - 1), axis=ax); a1 = lambda x: np.take(x, np.arange(1, x.shape[ax]), axis=ax)
        m = a0(ob) & a1(ob) & (a0(hid) | a1(hid)) & np.isfinite(dp)
        dp, dt = dp[m].astype("float64"), dt[m].astype("float64")
        s["gp"] += np.abs(dp).sum(); s["gt"] += np.abs(dt).sum(); s["gpt"] += (dp * dt).sum(); s["gpp"] += (dp * dp).sum()
        s["gtt"] += (dt * dt).sum(); s["n"] += int(m.sum())
    return dict(grad_ratio=s["gp"] / s["gt"], grad_corr=s["gpt"] / np.sqrt(s["gpp"] * s["gtt"]), n_pairs=s["n"])


def err_stats(p, t, m):
    e = (p - t)[m].astype("float64")
    return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()), n_px=int(m.sum()))


def score(D_):
    y, ocean, hid_up = D_["y"], D_["ocean"], D_["hid_up"]
    out, per_day = {}, {}
    for split, idx in D_["split"].items():
        t = y[idx]; fin = np.isfinite(t); hu = hid_up[idx] & fin
        allfin = np.ones_like(hu)
        for P in D_["preds"].values():
            allfin &= np.isfinite(P[idx])
        h_full = hu & allfin; h_oc = hu & ocean & allfin
        ob_oc = fin & ocean; vis = ob_oc & ~hid_up[idx] & allfin
        res = dict(n_days=len(idx), n_hidden_their_mask=int(hu.sum()), n_hidden_full=int(h_full.sum()), n_hidden_ocean=int(h_oc.sum()),
                   n_hidden_outside_ocean=int((hu & ~ocean).sum()), n_visible_ocean=int(vis.sum()),
                   n_hidden_dropped_nonfinite_pred=int((hu & ~allfin).sum()), last_batch_size=len(idx) % 4 or 4, models={})
        pdays = dict(hidden_full_mae={}, z_mae={})
        for m, P in D_["preds"].items():
            p = P[idx]
            r = dict(hidden_full=err_stats(p, t, h_full), hidden_ocean=err_stats(p, t, h_oc), visible_ocean=err_stats(p, t, vis),
                     grad_ocean=grad_stats(p, t, ob_oc, h_oc), grad_full=grad_stats(p, t, fin, h_full),
                     zmetric=z_allpx(p, t, D_["mu"], D_["sd"]))
            res["models"][m] = r
            ae = np.where(h_full, np.abs(p - t), 0.0).reshape(len(idx), -1).sum(1); nh = h_full.reshape(len(idx), -1).sum(1)
            pdays["hidden_full_mae"][m] = np.where(nh > 0, ae / np.maximum(nh, 1), np.nan)
            zt = np.where(fin, (t - D_["mu"]) / D_["sd"], 0.0); zp = (p - D_["mu"]) / D_["sd"]
            pdays["z_mae"][m] = np.abs(zp - zt).reshape(len(idx), -1).mean(1) if np.isfinite(zp).all() else \
                np.abs(np.where(fin & np.isfinite(zp), zp - zt, 0.0)).reshape(len(idx), -1).mean(1)
        zt = np.where(fin, (t - D_["mu"]) / D_["sd"], 0.0).astype("float64")
        res["zero_pred"] = dict(z_mae=float(np.abs(zt).mean()), z_mse=float((zt ** 2).mean()))
        out[split] = res
        per_day[split] = (D_["times"][idx], h_full.reshape(len(idx), -1).sum(1), pdays)
    return out, per_day


# ---------------------------------------------------------------- reporting
DESC = {**RUNS, **{k: v[0] for k, v in RUNS2.items()}}


def label(m):
    return UNET_LABEL if m == UNET else f"{m} ({DESC[m]})"


def write_tables(out, per_day, runinfo, prefix="recipe"):
    res = HERE / "results"; res.mkdir(exist_ok=True)
    (res / f"{prefix}_summary.json").write_text(json.dumps(dict(scores=out, runinfo=runinfo, published=PUBLISHED), indent=1))
    cols = ["split", "model", "description", "z_mse", "z_mae", "z_mse_lower_bound", "z_mae_lower_bound", "hidden_full_mae", "hidden_full_rmse",
            "hidden_full_bias", "hidden_ocean_mae", "hidden_ocean_rmse", "visible_ocean_mae", "grad_ratio_ocean", "grad_corr_ocean",
            "best_epoch", "stopped_epoch"] + (["final_lr", "n_lr_drops"] if prefix != "recipe" else []) + ["n_px_all", "n_px_hidden_full", "n_px_hidden_ocean", "n_px_visible_ocean", "n_grad_pairs_ocean"]
    with open(res / f"{prefix}_summary.csv", "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC); w.writerow(cols)
        for split in ("test", "val"):
            for m, r in out[split]["models"].items():
                ri = runinfo.get(m, {})
                w.writerow([split, m, UNET_LABEL if m == UNET else DESC[m], r["zmetric"]["z_mse"], r["zmetric"]["z_mae"], r["zmetric"]["z_mse_lb"],
                            r["zmetric"]["z_mae_lb"], r["hidden_full"]["mae"], r["hidden_full"]["rmse"], r["hidden_full"]["bias"],
                            r["hidden_ocean"]["mae"], r["hidden_ocean"]["rmse"], r["visible_ocean"]["mae"], r["grad_ocean"]["grad_ratio"],
                            r["grad_ocean"]["grad_corr"], ri.get("best_epoch", ""), ri.get("stopped_epoch", "")]
                           + ([ri.get("final_lr", ""), ri.get("n_lr_drops", "") if ri.get("n_lr_drops") is not None else ""] if prefix != "recipe" else [])
                           + [r["zmetric"]["n_px"],
                            r["hidden_full"]["n_px"], r["hidden_ocean"]["n_px"], r["visible_ocean"]["n_px"], r["grad_ocean"]["n_pairs"]])
            w.writerow([split, "published 2024 U-Net", PUBLISHED["source"], PUBLISHED["test_z_mse"] if split == "test" else "",
                        PUBLISHED["test_z_mae"] if split == "test" else ""] + [""] * (len(cols) - 5))
    for split, (tt, nh, pdays) in per_day.items():
        with open(res / f"{prefix}_per_day_{split}.csv", "w", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            names = list(pdays["z_mae"])
            w.writerow(["date", "n_hidden_full"] + [f"hidden_full_mae {m}" for m in names] + [f"allpx_z_mae {m}" for m in names])
            for i in range(len(tt)):
                w.writerow([str(tt[i])[:10], int(nh[i])] + [float(pdays["hidden_full_mae"][m][i]) for m in names]
                           + [float(pdays["z_mae"][m][i]) for m in names])


def plots(out, prefix="recipe"):
    from matplotlib.patches import Patch
    (HERE / "figures").mkdir(exist_ok=True)
    models = list(out["test"]["models"])
    names = [UNET_LABEL if m == UNET else m for m in models]
    x = np.arange(len(models)); w = 0.38
    fig, axes = plt.subplots(2, 3, figsize=(16, 10)); axes = axes.ravel()

    def labels(ax, bars, vals, fmt):
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt(v), ha="center", va="bottom", fontsize=6.5, rotation=90)

    # z panels: test only, both conventions, log axis
    for ax, k, lab in [(axes[0], "z_mse", "all-pixel MSE, z units, test 2019"), (axes[1], "z_mae", "all-pixel MAE, z units, test 2019")]:
        ev = [out["test"]["models"][m]["zmetric"][k] for m in models]
        lb = [out["test"]["models"][m]["zmetric"][k + "_lb"] for m in models]
        b1 = ax.bar(x - w / 2, [v if np.isfinite(v) else 0 for v in ev], w, color=OKABE_ITO[1])
        b2 = ax.bar(x + w / 2, lb, w, color="white", edgecolor=OKABE_ITO[2], hatch="//")
        labels(ax, b1, ev, lambda v: f"{v:.4f}"); labels(ax, b2, lb, lambda v: f"{v:.4f}")
        pub, zero = PUBLISHED["test_" + k], out["test"]["zero_pred"][k]
        ax.axhline(pub, color=OKABE_ITO[5], ls="--", lw=1); ax.axhline(zero, color=OKABE_ITO[7], ls=":", lw=1)
        ax.set_yscale("log")
        ax.legend(handles=[Patch(color=OKABE_ITO[1], label="raw output at every px (Keras evaluate)"),
                           Patch(facecolor="white", edgecolor=OKABE_ITO[2], hatch="//", label="0 error at NaN-target px (lower bound)"),
                           plt.Line2D([], [], color=OKABE_ITO[5], ls="--", label=f"published 2024 U-Net test {pub}"),
                           plt.Line2D([], [], color=OKABE_ITO[7], ls=":", label=f"all-zero prediction {zero:.4f}")], fontsize=6.5, loc="upper left")
        ax.set_ylim(min(lb) * 0.5, max(max(v for v in ev if np.isfinite(v)), zero) * 8)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
        ax.set(title=lab + f" (n {out['test']['models'][models[0]]['zmetric']['n_px']:,} px)", ylabel=lab.split(",")[0] + ", z units (log)")
        ax.grid(alpha=.3, axis="y")
    panels = [(axes[2], lambda r: r["hidden_full"]["mae"], "hidden-pixel MAE, ln CHL (full hidden set)", "n_hidden_full"),
              (axes[3], lambda r: r["visible_ocean"]["mae"], "visible-pixel MAE, ln CHL (ocean)", "n_visible_ocean"),
              (axes[4], lambda r: r["grad_ocean"]["grad_ratio"], "gradient ratio, hidden pairs (ocean)", None),
              (axes[5], lambda r: r["grad_ocean"]["grad_corr"], "gradient correlation, hidden pairs (ocean)", None)]
    for ax, f, lab, nk in panels:
        top = 0
        for j, split in enumerate(["val", "test"]):
            vals = [f(out[split]["models"][m]) for m in models]; top = max(top, max(vals))
            nlab = f"{split} (n {out[split][nk]:,} px)" if nk else split
            bars = ax.bar(x + (j - 0.5) * w, vals, w, color=OKABE_ITO[j], label=nlab)
            labels(ax, bars, vals, lambda v: f"{v:.4f}" if v < 0.1 else f"{v:.3f}")
        ax.set_ylim(0, top * 1.22); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
        ax.set(title=lab, ylabel=lab.split(" (")[0]); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Gap-fill scores on the 2024 hackweek protocol: val 2018 / test 2019, 366 days each, identical pixels for every model")
    fig.tight_layout(); fig.savefig(HERE / "figures" / f"{prefix}_summary.png", dpi=120); plt.close(fig)


def stage1_summary():
    """Val-only stage-1 numbers: warm-restart gain over the source's best val z MSE (and the restart epoch reaching it), lr drops,
    and val z MSE of each stage-1 run vs its reference run at epochs 10 / 20 / 30 / 40 / 50 and best."""
    hist = lambda r: json.loads((CATEGORIES["pinn/upstream"] / r / "history.json").read_text())
    out = {}
    for r, (desc, ref, warm) in RUNS2.items():
        if not (CATEGORIES["pinn/upstream"] / r / "history.json").exists():
            continue
        hn, hr = hist(r), hist(ref)
        vn, vr = hn["val_up_mse"], hr["val_up_mse"]
        o = dict(reference=ref, warm_restart=warm, best_val_z_mse=min(vn), best_epoch=int(np.argmin(vn)) + 1, stopped_epoch=len(vn),
                 ref_best_val_z_mse=min(vr), ref_best_epoch=int(np.argmin(vr)) + 1,
                 lr_used=hn.get("lr_used"),
                 n_lr_drops=(int(sum(b < a for a, b in zip(hn.get("lr_used", []), hn["lr"]))) if json.loads((CATEGORIES["pinn/upstream"] / r / "config.json").read_text()
                                                                                                      if (CATEGORIES["pinn/upstream"] / r / "config.json").exists() else "{}").get("sched") == "plateau" else None))
        if warm:
            src = hn.get("source") or {}
            sb = src.get("best_score", min(vr))
            o.update(source_best_epoch=src.get("best_epoch"), source_best_val_z_mse=sb, gain_vs_source=min(vn) - sb,
                     first_epoch_below_source=next((i + 1 for i, v in enumerate(vn) if v < sb), None), epoch1_val_z_mse=vn[0])
        else:
            o["by_epoch"] = {str(e): dict(run=(vn[e - 1] if e <= len(vn) else None), reference=(vr[e - 1] if e <= len(vr) else None))
                             for e in (10, 20, 30, 40, 50)}
        out[r] = o
    (HERE / "results" / "recipe2_stage1.json").write_text(json.dumps(out, indent=1))
    for r, o in out.items():
        print(f"stage1 {r}: best val z mse {o['best_val_z_mse']:.6f} (ep {o['best_epoch']} / {o['stopped_epoch']}), ref {o['reference']} "
              f"{o['ref_best_val_z_mse']:.6f} (ep {o['ref_best_epoch']}), lr drops {o['n_lr_drops']}"
              + (f", gain vs source best {o['gain_vs_source']:+.6f}, first epoch below source {o['first_epoch_below_source']}, epoch-1 val {o['epoch1_val_z_mse']:.6f}"
                 if o["warm_restart"] else "  by epoch " + " ".join(f"{e}: {v['run']:.5f}/{v['reference']:.5f}" for e, v in o["by_epoch"].items()
                                                                     if v["run"] is not None and v["reference"] is not None)))


def curves():
    """recipe2_curves.png: val all-pixel z MSE (top) and lr (bottom) vs epoch, one column per stage-1 run with its source / reference run.
    Warm restarts: epoch k of the restart is drawn at source best epoch + k."""
    hist = lambda r: json.loads((CATEGORIES["pinn/upstream"] / r / "history.json").read_text())
    runs = [r for r in RUNS2 if (CATEGORIES["pinn/upstream"] / r / "history.json").exists()]
    if not runs:
        return
    fig, axes = plt.subplots(2, len(runs), figsize=(4.6 * len(runs), 7.5), sharex="col", squeeze=False)
    for j, r in enumerate(runs):
        desc, ref, warm = RUNS2[r]
        hr, hn = hist(ref), hist(r)
        e_ref = np.arange(1, len(hr["val_up_mse"]) + 1)
        off = (hn.get("source") or {}).get("best_epoch", int(np.argmin(hr["val_up_mse"])) + 1) if warm else 0
        e_new = off + np.arange(1, len(hn["val_up_mse"]) + 1)
        ax, al = axes[0, j], axes[1, j]
        ax.plot(e_ref, hr["val_up_mse"], color=OKABE_ITO[0], lw=1.4, label=f"{ref} (reference)")
        ax.plot(e_new, hn["val_up_mse"], color=OKABE_ITO[1], lw=1.4, label=r)
        bi_r, bi_n = int(np.argmin(hr["val_up_mse"])), int(np.argmin(hn["val_up_mse"]))
        ax.plot(e_ref[bi_r], hr["val_up_mse"][bi_r], "o", color=OKABE_ITO[0], ms=5)
        ax.plot(e_new[bi_n], hn["val_up_mse"][bi_n], "o", color=OKABE_ITO[1], ms=5)
        if warm:
            ax.axvline(off, color=OKABE_ITO[7], ls=":", lw=1)
        lo = min(min(hr["val_up_mse"]), min(hn["val_up_mse"]))
        ax.set_ylim(lo * 0.97, lo * 1.6)
        ax.set(title=f"{r}\nval 2018 all-pixel z MSE by epoch", ylabel="val all-pixel MSE, z units"); ax.grid(alpha=.3)
        ax.legend(fontsize=7, title=("dots: best epoch; dotted line: restart point" if warm else "dots: best epoch"), title_fontsize=7)
        lr_ref = hr.get("lr_used") or [hr["lr"][0]] * len(hr["lr"])       # pre-10-21 runs: constant lr (lr after step = lr used)
        lr_new = hn.get("lr_used") or hn["lr"]
        al.step(e_ref, lr_ref, where="mid", color=OKABE_ITO[0], lw=1.4, label=ref)
        al.step(e_new, lr_new, where="mid", color=OKABE_ITO[1], lw=1.4, label=r)
        al.set_yscale("log"); al.set(title=f"{r}\nlearning rate by epoch", xlabel="epoch" + (" (restart drawn from source best epoch)" if warm else ""),
                                     ylabel="lr used in the epoch (Adam)"); al.grid(alpha=.3); al.legend(fontsize=7)
    fig.suptitle("2024-recipe runs, stage 1: val all-pixel z MSE and learning rate by epoch (366 val days, 15,459,840 px)")
    fig.tight_layout(); fig.savefig(HERE / "figures" / "recipe2_curves.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["recipe", "recipe2"], default="recipe", help="recipe: the 10-21 runs + 10-20 runs (original); "
                    "recipe2: upr_* runs of 10-21 + stage-1 runs")
    args = ap.parse_args()
    runs = RUNS if args.set == "recipe" else {**RUNS_UPR, **{k: v[0] for k, v in RUNS2.items()}}
    D_ = load(runs)
    out, per_day = score(D_)
    write_tables(out, per_day, D_["runinfo"], prefix=args.set)
    plots(out, prefix=args.set)
    if args.set == "recipe2":
        curves(); stage1_summary()
    for split in ("test", "val"):
        o = out[split]
        print(f"== {split}: days {o['n_days']} hidden full {o['n_hidden_full']} ocean {o['n_hidden_ocean']} visible {o['n_visible_ocean']} "
              f"(dropped non-finite pred {o['n_hidden_dropped_nonfinite_pred']}); zero-pred z mae {o['zero_pred']['z_mae']:.4f} mse {o['zero_pred']['z_mse']:.5f}")
        for m, r in o["models"].items():
            zm = r["zmetric"]; ri = D_["runinfo"].get(m, {})
            print(f"  {m:24s} z mse {zm['z_mse']:.5f} mae {zm['z_mae']:.4f} (lb {zm['z_mse_lb']:.5f} / {zm['z_mae_lb']:.4f})  hidden full {r['hidden_full']['mae']:.4f} "
                  f"ocean {r['hidden_ocean']['mae']:.4f} rmse {r['hidden_ocean']['rmse']:.4f}  vis {r['visible_ocean']['mae']:.4f}  grad {r['grad_ocean']['grad_ratio']:.3f} / "
                  f"{r['grad_ocean']['grad_corr']:.3f}  ep {ri.get('best_epoch', '-')} / {ri.get('stopped_epoch', '-')}  lr end {ri.get('final_lr', '-')}  "
                  f"drops {ri.get('n_lr_drops', '-')}  nonfinite {zm['n_pred_nonfinite']}")
    print(f"published 2024 test: z mse {PUBLISHED['test_z_mse']} mae {PUBLISHED['test_z_mae']}")


if __name__ == "__main__":
    main()
