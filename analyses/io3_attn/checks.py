"""Part A of the io3 efficient-attention track: cheap checks on io2_tattn8 (60 epochs, ±8 days, batch 4); no training.

Inputs (made by analyses/io3_attn/attn_eval.py runs, scored by analyses/io3_attn/score_runs.py):
  results/io3_eval_tattn8_{pre,post,rec,pool4,noself}.json   score.py metrics of the eval-only runs
  results/io3_eval_tattn8_rec_poolweights.json              pooling weights accumulated on every val hidden pixel (A1a)
  ~/mind_the_chl_gap/data/io3/io3_eval_tattn8_rec_weights.npz   weight maps (mean over the 4 heads) of every 15th val day (A4a)
  models/pinn/t3d/<run>/history.json, config.json            A2
  analyses/io_improve/results/{io2_tattn8,io2_base_eval,io2_seed1,io2_seed2}.json   A4b

A1a  mean pooling weight by day offset over val hidden pixels, split by whether that day was visible at the pixel (donor-masked
     context mask, as fed to the stem); share of weight on |offset| > 4.
A1b  val MAE with the pooling softmax restricted to |offset| <= 4 (eval-only, renormalised).
A2   val MAE vs optimiser steps (ceil(5540 / batch) per epoch) and vs training days seen, from history.json.
A3   val MAE with the self-attention block bypassed (eval-only).
A4a  spatial autocorrelation of the pooling-weight maps vs lag (px, along rows and columns, ocean pixel pairs): per sample day,
     rho(L) = sum_k C_k(L) / sum_k V_k over the 17 offsets k (C_k = mean product of deviations from the map's ocean mean over pixel
     pairs at lag L, V_k = the map's ocean variance), i.e. a variance-weighted mean over offsets; the lag where rho first drops below
     0.5 (linear interpolation), median over days. Secondary variant: only pairs visible on day k at both ends, deviations from the
     mean over visible pixels.
A4b  MAE gain (conv ±4 minus io2_tattn8) by age of the last real observation and by distance to the nearest visible pixel, val;
     conv seed 0 = t3d_dil_c4_60ep_gdl100 (io2_base_eval), range over seeds 0/1/2.
     "age" (score.py, analyses/forecast/forecast_eval.persistence) = days back to the last real (store-observed, not donor-masked)
     observation at the pixel within t-8..t-1, past days only. Two-sided version (added 2026-10-23, coordinator item 4): days to the
     nearest real observation within t-8..t+8 excluding t (calendar neighbours), same bins 1 d / 2-3 d / 4-8 d / none; from the
     runs' existing preds.npz (eval-only, no new run).

Usage (project root): python -m analyses.io3_attn.checks
Writes results/checks.json and figures/a1_pool_weights.png, a2_steps.png, a4_autocorr.png, a4_bins.png, a4_weight_maps.png.
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chl_cnn.plotting import OKABE_ITO, MAP_CMAP
from chl_cnn.runs import find_run

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
IO2 = HERE.parent / "io_improve" / "results"
DATA = Path.home() / "mind_the_chl_gap" / "data" / "io3"
N_TRAIN = 5540
LAGS = np.arange(1, 17)
CONV = ["t3d_dil_c4_60ep_gdl100", "io2_seed1", "io2_seed2"]
CONV_SCORE = {"t3d_dil_c4_60ep_gdl100": "io2_base_eval", "io2_seed1": "io2_seed1", "io2_seed2": "io2_seed2"}


def load(p):
    return json.loads(Path(p).read_text())


# ---------------------------------------------------------------- data / analysis
def a1a():
    r = load(RES / "io3_eval_tattn8_rec_poolweights.json")
    off = np.array(r["offsets"]); n_hid = r["n_hid"]
    wv, wn, nv = np.array(r["w_vis"]), np.array(r["w_nov"]), np.array(r["n_vis"])
    nn_ = n_hid - nv
    far = np.abs(off) > 4
    out = dict(n_val_days=r["n_val_days"], n_hidden_px=int(n_hid), offsets=off.tolist(),
               mean_w=((wv + wn) / n_hid).tolist(),                      # mean weight per hidden pixel (sums to 1 over offsets)
               mean_w_given_vis=np.where(nv > 0, wv / np.maximum(nv, 1), np.nan).tolist(),
               mean_w_given_novis=np.where(nn_ > 0, wn / np.maximum(nn_, 1), np.nan).tolist(),
               frac_vis=(nv / n_hid).tolist(), n_vis=nv.astype(int).tolist(), n_novis=nn_.astype(int).tolist(),
               mass_vis=(wv / n_hid).tolist(), mass_novis=(wn / n_hid).tolist(),
               share_far=float((wv + wn)[far].sum() / n_hid), share_far_vis=float(wv[far].sum() / n_hid),
               share_far_novis=float(wn[far].sum() / n_hid), share_vis_total=float(wv.sum() / n_hid),
               check_sum_w=float((wv + wn).sum() / n_hid))
    return out


def run_score(name):
    return load(RES / f"{name}.json")


def bit_identity():
    import numpy as np
    pre = np.load(find_run("io3_eval_tattn8_pre") / "preds.npz")["pred"]
    post = np.load(find_run("io3_eval_tattn8_post") / "preds.npz")["pred"]
    rec = np.load(find_run("io3_eval_tattn8_rec") / "preds.npz")["pred"]
    orig = np.load(find_run("io2_tattn8") / "preds.npz")["pred"]
    eq = lambda u, v: bool(np.array_equal(u, v, equal_nan=True))
    mx = lambda u, v: float(np.nanmax(np.abs(u.astype("float64") - v.astype("float64"))))
    return dict(post_eq_pre=eq(post, pre), rec_eq_post=eq(rec, post), pre_eq_io2_tattn8=eq(pre, orig),
                max_abs_post_pre=mx(post, pre), max_abs_pre_io2_tattn8=mx(pre, orig))


def a2():
    runs = {"io2_tattn8": "attention ±8, batch 4, 60 ep", "io2_tattn8_s30": "attention ±8, batch 4, stop 30",
            "t3d_dil_c4_60ep_gdl100": "conv ±4 seed 0, batch 8", "io2_seed1": "conv ±4 seed 1, batch 8", "io2_seed2": "conv ±4 seed 2, batch 8"}
    out = {}
    for r, lab in runs.items():
        p = find_run(r); c = load(p / "config.json"); h = load(p / "history.json")
        b = c["batch"]; spe = math.ceil(N_TRAIN / b)
        val = np.array(h["val"]); ep = np.arange(1, len(val) + 1)
        lr = h.get("lr_used") or [None] * len(val)
        if lr[0] is None:                                     # baseline predates lr_used: cosine over 60 epochs, lr of the epoch's updates
            lr = (0.5 * c["lr"] * (1 + np.cos(np.pi * (ep - 1) / c["epochs"]))).tolist()
        best = lambda N: float(np.min(val[10:N])) if len(val) >= N else None
        out[r] = dict(label=lab, batch=b, steps_per_epoch=spe, epochs_run=len(val), sched_epochs=c["epochs"], val=val.tolist(), lr_used=list(lr),
                      steps=(ep * spe).tolist(), days_seen=(ep * N_TRAIN).tolist(), best_11_30=best(30), best_11_60=best(60),
                      steps_at_30=30 * spe, steps_at_60=60 * spe)
    # matched optimiser steps: the attention run's epoch 30 (41,550 steps) vs the conv runs' epoch 60 (41,580 steps)
    out["matched_steps"] = dict(attn_epochs_1_30_best=out["io2_tattn8"]["best_11_30"], attn_s30_best=out["io2_tattn8_s30"]["best_11_30"],
                                conv_epochs_1_60_best={r: out[r]["best_11_60"] for r in CONV},
                                attn_steps=out["io2_tattn8"]["steps_at_30"], conv_steps=out[CONV[0]]["steps_at_60"])
    return out


def autocorr_day(w, ok_pix):
    """w (K, H, W) float maps; ok_pix (K, H, W) bool pixels used (ocean, or ocean & visible). -> rho(L) for LAGS, NaN if no variance."""
    K = w.shape[0]
    num = np.zeros(len(LAGS)); den = 0.0
    numv = np.zeros(len(LAGS)); npair = np.zeros(len(LAGS))
    covs = np.zeros((K, len(LAGS))); vars_ = np.zeros(K)
    for k in range(K):
        m = ok_pix[k]
        if m.sum() < 50:
            continue
        x = w[k].astype("float64"); mu = x[m].mean(); dx = np.where(m, x - mu, 0.0)
        v = (dx[m] ** 2).mean()
        vars_[k] = v
        for i, L in enumerate(LAGS):
            s = 0.0; n = 0
            for ax in (0, 1):
                a0 = [slice(None)] * 2; a1 = [slice(None)] * 2
                a0[ax] = slice(0, -L); a1[ax] = slice(L, None)
                pm = m[tuple(a0)] & m[tuple(a1)]
                s += (dx[tuple(a0)] * dx[tuple(a1)])[pm].sum(); n += int(pm.sum())
            covs[k, i] = s / max(n, 1)
    if vars_.sum() == 0:
        return np.full(len(LAGS), np.nan)
    return covs.sum(0) / vars_.sum()


def lag_below(rho, thr=0.5):
    r = np.r_[1.0, rho]; L = np.r_[0, LAGS]
    idx = np.flatnonzero(r < thr)
    if not len(idx):
        return np.inf
    i = idx[0]
    return float(L[i - 1] + (r[i - 1] - thr) / (r[i - 1] - r[i]) * (L[i] - L[i - 1]))


def a4a():
    z = np.load(DATA / "io3_eval_tattn8_rec_weights.npz")
    W, V, ocean = z["w"].astype("float32"), z["vis"], z["ocean"]
    n = len(W)
    rho_all = np.stack([autocorr_day(W[i], np.broadcast_to(ocean, W[i].shape)) for i in range(n)])
    rho_vis = np.stack([autocorr_day(W[i], V[i] & ocean) for i in range(n)])
    lag_all = np.array([lag_below(r) for r in rho_all]); lag_vis = np.array([lag_below(r) for r in rho_vis])
    q = lambda x: [float(np.nanpercentile(x[np.isfinite(x)], p)) for p in (25, 50, 75)] if np.isfinite(x).any() else None
    return dict(n_days=int(n), days=z["days"].tolist(), lags=LAGS.tolist(),
                rho_all_median=np.nanmedian(rho_all, 0).tolist(), rho_all_q25=np.nanpercentile(rho_all, 25, 0).tolist(),
                rho_all_q75=np.nanpercentile(rho_all, 75, 0).tolist(),
                rho_vis_median=np.nanmedian(rho_vis, 0).tolist(), rho_vis_q25=np.nanpercentile(rho_vis, 25, 0).tolist(),
                rho_vis_q75=np.nanpercentile(rho_vis, 75, 0).tolist(),
                lag05_all_q25_median_q75=q(lag_all), lag05_vis_q25_median_q75=q(lag_vis),
                n_days_lag_inf_all=int(np.isinf(lag_all).sum()), n_days_lag_inf_vis=int(np.isinf(lag_vis).sum()),
                n_days_nan_all=int(np.isnan(rho_all).any(1).sum()), n_days_nan_vis=int(np.isnan(rho_vis).any(1).sum()),
                lag05_all=lag_all.tolist(), lag05_vis=lag_vis.tolist()), (rho_all, rho_vis, z)


def a4b():
    at = load(IO2 / "io2_tattn8.json")["val"]
    cv = {r: load(IO2 / f"{CONV_SCORE[r]}.json")["val"] for r in CONV}
    out = {}
    for key in ("by_age", "by_dist"):
        out[key] = {}
        for b, v in at[key].items():
            g = [cv[r][key][b]["mae"] - v["mae"] for r in CONV]
            out[key][b] = dict(gain_seed0=g[0], gain_min=min(g), gain_max=max(g), mae_attn=v["mae"], mae_conv_seed0=cv[CONV[0]][key][b]["mae"], n=v["n"])
    g = [cv[r]["mae"] - at["mae"] for r in CONV]
    out["all"] = dict(gain_seed0=g[0], gain_min=min(g), gain_max=max(g), n=at["n_hidden"])
    return out


def age_twosided(obs, times, idx, K=8):
    """(len(idx), H, W) int8: days to the nearest real observation within t-K..t+K excluding t (0 = none)."""
    from chl_cnn import train2 as T2
    nb = {k: T2.neighbour_index(times, k) for k in range(-K, K + 1) if k}
    age = np.zeros((len(idx),) + obs.shape[1:], dtype=np.int8)
    for i, t in enumerate(idx):
        for k in range(1, K + 1):
            for j in (nb[-k][t], nb[k][t]):
                if j >= 0:
                    new = obs[j] & (age[i] == 0)
                    age[i][new] = k
    return age


def a4b_twosided():
    from chl_cnn import store as S, data as D
    from analyses.forecast.forecast_eval import BINS as AGE_BINS
    d = S.load_store(); times = d["time"]
    y = D.log_chl(np.asarray(d["chl_l3"])); obs = np.isfinite(y) & d["ocean"]
    _, va, _ = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    age = age_twosided(obs, times, va)
    runs = ["io2_tattn8"] + CONV
    err = {}
    for r in runs:
        z = np.load(find_run(CONV_SCORE.get(r, r)) / "preds.npz")             # seed 0: io2_base_eval (float32 preds), as A4b
        assert np.array_equal(z["val_idx"], va)
        hid = z["hidden"][:len(va)]
        err[r] = np.abs(z["pred"][:len(va)].astype("float32") - y[va])
    out = {}
    for name, lo, hi in AGE_BINS:
        m = hid & ((age == 0) if hi == 0 else ((age >= lo) & (age <= hi)))
        mae = {r: float(err[r][m].mean()) for r in runs}
        g = [mae[r] - mae["io2_tattn8"] for r in CONV]
        out[name] = dict(gain_seed0=g[0], gain_min=min(g), gain_max=max(g), mae_attn=mae["io2_tattn8"], mae_conv_seed0=mae[CONV[0]], n=int(m.sum()))
    out["n_hidden"] = int(hid.sum())
    np.save(DATA / "val_age_twosided.npy", age)
    return out


# ---------------------------------------------------------------- figures
def fig_a1(r):
    off = np.array(r["offsets"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(off - 0.2, r["mass_vis"], 0.4, color=OKABE_ITO[0], label="day visible at the pixel")
    ax[0].bar(off + 0.2, r["mass_novis"], 0.4, color=OKABE_ITO[1], label="day not visible at the pixel")
    ax[0].set(xlabel="day offset from target day t", ylabel="weight per hidden px (sums to 1)",
              title="Pooling weight mass by day offset, val hidden px"); ax[0].legend(fontsize=8)
    ax[1].plot(off, r["mean_w_given_vis"], "o-", color=OKABE_ITO[0], label="day visible at the pixel")
    ax[1].plot(off, r["mean_w_given_novis"], "s-", color=OKABE_ITO[1], label="day not visible at the pixel")
    ax[1].set(xlabel="day offset from target day t", ylabel="mean pooling weight of that day",
              title="Mean pooling weight given day visibility, val hidden px"); ax[1].legend(fontsize=8)
    ax[2].bar(off, r["frac_vis"], color=OKABE_ITO[2])
    ax[2].set(xlabel="day offset from target day t", ylabel="fraction of hidden px with the day visible",
              title="Context-day visibility at val hidden px")
    for a_ in ax:
        a_.grid(alpha=.3); a_.axvspan(-8.5, -4.5, color="0.9", zorder=0); a_.axvspan(4.5, 8.5, color="0.9", zorder=0)
    fig.suptitle(f"io2_tattn8 pooling weights (mean over 4 heads), {r['n_val_days']} val days, {r['n_hidden_px']:,} hidden px; grey = |offset| > 4")
    fig.tight_layout(); fig.savefig(FIG / "a1_pool_weights.png", dpi=130); plt.close(fig)


def fig_a2(r):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    sty = {"io2_tattn8": (OKABE_ITO[3], "-"), "io2_tattn8_s30": (OKABE_ITO[4], "--"), "t3d_dil_c4_60ep_gdl100": (OKABE_ITO[0], "-"),
           "io2_seed1": (OKABE_ITO[5], "-"), "io2_seed2": (OKABE_ITO[2], "-")}
    for k, (c, ls) in sty.items():
        h = r[k]
        ax[0].plot(np.array(h["steps"]) / 1e3, h["val"], color=c, ls=ls, label=h["label"])
        ax[1].plot(np.array(h["days_seen"]) / 1e3, h["val"], color=c, ls=ls, label=h["label"])
        ax[2].plot(np.array(h["steps"]) / 1e3, h["lr_used"], color=c, ls=ls, label=h["label"])
    for a_ in ax[:2]:
        a_.set_ylim(0.127, 0.140)
    ax[0].set(xlabel="optimiser steps (thousands)", ylabel="val hidden-pixel MAE, ln CHL", title="Val MAE vs optimiser steps")
    ax[1].set(xlabel="training days seen (thousands)", ylabel="val hidden-pixel MAE, ln CHL", title="Val MAE vs training days seen")
    ax[2].set(xlabel="optimiser steps (thousands)", ylabel="learning rate of the epoch's updates", title="Cosine lr schedule vs optimiser steps")
    for a_ in ax:
        a_.grid(alpha=.3); a_.legend(fontsize=7)
    fig.suptitle("io2 attention ±8 (batch 4) and conv ±4 (batch 8) runs; 5,540 training days per epoch; val n = 1,458 days")
    fig.tight_layout(); fig.savefig(FIG / "a2_steps.png", dpi=130); plt.close(fig)


def fig_a4a(r, rho_all, rho_vis):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for rho, key, c, lab in ((rho_all, "all", OKABE_ITO[0], "all ocean px"), (rho_vis, "vis", OKABE_ITO[1], "px visible on that day (both ends)")):
        ax[0].plot(LAGS, r[f"rho_{key}_median"], "o-", color=c, label=f"{lab}: median")
        ax[0].fill_between(LAGS, r[f"rho_{key}_q25"], r[f"rho_{key}_q75"], color=c, alpha=.25, label=f"{lab}: IQR")
        lag = np.array(r[f"lag05_{key}"]); lag = lag[np.isfinite(lag)]
        ax[1].hist(lag, bins=np.arange(0, 16.5, 0.5), color=c, alpha=.6, label=lab)
    ax[0].axhline(0.5, color="0.4", lw=1, ls=":")
    ax[0].set(xlabel="lag (px, 0.25°)", ylabel="autocorrelation of pooling-weight map", title="Pooling-weight map autocorrelation vs lag")
    ax[1].set(xlabel="lag where autocorrelation first drops below 0.5 (px)", ylabel="sample val days",
              title="Decorrelation lag per val day")
    for a_ in ax:
        a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.suptitle(f"io2_tattn8 pooling-weight maps (mean over heads, variance-weighted over 17 offsets), n = {r['n_days']} val days")
    fig.tight_layout(); fig.savefig(FIG / "a4_autocorr.png", dpi=130); plt.close(fig)


def fig_a4b(r):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a_, key, xl in ((ax[0], "by_age", "age of last real observation (t−8..t−1)"), (ax[1], "by_age_twosided", "days to nearest real observation (t±8, excl. t)"),
                        (ax[2], "by_dist", "distance to nearest visible px")):
        bins = list(r[key]); x = np.arange(len(bins))
        g0 = [r[key][b]["gain_seed0"] for b in bins]; lo = [r[key][b]["gain_min"] for b in bins]; hi = [r[key][b]["gain_max"] for b in bins]
        a_.bar(x, g0, color=OKABE_ITO[0], label="vs conv seed 0")
        a_.errorbar(x, g0, yerr=[np.array(g0) - lo, np.array(hi) - g0], fmt="none", ecolor="k", capsize=4, label="range over conv seeds 0/1/2")
        a_.set_xticks(x, [f"{b}\nn={r[key][b]['n'] / 1e6:.2f}M" for b in bins])
        a_.title.set_fontsize(9)
        a_.axhline(0, color="0.4", lw=1)
        a_.set(xlabel=xl, ylabel="MAE conv ±4 − MAE attention ±8, ln CHL", title=f"MAE difference by {xl.split(' (')[0]}")
        a_.title.set_fontsize(10)
        a_.grid(alpha=.3, axis="y"); a_.legend(fontsize=8)
    fig.suptitle("io2_tattn8 vs conv ±4: val MAE difference on hidden px (n = 4,998,642) by bin")
    fig.tight_layout(); fig.savefig(FIG / "a4_bins.png", dpi=130); plt.close(fig)


def fig_maps(z):
    W, V, H_, days, ti = z["w"].astype("float32"), z["vis"], z["hid"], z["days"], int(z["t_index"])
    ocean = z["ocean"]
    i = int(np.argsort(H_.reshape(len(H_), -1).sum(1))[len(H_) // 2])         # sample day with the median hidden count
    offs = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
    fig, ax = plt.subplots(2, len(offs), figsize=(2.4 * len(offs), 5.2))
    from chl_cnn import store as S
    times = S.load_store()["time"]
    for j, k in enumerate(offs):
        w = np.where(ocean, W[i, ti + k], np.nan)
        im = ax[0, j].imshow(w, cmap=MAP_CMAP, vmin=0, vmax=np.nanpercentile(W[i][:, ocean], 99))
        ax[0, j].set_title(f"weight, day t{k:+d}", fontsize=9)
        ax[1, j].imshow(np.where(ocean, V[i, ti + k], np.nan), cmap="cividis", vmin=0, vmax=1)
        ax[1, j].set_title(f"visible, day t{k:+d}", fontsize=9)
        for a_ in ax[:, j]:
            a_.set_xticks([]); a_.set_yticks([])
    fig.colorbar(im, ax=ax[0, :].tolist(), shrink=.8, label="pooling weight")
    fig.suptitle(f"io2_tattn8 pooling weights by context day (mean over heads) and input visibility, val day {str(times[days[i]])[:10]}")
    fig.savefig(FIG / "a4_weight_maps.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    out = dict(bit_identity=bit_identity())
    sc = {k: run_score(f"io3_eval_tattn8_{k}") for k in ("pre", "post", "rec", "pool4", "noself")}
    pick = lambda s: dict(mae=s["val"]["mae"], rmse=s["val"]["rmse"], bias=s["val"]["bias"], grad1_ratio=s["val"]["grad"]["1"]["ratio"],
                          grad1_corr=s["val"]["grad"]["1"]["corr"], n_hidden=s["val"]["n_hidden"], n_days=s["val"]["n_days"], n_nonfinite=s["val"]["n_nonfinite"])
    out["a1a"] = a1a()
    out["a1b"] = dict(full=pick(sc["post"]), pool_le4=pick(sc["pool4"]), delta=sc["pool4"]["val"]["mae"] - sc["post"]["val"]["mae"])
    out["a2"] = a2()
    out["a3"] = dict(full=pick(sc["post"]), no_self_attn=pick(sc["noself"]), delta=sc["noself"]["val"]["mae"] - sc["post"]["val"]["mae"])
    out["a4a"], (rho_all, rho_vis, z) = a4a()
    out["a4b"] = a4b()
    tw = a4b_twosided()
    out["a4b"]["by_age_twosided"] = {k: v for k, v in tw.items() if k != "n_hidden"}
    out["a4b"]["age_definition"] = ("by_age: days back to the last real (store-observed) observation at the pixel within t-8..t-1, past only "
                                    "(score.py / forecast_eval.persistence); by_age_twosided: days to the nearest real observation within t-8..t+8, excluding t")
    med = out["a4a"]["lag05_all_q25_median_q75"][1]
    out["score_arm"] = dict(rule="median A4a lag < 2 px -> pixel-query, else interpolated", median_lag_px=med,
                            arm="pixel-query" if med < 2 else "interpolated")
    (RES / "checks.json").write_text(json.dumps(out, indent=1))
    fig_a1(out["a1a"]); fig_a2(out["a2"]); fig_a4a(out["a4a"], rho_all, rho_vis); fig_a4b(out["a4b"]); fig_maps(z)
    # compact print
    print(json.dumps(out["bit_identity"]))
    r = out["a1a"]
    print(f"A1a n days {r['n_val_days']} hidden px {r['n_hidden_px']}: share |off|>4 {r['share_far']:.4f} (vis {r['share_far_vis']:.4f}, "
          f"not vis {r['share_far_novis']:.4f}); weight on visible days {r['share_vis_total']:.4f}; sum check {r['check_sum_w']:.6f}")
    for o, m, mv, mn, f in zip(r["offsets"], r["mean_w"], r["mean_w_given_vis"], r["mean_w_given_novis"], r["frac_vis"]):
        print(f"   off {o:+3d}  mean w {m:.4f}  | vis {mv:.4f}  | not vis {mn:.4f}  frac vis {f:.3f}")
    print(f"A1b val {out['a1b']['full']['mae']:.6f} -> pool<=4 {out['a1b']['pool_le4']['mae']:.6f} (delta {out['a1b']['delta']:+.6f})")
    print(f"A3  val {out['a3']['full']['mae']:.6f} -> no self-attn {out['a3']['no_self_attn']['mae']:.6f} (delta {out['a3']['delta']:+.6f})")
    print("A2 matched steps:", json.dumps(out["a2"]["matched_steps"]))
    a = out["a4a"]
    print(f"A4a n {a['n_days']}: lag<0.5 all q25/med/q75 {a['lag05_all_q25_median_q75']} (inf {a['n_days_lag_inf_all']}), vis {a['lag05_vis_q25_median_q75']} (inf {a['n_days_lag_inf_vis']})")
    print("    rho all median", np.round(a["rho_all_median"], 3).tolist())
    print("    rho vis median", np.round(a["rho_vis_median"], 3).tolist())
    print("A4b", json.dumps(out["a4b"], indent=0))
    print("score arm:", out["score_arm"])


if __name__ == "__main__":
    main()
