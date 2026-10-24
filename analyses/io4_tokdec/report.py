"""io4 track report: token-decoder stem runs (io4_tokdec, io4_tokdec_blend) and Job 1 (io3_ptok4 eval-only, blend off) vs the references.

Data (inputs, no GPU):
  analyses/io4_tokdec/results/<run>.json              score.py metrics of the io4 runs (score_runs.py)
  analyses/io3_attn/results/io3_ptok4.json, analyses/io_improve/results/{io2_base_eval,io2_seed1,io2_seed2,io2_tattn8,ens3_mean}.json
  models/pinn/t3d/<run>/{history,config}.json         s / epoch (median epoch_s), batch, peak GPU memory, params, val curve
  ~/mind_the_chl_gap/data/io4/<run>_maps_tokdec.npz   tokdec recorder: pooling weights of every val day, decoded channels of 6 val days
  ~/mind_the_chl_gap/data/io3/val_age_twosided.npy    two-sided age (io3 checks.py; recomputed if missing)
Analysis:
  table    val / test hidden-px MAE, delta vs conv seed 0, RMSE, bias, gradient ratio / corr at 1 / 2 / 4 px, cost, n;
  bins     val MAE by distance to the nearest visible px, past-only age and two-sided age (score.py bins), gain vs conv seed 0
           with the range over conv seeds 0 / 1 / 2;
  epochs   best val over epochs 11-30 and 11-60 (history.json), as io3 report_b.py;
  pooling  mean pooling weight by day offset over all val days x ocean patches, split by whether the patch has any visible px on that
           day (visible fraction > 0); spatial autocorrelation of the pooling-weight maps on the patch grid (lags 1-8 patches = 4-32 px;
           every 15th val day; variance-weighted over offsets, as io3 A4a), lag where it drops below 0.5;
  seams    on the 6 recorded val days, per decoded channel: mean |difference| between horizontally / vertically adjacent ocean px
           across patch edges (px x with x mod P = P - 1 -> x + 1) divided by the same within patches.
Figures: figures/io4_curves.png, io4_bins.png, io4_pool_weights.png, io4_decoded_<date>.png.

Usage (project root): python -m analyses.io4_tokdec.report
Writes results/io4_summary.json and prints the tables.
"""
import json
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
IO3 = HERE.parent / "io3_attn" / "results"
DATA4 = Path.home() / "mind_the_chl_gap" / "data" / "io4"
DATA3 = Path.home() / "mind_the_chl_gap" / "data" / "io3"
# key, label, score json dir, run dir name (None: no run dir)
ROWS = [("io4_eval_ptok4_noblend", "Job 1: arm A eval, blend off", RES, None),
        ("io4_tokdec", "Job 2: token decoder", RES, "io4_tokdec"),
        ("io4_tokdec_blend", "Job 3: token decoder + own-px blend", RES, "io4_tokdec_blend"),
        ("io3_ptok4", "io3_ptok4 (arm A)", IO3, "io3_ptok4"),
        ("io2_tattn8", "io2_tattn8 (attention ±8, b4)", IO2, "io2_tattn8"),
        ("io2_base_eval", "conv ±4 seed 0", IO2, "t3d_dil_c4_60ep_gdl100"),
        ("io2_seed1", "conv ±4 seed 1", IO2, "io2_seed1"), ("io2_seed2", "conv ±4 seed 2", IO2, "io2_seed2"),
        ("ens3_mean", "3-seed conv ensemble", IO2, None)]
CONV = ("io2_base_eval", "io2_seed1", "io2_seed2")
SMOKES = ("io4_smoke_tokdec", "io4_smoke_tokdec_blend")
EPOCH_RUNS = [("io4_tokdec", "token decoder (Job 2)"), ("io4_tokdec_blend", "token decoder + blend (Job 3)"), ("io3_ptok4", "io3_ptok4 (arm A)"),
              ("io2_tattn8", "io2_tattn8 (attention ±8, b4)"), ("t3d_dil_c4_60ep_gdl100", "conv ±4 seed 0"), ("io2_seed1", "conv ±4 seed 1"),
              ("io2_seed2", "conv ±4 seed 2")]
COL = {"io4_tokdec": OKABE_ITO[3], "io4_tokdec_blend": OKABE_ITO[4], "io3_ptok4": OKABE_ITO[1], "io2_tattn8": OKABE_ITO[2],
       "t3d_dil_c4_60ep_gdl100": OKABE_ITO[0], "io2_seed1": OKABE_ITO[5], "io2_seed2": OKABE_ITO[7]}
LAGS_P = np.arange(1, 9)


def load(p):
    return json.loads(Path(p).read_text())


# ================================================================ data / analysis
def table():
    rows = []
    for key, lab, d, run in ROWS:
        if not (d / f"{key}.json").exists():
            print(f"missing {key}; skipped")
            continue
        s = load(d / f"{key}.json")
        r = dict(run=key, label=lab)
        for sp in ("val", "test"):
            m = s[sp]
            r[sp] = dict(mae=m["mae"], rmse=m["rmse"], bias=m["bias"], n_hidden=m["n_hidden"], n_days=m["n_days"], n_nonfinite=m["n_nonfinite"],
                         grad={L: dict(ratio=m["grad"][L]["ratio"], corr=m["grad"][L]["corr"], n_pairs=m["grad"][L]["n_pairs"]) for L in ("1", "2", "4")},
                         by_age=m["by_age"], by_dist=m["by_dist"])
        if run:
            c = load(find_run(run) / "config.json"); h = load(find_run(run) / "history.json")
            r.update(batch=c["batch"], n_params=c["n_params"], best_epoch=c["best_epoch"], s_per_epoch=float(np.median(h["epoch_s"])),
                     peak_gpu_gb=c.get("peak_gpu_gb"), peak_gpu_reserved_gb=c.get("peak_gpu_reserved_gb"))
        rows.append(r)
    return rows


def smokes():
    out = {}
    for run in SMOKES:
        try:
            c = load(find_run(run) / "config.json"); h = load(find_run(run) / "history.json")
        except FileNotFoundError:
            continue
        out[run] = dict(s_per_epoch=h["epoch_s"], batch=c["batch"], n_params=c["n_params"], peak_gpu_gb=c["peak_gpu_gb"],
                        peak_gpu_reserved_gb=c["peak_gpu_reserved_gb"], val=h["val"])
    return out


def two_sided_age(va, times):
    f = DATA3 / "val_age_twosided.npy"
    if f.exists():
        return np.load(f)
    from chl_cnn import store as S, data as D
    from analyses.io3_attn.checks import age_twosided
    d = S.load_store(); obs = np.isfinite(D.log_chl(np.asarray(d["chl_l3"]))) & d["ocean"]
    return age_twosided(obs, times, va)


def add_age_twosided(rows):
    from chl_cnn import store as S, data as D
    from analyses.forecast.forecast_eval import BINS as AGE_BINS
    d = S.load_store(); times = d["time"]; y = D.log_chl(np.asarray(d["chl_l3"]))
    _, va, _ = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS); yv = y[va]
    age = two_sided_age(va, times)
    for r in rows:
        if r["run"] == "ens3_mean":
            continue
        z = np.load(find_run(r["run"]) / "preds.npz")
        assert np.array_equal(z["val_idx"], va)
        hid = z["hidden"][:len(va)]
        ae = np.abs(z["pred"][:len(va)].astype("float32") - yv)
        r["val"]["by_age2"] = {}
        for name, lo, hi in AGE_BINS:
            m = hid & ((age == 0) if hi == 0 else ((age >= lo) & (age <= hi)))
            r["val"]["by_age2"][name] = dict(mae=float(ae[m].mean()), n=int(m.sum()))


def epochs():
    out = {}
    for run, lab in EPOCH_RUNS:
        try:
            h = load(find_run(run) / "history.json")
        except FileNotFoundError:
            continue
        v = np.array(h["val"])
        if len(v) < 11:
            continue
        out[run] = dict(label=lab, best_11_30=float(v[10:30].min()), epoch_11_30=int(np.argmin(v[10:30]) + 11),
                        best_11_60=float(v[10:60].min()) if len(v) >= 60 else None,
                        epoch_11_60=int(np.argmin(v[10:60]) + 11) if len(v) >= 60 else None, n_epochs=int(len(v)), val=v.tolist())
    return out


def autocorr_grid(g, ok):
    """g (K, nPy, nPx) maps, ok (nPy, nPx) ocean patches -> rho(L) for LAGS_P (variance-weighted over the K maps)."""
    covs = np.zeros(len(LAGS_P)); var = 0.0
    for k in range(len(g)):
        x = g[k].astype("float64"); mu = x[ok].mean(); dx = np.where(ok, x - mu, 0.0)
        var += (dx[ok] ** 2).mean()
        for i, L in enumerate(LAGS_P):
            s = 0.0; n = 0
            for ax in (0, 1):
                a0 = [slice(None)] * 2; a1 = [slice(None)] * 2
                a0[ax] = slice(0, -L); a1[ax] = slice(L, None)
                pm = ok[tuple(a0)] & ok[tuple(a1)]
                s += (dx[tuple(a0)] * dx[tuple(a1)])[pm].sum(); n += int(pm.sum())
            covs[i] += s / max(n, 1)
    return covs / var if var > 0 else np.full(len(LAGS_P), np.nan)


def lag_below(rho, thr=0.5):
    r = np.r_[1.0, rho]; L = np.r_[0, LAGS_P]
    idx = np.flatnonzero(r < thr)
    if not len(idx):
        return np.inf
    i = idx[0]
    return float(L[i - 1] + (r[i - 1] - thr) / (r[i - 1] - r[i]) * (L[i] - L[i - 1]))


def pooling(run):
    f = DATA4 / f"{run}_maps_tokdec.npz"
    if not f.exists():
        return None, None
    z = np.load(f)
    W = z["pool_w_all"].astype("float32"); V = z["visfrac_all"].astype("float32")            # (n_val, Np, T)
    n, Np, T = W.shape; ti = T // 2
    offs = np.arange(T) - ti
    n_nonfinite = int((~np.isfinite(W)).sum())
    vis = V > 0
    w_mean = W.mean((0, 1))
    w_vis = np.array([W[..., k][vis[..., k]].mean() if vis[..., k].any() else np.nan for k in range(T)])
    w_nov = np.array([W[..., k][~vis[..., k]].mean() if (~vis[..., k]).any() else np.nan for k in range(T)])
    n_vis = vis.sum((0, 1)); n_nov = (~vis).sum((0, 1))
    share_far = float(W[..., np.abs(offs) > 4].sum(-1).mean())
    nPy, nPx, pidx = int(z["nPy"]), int(z["nPx"]), z["patch_idx"]
    ok = np.zeros(nPy * nPx, bool); ok[pidx] = True; ok = ok.reshape(nPy, nPx)
    days = np.arange(0, n, 15)
    rho = []
    for i in days:
        g = np.zeros((T, nPy * nPx)); g[:, pidx] = W[i].T
        rho.append(autocorr_grid(g.reshape(T, nPy, nPx), ok))
    rho = np.stack(rho); lag = np.array([lag_below(r) for r in rho])
    fin = np.isfinite(lag)
    out = dict(n_val_days=int(n), n_patches=int(Np), offsets=offs.tolist(), mean_w=w_mean.tolist(), mean_w_patch_visible=w_vis.tolist(),
               mean_w_patch_not_visible=w_nov.tolist(), n_patchdays_visible=n_vis.tolist(), n_patchdays_not_visible=n_nov.tolist(),
               w_day_t=float(w_mean[ti]), share_abs_offset_gt4=share_far, n_nonfinite=n_nonfinite,
               autocorr=dict(n_days=int(len(days)), lags_patches=LAGS_P.tolist(), rho_median=np.nanmedian(rho, 0).tolist(),
                             rho_q25=np.nanpercentile(rho, 25, 0).tolist(), rho_q75=np.nanpercentile(rho, 75, 0).tolist(),
                             lag05_patches_q25_median_q75=[float(np.percentile(lag[fin], q)) for q in (25, 50, 75)] if fin.any() else None,
                             n_days_lag_inf=int((~fin).sum()), n_days_nan=int(np.isnan(rho).any(1).sum())))
    return out, rho


def seams(run, ocean):
    f = DATA4 / f"{run}_maps_tokdec.npz"
    if not f.exists():
        return None
    z = np.load(f); P = int(z["P"])
    res = []
    for p in z["positions"]:
        dec = z[f"dec__{p}"]                                                    # (C, H, W)
        across, within = [], []
        for ax in (1, 2):
            n = dec.shape[ax]
            a0 = [slice(None)] * 3; a1 = [slice(None)] * 3; a0[ax] = slice(0, n - 1); a1[ax] = slice(1, n)
            d = np.abs(dec[tuple(a1)] - dec[tuple(a0)])                         # (C, ...)
            o0 = [slice(None)] * 2; o1 = [slice(None)] * 2; o0[ax - 1] = slice(0, n - 1); o1[ax - 1] = slice(1, n)
            okp = ocean[tuple(o0)] & ocean[tuple(o1)]
            pos = np.arange(n - 1) % P == P - 1
            edge = np.broadcast_to(pos.reshape((-1, 1) if ax == 1 else (1, -1)), okp.shape)
            across.append(d[:, okp & edge]); within.append(d[:, okp & ~edge])
        a = np.concatenate(across, 1).mean(1); w = np.concatenate(within, 1).mean(1)
        res.append(a / w)
    res = np.stack(res)                                                         # (days, C)
    return dict(n_days=int(len(res)), n_channels=int(res.shape[1]), ratio_median_over_channels_by_day=np.median(res, 1).tolist(),
                ratio_median=float(np.median(res)), ratio_q10_q90=[float(np.percentile(res, 10)), float(np.percentile(res, 90))])


# ================================================================ figures
def fig_curves(ep):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for run, _ in EPOCH_RUNS:
        if run not in ep:
            continue
        v = ep[run]["val"]
        ax.plot(np.arange(1, len(v) + 1), v, color=COL[run], label=ep[run]["label"], lw=2.2 if run.startswith("io4") else 1.1)
    ax.set_ylim(0.127, 0.145); ax.axvline(30, color="0.5", ls=":", lw=1)
    ax.set(xlabel="epoch (cosine over 60 epochs)", ylabel="val hidden-pixel MAE, ln CHL", title="Val MAE by epoch: token-decoder stems and reference runs")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.text(0.01, 0.01, "val n = 1,458 days, 4,998,642 hidden px; io2_tattn8 at batch 4, others batch 8", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "io4_curves.png", dpi=130); plt.close(fig)


def fig_bins(rows):
    R = {r["run"]: r for r in rows}
    runs = [(r, l, c) for r, l, c in (("io4_tokdec", "token decoder (Job 2)", OKABE_ITO[3]), ("io4_tokdec_blend", "token decoder + blend (Job 3)", OKABE_ITO[4]),
                                      ("io3_ptok4", "io3_ptok4 (arm A)", OKABE_ITO[1]), ("io2_tattn8", "io2_tattn8 (attention ±8)", OKABE_ITO[2])) if r in R]
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.6))
    for a_, key, xl in ((ax[0], "by_dist", "distance to nearest visible px"), (ax[1], "by_age", "age of last obs (t−8..t−1)"),
                        (ax[2], "by_age2", "days to nearest obs (t±8, excl. t)")):
        bins = list(R["io2_base_eval"]["val"][key]); x = np.arange(len(bins)); bw = 0.8 / len(runs)
        for j, (run, lab, c) in enumerate(runs):
            g = np.array([R["io2_base_eval"]["val"][key][b]["mae"] - R[run]["val"][key][b]["mae"] for b in bins])
            gs = np.array([[R[s]["val"][key][b]["mae"] - R[run]["val"][key][b]["mae"] for s in CONV] for b in bins])
            xo = x + (j - (len(runs) - 1) / 2) * bw
            a_.bar(xo, g, bw, color=c, label=lab)
            a_.errorbar(xo, g, yerr=[g - gs.min(1), gs.max(1) - g], fmt="none", ecolor="k", capsize=3, lw=0.8)
        a_.set_xticks(x, [f"{b}\nn={R['io2_base_eval']['val'][key][b]['n'] / 1e6:.2f}M" for b in bins])
        a_.axhline(0, color="0.4", lw=1)
        a_.set(xlabel=xl, ylabel="MAE conv ±4 seed 0 − MAE run, ln CHL", title=f"Val MAE gain over conv ±4 seed 0,\nby {xl}")
        a_.grid(alpha=.3, axis="y"); a_.legend(fontsize=8)
    fig.text(0.01, 0.01, "bars: gain vs conv seed 0; whiskers: range of the gain vs conv seeds 0/1/2; val hidden px n = 4,998,642 (1,458 days)", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "io4_bins.png", dpi=130); plt.close(fig)


def fig_pool(pl, rho, run):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    o = np.array(pl["offsets"])
    ax[0].plot(o, pl["mean_w"], "o-", color=OKABE_ITO[0], label="all ocean patches")
    ax[0].plot(o, pl["mean_w_patch_visible"], "s--", color=OKABE_ITO[1], label="patch has visible px on that day")
    ax[0].plot(o, pl["mean_w_patch_not_visible"], "^:", color=OKABE_ITO[2], label="patch has no visible px on that day")
    ax[0].set(xlabel="day offset from t", ylabel="mean pooling weight (mean over 4 heads)", title=f"Pooling weight by day offset, {run}")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    a = pl["autocorr"]; L = np.array(a["lags_patches"])
    ax[1].plot(L * 4, a["rho_median"], "o-", color=OKABE_ITO[0], label="median over days")
    ax[1].fill_between(L * 4, a["rho_q25"], a["rho_q75"], color=OKABE_ITO[0], alpha=.25, label="IQR over days")
    ax[1].axhline(0.5, color="0.5", ls=":", lw=1)
    ax[1].set(xlabel="lag (px; 1 patch = 4 px)", ylabel="autocorrelation of pooling-weight map", title=f"Pooling-weight map autocorrelation vs lag, {run}")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.text(0.01, 0.01, f"left: {pl['n_val_days']} val days x {pl['n_patches']} ocean patches; right: {a['n_days']} val days (every 15th), "
             f"ocean-patch pairs on the {38} x {46} patch grid, variance-weighted over the 17 offsets", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "io4_pool_weights.png", dpi=130); plt.close(fig)


def fig_decoded(run, ocean, times, va, pos_pick=None):
    f = DATA4 / f"{run}_maps_tokdec.npz"
    z = np.load(f); P = int(z["P"])
    p = int(z["positions"][0] if pos_pick is None else pos_pick)
    dec = z[f"dec__{p}"]
    v = np.array([dec[c][ocean].var() for c in range(len(dec))])
    chans = np.argsort(v)[::-1][:4]
    # zoom window: a 40 x 40 px all-ocean-majority box near the centre of the ocean
    ys, xs = np.nonzero(ocean); cy, cx = int(np.median(ys)), int(np.median(xs))
    y0, x0 = (cy // P) * P - 20, (cx // P) * P - 20
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.2))
    for j, c in enumerate(chans):
        m = np.where(ocean, dec[c], np.nan)
        lo, hi = np.nanpercentile(m, [2, 98])
        im = ax[0, j].imshow(m, cmap=MAP_CMAP, vmin=lo, vmax=hi)
        ax[0, j].add_patch(plt.Rectangle((x0 - .5, y0 - .5), 40, 40, fill=False, ec="#D55E00", lw=1.2))
        ax[0, j].set_title(f"decoded channel {c}, full grid", fontsize=9); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
        fig.colorbar(im, ax=ax[0, j], shrink=.75)
        im = ax[1, j].imshow(m[y0:y0 + 40, x0:x0 + 40], cmap=MAP_CMAP, vmin=lo, vmax=hi, extent=(x0 - .5, x0 + 39.5, y0 + 39.5, y0 - .5))
        for k in range(0, 41, P):
            ax[1, j].axvline(x0 - .5 + k, color="w", lw=0.4, alpha=.6); ax[1, j].axhline(y0 - .5 + k, color="w", lw=0.4, alpha=.6)
        ax[1, j].set_title(f"channel {c}, 40 x 40 px zoom\n(white lines: 4 x 4 px patch edges)", fontsize=9)
        ax[1, j].set_xlabel("column (px)"); ax[1, j].set_ylabel("row (px)")
        fig.colorbar(im, ax=ax[1, j], shrink=.75)
    date = str(times[va[p]])[:10]
    fig.suptitle(f"{run}: decoded context channels after Conv 3x3 + GELU, val day {date} (4 of 32, highest ocean variance)")
    fig.tight_layout(); fig.savefig(FIG / f"io4_decoded_{date}.png", dpi=120); plt.close(fig)
    return date, chans.tolist()


# ================================================================ report
def main():
    from chl_cnn import store as S, data as D
    st = S.load_store(); times = st["time"]; ocean = np.asarray(st["ocean"]).astype(bool)
    _, va, _ = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    rows = table(); add_age_twosided(rows); ep = epochs(); sm = smokes()
    pl, rho = {}, {}
    for run in ("io4_tokdec", "io4_tokdec_blend"):
        p_, r_ = pooling(run)
        if p_ is not None:
            pl[run], rho[run] = p_, r_
    sea = {run: seams(run, ocean) for run in ("io4_tokdec", "io4_tokdec_blend")}
    leak = {s: load(RES / f"leak_{s}.json") for s in SMOKES if (RES / f"leak_{s}.json").exists()}
    out = dict(rows=rows, epochs={k: {kk: vv for kk, vv in v.items() if kk != "val"} for k, v in ep.items()}, smokes=sm, pooling=pl, seams=sea, leak=leak,
               seed_spread=dict(S60=0.00024, S30=0.0015))
    FIG.mkdir(exist_ok=True)
    fig_curves(ep); fig_bins(rows)
    if "io4_tokdec" in pl:
        fig_pool(pl["io4_tokdec"], rho["io4_tokdec"], "io4_tokdec")
        out["decoded_fig"] = dict(zip(("date", "channels"), fig_decoded("io4_tokdec", ocean, times, va)))
    (RES / "io4_summary.json").write_text(json.dumps(out, indent=1))
    R = {r["run"]: r for r in rows}
    b0v, b0t = R["io2_base_eval"]["val"]["mae"], R["io2_base_eval"]["test"]["mae"]
    print(f"{'run':24s} {'val':>8s} {'dval':>8s} {'test':>8s} {'dtest':>8s} {'rmse_v':>7s} {'bias_v':>7s} g1r/c   g2r/c   g4r/c   s/ep batch GBalloc/res params best_ep  n_val_px nonfinite")
    for r in rows:
        v, t = r["val"], r["test"]; g = v["grad"]
        print(f"{r['run']:24s} {v['mae']:.6f} {v['mae'] - b0v:+.5f} {t['mae']:.6f} {t['mae'] - b0t:+.5f} {v['rmse']:.4f} {v['bias']:+.4f} "
              f"{g['1']['ratio']:.3f}/{g['1']['corr']:.3f} {g['2']['ratio']:.3f}/{g['2']['corr']:.3f} {g['4']['ratio']:.3f}/{g['4']['corr']:.3f} "
              f"{r.get('s_per_epoch', float('nan')):.0f} {r.get('batch', '-')} {r.get('peak_gpu_gb') or float('nan'):.1f}/{r.get('peak_gpu_reserved_gb') or float('nan'):.1f} "
              f"{r.get('n_params', '-')} {r.get('best_epoch', '-')}  {v['n_hidden']} {v['n_nonfinite']}/{t['n_nonfinite']}")
    print("test grad:", {r["run"]: [round(r["test"]["grad"][L][q], 3) for L in ("1", "2", "4") for q in ("ratio", "corr")] for r in rows})
    for key in ("by_dist", "by_age", "by_age2"):
        print(key, "n:", {b: R["io2_base_eval"]["val"][key][b]["n"] for b in R["io2_base_eval"]["val"][key]})
        for r in rows:
            if key in r["val"]:
                print(f"   {r['run']:24s}", {b: (round(r["val"][key][b]["mae"], 5), round(R["io2_base_eval"]["val"][key][b]["mae"] - r["val"][key][b]["mae"], 5))
                                              for b in r["val"][key]})
    print("epochs:", json.dumps(out["epochs"]))
    print("smokes:", json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in sm.items()}))
    print("pooling:", json.dumps(pl))
    print("seams:", json.dumps(sea))


if __name__ == "__main__":
    main()
