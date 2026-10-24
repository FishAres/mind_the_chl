"""io6 track report: learned-kriging stem (io6_tokdec_krig, io6_tokdec_krig_signed) and spatial token attention (io6_tokdec_blend_spatial)
vs the references. Runs that the queue's gates skipped are left out (logs/io6/gates.log).

Data (inputs, no GPU):
  analyses/io6_kriging/results/<run>.json             score_runs.py metrics (incl. by_age2, by_key, spread)
  analyses/io4_tokdec/results/io4_tokdec_blend.json, analyses/io3_attn/results/io3_ptok4.json,
  analyses/io_improve/results/{io2_tattn8,io2_base_eval,io2_seed1,io2_seed2,ens3_mean}.json
  models/pinn/t3d/<run>/{history,config}.json, model.pt  val curve, s / epoch (median epoch_s), peak memory, params; learned s, lambda,
                                                      relative-position bias, no-key logits
  ~/mind_the_chl_gap/data/io6/<run>_diag.npz          io6_eval.py --diag-every: weights per stencil offset and per |dSST| bin (val hidden px)
  ~/mind_the_chl_gap/data/io3/val_age_twosided.npy    two-sided age
  analyses/io6_kriging/results/{leak_*,gpu_check,unit_tests}.json, ~/mind_the_chl_gap/logs/io6/gates.log
Analysis:
  table     val / test hidden-px MAE, delta vs conv seed 0 and vs io4_tokdec_blend, RMSE, bias, gradient ratio / corr at 1 / 2 / 4 px, cost, n
  bins      val MAE gain over conv seed 0 by distance, past-only age, two-sided age (whiskers: range over conv seeds 0 / 1 / 2), and by the
            kriging key split (">= 1 visible key" / "no key"; conv seed 0 only)
  epochs    best val over epochs 11-30 and 11-60
  kriging   weight share and mean weight per visible candidate by dt, dx, dy (val hidden px, every --diag-every-th val day), maps per dt;
            relative-position bias maps (mean over the value heads); mean weight per visible candidate by |dSST|; learned s, lambda;
            kriging spread vs |error| (score_runs.py)
Figures (figures/): io6_curves, io6_bins, io6_variogram_lines_<run>, io6_variogram_maps_<run>, io6_weight_vs_sst, io6_learned_params,
io6_spread_vs_error (.png). Writes results/io6_summary.json and prints the tables.

Usage (project root): python -m analyses.io6_kriging.report
"""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chl_cnn.plotting import OKABE_ITO
from chl_cnn.runs import find_run

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
IO2 = HERE.parent / "io_improve" / "results"
IO3 = HERE.parent / "io3_attn" / "results"
IO4 = HERE.parent / "io4_tokdec" / "results"
DATA6 = Path.home() / "mind_the_chl_gap" / "data" / "io6"
DATA3 = Path.home() / "mind_the_chl_gap" / "data" / "io3"
GATES = Path.home() / "mind_the_chl_gap" / "logs" / "io6" / "gates.log"
KRIG = ("io6_tokdec_krig", "io6_tokdec_krig_signed")
IO6 = ("io6_tokdec_krig", "io6_tokdec_blend_spatial", "io6_tokdec_krig_signed")
LABEL = {"io6_tokdec_krig": "K1: tokens + kriging", "io6_tokdec_krig_signed": "K1 + signed head pair",
         "io6_tokdec_blend_spatial": "tokens + blend + spatial attention", "io4_tokdec_blend": "io4_tokdec_blend (tokens + blend)",
         "io3_ptok4": "io3_ptok4 (arm A)", "io2_tattn8": "io2_tattn8 (attention ±8, b4)", "io2_base_eval": "conv ±4 seed 0",
         "io2_seed1": "conv ±4 seed 1", "io2_seed2": "conv ±4 seed 2", "ens3_mean": "3-seed conv ensemble"}
# key, score json dir, run dir (None: no run dir)
ROWS = [("io6_tokdec_krig", RES, "io6_tokdec_krig"), ("io6_tokdec_krig_signed", RES, "io6_tokdec_krig_signed"),
        ("io6_tokdec_blend_spatial", RES, "io6_tokdec_blend_spatial"), ("io4_tokdec_blend", IO4, "io4_tokdec_blend"),
        ("io3_ptok4", IO3, "io3_ptok4"), ("io2_tattn8", IO2, "io2_tattn8"), ("io2_base_eval", IO2, "t3d_dil_c4_60ep_gdl100"),
        ("io2_seed1", IO2, "io2_seed1"), ("io2_seed2", IO2, "io2_seed2"), ("ens3_mean", IO2, None)]
CONV = ("io2_base_eval", "io2_seed1", "io2_seed2")
PRED_RUN = {"io2_base_eval": "io2_base_eval"}                     # run dir holding the float32 preds of conv seed 0
EPOCH_RUNS = [("io6_tokdec_krig", "io6_tokdec_krig"), ("io6_tokdec_krig_signed", "io6_tokdec_krig_signed"),
              ("io6_tokdec_blend_spatial", "io6_tokdec_blend_spatial"), ("io4_tokdec_blend", "io4_tokdec_blend"), ("io3_ptok4", "io3_ptok4"),
              ("io2_tattn8", "io2_tattn8"), ("io2_base_eval", "t3d_dil_c4_60ep_gdl100"), ("io2_seed1", "io2_seed1"), ("io2_seed2", "io2_seed2")]
COL = {"io6_tokdec_krig": OKABE_ITO[3], "io6_tokdec_krig_signed": OKABE_ITO[7], "io6_tokdec_blend_spatial": OKABE_ITO[2],
       "io4_tokdec_blend": OKABE_ITO[4], "io3_ptok4": OKABE_ITO[1], "io2_tattn8": OKABE_ITO[5], "io2_base_eval": OKABE_ITO[0],
       "io2_seed1": "#777777", "io2_seed2": "#BBBBBB"}
SMOKES = ("io6_smoke_krig", "io6_smoke_spatial", "io6_smoke_krig_signed", "io6_smoke_krig_d24", "io6_smoke_krig_signed_d24")


def load(p):
    return json.loads(Path(p).read_text())


# ================================================================ data / analysis
def table():
    rows = []
    for key, d, run in ROWS:
        if not (d / f"{key}.json").exists():
            print(f"missing {key}; skipped")
            continue
        s = load(d / f"{key}.json")
        r = dict(run=key, label=LABEL[key])
        for sp in ("val", "test"):
            m = s[sp]
            r[sp] = dict(mae=m["mae"], rmse=m["rmse"], bias=m["bias"], n_hidden=m["n_hidden"], n_days=m["n_days"], n_nonfinite=m["n_nonfinite"],
                         grad={L: dict(ratio=m["grad"][L]["ratio"], corr=m["grad"][L]["corr"], n_pairs=m["grad"][L]["n_pairs"]) for L in ("1", "2", "4")},
                         by_age=m["by_age"], by_dist=m["by_dist"])
            for extra in ("by_age2", "by_key", "spread"):
                if extra in m:
                    r[sp][extra] = m[extra]
        if run:
            try:
                c = load(find_run(run) / "config.json"); h = load(find_run(run) / "history.json")
                r.update(batch=c["batch"], n_params=c["n_params"], best_epoch=c["best_epoch"], s_per_epoch=float(np.median(h["epoch_s"])),
                         peak_gpu_gb=c.get("peak_gpu_gb"), peak_gpu_reserved_gb=c.get("peak_gpu_reserved_gb"), in_channels=c.get("in_channels"))
            except FileNotFoundError:
                pass
        rows.append(r)
    return rows


def add_age2_refs(rows):
    """two-sided age bins for reference rows without them (from their preds.npz)."""
    from chl_cnn import store as S, data as D
    from analyses.forecast.forecast_eval import BINS as AGE_BINS
    need = [r for r in rows if "by_age2" not in r["val"] and r["run"] != "ens3_mean"]
    if not need:
        return
    d = S.load_store(); times = d["time"]; y = D.log_chl(np.asarray(d["chl_l3"]))
    _, va, _ = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS); yv = y[va]
    age = np.load(DATA3 / "val_age_twosided.npy")
    for r in need:
        z = np.load(find_run(PRED_RUN.get(r["run"], r["run"])) / "preds.npz")
        assert np.array_equal(z["val_idx"], va)
        hid = z["hidden"][:len(va)]
        ae = np.abs(z["pred"][:len(va)].astype("float32") - yv)
        r["val"]["by_age2"] = {}
        for name, lo, hi in AGE_BINS:
            m = hid & ((age == 0) if hi == 0 else ((age >= lo) & (age <= hi)))
            r["val"]["by_age2"][name] = dict(mae=float(ae[m].mean()), n=int(m.sum()))


def epochs():
    out = {}
    for key, run in EPOCH_RUNS:
        try:
            h = load(find_run(run) / "history.json")
        except FileNotFoundError:
            continue
        v = np.array(h["val"])
        if len(v) < 11:
            continue
        out[key] = dict(best_11_30=float(v[10:30].min()), epoch_11_30=int(np.argmin(v[10:30]) + 11),
                        best_11_60=float(v[10:60].min()) if len(v) >= 60 else None,
                        epoch_11_60=int(np.argmin(v[10:60]) + 11) if len(v) >= 60 else None, n_epochs=int(len(v)), val=v.tolist())
    return out


def smokes():
    out = {}
    for run in SMOKES:
        try:
            c = load(find_run(run) / "config.json"); h = load(find_run(run) / "history.json")
        except FileNotFoundError:
            continue
        out[run] = dict(s_per_epoch=h["epoch_s"], batch=c["batch"], n_params=c["n_params"], in_channels=c["in_channels"], peak_gpu_gb=c["peak_gpu_gb"],
                        peak_gpu_reserved_gb=c["peak_gpu_reserved_gb"], val=h["val"], krig_dim=c.get("krig_dim"), krig_chunk=c.get("krig_chunk"))
    return out


def learned(run):
    """s, lambda, no-key logits, relative-position bias of a kriging run (model.pt)."""
    try:
        sd = torch.load(find_run(run) / "model.pt", map_location="cpu", weights_only=False)["model"]
        cfg = load(find_run(run) / "config.json")
    except FileNotFoundError:
        return None
    heads = cfg["krig_heads"]
    out = dict(s=float(sd["krig.s_adv"]), s_init=cfg["krig_s0"], none_logit=sd["krig.none_logit"].tolist(), heads=heads,
               relbias=sd["krig.relbias"].numpy())
    if "krig.lam_logit" in sd:
        out.update(lam=float(torch.sigmoid(sd["krig.lam_logit"])), lam_init=0.1)
    return out


def variogram(run):
    f = DATA6 / f"{run}_diag.npz"
    if not f.exists():
        return None
    z = np.load(f)
    off = z["offsets"]                                                             # (K, 3) dt, dy, dx
    sw, nv, nh = z["sum_w"], z["n_valid"], int(z["n_hidden_px"])
    cfg = load(find_run(run) / "config.json"); hv = cfg["krig_heads"]
    swv = sw[:, :hv].mean(1)                                                       # head mean over the value heads
    dts = np.arange(-4, 5)
    share = {int(t): float(swv[off[:, 0] == t].sum() / nh) for t in dts}            # weight share per lag (per hidden px)
    share_heads = {int(t): (sw[off[:, 0] == t].sum(0) / nh).tolist() for t in dts}
    per_cand = lambda m: float(swv[m].sum() / max(nv[m].sum(), 1))                 # mean weight per visible candidate
    out = dict(n_days=int(z["n_days"]), n_hidden_px=nh, n_hidden_px_with_key=int(z["n_hidden_px_with_key"]), share_by_dt=share,
               share_by_dt_per_head=share_heads, share_total=float(swv.sum() / nh),
               w_per_visible_by_dt={int(t): per_cand(off[:, 0] == t) for t in dts}, n_visible_by_dt={int(t): int(nv[off[:, 0] == t].sum()) for t in dts},
               w_per_visible_by_dx={int(x): per_cand(off[:, 2] == x) for x in range(-6, 7)},
               w_per_visible_by_dy={int(y): per_cand(off[:, 1] == y) for y in range(-6, 7)},
               n_visible_by_dx={int(x): int(nv[off[:, 2] == x].sum()) for x in range(-6, 7)},
               n_visible_by_dy={int(y): int(nv[off[:, 1] == y].sum()) for y in range(-6, 7)})
    e = z["sst_edges_k"]
    out["sst"] = dict(edges_k=[float(v) for v in e], w_per_visible=(z["sst_sum_w"] / np.maximum(z["sst_n"], 1)).tolist(), n=z["sst_n"].astype(int).tolist(),
                      w_per_visible_dt0=(z["sst_sum_w_t0"] / np.maximum(z["sst_n_t0"], 1)).tolist(), n_dt0=z["sst_n_t0"].astype(int).tolist())
    if sw.shape[1] > hv:                                                            # signed pair: weight share per lag of s+ and s-
        out["signed_share_by_dt"] = {int(t): (sw[off[:, 0] == t, hv:].sum(0) / nh).tolist() for t in dts}
    maps = np.full((9, 13, 13), np.nan)
    for (t, y, x), w_, n_ in zip(off, swv, nv):
        maps[t + 4, y + 6, x + 6] = w_ / n_ if n_ > 0 else np.nan
    return out, maps, off


def gates():
    if not GATES.exists():
        return []
    return [l.strip() for l in GATES.read_text().splitlines() if l.strip()]


# ================================================================ figures
def fig_curves(ep):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for key, _ in EPOCH_RUNS:
        if key not in ep:
            continue
        v = ep[key]["val"]
        ax.plot(np.arange(1, len(v) + 1), v, color=COL[key], label=LABEL[key], lw=2.2 if key.startswith("io6") else 1.1)
    ax.set_ylim(0.125, 0.145); ax.axvline(30, color="0.5", ls=":", lw=1)
    ax.set(xlabel="epoch (cosine over 60 epochs)", ylabel="val hidden-pixel MAE, ln CHL", title="Val MAE by epoch: io6 runs and reference runs")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.text(0.01, 0.01, "val n = 1,458 days, 4,998,642 hidden px; io2_tattn8 at batch 4, others batch 8; single seeds except the conv runs", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "io6_curves.png", dpi=130); plt.close(fig)


def fig_bins(rows):
    R = {r["run"]: r for r in rows}
    runs = [k for k in IO6 + ("io4_tokdec_blend",) if k in R]
    fig, ax = plt.subplots(1, 4, figsize=(24, 4.8))
    for a_, key, xl in ((ax[0], "by_dist", "distance to nearest visible px"), (ax[1], "by_age", "age of last obs (t−8..t−1)"),
                        (ax[2], "by_age2", "days to nearest obs (t±8, excl. t)")):
        bins = list(R["io2_base_eval"]["val"][key]); x = np.arange(len(bins)); bw = 0.8 / len(runs)
        for j, run in enumerate(runs):
            g = np.array([R["io2_base_eval"]["val"][key][b]["mae"] - R[run]["val"][key][b]["mae"] for b in bins])
            gs = np.array([[R[s]["val"][key][b]["mae"] - R[run]["val"][key][b]["mae"] for s in CONV] for b in bins])
            xo = x + (j - (len(runs) - 1) / 2) * bw
            a_.bar(xo, g, bw, color=COL[run], label=LABEL[run])
            a_.errorbar(xo, g, yerr=[g - gs.min(1), gs.max(1) - g], fmt="none", ecolor="k", capsize=3, lw=0.8)
        a_.set_xticks(x, [f"{b}\nn={R['io2_base_eval']['val'][key][b]['n'] / 1e6:.2f}M" for b in bins])
        a_.axhline(0, color="0.4", lw=1)
        a_.set(xlabel=xl, ylabel="MAE conv ±4 seed 0 − MAE run, ln CHL", title=f"Val MAE gain over conv ±4 seed 0,\nby {xl}")
        a_.grid(alpha=.3, axis="y"); a_.legend(fontsize=7)
    a_ = ax[3]
    kr = next((R[k] for k in KRIG if k in R and "by_key" in R[k]["val"]), None)
    keyed = [k for k in IO6 if k in R and "by_key" in R[k]["val"]]
    if kr is not None:
        groups = [">=1 key", "no key"]; x = np.arange(2); bw = 0.8 / (len(keyed) + 1)
        for j, run in enumerate(keyed + ["io4_tokdec_blend"]):
            src = R[run]["val"]["by_key"] if run != "io4_tokdec_blend" else kr["val"]["by_key"]
            g = np.array([src[gg]["conv_seed0"]["mae"] - (src[gg]["run"]["mae"] if run != "io4_tokdec_blend" else src[gg]["io4_tokdec_blend"]["mae"])
                          for gg in groups])
            a_.bar(x + (j - len(keyed) / 2) * bw, g, bw, color=COL[run], label=LABEL[run])
        a_.set_xticks(x, [f"{gg} in the stencil\nn={kr['val']['by_key'][gg]['run']['n'] / 1e6:.2f}M" for gg in groups])
        a_.axhline(0, color="0.4", lw=1); a_.grid(alpha=.3, axis="y"); a_.legend(fontsize=7)
    a_.set(xlabel="visible kriging-stencil keys at the hidden px", ylabel="MAE conv ±4 seed 0 − MAE run, ln CHL",
           title="Val MAE gain over conv ±4 seed 0,\nby visible keys in the kriging stencil")
    fig.text(0.01, 0.01, "bars: gain vs conv seed 0; whiskers (panels 1-3): range of the gain vs conv seeds 0/1/2; panel 4: seed 0 only; "
             "val hidden px n = 4,998,642 (1,458 days)", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "io6_bins.png", dpi=130); plt.close(fig)


def fig_variogram(run, vg, lp):
    out, maps, off = vg
    dts = np.arange(-4, 5)
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.4))
    ax[0].bar(dts, [out["share_by_dt"][int(t)] for t in dts], color=COL[run], label="mean over the value heads")
    if "signed_share_by_dt" in out:
        ax[0].plot(dts, [out["signed_share_by_dt"][int(t)][0] for t in dts], "o-", color=OKABE_ITO[0], label="signed pair: softmax(s+)")
        ax[0].plot(dts, [out["signed_share_by_dt"][int(t)][1] for t in dts], "s--", color=OKABE_ITO[1], label="signed pair: softmax(s−)")
    ax[0].set(xlabel="day offset dt of the key", ylabel="weight per hidden px (sum over the lag's keys)", title=f"Weight share by day offset, {run}")
    ax[0].legend(fontsize=7)
    ax[1].plot(dts, [out["w_per_visible_by_dt"][int(t)] for t in dts], "o-", color=COL[run])
    ax[1].set(xlabel="day offset dt", ylabel="mean weight per visible key", title="Mean weight per visible key by dt")
    xs = np.arange(-6, 7)
    ax[2].plot(xs, [out["w_per_visible_by_dx"][int(v)] for v in xs], "o-", color=COL[run])
    ax[2].set(xlabel="dx (px, columns; 1 px = 0.25°)", ylabel="mean weight per visible key", title="Mean weight per visible key by dx (all dt, dy)")
    ax[3].plot(xs, [out["w_per_visible_by_dy"][int(v)] for v in xs], "o-", color=COL[run])
    ax[3].set(xlabel="dy (px, rows; + = southward)", ylabel="mean weight per visible key", title="Mean weight per visible key by dy (all dt, dx)")
    for a_ in ax:
        a_.grid(alpha=.3)
    fig.text(0.01, 0.01, f"val hidden px of {out['n_days']} val days (every 10th): n = {out['n_hidden_px']:,} px, {out['n_hidden_px_with_key']:,} with "
             f">= 1 visible key; weights = final softmax weights, mean over the {lp['heads'] if lp else 4} value heads", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / f"io6_variogram_lines_{run}.png", dpi=130); plt.close(fig)
    if lp is None:
        return
    rb = np.full((9, 13, 13), np.nan)
    rbm = lp["relbias"][:len(off), :lp["heads"]].mean(1)
    for (t, y, x), v in zip(off, rbm):
        rb[t + 4, y + 6, x + 6] = v
    fig, ax = plt.subplots(2, 9, figsize=(27, 7.2), constrained_layout=True)
    lo, hi = np.nanpercentile(maps, [2, 98]); bl, bh = np.nanpercentile(rb, [2, 98])
    for i, t in enumerate(dts):
        im = ax[0, i].imshow(maps[i], cmap="viridis", vmin=lo, vmax=hi, extent=(-6.5, 6.5, 6.5, -6.5))
        ax[0, i].set_title(f"dt = {t:+d}", fontsize=10)
        im2 = ax[1, i].imshow(rb[i], cmap="PuOr_r", vmin=-max(abs(bl), abs(bh)), vmax=max(abs(bl), abs(bh)), extent=(-6.5, 6.5, 6.5, -6.5))
        ax[1, i].set_title(f"dt = {t:+d}", fontsize=10)
        for a_ in ax[:, i]:
            a_.set_xlabel("dx (px)")
    ax[0, 0].set_ylabel("mean weight per visible key\ndy (px)"); ax[1, 0].set_ylabel("relative-position bias\n(mean over value heads)\ndy (px)")
    fig.colorbar(im, ax=ax[0].tolist(), shrink=0.8, label="mean weight per visible key")
    fig.colorbar(im2, ax=ax[1].tolist(), shrink=0.8, label="relative-position bias (logit)")
    fig.suptitle(f"{run}: weights and relative-position bias by stencil offset (val hidden px; blank = offset not in the stencil)")
    fig.savefig(FIG / f"io6_variogram_maps_{run}.png", dpi=110); plt.close(fig)


def fig_sst(vgs):
    if not vgs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    for run, (out, _, _) in vgs.items():
        e = np.array(out["sst"]["edges_k"]); c = [0, 0.25, 0.5, 0.75, 1.0, 1.375, 1.875, 2.625, 3.5]
        lab = ["0", "0.25", "0.5", "0.75", "1.0", "1.25-1.5", "1.75-2", "2.25-3", ">3"]
        x = np.arange(len(lab))
        ax[0].plot(x, out["sst"]["w_per_visible"], "o-", color=COL[run], label=LABEL[run])
        ax[1].plot(x, out["sst"]["w_per_visible_dt0"], "o-", color=COL[run], label=LABEL[run])
    for a_, nkey, tt in ((ax[0], "n", "all dt"), (ax[1], "n_dt0", "dt = 0 keys only")):
        a_.set_xticks(x, [f"{l}\nn={n / 1e3:,.0f}k" for l, n in zip(lab, out["sst"][nkey])], fontsize=7)
        a_.set(xlabel="|SST(key) − SST(target px, day t)| (K; store SST is quantised to 0.25 K)", ylabel="mean weight per visible key",
               title=f"Mean weight per visible key by |SST difference|, {tt}")
        a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.text(0.01, 0.01, "val hidden px of every 10th val day; weights = mean over the value heads; n = visible keys (counts of the last run plotted)", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "io6_weight_vs_sst.png", dpi=130); plt.close(fig)


def fig_params(lps):
    if not lps:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    runs = list(lps)
    ax[0].bar(range(len(runs)), [lps[r]["s"] for r in runs], color=[COL[r] for r in runs])
    ax[0].axhline(lps[runs[0]]["s_init"], color="0.3", ls="--", lw=1, label=f"init {lps[runs[0]]['s_init']}")
    ax[0].set_xticks(range(len(runs)), [LABEL[r] for r in runs], fontsize=8)
    ax[0].set(ylabel="s (advected-offset scale of the currents)", title="Learned advected-offset scale s"); ax[0].legend(fontsize=8)
    sg = [r for r in runs if "lam" in lps[r]]
    if sg:
        ax[1].bar(range(len(sg)), [lps[r]["lam"] for r in sg], color=[COL[r] for r in sg])
        ax[1].axhline(0.1, color="0.3", ls="--", lw=1, label="init 0.1"); ax[1].legend(fontsize=8)
        ax[1].set_xticks(range(len(sg)), [LABEL[r] for r in sg], fontsize=8)
    ax[1].set(ylabel="lambda = sigmoid(theta)", title="Learned signed-pair weight lambda")
    for j, r in enumerate(runs):
        nl = lps[r]["none_logit"]
        ax[2].plot(np.arange(len(nl)), nl, "o", color=COL[r], label=LABEL[r])
    ax[2].set(xlabel="head (value heads first; signed pair last)", ylabel="no-key logit", title="Learned no-key logit per head"); ax[2].legend(fontsize=8)
    for a_ in ax:
        a_.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "io6_learned_params.png", dpi=130); plt.close(fig)


def fig_spread(rows):
    R = {r["run"]: r for r in rows}
    ks = [k for k in KRIG if k in R and "spread" in R[k]["val"]]
    if not ks:
        return
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.6))
    for k in ks:
        s = R[k]["val"]["spread"]
        ax[0].plot(s["spread_mean_by_decile"], s["mae_by_decile"], "o-", color=COL[k], label=LABEL[k])
        b = list(s["by_dist"]); ax[1].plot(range(len(b)), [s["by_dist"][x]["spearman"] for x in b], "o-", color=COL[k], label=LABEL[k])
        ax[1].set_xticks(range(len(b)), [f"{x}\nn={s['by_dist'][x]['n'] / 1e6:.2f}M" for x in b])
        b2 = list(s["by_age2"]); ax[2].plot(range(len(b2)), [s["by_age2"][x]["spearman"] for x in b2], "o-", color=COL[k], label=LABEL[k])
        ax[2].set_xticks(range(len(b2)), [f"{x}\nn={s['by_age2'][x]['n'] / 1e6:.2f}M" for x in b2])
    ax[0].set(xlabel="kriging spread (mean within spread decile), anomaly z units", ylabel="val hidden-px MAE, ln CHL",
              title="Val MAE by kriging-spread decile (hidden px with >= 1 key)")
    ax[1].set(xlabel="distance to nearest visible px (px)", ylabel="Spearman rho(spread, |error|)", title="Spread vs |error| rank correlation by distance")
    ax[2].set(xlabel="days to nearest obs (t±8, excl. t)", ylabel="Spearman rho(spread, |error|)", title="Spread vs |error| rank correlation by two-sided age")
    for a_ in ax:
        a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "io6_spread_vs_error.png", dpi=130); plt.close(fig)


# ================================================================ report
def main():
    FIG.mkdir(exist_ok=True)
    rows = table(); add_age2_refs(rows); ep = epochs(); sm = smokes()
    lps = {r: learned(r) for r in KRIG}
    lps = {k: v for k, v in lps.items() if v is not None}
    vgs = {r: variogram(r) for r in KRIG}
    vgs = {k: v for k, v in vgs.items() if v is not None}
    leak = {f.stem: load(f) for f in sorted(RES.glob("leak_*.json"))}
    out = dict(rows=rows, epochs={k: {kk: vv for kk, vv in v.items() if kk != "val"} for k, v in ep.items()}, smokes=sm, leak=leak,
               learned={k: {kk: vv for kk, vv in v.items() if kk != "relbias"} | dict(relbias_mean_over_value_heads_by_dt=
                        {int(t): float(v["relbias"][:vgs[k][2].shape[0], :v["heads"]].mean(1)[vgs[k][2][:, 0] == t].mean()) for t in range(-4, 5)}
                        if k in vgs else None) for k, v in lps.items()},
               kriging={k: v[0] for k, v in vgs.items()}, gates=gates(),
               gpu_check=load(RES / "gpu_check.json") if (RES / "gpu_check.json").exists() else None,
               unit_tests=load(RES / "unit_tests.json") if (RES / "unit_tests.json").exists() else None, seed_spread=dict(S60=0.00024, S30=0.0015))
    fig_curves(ep); fig_bins(rows); fig_sst(vgs); fig_params(lps); fig_spread(rows)
    for k, v in vgs.items():
        fig_variogram(k, v, lps.get(k))
    (RES / "io6_summary.json").write_text(json.dumps(out, indent=1))
    R = {r["run"]: r for r in rows}
    b0v, b0t = R["io2_base_eval"]["val"]["mae"], R["io2_base_eval"]["test"]["mae"]
    bbv, bbt = R["io4_tokdec_blend"]["val"]["mae"], R["io4_tokdec_blend"]["test"]["mae"]
    print(f"{'run':26s} {'val':>8s} {'dconv':>8s} {'dblend':>8s} {'test':>8s} {'dconv':>8s} {'dblend':>8s} {'rmse_v':>7s} {'bias_v':>7s} g1r/c   g2r/c   g4r/c   "
          f"s/ep batch GBalloc/res params best_ep  n_val_px nonfinite")
    for r in rows:
        v, t = r["val"], r["test"]; g = v["grad"]
        print(f"{r['run']:26s} {v['mae']:.6f} {v['mae'] - b0v:+.5f} {v['mae'] - bbv:+.5f} {t['mae']:.6f} {t['mae'] - b0t:+.5f} {t['mae'] - bbt:+.5f} "
              f"{v['rmse']:.4f} {v['bias']:+.4f} {g['1']['ratio']:.3f}/{g['1']['corr']:.3f} {g['2']['ratio']:.3f}/{g['2']['corr']:.3f} "
              f"{g['4']['ratio']:.3f}/{g['4']['corr']:.3f} {r.get('s_per_epoch', float('nan')):.0f} {r.get('batch', '-')} "
              f"{r.get('peak_gpu_gb') or float('nan'):.1f}/{r.get('peak_gpu_reserved_gb') or float('nan'):.1f} {r.get('n_params', '-')} {r.get('best_epoch', '-')}  "
              f"{v['n_hidden']} {v['n_nonfinite']}/{t['n_nonfinite']}")
    for key in ("by_dist", "by_age", "by_age2"):
        print(key, "n:", {b: R["io2_base_eval"]["val"][key][b]["n"] for b in R["io2_base_eval"]["val"][key]})
        for r in rows:
            if key in r["val"]:
                print(f"   {r['run']:26s}", {b: (round(r["val"][key][b]["mae"], 5), round(R["io2_base_eval"]["val"][key][b]["mae"] - r["val"][key][b]["mae"], 5))
                                              for b in r["val"][key]})
    for r in rows:
        if "by_key" in r["val"]:
            print("by_key", r["run"], json.dumps(r["val"]["by_key"]))
    print("epochs:", json.dumps(out["epochs"]))
    print("smokes:", json.dumps(sm))
    print("learned:", json.dumps(out["learned"]))
    print("kriging:", json.dumps(out["kriging"]))
    print("gates:", json.dumps(out["gates"]))


if __name__ == "__main__":
    main()
