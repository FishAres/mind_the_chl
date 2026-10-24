"""Part B of the io3 track: the patch-token stem run (io3_ptok4) against the reference runs. Analysis and figures only.

Inputs: analyses/io3_attn/results/io3_ptok4.json (score.py metrics), analyses/io_improve/results/<ref>.json (references), the runs'
history.json / config.json, ~/mind_the_chl_gap/data/io3/io3_ptok4_maps_ptok.npz (ptok_eval.py recorder, 6 val days).

Comparisons (as analyses/io_improve/PLAN.md section 3): at 60 epochs the selected model's val (best over epochs 11..60) and its test;
at 30 epochs the best val over epochs 11..30 from history.json (io2_tattn8_s30, the conv seeds 0/1/2).
Score-term diagnostic: on the softmax domain (context pixel-days with conf(m) > 0 for lavg, visible for own) of the 6 recorded val days, the std of the interpolated patch score and of the
pixel term, their correlation, and the variance share of each in the logit (var(term) / var(logit)); mean "no day" weight and
mean weight entropy over all ocean pixels of the store mask (hidden pixels are not known to the recorder).

Usage (project root): python -m analyses.io3_attn.report_b
Writes results/partb.json, figures/b_curves.png, b_bins.png, b_weight_maps.png.
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
DATA = Path.home() / "mind_the_chl_gap" / "data" / "io3"
ARMS = [("io3_ptok4_lavg", "arm B: local masked average"), ("io3_ptok4", "arm A: own pixel")]
REFS = [("io3_ptok4_lavg", "patch tokens P4, arm B (local avg), b8", RES, "io3_ptok4_lavg"),
        ("io3_ptok4", "patch tokens P4, arm A (own px), b8", RES, "io3_ptok4"),
        ("io2_base_eval", "conv ±4 seed 0, b8", IO2, "t3d_dil_c4_60ep_gdl100"),
        ("io2_seed1", "conv ±4 seed 1, b8", IO2, "io2_seed1"), ("io2_seed2", "conv ±4 seed 2, b8", IO2, "io2_seed2"),
        ("io2_tattn8", "attention ±8, b4", IO2, "io2_tattn8"),
        ("ens3_mean", "conv 3-seed ensemble (restarted), mean", IO2, None)]
S30 = [("io3_ptok4_lavg", "patch tokens, arm B"), ("io3_ptok4", "patch tokens, arm A"), ("t3d_dil_c4_60ep_gdl100", "conv seed 0"), ("io2_seed1", "conv seed 1"), ("io2_seed2", "conv seed 2"),
       ("io2_tattn8_s30", "attention ±8, stop 30"), ("io2_tattn8", "attention ±8, 60-ep run")]


def load(p):
    return json.loads(Path(p).read_text())


def table():
    rows = []
    for key, lab, d, run in REFS:
        if not (d / f"{key}.json").exists():
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


def at30():
    out = {}
    for run, lab in S30:
        try:
            if not (find_run(run) / "history.json").exists():
                continue
        except FileNotFoundError:
            continue
        h = load(find_run(run) / "history.json"); v = np.array(h["val"])
        if len(v) < 11:
            continue
        out[run] = dict(label=lab, best_11_30=float(v[10:30].min()), epoch=int(np.argmin(v[10:30]) + 11), raw_30=float(v[29]),
                        best_11_60=float(v[10:60].min()) if len(v) >= 60 else None, val=v.tolist())
    conv = [out[r]["best_11_30"] for r in ("t3d_dil_c4_60ep_gdl100", "io2_seed1", "io2_seed2")]
    out["S30"] = max(conv) - min(conv)
    return out


def weights_diag(run):
    from chl_cnn import store as S
    ocean_store = np.asarray(S.load_store()["ocean"]).astype(bool)
    z = np.load(DATA / f"{run}_maps_ptok.npz")
    pos = z["positions"].tolist()
    sp, px, wnone, ent = [], [], [], []
    for p in pos:
        vis, w = z[f"vis__{p}"], z[f"w__{p}"]
        T = vis.shape[0]; ti = T // 2
        ctx = (z[f"conf__{p}"] > 1e-5) if f"conf__{p}" in z else vis.copy()     # softmax domain: lavg conf(m) > 0, own: visible
        ctx[ti] = False
        sp.append(z[f"s_patch__{p}"][ctx]); px.append(z[f"s_pixel__{p}"][ctx])
        ocean = ocean_store
        wnone.append(w[T][ocean]); ent.append(z[f"ent__{p}"][0][ocean])
    sp, px = np.concatenate(sp), np.concatenate(px)
    lg = sp + px
    return dict(n_days=len(pos), positions=pos, n_pixel_days=int(sp.size), std_patch=float(sp.std()), std_pixel=float(px.std()),
                corr_patch_pixel=float(np.corrcoef(sp, px)[0, 1]), var_share_patch=float(sp.var() / lg.var()), var_share_pixel=float(px.var() / lg.var()),
                mean_w_none_all_ocean_px=float(np.concatenate(wnone).mean()), mean_entropy_all_ocean_px=float(np.concatenate(ent).mean()))


def fig_curves(s30, rows):
    s30 = {k: v for k, v in s30.items() if isinstance(v, dict)}
    fig, ax = plt.subplots(figsize=(8, 4.3))
    cols = {"io3_ptok4_lavg": OKABE_ITO[1], "io3_ptok4": OKABE_ITO[3], "t3d_dil_c4_60ep_gdl100": OKABE_ITO[0], "io2_seed1": OKABE_ITO[5], "io2_seed2": OKABE_ITO[2], "io2_tattn8": OKABE_ITO[4]}
    for r, c in cols.items():
        if r not in s30:
            continue
        v = s30[r]["val"]
        ax.plot(np.arange(1, len(v) + 1), v, color=c, label=s30[r]["label"], lw=2 if r.startswith("io3") else 1)
    ax.set_ylim(0.127, 0.145); ax.axvline(30, color="0.5", ls=":", lw=1)
    ax.set(xlabel="epoch (cosine over 60 epochs)", ylabel="val hidden-pixel MAE, ln CHL", title="Val MAE by epoch: patch-token stem and reference runs")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.text(0.01, 0.01, "val n = 1,458 days, 4,998,642 hidden px; attention ±8 runs at batch 4, others batch 8", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "b_curves.png", dpi=130); plt.close(fig)


def add_age_twosided(rows):
    """val MAE by two-sided age (days to the nearest real obs in t-8..t+8 excl. t; checks.py cache) into rows[i]['val']['by_age2']."""
    from chl_cnn import store as S, data as D
    from analyses.forecast.forecast_eval import BINS as AGE_BINS
    age = np.load(DATA / "val_age_twosided.npy")
    d = S.load_store(); times = d["time"]; y = D.log_chl(np.asarray(d["chl_l3"]))
    _, va, _ = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS); yv = y[va]
    src = {"io2_base_eval": "io2_base_eval"}
    for r in rows:
        if r["run"] == "ens3_mean":
            continue
        z = np.load(find_run(r["run"]) / "preds.npz"); hid = z["hidden"][:len(va)]
        ae = np.abs(z["pred"][:len(va)].astype("float32") - yv)
        r["val"]["by_age2"] = {}
        for name, lo, hi in AGE_BINS:
            m = hid & ((age == 0) if hi == 0 else ((age >= lo) & (age <= hi)))
            r["val"]["by_age2"][name] = dict(mae=float(ae[m].mean()), n=int(m.sum()))


def fig_bins(rows):
    R = {r["run"]: r for r in rows}
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.3))
    for a_, key, xl in ((ax[0], "by_age", "age of last obs (t−8..t−1)"), (ax[1], "by_age2", "days to nearest obs (t±8, excl. t)"),
                        (ax[2], "by_dist", "distance to visible px")):
        bins = list(R["io2_tattn8"]["val"][key]); x = np.arange(len(bins))
        runs = [(r, l, c) for r, l, c in (("io3_ptok4_lavg", "patch tokens arm B", OKABE_ITO[1]), ("io3_ptok4", "patch tokens arm A", OKABE_ITO[3]),
                                          ("io2_tattn8", "attention ±8", OKABE_ITO[4])) if r in R]
        bw = 0.8 / len(runs)
        for j, (run, lab, c) in enumerate(runs):
            g = [R["io2_base_eval"]["val"][key][b]["mae"] - R[run]["val"][key][b]["mae"] for b in bins]
            gs = [[R[s]["val"][key][b]["mae"] - R[run]["val"][key][b]["mae"] for s in ("io2_base_eval", "io2_seed1", "io2_seed2")] for b in bins]
            xo = x + (j - (len(runs) - 1) / 2) * bw
            a_.bar(xo, g, bw, color=c, label=f"{lab} vs conv seed 0")
            a_.errorbar(xo, g, yerr=[np.array(g) - [min(v) for v in gs], [max(v) for v in gs] - np.array(g)], fmt="none", ecolor="k", capsize=3)
        a_.set_xticks(x, [f"{b}\nn={R['io2_tattn8']['val'][key][b]['n'] / 1e6:.2f}M" for b in bins])
        a_.axhline(0, color="0.4", lw=1)
        a_.set(xlabel=xl, ylabel="MAE conv ±4 − MAE run, ln CHL", title=f"Val MAE gain over conv ±4, by {xl}")
        a_.grid(alpha=.3, axis="y"); a_.legend(fontsize=8)
    fig.text(0.01, 0.01, "bars: vs conv seed 0; whiskers: range vs conv seeds 0/1/2; val hidden px n = 4,998,642", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "b_bins.png", dpi=130); plt.close(fig)


def fig_maps(run):
    z = np.load(DATA / f"{run}_maps_ptok.npz")
    from chl_cnn import store as S, data as D
    st = S.load_store(); times = st["time"]; ocean = np.asarray(st["ocean"]).astype(bool)   # store ocean mask (north-first rows)
    _, va, _ = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    for p in z["positions"][:3]:
        vis, w, sp, px = z[f"vis__{p}"], z[f"w__{p}"], z[f"s_patch__{p}"], z[f"s_pixel__{p}"]
        T = vis.shape[0]; ti = T // 2
        offs = [-4, -2, -1, 1, 2, 4]
        fig, ax = plt.subplots(4, len(offs), figsize=(2.6 * len(offs), 9))
        lims = dict(sp=np.nanpercentile(sp[:, ocean], [2, 98]), px=np.nanpercentile(px[:, ocean], [2, 98]))
        for j, k in enumerate(offs):
            for i, (arr, ttl, cm, lo_hi) in enumerate(((sp[ti + k], "patch score (interp.)", MAP_CMAP, lims["sp"]), (px[ti + k], "pixel term", MAP_CMAP, lims["px"]),
                                                     (w[ti + k], "day weight", MAP_CMAP, (0, 1)),
                                                     ((z[f"conf__{p}"][ti + k], "local confidence conv(m)", "cividis", (0, 1)) if f"conf__{p}" in z
                                                      else (vis[ti + k].astype(float), "visible", "cividis", (0, 1))))):
                im = ax[i, j].imshow(np.where(ocean, arr, np.nan), cmap=cm, vmin=lo_hi[0], vmax=lo_hi[1])
                ax[i, j].set_title(f"{ttl}, t{k:+d}", fontsize=8); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
                if j == len(offs) - 1:
                    fig.colorbar(im, ax=ax[i, j], shrink=.8)
        fig.suptitle(f"{run} day-score terms, day weights and input availability, val day {str(times[va[p]])[:10]}")
        fig.tight_layout(); fig.savefig(FIG / f"b_weight_maps_{run}_{str(times[va[p]])[:10]}.png", dpi=110); plt.close(fig)


def main():
    rows = table(); add_age_twosided(rows); s30 = at30()
    have = [r for r, _ in ARMS if (DATA / f"{r}_maps_ptok.npz").exists()]
    wd = {r: weights_diag(r) for r in have}
    out = dict(rows=rows, at30={k: ({kk: vv for kk, vv in v.items() if kk != "val"} if isinstance(v, dict) else v) for k, v in s30.items()},
               score_terms=wd, leak={s: load(RES / f"leak_{s}.json") for s in ("io3_smoke_ptok4_lavg", "io3_smoke_ptok4_own")})
    (RES / "partb.json").write_text(json.dumps(out, indent=1))
    fig_curves(s30, rows); fig_bins(rows)
    for r in have:
        fig_maps(r)
    print(f"{'run':16s} {'val':>8s} {'dval':>8s} {'test':>8s} {'rmse_v':>7s} {'bias_v':>7s} g1r/c   g2r/c   g4r/c   s/ep  GB  params batch")
    b0 = [r for r in rows if r["run"] == "io2_base_eval"][0]["val"]["mae"]
    for r in rows:
        v, t = r["val"], r["test"]; g = v["grad"]
        print(f"{r['run']:16s} {v['mae']:.6f} {v['mae'] - b0:+.5f} {t['mae']:.6f} {v['rmse']:.4f} {v['bias']:+.4f} "
              f"{g['1']['ratio']:.3f}/{g['1']['corr']:.3f} {g['2']['ratio']:.3f}/{g['2']['corr']:.3f} {g['4']['ratio']:.3f}/{g['4']['corr']:.3f} "
              f"{r.get('s_per_epoch', float('nan')):.0f} {r.get('peak_gpu_reserved_gb') or float('nan'):.1f} {r.get('n_params', '-')} {r.get('batch', '-')}"
              f"  best ep {r.get('best_epoch', '-')}  n {v['n_hidden']} nonfinite {v['n_nonfinite']}/{t['n_nonfinite']}")
    print("test grad:", {r["run"]: [round(r["test"]["grad"][L][q], 3) for L in ("1", "2", "4") for q in ("ratio", "corr")] for r in rows})
    print("at 30:", json.dumps(out["at30"]))
    for key in ("by_age", "by_age2", "by_dist"):
        print(key, {b: [round(r["val"][key][b]["mae"], 4) for r in rows if key in r["val"]] for b in rows[0]["val"][key]})
    print("score terms:", json.dumps(wd))


if __name__ == "__main__":
    main()
