"""Report for the tracer diagnostics (CPU): reads results/diag_landscape_*.npz, diag_synth_*.npz / diag_synth_train_*.json,
diag_scalar_lr*.json, models/tracer/diag_src_*/{history.json, perday_val.npz}; writes results/diag_tables.md,
results/diag_summary.json and figures/diag_*.png.

Definitions
- Landscape minimum: grid argmin of the MAE curve over a, refined by the vertex of the parabola through the argmin and
  its two neighbours ("vertex"; grid value if the argmin is at the grid edge). Mean-lead curve = unweighted mean of the
  five per-lead MAEs (the training loss weighting).
- Bootstrap: 300 resamples of val start days (with replacement), pooled MAE recomputed from per-day sums; 95 %
  percentile intervals for the fixed-set pixel argmin vertex and for MAE(a = 0.5) - MAE(a = 0.17).
- Pearson r: pooled over pixels (and start days) of the set.
- Source runs, per-day contributions: day d's contribution to the val MAE of eval e at lead k is E_ed / N_e (sum |error|
  of day d over the pooled pixel count), so the eval MAE is the sum over days; between consecutive evals the change is
  the sum of per-day changes c_d. Concentration = share of sum |c_d| carried by the 10 days with the largest |c_d|
  (n days 351; equal shares would give 10 / 351 = 2.8 %; share of pixel count of the same days given for reference).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from chl_cnn.plotting import OKABE_ITO as C
from chl_cnn.tracer_diag import curve_metrics, vertex

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
MODELS = ROOT / "models/tracer"
K = 5
VARIANTS = ["base", "shift_m1", "shift_p1", "nsub1", "nsub16", "sampling:nearest", "sampling:bilinear", "sampling:bicubic", "geo_mean", "train_base"]
VLAB = {"base": "base (u_rea, n_sub 4, bilinear)", "shift_m1": "currents day j-1", "shift_p1": "currents day j+1", "nsub1": "n_sub 1",
        "nsub16": "n_sub 16", "sampling:nearest": "nearest (strict set)", "sampling:bilinear": "bilinear (strict set)",
        "sampling:bicubic": "bicubic (strict set)", "geo_mean": "geo_mean component", "train_base": "base, train starts (every 3rd)"}
SYN = ["anom_only", "none", "iid", "corr_noise", "static30", "static50", "decay", "all"]
SLAB = {"anom_only": "anomaly only", "none": "none (clim base + anomaly)", "iid": "+ iid noise", "corr_noise": "+ correlated noise",
        "static30": "+ static 30 %", "static50": "+ static 50 %", "decay": "+ decay", "all": "+ corr. noise, static 30 %, decay"}
SRC = ["diag_src_ctrl", "diag_src_lr0p1", "diag_src_lrwind0p1", "diag_src_g0p1", "diag_src_wd", "diag_src_drop"]
SRCLAB = {"diag_src_ctrl": "control (lr 3e-3, wind 1e-2)", "diag_src_lr0p1": "all lr x0.1", "diag_src_lrwind0p1": "wind lr x0.1",
          "diag_src_g0p1": "g_max 0.1", "diag_src_wd": "AdamW wd 1.0 (source)", "diag_src_drop": "dropout 0.2 (source)"}
NB = 300


def fmt(x, d=3):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def vtx(al, y, find="min"):
    g, v = vertex(al, y, find)
    return v if np.isfinite(v) else g


# ============================== landscape ==============================
def load_landscape():
    out = {}
    for f in sorted(RES.glob("diag_landscape_*.npz")):
        v = f.stem.replace("diag_landscape_", "")
        if v.endswith("_debug") or v == "check":
            continue
        z = np.load(f)
        info = json.loads(str(z["info"]))
        for m in info["modes"]:
            key = v if v != "sampling" else f"sampling:{m}"
            out[key] = dict(al=z["alphas"], info=info, stats={s: z[f"stats__{m}__{s}"] for s in ["fixed", "own", "fixed_moving", "fixed_zero"] if f"stats__{m}__{s}" in z},
                            perday=z[f"perday__{m}"])
    return out


def boot(al, perday, j, rng, i17, i5):
    """perday (n_days, n_a, K, 2, 2) -> bootstrap of the vertex and of MAE(0.5) - MAE(0.17) per lead (+ mean lead)."""
    nd = perday.shape[0]
    vs = np.zeros((NB, K + 1)); dd = np.zeros((NB, K + 1))
    for b in range(NB):
        s = perday[rng.integers(0, nd, nd)].sum(0)               # (n_a, K, 2, 2)
        mae = s[:, :, j, 0] / s[:, :, j, 1]
        ml = mae.mean(1)
        for k in range(K + 1):
            y = mae[:, k] if k < K else ml
            vs[b, k] = vtx(al, y); dd[b, k] = y[i5] - y[i17]
    q = lambda x: np.percentile(x, [2.5, 97.5], 0)
    return q(vs), q(dd)


def landscape_tables(L, rng):
    rows, summ = [], {}
    for v in VARIANTS:
        if v not in L:
            continue
        al = L[v]["al"]; st = L[v]["stats"]
        i17, i5 = list(al).index(0.17), list(al).index(0.5)
        res = {}
        for sname in ("fixed", "own"):
            for j, met in ((0, "pix"), (1, "band")):
                if v == "train_base" and j == 1:
                    continue
                mae, r, n = curve_metrics(st[sname][:, :, j])
                ml = mae.mean(1)
                res[f"{sname}_{met}"] = dict(argmin=[vtx(al, mae[:, k]) for k in range(K)] + [vtx(al, ml)],
                                             argmax_r=[vtx(al, r[:, k], "max") for k in range(K)],
                                             d50_17=[float(mae[i5, k] - mae[i17, k]) for k in range(K)] + [float(ml[i5] - ml[i17])],
                                             mae_min=[float(np.nanmin(mae[:, k])) for k in range(K)],
                                             mae_a0=[float(mae[0, k]) for k in range(K)],
                                             n_a0=n[0].astype(int).tolist(), n_amax=n[-1].astype(int).tolist())
        ci_v, ci_d = boot(al, L[v]["perday"], 0, rng, i17, i5)
        res["fixed_pix"]["argmin_ci"] = ci_v.T.tolist(); res["fixed_pix"]["d50_17_ci"] = ci_d.T.tolist()
        if v != "train_base":
            ci_vb, ci_db = boot(al, L[v]["perday"], 1, rng, i17, i5)
            res["fixed_band"]["argmin_ci"] = ci_vb.T.tolist(); res["fixed_band"]["d50_17_ci"] = ci_db.T.tolist()
        for sub in ("fixed_moving", "fixed_zero"):
            if sub in st:
                for j, met in ((0, "pix"), (1, "band")):
                    mae, r, n = curve_metrics(st[sub][:, :, j])
                    res[f"{sub}_{met}"] = dict(argmin=[vtx(al, mae[:, k]) for k in range(K)] + [vtx(al, mae.mean(1))],
                                               d50_17=[float(mae[i5, k] - mae[i17, k]) for k in range(K)],
                                               argmax_r=[vtx(al, r[:, k], "max") for k in range(K)], n=n[0].astype(int).tolist())
        res["info"] = {k: L[v]["info"][k] for k in ("n_starts", "n_fixed_px", "nonfinite_in_own") if k in L[v]["info"]}
        summ[v] = res
    return summ


def fig_landscape_base(L):
    b = L["base"]; al = b["al"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for j, met in ((0, "pixel"), (1, "band-passed 24-150 km")):
        mae, r, n = curve_metrics(b["stats"]["fixed"][:, :, j])
        for k in range(K):
            ax[j, 0].plot(al, mae[:, k] - np.nanmin(mae[:, k]), color=C[k], label=f"lead {k + 1} (n {int(n[0, k]) / 1e6:.2f} M)")
            ax[j, 0].axvline(vtx(al, mae[:, k]), color=C[k], ls=":", lw=1)
            ax[j, 1].plot(al, r[:, k], color=C[k], label=f"lead {k + 1}")
        ax[j, 0].set(title=f"{met.capitalize()} MAE - min vs a (fixed set)", xlabel="a (u_eff = a x u_rea)", ylabel="MAE - min MAE (ln CHL)")
        ax[j, 0].set_ylim(0, 0.03 if j == 0 else 0.012); ax[j, 0].legend(fontsize=8)
        ax[j, 1].set(title=f"{met.capitalize()}: Pearson r, prediction vs target", xlabel="a", ylabel="Pearson r (pooled pixels)")
        ax[j, 1].legend(fontsize=8)
    for j, (sub, lab) in enumerate((("fixed_moving", "moving patches"), ("fixed_zero", "zero-shift patches"))):
        mae, r, n = curve_metrics(b["stats"][sub][:, :, 0])
        for k in range(K):
            ax[j, 2].plot(al, mae[:, k] - np.nanmin(mae[:, k]), color=C[k], label=f"lead {k + 1} (n {int(n[0, k]) / 1e3:.0f} k)")
            ax[j, 2].axvline(vtx(al, mae[:, k]), color=C[k], ls=":", lw=1)
        ax[j, 2].set(title=f"Pixel MAE - min vs a: {lab}", xlabel="a", ylabel="MAE - min MAE (ln CHL)")
        ax[j, 2].set_ylim(0, 0.03); ax[j, 2].legend(fontsize=8)
    fig.suptitle("Val loss landscape of u_eff = a x u_rea (fixed pixel set; dotted lines = parabolic minimum per lead; moving: |shift| >= 1 px)")
    fig.tight_layout(); fig.savefig(FIG / "diag_landscape_base.png", dpi=110); plt.close(fig)


def fig_landscape_variants(S):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    leads = np.arange(1, K + 1)
    sty = {"base": (C[0], "-", "o"), "shift_m1": (C[1], "-", "v"), "shift_p1": (C[2], "-", "^"), "nsub1": (C[3], "-", "s"),
           "nsub16": (C[4], "-", "D"), "sampling:nearest": (C[5], "--", "o"), "sampling:bilinear": (C[6], "--", "o"),
           "sampling:bicubic": (C[7], "--", "o"), "geo_mean": (C[0], ":", "P"), "train_base": (C[3], ":", "X")}
    for i, v in enumerate([v for v in VARIANTS if v in S]):
        col, ls, mk = sty[v]
        for j, key in enumerate(("fixed_pix", "fixed_band")):
            if key in S[v]:
                ax[j].plot(leads, S[v][key]["argmin"][:K], marker=mk, color=col, ls=ls, label=VLAB[v])
        ax[2].plot(leads, S[v]["fixed_pix"]["d50_17"][:K], marker=mk, color=col, ls=ls, label=VLAB[v])
    ci = np.array(S["base"]["fixed_pix"]["argmin_ci"])[:K]
    ax[0].fill_between(leads, ci[:, 0], ci[:, 1], color=C[0], alpha=0.2, label="base 95 % day-bootstrap")
    for j, t in enumerate(("pixel MAE", "band-passed MAE (24-150 km)")):
        ax[j].set(title=f"Landscape minimum a by lead, {t}, fixed set", xlabel="lead (days)", ylabel="a at the MAE minimum")
    ax[2].axhline(0, color="grey", lw=0.8)
    ax[2].set(title="Pixel MAE(a = 0.5) - MAE(a = 0.17), fixed set", xlabel="lead (days)", ylabel="MAE difference (ln CHL)")
    h, l = ax[1].get_legend_handles_labels()
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "diag_landscape_variants.png", dpi=110); plt.close(fig)


# ============================== synthetic ==============================
def load_synth():
    out = {}
    for c in SYN:
        f = RES / f"diag_synth_{c}.npz"
        if f.exists():
            z = np.load(f)
            out[c] = dict(al=z["alphas"], info=json.loads(str(z["info"])), stats={s: z[f"stats__bilinear__{s}"] for s in ("fixed", "own")}, perday=z["perday__bilinear"])
        ft = RES / f"diag_synth_train_{c}.json"
        if ft.exists() and c in out:
            out[c]["train"] = json.load(open(ft))
        ft2 = RES / f"diag_synth_train_{c}_lr0.03_ep2.json"
        if ft2.exists() and c in out:
            out[c]["train2"] = json.load(open(ft2))
    return out


def synth_tables(Sy, rng):
    res = {}
    for c, s in Sy.items():
        al = s["al"]; i17, i5 = list(al).index(0.17), list(al).index(0.5)
        r = {}
        for j, met in ((0, "pix"), (1, "band")):
            mae, rr, n = curve_metrics(s["stats"]["fixed"][:, :, j])
            r[met] = dict(argmin=[vtx(al, mae[:, k]) for k in range(K)] + [vtx(al, mae.mean(1))],
                          argmax_r=[vtx(al, rr[:, k], "max") for k in range(K)], n=n[0].astype(int).tolist(),
                          d50_17=[float(mae[i5, k] - mae[i17, k]) for k in range(K)])
        ci_v, _ = boot(al, s["perday"], 0, rng, i17, i5)
        r["pix"]["argmin_ci"] = ci_v.T.tolist()
        if "train" in s:
            t = s["train"]
            r["trained_a"] = t["a"][-1]; r["trained_steps"] = len(t["a"])
        if "train2" in s:
            r["trained_a_lr3e-2"] = s["train2"]["a"][-1]; r["trained_a_lr3e-2_last50_sd"] = float(np.std(s["train2"]["a"][-50:]))
        r["anomaly_var_lead0"] = s["info"]["anomaly_var_obs_px_lead0"]
        res[c] = r
    return res


def fig_synth(Sy, T):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    leads = np.arange(1, K + 1)
    for i, c in enumerate([c for c in SYN if c in T]):
        col = C[i % len(C)]
        ax[0].plot(leads, T[c]["pix"]["argmin"][:K], marker="o", color=col, label=SLAB[c])
        ax[1].plot(leads, T[c]["band"]["argmin"][:K], marker="o", color=col, label=SLAB[c])
        mae, r, n = curve_metrics(Sy[c]["stats"]["fixed"][:, :, 0])
        ax[2].plot(Sy[c]["al"], mae[:, 0] - np.nanmin(mae[:, 0]), color=col, label=SLAB[c])
    for j in range(2):
        ax[j].axhline(0.5, color="grey", ls="--", lw=1, label="true a = 0.5")
        ax[j].set(xlabel="lead (days)", ylabel="a at the MAE minimum", ylim=(0, 0.65))
    ax[0].set_title("Synthetic: landscape minimum a by lead, pixel MAE, fixed set")
    ax[1].set_title("Synthetic: landscape minimum a by lead, band MAE (24-150 km)")
    ax[2].axvline(0.5, color="grey", ls="--", lw=1)
    ax[2].set(title="Synthetic, lead 1: pixel MAE minus its minimum vs a", xlabel="a", ylabel="MAE - min MAE", ylim=(0, 0.02))
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "diag_synth.png", dpi=110); plt.close(fig)


# ============================== scalar training ==============================
def load_scalar():
    out = {}
    for f in sorted(RES.glob("diag_scalar_lr*.json")):
        if "debug" in f.stem:
            continue
        j = json.load(open(f)); out[j["lr"]] = j
    return out


def fig_scalar(Sc, S):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ls = {0.1: ":", 0.5: "-", 1.0: "--"}
    for i, lr in enumerate(sorted(Sc)):
        for a0, r in Sc[lr]["runs"].items():
            ax[0].plot(np.arange(1, len(r["a"]) + 1), r["a"], color=C[i], ls=ls.get(float(a0), "-"), label=f"lr {lr:g}, a0 {a0}")
            ax[1].plot(r["eval_steps"], np.mean(r["val_mae"], 1), color=C[i], ls=ls.get(float(a0), "-"), marker=".", label=f"lr {lr:g}, a0 {a0}")
    if "train_base" in S:
        ax[0].axhline(S["train_base"]["own_pix"]["argmin"][K], color="k", lw=1, label="landscape min, train starts (mean lead, own sets)")
    if "base" in S:
        ax[0].axhline(S["base"]["own_pix"]["argmin"][K], color="grey", lw=1, ls="--", label="landscape min, val (mean lead, own sets)")
    ax[0].set(title="Scalar a during training (Adam, batch 16, train starts)", xlabel="optimiser step", ylabel="a", xscale="log")
    ax[1].set(title="Val MAE (mean of leads 1-5, own set) during training", xlabel="optimiser step", ylabel="MAE (ln CHL)")
    ax[1].set_ylim(None, None)
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "diag_scalar_train.png", dpi=110); plt.close(fig)


# ============================== source runs ==============================
def load_src():
    out = {}
    for r in SRC + ["tr_src"]:
        p = MODELS / r
        if not (p / "history.json").exists():
            continue
        h = json.load(open(p / "history.json"))
        o = dict(hist=h, cfg=json.load(open(p / "config.json")))
        if (p / "perday_val.npz").exists():
            z = np.load(p / "perday_val.npz"); o.update(e=z["e"], n=z["n"], check=float(z["check_max_abs_diff"]))
        mt = json.load(open(p / "metrics.json"))
        o["init_common"] = [mt["init"]["common"]["model"][str(k)]["mae"] for k in range(1, K + 1)]
        o["val_days"] = mt["best"]["val_days"]
        out[r] = o
    return out


def src_tables(Sr):
    res = {}
    for r, o in Sr.items():
        v = np.array(o["hist"]["val_model_common"]); trn = np.array(o["hist"]["train_l1"])
        dv = np.diff(v, axis=0)
        x = dict(n_epochs=len(v), steps_per_epoch=o["hist"]["steps"][0], val_range=(v.max(0) - v.min(0)).tolist(),
                 val_absdiff_mean=np.abs(dv).mean(0).tolist() if len(dv) else None, val_absdiff_max=np.abs(dv).max(0).tolist() if len(dv) else None,
                 val_last=v[-1].tolist(), val_min=v.min(0).tolist(), train_first=float(trn[0]), train_last=float(trn[-1]),
                 a_last=o["hist"]["factors"][-1].get("a", {}).get("mean"), S_mean=[s["mean"] if s else None for s in o["hist"]["source"]])
        if "e" in o:
            E, N = o["e"], o["n"]                                   # (n_evals, n_days, K)
            x["perday_check_max_abs_diff"] = o["check"]
            Np = N.sum(1, keepdims=True)
            contrib = E / Np                                        # (evals, days, K)
            ep = contrib[1:1 + len(v)]                              # epochs only
            c = np.diff(ep, axis=0)                                 # (transitions, days, K)
            conc, nshare, top_days = [], [], []
            for k in range(K):
                cs, ns = [], []
                for t in range(c.shape[0]):
                    o_ = np.argsort(-np.abs(c[t, :, k]))[:10]
                    cs.append(np.abs(c[t, o_, k]).sum() / np.abs(c[t, :, k]).sum())
                    ns.append(N[t + 2, o_, k].sum() / N[t + 2, :, k].sum())
                conc.append(float(np.mean(cs))); nshare.append(float(np.mean(ns)))
            x["top10_share_abs_change"] = conc; x["top10_share_pixels"] = nshare
            # coherence of the change across days: pixel-weighted share of days whose change has the sign of the total
            agree = []
            for k in range(K):
                ag = []
                for t in range(c.shape[0]):
                    tot = c[t, :, k].sum(); w = N[t + 2, :, k]
                    ag.append(float((w * (np.sign(c[t, :, k]) == np.sign(tot))).sum() / w.sum()))
                agree.append(float(np.mean(ag)))
            x["sign_agreement_share"] = agree
            sm = np.array([q if q is not None else np.nan for q in x["S_mean"]])
            x["r_valS_vs_val"] = [float(np.corrcoef(sm, v[:, k])[0, 1]) for k in range(K)]
            # per-day variance of the contribution across epochs; leave-top-10-out swing
            var = ep.var(0)                                         # (days, K)
            lo = []
            for k in range(K):
                o_ = np.argsort(-var[:, k])[:10]
                keep = np.setdiff1d(np.arange(var.shape[0]), o_)
                m_all = E[1:1 + len(v), :, k].sum(1) / N[1:1 + len(v), :, k].sum(1)
                m_kept = E[1:1 + len(v)][:, keep, k].sum(1) / N[1:1 + len(v)][:, keep, k].sum(1)
                lo.append(dict(range_all=float(np.ptp(m_all)), range_without_top10=float(np.ptp(m_kept)),
                               sd_diff_all=float(np.std(np.diff(m_all))), sd_diff_without_top10=float(np.std(np.diff(m_kept))),
                               top10_var_share=float(var[o_, k].sum() / var[:, k].sum()),
                               top10_days=[o["val_days"][i] for i in o_]))
            x["leave_top10_out"] = lo
        res[r] = x
    if "diag_src_ctrl" in res:
        c0 = np.diff(np.array(Sr["diag_src_ctrl"]["hist"]["val_model_common"])[:, 4])
        for r in res:
            v = np.diff(np.array(Sr[r]["hist"]["val_model_common"])[:, 4])
            res[r]["r_dval_l5_vs_control"] = float(np.corrcoef(c0, v)[0, 1]) if len(v) == len(c0) else None
    return res


def fig_src(Sr):
    runs = [r for r in SRC if r in Sr]
    if not runs:
        return
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for i, r in enumerate(runs):
        h = Sr[r]["hist"]; v = np.array(h["val_model_common"]); st = np.array(h["steps"])
        x0 = np.concatenate([[0], st])
        for j, k in enumerate((0, 4)):
            ax[j].plot(x0, np.concatenate([[Sr[r]["init_common"][k]], v[:, k]]), marker="o", ms=3, color=C[i], label=SRCLAB[r])
        ax[2].plot(st, h["train_l1"], marker="o", ms=3, color=C[i], label=SRCLAB[r])
    for j, k in enumerate((1, 5)):
        ax[j].set(title=f"Source runs: val MAE, lead {k} (common set)", xlabel="optimiser step (eval every 50 steps; same batch order, seed 0)", ylabel="MAE (ln CHL)")
    ax[2].set(title="Source runs: train L1 per eval interval (mean of leads)", xlabel="optimiser step", ylabel="train L1 (ln CHL)")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "diag_src_val_curves.png", dpi=110); plt.close(fig)
    # per-day contributions, control
    r = "diag_src_ctrl"
    if r in Sr and "e" in Sr[r]:
        E, N = Sr[r]["e"], Sr[r]["n"]
        nep = len(Sr[r]["hist"]["val_model_common"])
        contrib = (E / N.sum(1, keepdims=True))[1:1 + nep, :, 4]          # lead 5
        dev_ = contrib - contrib.mean(0, keepdims=True)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        o_ = np.argsort(-contrib.var(0))
        im = ax[0].imshow(dev_[:, o_[:60]].T * 1e3, aspect="auto", cmap="PuOr", vmin=-np.abs(dev_).max() * 1e3, vmax=np.abs(dev_).max() * 1e3)
        ax[0].set(title="Control run, lead 5: per-day contribution minus its mean over evals (60 most variable days)", xlabel="eval (epoch of 50 steps)", ylabel="val start day (ranked by variance)")
        fig.colorbar(im, ax=ax[0], label="contribution to val MAE x 1e3 (ln CHL)")
        m_all = E[1:1 + nep, :, 4].sum(1) / N[1:1 + nep, :, 4].sum(1)
        ax[1].plot(np.arange(1, nep + 1), m_all, marker="o", color=C[0], label="all 351 days")
        for q, ntop in ((1, 10), (2, 30)):
            keep = np.setdiff1d(np.arange(E.shape[1]), o_[:ntop])
            ax[1].plot(np.arange(1, nep + 1), E[1:1 + nep][:, keep, 4].sum(1) / N[1:1 + nep][:, keep, 4].sum(1), marker="o", color=C[q], label=f"without the {ntop} most variable days")
        ax[1].set(title="Control run: val MAE lead 5 by eval, with / without the most variable days", xlabel="eval (epoch of 50 steps)", ylabel="MAE (ln CHL)")
        ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "diag_src_perday.png", dpi=110); plt.close(fig)


# ============================== tables ==============================
def write_tables(S, T, Sc, Sr):
    L = ["# Tracer diagnostics (analyses/tracer/diag_report.py)", ""]
    if S:
        L += ["## Test 1: loss landscape of a x u_rea (val starts; fixed set unless noted)", "",
              "Minimum a by lead (parabolic vertex) and mean-lead; pixel MAE, fixed set [95 % day-bootstrap CI]", "",
              "| variant | l1 | l2 | l3 | l4 | l5 | mean lead | n starts | n fixed px l1 / l5 |", "|---|---|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            if v not in S:
                continue
            f = S[v]["fixed_pix"]; ci = f["argmin_ci"]
            L.append(f"| {VLAB[v]} | " + " | ".join(f"{fmt(f['argmin'][k], 2)} [{fmt(ci[k][0], 2)}, {fmt(ci[k][1], 2)}]" for k in range(K + 1))
                     + f" | {S[v]['info']['n_starts']} | {int(f['n_a0'][0]):,} / {int(f['n_a0'][4]):,} |")
        L += ["", "Minimum a, band-passed (24-150 km) MAE, fixed set; own-set pixel and band minima; Pearson-r maxima (fixed set, pixel / band)", "",
              "| variant | band fixed l1-l5 | pixel own l1-l5 | band own l1-l5 | r max pixel l1-l5 | r max band l1-l5 |", "|---|---|---|---|---|---|"]
        j = lambda xs: " / ".join(fmt(x, 2) for x in xs[:K])
        for v in VARIANTS:
            if v not in S or "fixed_band" not in S[v]:
                continue
            L.append(f"| {VLAB[v]} | {j(S[v]['fixed_band']['argmin'])} | {j(S[v]['own_pix']['argmin'])} | {j(S[v]['own_band']['argmin'])} | "
                     f"{j(S[v]['fixed_pix']['argmax_r'])} | {j(S[v]['fixed_band']['argmax_r'])} |")
        L += ["", "Flatness: MAE(a = 0.5) - MAE(a = 0.17), fixed set, pixel [95 % CI] and band", "",
              "| variant | pixel l1 | pixel l3 | pixel l5 | band l1 | band l3 | band l5 | min pixel MAE l1 / l5 |", "|---|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            if v not in S:
                continue
            f = S[v]["fixed_pix"]; b = S[v].get("fixed_band")
            cell = lambda k: f"{f['d50_17'][k]:+.4f} [{f['d50_17_ci'][k][0]:+.4f}, {f['d50_17_ci'][k][1]:+.4f}]"
            bc = lambda k: f"{b['d50_17'][k]:+.4f}" if b else "-"
            L.append(f"| {VLAB[v]} | {cell(0)} | {cell(2)} | {cell(4)} | {bc(0)} | {bc(2)} | {bc(4)} | {f['mae_min'][0]:.4f} / {f['mae_min'][4]:.4f} |")
        if "fixed_moving_pix" in S.get("base", {}):
            L += ["", "Base, pixel subsets inside accepted lead-1 spectral-motion patches of the start day (intersected with the fixed set)", "",
                  "| subset | min a pixel l1-l5 | min a band l1-l5 | MAE(0.5)-MAE(0.17) pixel l1 / l3 / l5 | n px l1 / l5 |", "|---|---|---|---|---|"]
            for sub, lab in (("fixed_moving", "moving (|shift| >= 1 px)"), ("fixed_zero", "zero shift"), ("fixed", "all fixed")):
                p = S["base"][f"{sub}_pix"]; bb = S["base"][f"{sub}_band"]
                n = p.get("n", p.get("n_a0"))
                L.append(f"| {lab} | {j(p['argmin'])} | {j(bb['argmin'])} | {p['d50_17'][0]:+.4f} / {p['d50_17'][2]:+.4f} / {p['d50_17'][4]:+.4f} | {int(n[0]):,} / {int(n[4]):,} |")
    if T:
        L += ["", "## Test 2: synthetic recovery (true a = 0.5; val starts, real cloud masks)", "",
              "| case | min a pixel l1-l5 | mean lead | 95 % CI l1 | min a band l1-l5 | trained a, lr 3e-3, 153 steps | trained a, lr 3e-2, 306 steps | anomaly var lead 0 | n px l1 / l5 |", "|---|---|---|---|---|---|---|---|---|"]
        j = lambda xs: " / ".join(fmt(x, 2) for x in xs[:K])
        for c in SYN:
            if c not in T:
                continue
            t = T[c]
            L.append(f"| {SLAB[c]} | {j(t['pix']['argmin'])} | {fmt(t['pix']['argmin'][K], 2)} | [{fmt(t['pix']['argmin_ci'][0][0], 2)}, {fmt(t['pix']['argmin_ci'][0][1], 2)}] | "
                     f"{j(t['band']['argmin'])} | {fmt(t.get('trained_a'), 3)} | {fmt(t.get('trained_a_lr3e-2'), 3)} | {t['anomaly_var_lead0']:.3f} | {t['pix']['n'][0]:,} / {t['pix']['n'][4]:,} |")
    if Sc:
        ref = lambda v, st: fmt(S[v][st]["argmin"][K], 3) if v in S else "-"
        L += ["", "## Test 3a: scalar a training (train starts, batch 16)", "",
              f"Landscape minimum, mean of leads 1-5: train starts (every 3rd) own sets {ref('train_base', 'own_pix')}, fixed set {ref('train_base', 'fixed_pix')}; "
              f"val own sets {ref('base', 'own_pix')}, fixed set {ref('base', 'fixed_pix')}.", "",
              "| lr | a0 | a after epoch 1 | a after epoch 2 | final a | final val MAE mean lead (own) | steps |", "|---|---|---|---|---|---|---|"]
        for lr in sorted(Sc):
            spe = None
            for a0, r in Sc[lr]["runs"].items():
                spe = r["steps_per_epoch"]
                ae = [r["a"][min(spe * e, len(r["a"])) - 1] for e in (1, 2)]
                L.append(f"| {lr:g} | {a0} | {ae[0]:.4f} | {ae[1]:.4f} | {r['a'][-1]:.4f} | {np.mean(r['val_mae'][-1]):.5f} | {len(r['a'])} |")
    gc, fd = RES / "diag_gradcheck.json", RES / "diag_gradcheck_fd.json"
    if gc.exists():
        g = json.load(open(gc))
        L += ["", f"Training objective vs a, all train starts ({g['n_train']}; {g['n_batches']} batches of 16): batch-mean loss, autograd d loss / d a (batch mean), pooled own-set MAE (mean of leads)", "",
              "| a | batch-mean loss | autograd gradient | pooled own MAE |", "|---|---|---|---|"]
        for i, al in enumerate(g["grid"][:len(g["loss"])]):
            L.append(f"| {al:.3f} | {g['loss'][i]:.6f} | {g['grad'][i]:+.5f} | {g['pooled'][i]:.6f} |")
    if fd.exists():
        f = json.load(open(fd))
        L += ["", f"Autograd vs central finite difference (a = {f['a0']}, delta {f['delta']}, first {f['n_batches']} train batches, pixel-pooled per lead)", "",
              "| lead | FD, own set (mask varies with a) | autograd, own set | FD, mask fixed | autograd, mask fixed | px whose validity differs between a -/+ delta | n px |", "|---|---|---|---|---|---|---|"]
        for k in range(K):
            L.append(f"| {k + 1} | {f['fd_own'][k]:+.5f} | {f['autograd_own'][k]:+.5f} | {f['fd_fixed'][k]:+.5f} | {f['autograd_fixed'][k]:+.5f} | {int(f['n_mask_changed'][k]):,} | {int(f['n_own'][k]):,} |")
    if Sr:
        L += ["", "## Test 3b: source runs (12 evals x 50 steps; val common set, every 2nd val start)", "",
              "| run | val range l1 / l3 / l5 (max-min over evals) | mean abs change between evals l1 / l3 / l5 | max abs change l5 | train L1 first / last | top-10-day share of abs change l5 (pixel share) | share of px-days changing with the total, l5 | r(val mean S, val l5) | r(eval-to-eval change l5, control) | final a |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in SRC + ["tr_src"]:
            if r not in Sr:
                continue
            x = Sr[r]
            ad = x["val_absdiff_mean"]
            L.append(f"| {SRCLAB.get(r, r + ' (20 epochs x 153 steps, earlier run)')} | {x['val_range'][0]:.4f} / {x['val_range'][2]:.4f} / {x['val_range'][4]:.4f} | "
                     f"{ad[0]:.4f} / {ad[2]:.4f} / {ad[4]:.4f} | {x['val_absdiff_max'][4]:.4f} | {x['train_first']:.4f} / {x['train_last']:.4f} | "
                     + (f"{x['top10_share_abs_change'][4]:.2f} ({x['top10_share_pixels'][4]:.2f}) | {x['sign_agreement_share'][4]:.2f} | {x['r_valS_vs_val'][4]:+.2f}" if "top10_share_abs_change" in x else "- | - | -")
                     + f" | {fmt(x.get('r_dval_l5_vs_control'), 2)} | {fmt(x['a_last'], 3)} |")
        if "diag_src_ctrl" in Sr and "leave_top10_out" in Sr["diag_src_ctrl"]:
            L += ["", "Control run, leave the 10 most variable days out (by variance of per-day contribution over evals)", "",
                  "| lead | val range all days | without top 10 | SD of eval-to-eval change all | without top 10 | top-10 share of contribution variance |", "|---|---|---|---|---|---|"]
            for k, lo in enumerate(Sr["diag_src_ctrl"]["leave_top10_out"]):
                L.append(f"| {k + 1} | {lo['range_all']:.4f} | {lo['range_without_top10']:.4f} | {lo['sd_diff_all']:.4f} | {lo['sd_diff_without_top10']:.4f} | {lo['top10_var_share']:.2f} |")
    (RES / "diag_tables.md").write_text("\n".join(L) + "\n")


def fig_gradcheck():
    f = RES / "diag_gradcheck.json"
    if not f.exists():
        return
    g = json.load(open(f)); x = np.array(g["grid"][:len(g["loss"])])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(x, g["loss"], marker="o", color=C[0], label="batch-mean training loss")
    ax[0].plot(x, np.array(g["pooled"]) + (g["loss"][0] - g["pooled"][0]), marker="s", color=C[1], label="pooled own-set MAE (shifted to match at a = 0.1)")
    ax[0].set(title="Training objective vs a, all train starts (mean of leads 1-5)", xlabel="a (u_eff = a x u_rea)", ylabel="loss (ln CHL)")
    ax[0].legend(fontsize=8)
    ax[1].plot(x, g["grad"], marker="o", color=C[0], label="autograd d loss / d a (batch mean)")
    ax[1].plot(x, np.gradient(g["loss"], x), marker="s", color=C[3], label="finite difference of the batch-mean loss curve")
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set(title="Gradient of the training objective vs a", xlabel="a", ylabel="d loss / d a")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "diag_gradcheck.png", dpi=110); plt.close(fig)


def main():
    rng = np.random.default_rng(0)
    fig_gradcheck()
    L = load_landscape(); S = landscape_tables(L, rng) if L else {}
    if "base" in L:
        fig_landscape_base(L); fig_landscape_variants(S)
    Sy = load_synth(); T = synth_tables(Sy, rng) if Sy else {}
    if T:
        fig_synth(Sy, T)
    Sc = load_scalar()
    if Sc:
        fig_scalar(Sc, S)
    Sr = load_src(); R = src_tables(Sr) if Sr else {}
    fig_src(Sr)
    write_tables(S, T, Sc, R)
    (RES / "diag_summary.json").write_text(json.dumps(dict(landscape=S, synth=T, scalar={str(k): {a0: dict(a_final=r["a"][-1], val_final=r["val_mae"][-1]) for a0, r in v["runs"].items()} for k, v in Sc.items()},
                                                         src=R), indent=1, default=float))
    print((RES / "diag_tables.md").read_text())


if __name__ == "__main__":
    main()
