"""Tables and figures for the tracer seed check (Task A) and step 2 (pointwise source term; train_tracer.py --source).
Val years only (2017, 2021; every 2nd start day); test years not scored.

Reads models/tracer/<run>/{metrics,history,config}.json, factor_maps.npz, source_maps.npz, source_pd.npz.
  metrics["best"]  evaluation at the best epoch (mean-lead val MAE on the model's own pixel set), common set without ref.
  metrics["final"] (--source runs) re-evaluation of the best model with the step-1 run tr_rea_geo_wind (seed 0) scored
                   as predictor "ref" on the same pixels (its validity is part of the common set).
Writes analyses/tracer/results/step2_tables.md and analyses/tracer/figures/step2_*.png.
Run from the repo root: python -m analyses.tracer.step2_report
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from chl_cnn.plotting import OKABE_ITO, ERR_CMAP

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True, parents=True); FIG.mkdir(exist_ok=True, parents=True)
STEP1 = {"step-1 seed 0": "tr_rea_geo_wind", "step-1 seed 1": "tr_rea_geo_wind_s1", "step-1 seed 2": "tr_rea_geo_wind_s2"}
STEP2 = {"source seed 0": "tr_src", "source seed 1": "tr_src_s1", "source no ln C": "tr_src_noC"}
REG = {1: "NW shelf", 2: "W open sea", 3: "E open sea", 4: "Anatolian shelf", 5: "Crimea/Kerch/Caucasus", 6: "Azov+Kerch", 7: "Marmara/Bosporus"}
SEAS = ["DJF", "MAM", "JJA", "SON"]
K = 5
MIN_N = 5
f4 = lambda x: f"{x:.4f}"


def load(runs):
    out = {}
    for lab, run in runs.items():
        p = ROOT / "models/tracer" / run
        if not (p / "metrics.json").exists() or "best_epoch" not in json.load(open(p / "metrics.json")):
            print(f"skip {run}: not finished"); continue
        r = dict(run=run, m=json.load(open(p / "metrics.json")), h=json.load(open(p / "history.json")), c=json.load(open(p / "config.json")),
                 f=np.load(p / "factor_maps.npz"))
        for nm in ("source_maps", "source_pd"):
            if (p / f"{nm}.npz").exists():
                r[nm] = np.load(p / f"{nm}.npz")
        out[lab] = r
    return out


def cmean(th):
    """Circular mean of angles in degrees."""
    return float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(th))))))


def mae(ev, p, k, s="common"):
    return ev[s][p][str(k)]["mae"]


# ------------------------------------------------------------------ Task A
def seed_tables(R1):
    L = ["## Task A: step-1 seed check (rea+geo+wind, 20 epochs, seeds 0 / 1 / 2)", "",
         "Each run's best epoch, its own common set (target observed; model, persistence, x1.0, x0.25, clim valid). MAE ln CHL.", "",
         "| lead | " + " | ".join(f"model {l[-6:]}" for l in R1) + " | model spread (max-min) | " + " | ".join(f"model - x0.25 {l[-6:]}" for l in R1)
         + " | (model - x0.25) spread | x0.25 range over runs | n (seed 0) |", "|---" * (2 * len(R1) + 5) + "|"]
    sp = {}
    for k in range(1, K + 1):
        m = [mae(r["m"]["best"], "model", k) for r in R1.values()]
        g = [mae(r["m"]["best"], "model", k) - mae(r["m"]["best"], "adv_x0.25", k) for r in R1.values()]
        b = [mae(r["m"]["best"], "adv_x0.25", k) for r in R1.values()]
        sp[k] = dict(model_spread=max(m) - min(m), gain_spread=max(g) - min(g), gain=g, model=m)
        n0 = list(R1.values())[0]["m"]["best"]["common"]["model"][str(k)]["n"]
        L.append(f"| {k} | " + " | ".join(f4(x) for x in m) + f" | {max(m) - min(m):.4f} | " + " | ".join(f"{x:+.4f}" for x in g)
                 + f" | {max(g) - min(g):.4f} | {min(b):.4f}-{max(b):.4f} | {n0:,} |")
    L += ["", "| run | best epoch | a mean / sd | b mean / sd | wind c mean (m/s per Pa) | wind theta circular mean (deg) |", "|---|---|---|---|---|---|"]
    for lab, r in R1.items():
        fs = r["m"]["best"]["factors_day_mean_over_ocean"]; w = r["m"]["wind"]
        L.append(f"| {lab} | {r['m']['best_epoch']} | {fs['a']['mean']:.3f} / {fs['a']['sd']:.3f} | {fs['b']['mean']:.3f} / {fs['b']['sd']:.3f} | "
                 f"{np.mean(w['c_m_s_per_Pa']):.3f} | {cmean(w['theta_deg']):.0f} |")
    return L, sp


# ------------------------------------------------------------------ step 2
def step2_tables(R1, R2, sp):
    L = ["", "## Step 2: pointwise source", ""]
    for lab, r in R2.items():
        ev = r["m"]["final"]; c = r["c"]
        L += [f"### {lab} ({r['run']}; best epoch {r['m']['best_epoch']} of {len(r['h']['epoch'])}, {r['m']['steps']} steps, {r['m']['wall_s'] / 60:.0f} min, "
              f"peak {r['m']['peak_gb']:.1f} GB; inputs {len(c['data']['source']['channels'])})", "",
              "Final re-evaluation of the best model; common set includes the step-1 seed-0 run ('ref') on the same pixels. MAE ln CHL.", "",
              "| lead | source model | step-1 (ref) | x0.25 | persistence | x1.0 | DOY clim | source - ref | source - x0.25 | source - persistence | n |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in range(1, K + 1):
            v = {p: mae(ev, p, k) for p in ("model", "ref", "adv_x0.25", "persistence", "adv_x1.0", "doy_clim")}
            L.append(f"| {k} | " + " | ".join(f4(x) for x in v.values()) + f" | {v['model'] - v['ref']:+.4f} | {v['model'] - v['adv_x0.25']:+.4f} | "
                     f"{v['model'] - v['persistence']:+.4f} | {ev['common']['model'][str(k)]['n']:,} |")
        L += ["", "Own sets, MAE (n): " + "; ".join(f"l{k} model {mae(ev, 'model', k, 'own'):.4f} ({ev['own']['model'][str(k)]['n'] / 1e6:.2f} M), ref {mae(ev, 'ref', k, 'own'):.4f} ({ev['own']['ref'][str(k)]['n'] / 1e6:.2f} M)" for k in (1, 5)), ""]
    # compact gain table
    L += ["### Source model minus step-1 (ref, same pixels) by lead, with the Task A step-1 seed spread", "",
          "| lead | " + " | ".join(R2) + " | step-1 seed spread of model MAE (max-min, own common sets) | step-1 seed spread of model - x0.25 |", "|---" * (len(R2) + 3) + "|"]
    for k in range(1, K + 1):
        L.append(f"| {k} | " + " | ".join(f"{mae(r['m']['final'], 'model', k) - mae(r['m']['final'], 'ref', k):+.4f}" for r in R2.values())
                 + (f" | {sp[k]['model_spread']:.4f} | {sp[k]['gain_spread']:.4f} |" if sp else " | n/a | n/a |"))
    # velocity factors
    L += ["", "### Learned velocity factors with and without the source (best epoch; ocean mean per val day: mean / sd over days)", "",
          "| run | best epoch | a | b | wind c mean (m/s per Pa) | wind theta circular mean (deg) | model speed (px/day) | a, epochs 11-20: mean (min-max) | b, epochs 11-20: mean (min-max) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for lab, r in {**R1, **R2}.items():
        fs = r["m"]["best"]["factors_day_mean_over_ocean"]; w = r["m"]["wind"]
        late = {k_: np.array([e[k_]["mean"] for e in r["h"]["factors"][10:]]) for k_ in ("a", "b")}
        L.append(f"| {lab} | {r['m']['best_epoch']} | {fs['a']['mean']:.3f} / {fs['a']['sd']:.3f} | {fs['b']['mean']:.3f} / {fs['b']['sd']:.3f} | {np.mean(w['c_m_s_per_Pa']):.3f} | "
                 f"{cmean(w['theta_deg']):.0f} | {fs['speed_model']['mean']:.3f} | "
                 + " | ".join(f"{late[k_].mean():.3f} ({late[k_].min():.3f}-{late[k_].max():.3f})" for k_ in ("a", "b")) + " |")
    allr = {**R1, **R2}
    labs = list(allr)
    L += ["", "Pattern correlation of the factor-a maps (12 mid-month maps x ocean px, Pearson) between runs:", "",
          "| | " + " | ".join(labs) + " |", "|---" * (len(labs) + 1) + "|"]
    A = {l: allr[l]["f"]["a"][:, allr[l]["f"]["ocean"]].ravel() for l in labs}
    for l1 in labs:
        L.append(f"| {l1} | " + " | ".join(f"{np.corrcoef(A[l1], A[l2])[0, 1]:.2f}" for l2 in labs) + " |")
    # source by season / region
    for lab, r in R2.items():
        if "source_maps" not in r:
            continue
        sm = r["source_maps"]; mean, n, reg = sm["mean"], sm["n"].astype(float), sm["region"]
        s = np.nan_to_num(mean) * n
        L += ["", f"### Lead-1 source S (ln CHL / day; prediction minus departure value over 1 day), {lab}: mean by region and season (n pixel-days)", "",
              "| region | " + " | ".join(SEAS) + " | all |", "|---" * 6 + "|"]
        for rk, rn in REG.items():
            cells = []
            for q in list(range(4)) + [None]:
                ss = s[q][reg == rk].sum() if q is not None else s[:, reg == rk].sum()
                nn = n[q][reg == rk].sum() if q is not None else n[:, reg == rk].sum()
                cells.append(f"(n < {MIN_N})" if nn < MIN_N else f"{ss / nn:+.4f} ({int(nn):,})")
            L.append(f"| {rn} | " + " | ".join(cells) + " |")
        cells = [f"{s[q].sum() / n[q].sum():+.4f} ({int(n[q].sum()):,})" for q in range(4)] + [f"{s.sum() / n.sum():+.4f} ({int(n.sum()):,})"]
        L.append("| all | " + " | ".join(cells) + " |")
        if "source_pd" in r:
            a_ = np.abs(r["source_pd"]["s"]); gm = r["c"]["g_max"]
            L.append(f"\n|S| at sub-step samples (n {len(a_):,}): median {np.median(a_):.4f}, p90 {np.quantile(a_, .9):.4f}, p99 {np.quantile(a_, .99):.4f}, "
                     f"max {a_.max():.4f}; fraction above 0.9 g_max ({0.9 * gm:.2f}): {(a_ > 0.9 * gm).mean():.2e}")
    # PD ranges
    L += ["", "### S by input decile (mean S over sub-step samples, val lead 1, every ~50th valid pixel): range of decile means (max - min), ln CHL / day", ""]
    have = [l for l, r in R2.items() if "source_pd" in r]
    if have:
        chans = list(R2[have[0]]["source_pd"]["channels"])
        L += ["| input | " + " | ".join(have) + " |", "|---" * (len(have) + 1) + "|"]
        pdr = {l: pd_curves(R2[l]["source_pd"]) for l in have}
        for ch in chans:
            L.append(f"| {ch} | " + " | ".join((f"{np.nanmax(pdr[l][ch][1]) - np.nanmin(pdr[l][ch][1]):.4f}" if ch in pdr[l] else "-") for l in have) + " |")
        L.append("\nso_ok is the bilinear-sampled flag: two bins (< 0.999, >= 0.999); n in the low bin: " + "; ".join(f"{l} {pdr[l]['so_ok'][2][0]:,}" for l in have))
        L.append("\nn samples: " + "; ".join(f"{l} {len(R2[l]['source_pd']['s']):,}" for l in have))
    # epochs
    L += ["", "### Per-epoch history (val: 351 start days; S = basin-mean lead-1 source on val)", ""]
    for lab, r in R2.items():
        h = r["h"]
        L += [f"{lab}:", "", "| epoch | train L1 (mean lead) | val MAE mean lead (own) | val lead-5 MAE (common) | val mean S (ln CHL / day) | a | b |", "|---|---|---|---|---|---|---|"]
        for i, e in enumerate(h["epoch"]):
            L.append(f"| {e} | {h['train_l1'][i]:.4f} | {h['val_mean_lead'][i]:.4f} | {h['val_model_common'][i][4]:.4f} | {h['source'][i]['mean']:+.4f} | "
                     f"{h['factors'][i]['a']['mean']:.3f} | {h['factors'][i]['b']['mean']:.3f} |")
        sm = np.array([x["mean"] for x in h["source"]]); l5 = np.array([x[4] for x in h["val_model_common"]])
        L += ["", f"Pearson r over epochs, val mean S vs val lead-5 MAE: {np.corrcoef(sm, l5)[0, 1]:.2f} (n {len(sm)} epochs)", ""]
    # gradients
    L += ["", "### Gradient ratio / corr at lead 1, common-set pixel pairs (final evaluation; lag in px; n pairs)", ""]
    for lab, r in R2.items():
        ev = r["m"]["final"]
        L += [f"{lab}:", "", "| predictor | lag 1 | lag 4 | lag 16 |", "|---|---|---|---|"]
        for p in ("model", "ref", "adv_x0.25", "persistence", "adv_x1.0", "doy_clim"):
            g = ev["grad"][p]["1"]
            L.append(f"| {p} | " + " | ".join(f"{g[str(l)]['ratio']:.3f} / {g[str(l)]['corr']:.3f} ({g[str(l)]['n_pairs'] / 1e6:.2f} M)" for l in (1, 4, 16)) + " |")
        L.append("")
    # regions
    L += ["### Lead-1 MAE by region, common set (final evaluation)", ""]
    for lab, r in R2.items():
        ev = r["m"]["final"]["region_lead1_common"]
        L += [f"{lab}:", "", "| region | source model | step-1 (ref) | x0.25 | persistence | source - ref | n |", "|---|---|---|---|---|---|---|"]
        for rk, rn in REG.items():
            n = ev["model"][str(rk)]["n"]
            if n < MIN_N:
                L.append(f"| {rn} | (n < {MIN_N}) |||||{n} |"); continue
            v = [ev[p][str(rk)]["mae"] for p in ("model", "ref", "adv_x0.25", "persistence")]
            L.append(f"| {rn} | " + " | ".join(f4(x) for x in v) + f" | {v[0] - v[1]:+.4f} | {n:,} |")
        L.append("")
    return L


def pd_curves(z, nbin=10):
    """{channel: (decile midpoints in physical units, mean S per decile, n per decile)}."""
    x, s, ch = z["x"], z["s"], list(z["channels"])
    out = {}
    for i, c in enumerate(ch):
        v = x[:, i].astype("float64")
        if i < 7:
            v = v * z["z_sd"][i] + z["z_mean"][i]
        elif c.startswith("c ="):
            v = v * float(z["c_sd"]) + float(z["c_mean"])
        if c == "so_ok":                                                    # sampled flag: 1 vs < 1 (partly invalid neighbourhood)
            v = (v >= 0.9999).astype("float64")
        qs = np.unique(np.quantile(v, np.linspace(0, 1, nbin + 1)))
        if c == "so_ok" or len(qs) < 3:                                     # flag (bilinear-sampled, mostly 1): two bins
            b = (v >= 0.999).astype(int)
            out[c] = (np.array([v[b == 0].mean() if (b == 0).any() else np.nan, 1.0]), np.array([s[b == j].mean() if (b == j).any() else np.nan for j in (0, 1)]),
                      np.array([(b == j).sum() for j in (0, 1)])); continue
        b = np.clip(np.searchsorted(qs, v, side="right") - 1, 0, len(qs) - 2)
        mid = np.array([np.median(v[b == j]) for j in range(len(qs) - 1)])
        out[c] = (mid, np.array([s[b == j].mean() for j in range(len(qs) - 1)]), np.array([(b == j).sum() for j in range(len(qs) - 1)]))
    return out


# ------------------------------------------------------------------ figures
def figures(R1, R2, sp):
    C = OKABE_ITO; ks = np.arange(1, K + 1)
    # seed check
    if R1:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for j, (lab, r) in enumerate(R1.items()):
            b = r["m"]["best"]
            ax[0].plot(ks, [mae(b, "model", k) for k in ks], "o-", color=C[j], label=lab)
            ax[1].plot(ks, [mae(b, "model", k) - mae(b, "adv_x0.25", k) for k in ks], "o-", color=C[j], label=lab)
        b0 = list(R1.values())[0]["m"]["best"]
        ax[0].plot(ks, [mae(b0, "adv_x0.25", k) for k in ks], "s:", color=C[7], label="x0.25 (seed-0 set)")
        ax[0].set(title="Step-1 val MAE by lead, three seeds (own common sets)", xlabel="lead (days)", ylabel="MAE (ln CHL)", xticks=ks)
        ax[1].axhline(0, color="k", lw=.6)
        ax[1].set(title="Step-1 model minus x0.25 MAE by lead, three seeds", xlabel="lead (days)", ylabel="ΔMAE (ln CHL)", xticks=ks)
        for a_ in ax: a_.legend(fontsize=8); a_.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(FIG / "step2_seed_check.png", dpi=130); plt.close(fig)
    if not R2:
        return
    # MAE vs lead (final eval, first source run's common set) + gain vs ref
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ev = list(R2.values())[0]["m"]["final"]
    for j, (p, lab) in enumerate([("ref", "step-1 (seed 0)"), ("adv_x0.25", "x0.25"), ("persistence", "persistence"), ("adv_x1.0", "x1.0")]):
        ax[0].plot(ks, [mae(ev, p, k) for k in ks], "s--", color=C[4 + j if 4 + j < 8 else 7], label=lab)
    for j, (lab, r) in enumerate(R2.items()):
        ax[0].plot(ks, [mae(r["m"]["final"], "model", k) for k in ks], "o-", color=C[j], label=lab)
        ax[1].plot(ks, [mae(r["m"]["final"], "model", k) - mae(r["m"]["final"], "ref", k) for k in ks], "o-", color=C[j], label=lab)
    ax[0].set(title=f"Val MAE by lead, common set of {list(R2.values())[0]['run']} (incl. step-1)", xlabel="lead (days)", ylabel="MAE (ln CHL)", xticks=ks)
    if sp:
        sv = np.array([sp[k]["model_spread"] for k in ks])
        ax[1].fill_between(ks, -sv / 2, sv / 2, color="0.8", label="± half step-1 seed spread (max-min)")
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].set(title="Source model minus step-1 (same pixels), val", xlabel="lead (days)", ylabel="ΔMAE (ln CHL)", xticks=ks)
    for a_ in ax: a_.legend(fontsize=8); a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "step2_mae_vs_lead.png", dpi=130); plt.close(fig)
    # velocity factors seasonal, all runs
    allr = {**R1, **R2}
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    for j, (lab, r) in enumerate(allr.items()):
        fm = r["f"]; oc = fm["ocean"]; ls = "--" if lab.startswith("step-1") else "-"
        ax[0].plot(fm["doy"], [fm["a"][m][oc].mean() for m in range(12)], ls, marker="o", ms=3, color=C[j], label=lab)
        ax[1].plot(fm["doy"], [fm["b"][m][oc].mean() for m in range(12)], ls, marker="o", ms=3, color=C[j], label=lab)
        w = r["m"]["wind"]
        ax[2].plot(w["doy"], w["c_m_s_per_Pa"], ls, marker="o", ms=3, color=C[j], label=lab)
        ax[3].plot(w["doy"], (np.asarray(w["theta_deg"]) + 270) % 360 - 270, ls, marker="o", ms=3, color=C[j], label=lab)
    ax[0].set(title="Ocean-mean factor a (reanalysis current) by day of year", xlabel="day of year", ylabel="a (dimensionless)")
    ax[1].set(title="Ocean-mean factor b (geostrophic) by day of year", xlabel="day of year", ylabel="b (dimensionless)")
    ax[2].set(title="Wind coefficient c by day of year", xlabel="day of year", ylabel="c (m/s per Pa)")
    ax[3].set(title="Wind turning angle θ by day of year", xlabel="day of year", ylabel="θ (deg, ccw from τ; wrapped to [-270, 90))")
    for a_ in ax: a_.legend(fontsize=7); a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "step2_velocity_factors.png", dpi=120); plt.close(fig)
    # source maps
    have = [(l, r) for l, r in R2.items() if "source_maps" in r]
    if have:
        lim = max(np.nanpercentile(np.abs(r["source_maps"]["mean"]), 99) for _, r in have)
        fig, ax = plt.subplots(len(have), 4, figsize=(18, 3.3 * len(have)), squeeze=False)
        for i, (lab, r) in enumerate(have):
            sm = r["source_maps"]; lat, lon = sm["lat"], sm["lon"]
            for q in range(4):
                x = np.where(sm["ocean"] & (sm["n"][q] >= MIN_N), sm["mean"][q], np.nan)
                im = ax[i, q].imshow(x, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap=ERR_CMAP, vmin=-lim, vmax=lim, aspect="auto")
                ax[i, q].set(title=f"Mean lead-1 source S, {SEAS[q]}, {lab}", xlabel="lon (°E)", ylabel="lat (°N)")
                fig.colorbar(im, ax=ax[i, q], label="S (ln CHL / day)", shrink=.85)
        fig.suptitle(f"Time-mean source along lead-1 trajectories by season, val days (pixels with n ≥ {MIN_N} days)", fontsize=11)
        fig.tight_layout(); fig.savefig(FIG / "step2_source_maps.png", dpi=110); plt.close(fig)
    # PD
    havep = [(l, r) for l, r in R2.items() if "source_pd" in r]
    if havep:
        chans = list(havep[0][1]["source_pd"]["channels"])
        nc = 4; nr = int(np.ceil(len(chans) / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.3 * nr), squeeze=False)
        for j, (lab, r) in enumerate(havep):
            cur = pd_curves(r["source_pd"])
            for i, ch in enumerate(chans):
                if ch in cur:
                    a_ = ax.flat[i]; xm, ym, _ = cur[ch]
                    a_.plot(xm, ym, "o-", ms=3, color=C[j], label=lab)
        short = {"sst32": ("SST", "°C"), "rsntds": ("net shortwave rsntds", "W m⁻²"), "log10 mlotst": ("mixed-layer depth", "log10 m"),
                 "dilution max(dMLD,0)/MLD per day": ("dilution max(ΔMLD,0)/MLD", "1/day"), "|tau|": ("wind-stress magnitude |τ|", "Pa"),
                 "log10(1 + river influence m3/s)": ("river influence", "log10(1 + m³/s), L = 50 km"), "so": ("salinity so", "psu"),
                 "so_ok": ("physics-valid flag so_ok", "sampled fraction"), "log10 depth (static)": ("log10 depth", "z-score"),
                 "log10(1 + distance to coast) (static)": ("log10(1 + distance to coast)", "z-score"), "doy sin": ("day-of-year sin", "sin"),
                 "doy cos": ("day-of-year cos", "cos"), "c = carried ln C (z)": ("carried ln C", "ln CHL")}
        for i, ch in enumerate(chans):
            nm, un = short.get(ch, (ch, ""))
            a_ = ax.flat[i]; a_.axhline(0, color="k", lw=.5)
            a_.set(title=f"Mean S by decile of {nm}", xlabel=f"{nm} ({un})", ylabel="mean S (ln CHL / day)"); a_.grid(alpha=.3)
        ax.flat[0].legend(fontsize=7)
        for i in range(len(chans), nr * nc):
            ax.flat[i].axis("off")
        fig.suptitle("Learned source S vs each input: mean S per input decile (val lead-1 sub-step samples)", fontsize=11)
        fig.tight_layout(); fig.savefig(FIG / "step2_source_pd.png", dpi=110); plt.close(fig)
    # training curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for j, (lab, r) in enumerate({**{k: v for k, v in R1.items() if k.endswith("0")}, **R2}.items()):
        h = r["h"]
        ax[0].plot(h["epoch"], h["val_mean_lead"], "o-", ms=3, color=C[j], label=f"val ({lab})")
        ax[0].plot(h["epoch"], h["train_l1"], "x--", ms=3, color=C[j], label=f"train ({lab})")
        if h.get("source") and h["source"][0]:
            ax[1].plot(h["epoch"], [s["mean"] for s in h["source"]], "o-", ms=3, color=C[j], label=lab)
    ax[0].set(title="Mean-lead MAE by epoch (train L1; val own set)", xlabel="epoch", ylabel="MAE (ln CHL)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].set(title="Basin-mean lead-1 source S by epoch (val)", xlabel="epoch", ylabel="mean S (ln CHL / day)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "step2_training_curves.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    R1, R2 = load(STEP1), load(STEP2)
    L, sp = seed_tables(R1) if len(R1) == 3 else ([], {})
    if R2:
        L += step2_tables(R1, R2, sp)
    (RES / "step2_tables.md").write_text("\n".join(L))
    print("\n".join(L))
    figures(R1, R2, sp)
