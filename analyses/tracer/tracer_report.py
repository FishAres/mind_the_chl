"""Tables and figures for the tracer step-1 runs (chl_cnn/train_tracer.py; velocity terms only), val years only.

Reads models/tracer/<run>/{metrics,history,config}.json and factor_maps.npz. metrics["best"] = evaluation at the
best epoch (mean-lead val MAE on the model's own pixel set). Writes analyses/tracer/results/step1_tables.md and
figures in analyses/tracer/figures/.
Run from the repo root: python -m analyses.tracer.tracer_report
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from chl_cnn.plotting import OKABE_ITO

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True, parents=True); FIG.mkdir(exist_ok=True, parents=True)
RUNS = {"rea": "tr_rea", "rea+geo": "tr_rea_geo", "rea+geo+wind": "tr_rea_geo_wind"}
PRED = ["model", "persistence", "adv_x1.0", "adv_x0.25", "doy_clim"]
REG = {1: "NW shelf", 2: "W open sea", 3: "E open sea", 4: "Anatolian shelf", 5: "Crimea/Kerch/Caucasus", 6: "Azov+Kerch", 7: "Marmara/Bosporus"}
MIN_N = 5


def load():
    out = {}
    for lab, run in RUNS.items():
        p = ROOT / "models/tracer" / run
        if (p / "metrics.json").exists():
            out[lab] = dict(m=json.load(open(p / "metrics.json")), h=json.load(open(p / "history.json")),
                            c=json.load(open(p / "config.json")), f=np.load(p / "factor_maps.npz"))
    return out


def tables(R):
    L = []
    K = 5
    f = lambda x: f"{x:.4f}"
    for lab, r in R.items():
        b = r["m"]["best"]
        L += [f"### {lab} ({RUNS[lab]}, best epoch {r['m']['best_epoch']} of {len(r['h']['epoch'])}, {r['m']['steps']} steps, {r['m']['wall_s'] / 60:.1f} min)", "",
              "Common set (target observed, every predictor valid). MAE ln CHL; n = pixels.", "",
              "| lead | model | persistence | adv x1.0 | adv x0.25 | DOY clim | model - pers | model - x0.25 | n |", "|---|---|---|---|---|---|---|---|---|"]
        for k in range(1, K + 1):
            c = {p: b["common"][p][str(k)] for p in PRED}
            L.append(f"| {k} | " + " | ".join(f(c[p]["mae"]) for p in PRED) + f" | {c['model']['mae'] - c['persistence']['mae']:+.4f} | {c['model']['mae'] - c['adv_x0.25']['mae']:+.4f} | {c['model']['n']:,} |")
        L += ["", "Each predictor on its own set (target observed and that predictor valid): MAE (n).", "",
              "| lead | " + " | ".join(PRED) + " |", "|---" * (len(PRED) + 1) + "|"]
        for k in range(1, K + 1):
            L.append(f"| {k} | " + " | ".join(f"{f(b['own'][p][str(k)]['mae'])} ({b['own'][p][str(k)]['n'] / 1e6:.2f} M)" for p in PRED) + " |")
        L += ["", "Gradient ratio / corr on common-set pixel pairs, lead 1 (lag in px; n pairs):", "",
              "| predictor | lag 1 | lag 4 | lag 16 |", "|---|---|---|---|"]
        for p in PRED:
            g = b["grad"][p]["1"]
            L.append(f"| {p} | " + " | ".join(f"{g[str(l)]['ratio']:.3f} / {g[str(l)]['corr']:.3f} ({g[str(l)]['n_pairs'] / 1e6:.2f} M)" for l in (1, 4, 16)) + " |")
        L += ["", "Lead-1 MAE by region, common set:", "", "| region | model | persistence | adv x1.0 | adv x0.25 | DOY clim | n |", "|---|---|---|---|---|---|---|"]
        for rk, rn in REG.items():
            n = b["region_lead1_common"]["model"][str(rk)]["n"]
            if n < MIN_N:
                L.append(f"| {rn} | (n < {MIN_N}) |||||{n} |"); continue
            L.append(f"| {rn} | " + " | ".join(f(b["region_lead1_common"][p][str(rk)]["mae"]) for p in PRED) + f" | {n:,} |")
        fs = b["factors_day_mean_over_ocean"]
        L += ["", "Ocean-mean factor per val day (mean, sd over days, p10, p90): " + "; ".join(f"{k_} {v['mean']:.3f} / {v['sd']:.3f} / {v['p10']:.3f} / {v['p90']:.3f}" for k_, v in fs.items())]
        if r["m"].get("wind"):
            w = r["m"]["wind"]
            L.append(f"Wind coefficient by doy {w['doy']}: c (m/s per Pa) {[round(x, 3) for x in w['c_m_s_per_Pa']]}; theta (deg, ccw) {[round(x) for x in w['theta_deg']]}")
        L.append("")
    # gating
    L += ["### Gating (val, common set of each run): best model minus baseline, leads 1-4", "", "| run | vs persistence (l1..l4) | vs adv x0.25 (l1..l4) |", "|---|---|---|"]
    for lab, r in R.items():
        c = r["m"]["best"]["common"]
        dp = [c["model"][str(k)]["mae"] - c["persistence"][str(k)]["mae"] for k in range(1, 5)]
        da = [c["model"][str(k)]["mae"] - c["adv_x0.25"][str(k)]["mae"] for k in range(1, 5)]
        L.append(f"| {lab} | {' '.join(f'{x:+.4f}' for x in dp)} | {' '.join(f'{x:+.4f}' for x in da)} |")
    (RES / "step1_tables.md").write_text("\n".join(L))
    print("\n".join(L))


def figures(R):
    C = OKABE_ITO
    K = np.arange(1, 6)
    # 1 MAE vs lead, common set, one panel per run
    fig, ax = plt.subplots(1, len(R), figsize=(5 * len(R), 4), squeeze=False)
    for a_, (lab, r) in zip(ax[0], R.items()):
        b = r["m"]["best"]["common"]
        for j, p in enumerate(PRED[:4]):
            a_.plot(K, [b[p][str(k)]["mae"] for k in K], "o-", color=C[j], label=p)
        a_.set(title=f"Val MAE by lead, common pixel set ({lab})", xlabel="lead (days)", ylabel="MAE (ln CHL)", xticks=K)
        a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step1_mae_vs_lead.png", dpi=130); plt.close(fig)
    # 2 model - baseline by lead
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, base in enumerate(["persistence", "adv_x0.25"]):
        for j, (lab, r) in enumerate(R.items()):
            b = r["m"]["best"]["common"]
            ax[i].plot(K, [b["model"][str(k)]["mae"] - b[base][str(k)]["mae"] for k in K], "o-", color=C[j], label=lab)
        ax[i].axhline(0, color="k", lw=0.6)
        ax[i].set(title=f"Model minus {base} MAE, val common set", xlabel="lead (days)", ylabel="ΔMAE (ln CHL)", xticks=K); ax[i].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step1_gain_vs_lead.png", dpi=130); plt.close(fig)
    # 3 factor maps: a (and b) for Jan / Apr / Jul / Oct, last run that has them
    for nm in ("a", "b"):
        have = [(lab, r) for lab, r in R.items() if nm in r["f"].files]
        if not have:
            continue
        fig, ax = plt.subplots(len(have), 4, figsize=(18, 3.2 * len(have)), squeeze=False)
        for i, (lab, r) in enumerate(have):
            fm = r["f"]; lat, lon, oc = fm["lat"], fm["lon"], fm["ocean"]
            for j, mo in enumerate([0, 3, 6, 9]):
                x = np.where(oc, fm[nm][mo], np.nan)
                im = ax[i, j].imshow(x, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap="cividis",
                                     vmin=-0.5 if nm == "b" else 0, vmax=1.5 if nm == "b" else 0.5, aspect="auto")
                ax[i, j].set(title=f"Factor {nm} on {'u_rea' if nm == 'a' else 'u_geo'}, doy {int(fm['doy'][mo])} ({lab})", xlabel="lon (°E)", ylabel="lat (°N)")
                fig.colorbar(im, ax=ax[i, j], label=f"{nm} (dimensionless)", shrink=.85)
        fig.tight_layout(); fig.savefig(FIG / f"step1_factor_map_{nm}.png", dpi=110); plt.close(fig)
    # 4 seasonal curves of ocean-mean a, b (factor maps) and wind c / theta
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for j, (lab, r) in enumerate(R.items()):
        fm = r["f"]; oc = fm["ocean"]
        for nm, ls in (("a", "-"), ("b", "--")):
            if nm in fm.files:
                ax[0].plot(fm["doy"], [fm[nm][m][oc].mean() for m in range(12)], ls, marker="o", color=C[j], label=f"{nm} ({lab})")
        if r["m"].get("wind"):
            w = r["m"]["wind"]
            ax[1].plot(w["doy"], w["c_m_s_per_Pa"], "o-", color=C[j], label=lab)
            ax[2].plot(w["doy"], w["theta_deg"], "o-", color=C[j], label=lab)
    ax[0].set(title="Ocean-mean learned factors by day of year", xlabel="day of year", ylabel="factor (dimensionless)"); ax[0].legend(fontsize=8)
    ax[1].set(title="Wind coefficient c by day of year", xlabel="day of year", ylabel="c (m/s per Pa)"); ax[1].legend(fontsize=8)
    ax[2].set(title="Wind rotation angle θ by day of year", xlabel="day of year", ylabel="θ (deg, counter-clockwise from τ)"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step1_seasonal_factors.png", dpi=130); plt.close(fig)
    # 5 training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, (lab, r) in enumerate(R.items()):
        h = r["h"]
        ax[0].plot(h["epoch"], h["val_mean_lead"], "o-", color=C[j], label=f"val, own set ({lab})")
        ax[0].plot(h["epoch"], h["train_l1"], "x--", color=C[j], label=f"train L1 ({lab})")
        ax[1].plot(h["epoch"], [np.mean(x) for x in h["val_model_common"]], "o-", color=C[j], label=lab)
    ax[0].set(title="Mean-lead MAE by epoch (train, val own set)", xlabel="epoch", ylabel="MAE (ln CHL)"); ax[0].legend(fontsize=7)
    ax[1].set(title="Mean-lead val MAE by epoch, common set", xlabel="epoch", ylabel="MAE (ln CHL)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step1_training_curves.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    R = load()
    tables(R); figures(R)
