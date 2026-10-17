"""UDE stage B summary (Black Sea, learned advection factor): tables + figures from models/ude/{udeA_*,udeB_*}/metrics.json.

Run: python analyses/ude/stageB_summary.py   -> prints markdown tables, writes analyses/ude/figures/stageB_{mae_vs_lead,regions_l1,alpha_epochs}.png
"""
import json, shutil, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from chl_cnn.plotting import OKABE_ITO
M = ROOT / "models" / "ude"; FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
KINDS = ["model", "persistence", "climatology", "transport_only", "transport_alpha1", "g_only"]


def load(run):
    p = M / run / "metrics.json"
    if not p.exists(): return None
    m = json.load(open(p)); c = json.load(open(M / run / "config.json")); h = json.load(open(M / run / "history.json"))
    return dict(run=run, m=m, c=c, h=h)


def alpha_str(r):
    c, m = r["c"], r["m"]; a = m.get("alpha")
    if c.get("adv_scale", "fixed") == "fixed" or a is None:
        return "1 (fixed)" if c.get("transport") == "sl" else "-"
    if isinstance(a, dict):
        return f"{a['mean']:.3f} ({a['p10']:.3f}/{a['p90']:.3f})"
    return f"{a:.3f}"


def f3(x): return f"{x:.4f}"


def main():
    runs = [load(p.name) for p in sorted(M.glob("udeA_*")) + sorted(M.glob("udeB_*")) if "smoke" not in p.name and "smk" not in p.name]
    runs = [r for r in runs if r]
    hdr = "| run | transport | adv-scale (alpha / mean (p10/p90)) | g | epochs | model L1/L2/L3 | persistence | climatology | T-only (learned alpha) | T alpha=1 | g-only | RMSE L1 | grad ratio/corr L1 | min | GPU GB |"
    print(hdr); print("|" + "---|" * hdr.count("|")[:-0] if False else "|" + "---|" * (hdr.count("|") - 1))
    for r in runs:
        v, c, m = r["m"]["val"], r["c"], r["m"]
        cell = lambda k: "/".join(f3(x) for x in v[k]["mae"]) if k in v else "-"
        ne = len(r["h"]["val"]) - 1
        print(f"| {r['run']} | {c['transport']} | {alpha_str(r)} | {c['g']} | {ne} | {cell('model')} | {cell('persistence')} | {cell('climatology')} | {cell('transport_only')} | {cell('transport_alpha1')} | {cell('g_only')} | "
              f"{v['model']['rmse'][0]:.4f} | {v['grad_ratio_l1']:.3f}/{v['grad_corr_l1']:.3f} | {m['minutes']:.0f} | {m['max_mem_gb']:.1f} |")
    print(f"\nval t0: {[r['m']['val_t0'] for r in runs]}  n_obs by lead: {runs[0]['m']['val']['n_obs']}")
    # region table at lead 1
    A = [r for r in runs if r["run"].startswith("udeA_") and r["c"]["g"] != "none"]
    bestA = min(A, key=lambda r: r["m"]["val"]["model"]["mae"][0]) if A else None
    sel = [r for r in runs if r["run"] in ("udeB_mlp_scalar", "udeB_mlp_field")] + ([bestA] if bestA else [])
    if sel and "regions_l1" in sel[0]["m"]["val"]:
        regs = list(sel[0]["m"]["val"]["regions_l1"])
        print("\n| region | " + " | ".join(r["run"] for r in sel) + " | persistence | climatology | n obs |"); print("|" + "---|" * (len(sel) + 4))
        for g in regs:
            rr = sel[0]["m"]["val"]["regions_l1"][g]
            print(f"| {g} | " + " | ".join(f3(r['m']['val']['regions_l1'][g]['model']) for r in sel) + f" | {f3(rr['persistence'])} | {f3(rr['climatology'])} | {rr['n_obs']:,} |")
        fig, ax = plt.subplots(figsize=(10, 4.5)); x = np.arange(len(regs)); w = 0.8 / (len(sel) + 1)
        for i, r in enumerate(sel + [None]):
            vals = [sel[0]["m"]["val"]["regions_l1"][g]["persistence"] if r is None else r["m"]["val"]["regions_l1"][g]["model"] for g in regs]
            ax.bar(x + (i - len(sel) / 2) * w, vals, w, color=OKABE_ITO[i], label="persistence (C_0)" if r is None else r["run"])
        ax.set_xticks(x); ax.set_xticklabels(regs, rotation=20, ha="right", fontsize=8); ax.set_ylabel("val MAE at lead 1 (ln CHL, observed px)")
        ax.set_title("Black Sea UDE: val MAE at lead 1 by region"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "stageB_regions_l1.png", dpi=130)
    # MAE vs lead
    fig, ax = plt.subplots(figsize=(7.5, 5)); leads = [1, 2, 3]
    for i, r in enumerate(runs):
        ax.plot(leads, r["m"]["val"]["model"]["mae"], marker="o", color=OKABE_ITO[i % 8], ls="-" if r["run"].startswith("udeB") else "--", label=r["run"])
    ax.plot(leads, runs[0]["m"]["val"]["persistence"]["mae"], "k:", label="persistence (C_0)"); ax.plot(leads, runs[0]["m"]["val"]["climatology"]["mae"], "k-.", label="DOY climatology")
    ax.set_xlabel("lead (days)"); ax.set_ylabel("val MAE (ln CHL, observed px)"); ax.set_xticks(leads); ax.set_title("Black Sea UDE stage A (dashed) and B (solid): val MAE vs lead"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "stageB_mae_vs_lead.png", dpi=130)
    # alpha per epoch
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, r in enumerate(runs):
        if r["c"].get("adv_scale", "fixed") == "fixed": continue
        ps = r["h"]["params"]; ep = np.arange(len(ps))
        if "alpha" in ps[0]: ax.plot(ep, [p["alpha"] for p in ps], "o-", color=OKABE_ITO[i % 8], label=f"{r['run']} alpha")
        elif "alpha_mean" in ps[-1]:
            ax.plot(ep, [p.get("alpha_mean", np.nan) for p in ps], "o-", color=OKABE_ITO[i % 8], label=f"{r['run']} alpha mean")
            ax.fill_between(ep, [p.get("alpha_p10", np.nan) for p in ps], [p.get("alpha_p90", np.nan) for p in ps], color=OKABE_ITO[i % 8], alpha=0.2, label="p10-p90 over ocean px-days")
    ax.set_xlabel("epoch (0 = init)"); ax.set_ylabel("advection factor alpha"); ax.set_title("Learned advection factor alpha per epoch (val evaluation)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "stageB_alpha_epochs.png", dpi=130)
    for r in runs:
        figs = sorted((M / r["run"] / "figures").glob("alpha_map_epoch_*.png"))
        if figs: shutil.copy(figs[-1], FIG / f"stageB_alpha_map_{r['run']}.png")


if __name__ == "__main__":
    main()
