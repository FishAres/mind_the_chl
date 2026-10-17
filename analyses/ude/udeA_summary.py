"""UDE stage A (Black Sea, transport off; models/ude/udeA_*): markdown tables of the final val metrics per lead (model /
persistence / climatology / g-only), the region table at lead 1 for one run, and two figures: val MAE per lead per epoch for
every run, and the final MAE gain over persistence per lead.
Usage: python analyses/ude/udeA_summary.py [--region RUN] [run ...]   (default: udeA_* in models/ude)"""
import sys, json, glob, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chl_cnn.plotting import OKABE_ITO

ROOT = Path(__file__).resolve().parents[2]; M = ROOT / "models/ude"; FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)


def row(run):
    d = M / run
    if not (d / "metrics.json").exists():
        return None
    c, m, h = json.load(open(d / "config.json")), json.load(open(d / "metrics.json")), json.load(open(d / "history.json"))
    v = m["val"]
    return dict(run=run, c=c, v=v, m=m, h=h)


def f3(xs): return "/".join(f"{x:.4f}" for x in xs)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--region", default="udeA_mlp"); p.add_argument("runs", nargs="*"); a = p.parse_args()
    runs = a.runs or sorted(x.name for x in M.glob("udeA_*") if x.name != "udeA_smoke")
    rows = [r for r in map(row, runs) if r]
    print("| run | g | g-max | lr-net | IC | ep | stride | val t0 | model MAE 1/2/3 | persistence 1/2/3 | climatology 1/2/3 | RMSE l1 | grad ratio/corr l1 | lambda_r | min | GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c, v, m = r["c"], r["v"], r["m"]
        gm = "-" if c["g"] in ("none", "relax") else c["g_max"]; lr = "-" if c["g"] in ("none", "relax") else c["lr_net"]
        lam = f"{m['params']['lambda_r']:.3f}" if "lambda_r" in m["params"] else "-"
        ep = 0 if c["eval_only"] or c["g"] == "none" else c["epochs"]
        gonly = "" if np.allclose(v["g_only"]["mae"], v["model"]["mae"], atol=1e-6) else f" (g-only {f3(v['g_only']['mae'])})"
        print(f"| {r['run']} | {c['g']} | {gm} | {lr} | {c['ic']} | {ep} | {c['train_stride']} | {m['val_t0']} | {f3(v['model']['mae'])}{gonly} | {f3(v['persistence']['mae'])} | "
              f"{f3(v['climatology']['mae'])} | {v['model']['rmse'][0]:.4f} | {v['grad_ratio_l1']:.3f}/{v['grad_corr_l1']:.3f} | {lam} | {m['minutes']:.1f} | {m['max_mem_gb']:.2f} |")
    # best-epoch note
    print()
    for r in rows:
        hv = r["h"]["val"]
        if len(hv) > 1:
            means = [np.mean(x["model"]["mae"]) for x in hv]
            print(f"{r['run']}: mean-lead val MAE per epoch (0 = init) {' '.join(f'{x:.4f}' for x in means)}  best epoch {int(np.argmin(means))}  train L1 {' '.join(f'{x:.4f}' for x in r['h']['train'][1:])}")
    # region table
    rr = next((r for r in rows if r["run"] == a.region), None)
    if rr and "regions_l1" in rr["v"]:
        print(f"\nRegion MAE at lead 1, {a.region}:\n| region | model | persistence | climatology | model - persistence | n obs |\n|---|---|---|---|---|---|")
        for name, d in rr["v"]["regions_l1"].items():
            print(f"| {name} | {d['model']:.4f} | {d['persistence']:.4f} | {d['climatology']:.4f} | {d['model'] - d['persistence']:+.4f} | {d['n_obs']} |")
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for k in range(3):
        for i, r in enumerate(rows):
            hv = r["h"]["val"]
            if len(hv) <= 1: continue
            ax[k].plot(range(len(hv)), [x["model"]["mae"][k] for x in hv], "o-", color=OKABE_ITO[i % len(OKABE_ITO)], label=r["run"])
        ax[k].axhline(rows[0]["v"]["persistence"]["mae"][k], color="k", ls="--", lw=1, label="persistence (IC fill)")
        ax[k].set_title(f"Val MAE at lead {k + 1} per epoch (observed px, ln CHL)"); ax[k].set_xlabel("epoch (0 = init)"); ax[k].set_ylabel("val MAE (ln CHL)")
    ax[0].legend(fontsize=7); fig.suptitle("UDE stage A, Black Sea, transport off: val MAE by lead per epoch", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "udeA_curves.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2)); w = 0.8 / max(len(rows), 1)
    for i, r in enumerate(rows):
        g = np.array(r["v"]["model"]["mae"]) - np.array(r["v"]["persistence"]["mae"])
        ax.bar(np.arange(3) + (i - len(rows) / 2 + 0.5) * w, g, w, color=OKABE_ITO[i % len(OKABE_ITO)], label=r["run"])
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(["lead 1", "lead 2", "lead 3"])
    ax.set_ylabel("model MAE minus persistence MAE (ln CHL)"); ax.set_title("UDE stage A: final val MAE minus persistence MAE, by lead"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "udeA_gain_by_lead.png", dpi=110); plt.close(fig)
    print(f"\nfigures: {FIG / 'udeA_curves.png'}, {FIG / 'udeA_gain_by_lead.png'}")


if __name__ == "__main__":
    main()
