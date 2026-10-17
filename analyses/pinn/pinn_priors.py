"""Summary of the prior-knowledge PINN runs (models/pinn_v1_*) against the v0 references (pinn_lam0, pinn_lam0p3).
  python analyses/pinn/pinn_priors.py   -> analyses/pinn/figures/pinn_priors.png and a table on stdout
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chl_cnn.plotting import OKABE_ITO
from chl_cnn.runs import find_run

FIG = Path(__file__).resolve().parent / "figures"
ORDER = ["pinn_lam0", "pinn_lam0p3", "pinn_v1_clim", "pinn_v1_ekman", "pinn_v1_courant", "pinn_v1_clim_ekman", "pinn_v1_all"]
LABEL = {"pinn_lam0": "no physics (lam 0)", "pinn_lam0p3": "transport only (lam 0.3)", "pinn_v1_clim": "+ seasonal source",
         "pinn_v1_ekman": "+ Ekman source", "pinn_v1_courant": "+ Courant weighting", "pinn_v1_clim_ekman": "+ seasonal + Ekman",
         "pinn_v1_all": "+ seasonal + Ekman + Courant"}
def _d(r):
    try: return find_run(r)
    except FileNotFoundError: return None
runs = [r for r in ORDER if _d(r) is not None and (_d(r) / "metrics.json").exists()]

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
rows = []
for i, r in enumerate(runs):
    D = find_run(r); cfg = json.load(open(D / "config.json")); m = json.load(open(D / "metrics.json"))["scores"]; h = json.load(open(D / "history.json"))
    v = m["val"]; ph = v["physics"]
    base_cloud = ph.get("abs_rbase_cloud", ph["abs_r_cloud"])      # v0 runs: no source, so base == full
    rows.append((LABEL[r], v["cnn_hidden"]["mae"], v["cnn_hidden"]["rmse"], v["cnn_hidden"]["bias"], ph["abs_r_cloud"], base_cloud, ph["abs_r_obs"],
                 cfg.get("ekman_alpha"), cfg.get("ekman_kappa"), v["cnn_hidden"]["n"]))
    c = OKABE_ITO[i % len(OKABE_ITO)]; ls = "-" if i < len(OKABE_ITO) else "--"
    ep = range(1, len(h["val"]) + 1)
    ax[0].plot(ep, h["val"], color=c, ls=ls, marker=".", label=LABEL[r])
    ax[1].plot(ep, h.get("val_physbase_cloud", h["val_phys_cloud"]), color=c, ls=ls, marker=".", label=LABEL[r])
ax[0].set(xlabel="epoch", ylabel="val MAE on hidden pixels, ln CHL", title="Validation hidden-pixel MAE by epoch"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
ax[1].set(xlabel="epoch", ylabel="val mean |transport residual| under real clouds, ln CHL / day",
          title="Transport-only residual under real clouds by epoch"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
xi = np.arange(len(rows)); R = rows
ax[2].bar(xi, [r[1] for r in R], color=[OKABE_ITO[i % len(OKABE_ITO)] for i in xi])
ax[2].set(xticks=xi, xticklabels=[r[0] for r in R], ylabel="val MAE on hidden pixels, ln CHL", title="Final val hidden-pixel MAE by run",
          ylim=(min(r[1] for r in R) - 0.002, max(r[1] for r in R) + 0.002))
ax[2].tick_params(axis="x", rotation=35, labelsize=7); ax[2].grid(alpha=.3, axis="y")
fig.suptitle("PINN prior-knowledge runs: dilated CNN, lambda 0.3, 12 epochs, 1997-2020 year split"); fig.tight_layout()
fig.savefig(FIG / "pinn_priors.png", dpi=110)

print(f"{'run':30s} {'val MAE':>8} {'val RMSE':>9} {'bias':>7} {'|r| cloud':>10} {'|r_base| cloud':>15} {'|r| obs':>8} {'alpha':>7} {'kappa (0-3,4-10,11-20,21-30 d)':>32} {'n hidden px':>11}")
for r in rows:
    al = f"{r[7]:.3g}" if r[7] is not None else "-"
    ka = "[" + ", ".join(f"{k:.2f}" for k in r[8]) + "]" if r[8] else "-"
    print(f"{r[0]:30s} {r[1]:8.4f} {r[2]:9.4f} {r[3]:7.4f} {r[4]:10.4f} {r[5]:15.4f} {r[6]:8.4f} {al:>7} {ka:>32} {r[9]:11d}")
