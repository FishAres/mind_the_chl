"""Summary of the PINN lambda sweep (models/pinn_lam*): val scores table + figure.
  python analyses/pinn/pinn_sweep.py   -> analyses/pinn/figures/pinn_lambda_sweep.png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chl_cnn.plotting import OKABE_ITO
from chl_cnn.runs import list_runs

FIG = Path(__file__).resolve().parent / "figures"
runs = sorted((p / "metrics.json" for p in list_runs(["pinn/cnn"], "pinn_lam*") if (p / "metrics.json").exists()), key=lambda p: json.load(open(p.parent / "config.json"))["lam"])
rows = []
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for i, p in enumerate(runs):
    cfg = json.load(open(p.parent / "config.json")); m = json.load(open(p))["scores"]; h = json.load(open(p.parent / "history.json"))
    lam = cfg["lam"]; v = m["val"]; t = m["test"]
    rows.append((lam, v["cnn_hidden"]["mae"], v["cnn_hidden"]["rmse"], v["cnn_hidden"]["bias"], v["physics"]["abs_r_cloud"], v["physics"]["abs_r_obs"],
                 t["cnn_hidden"]["mae"], t["physics"]["abs_r_cloud"], v["cnn_hidden"]["n"], v["physics"]["n_cloud"]))
    ep = range(1, len(h["val"]) + 1); c = OKABE_ITO[i]
    ax[0].plot(ep, h["val"], color=c, marker=".", label=f"lambda {lam:g}")
    ax[1].plot(ep, h["val_phys_cloud"], color=c, ls="-", marker=".", label=f"lambda {lam:g}, under real clouds")
    ax[1].plot(ep, h["val_phys_obs"], color=c, ls=":", label=f"lambda {lam:g}, observed pixels")
ax[0].set(xlabel="epoch", ylabel="val MAE on hidden pixels, ln CHL", title="Validation hidden-pixel MAE by epoch"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
ax[1].set(xlabel="epoch", ylabel="val mean |r|, ln CHL per day", title="Transport residual on val days by epoch"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
R = np.array(rows)
xi = np.arange(len(R))
ax[2].plot(xi, R[:, 1], marker="o", color=OKABE_ITO[0], label="val hidden-pixel MAE")
ax[2].set(xlabel="lambda (physics loss weight)", ylabel="val MAE on hidden pixels, ln CHL", title="Final val MAE and residual vs lambda", xticks=xi, xticklabels=[f"{l:g}" for l in R[:, 0]])
ax2 = ax[2].twinx(); ax2.plot(xi, R[:, 4], marker="s", color=OKABE_ITO[1], label="val |r| under real clouds"); ax2.set_ylabel("val mean |r|, ln CHL per day")
h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2, fontsize=8); ax[2].grid(alpha=.3)
fig.suptitle("PINN v0 lambda sweep: dilated CNN, 12 epochs, 1997-2020 year split"); fig.tight_layout()
fig.savefig(FIG / "pinn_lambda_sweep.png", dpi=110)
print(f"{'lam':>5} {'val MAE':>8} {'val RMSE':>9} {'val bias':>9} {'|r| cloud':>10} {'|r| obs':>8} {'test MAE':>9} {'test |r| cloud':>14} {'n val px':>9} {'n val cloud px':>14}")
for r in rows:
    print(f"{r[0]:5g} {r[1]:8.4f} {r[2]:9.4f} {r[3]:9.4f} {r[4]:10.4f} {r[5]:8.4f} {r[6]:9.4f} {r[7]:14.4f} {int(r[8]):9d} {int(r[9]):14d}")
