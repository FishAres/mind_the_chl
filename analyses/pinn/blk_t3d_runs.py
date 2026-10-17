"""Black Sea 3D temporal stem + gradient loss, with / without the semi-Lagrangian physics loss (blk_t3d_* runs).

Table: best val hidden-pixel MAE and epoch, test MAE, grad ratio / corr at the best epoch, val |r| (semi-Lagrangian
residual) under clouds / on observed pixels at the last epoch, s/epoch, peak GPU.
Figure: figures/blk_t3d_val_by_epoch.png (val hidden-pixel MAE by epoch for the two runs).
Usage: python analyses/pinn/blk_t3d_runs.py   (from the project root)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chl_cnn.plotting import OKABE_ITO  # noqa: E402

RUNS = [("blk_t3d_c4_gdl100_lam0", "no physics loss (lambda 0)"), ("blk_t3d_c4_gdl100_sl_lam0p3", "semi-Lagrangian physics loss, lambda 0.3")]
OUT = Path(__file__).resolve().parent / "figures"


def main():
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (run, lab) in enumerate(RUNS):
        d = ROOT / "models" / "pinn" / "t3d" / run
        if not (d / "history.json").exists():
            print("missing", run); continue
        h = json.loads((d / "history.json").read_text())
        ep = np.arange(1, len(h["val"]) + 1)
        ax.plot(ep, h["val"], "o-", ms=3, color=OKABE_ITO[i], label=lab)
        c0 = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        if c0.get("legacy_resume"):
            ax.axvline(c0["legacy_resume_epoch"] + 0.5, color=OKABE_ITO[i], ls=":", lw=1.2,
                       label=f"{lab}: restarted after epoch {c0['legacy_resume_epoch']} (optimizer state reset)")
        if (d / "metrics.json").exists():
            c = json.loads((d / "config.json").read_text()); m = json.loads((d / "metrics.json").read_text())["scores"]
            b = c["best_epoch"] - 1
            rows.append(dict(run=run, best_val=c["best_val_mae"], best_epoch=c["best_epoch"], test=m["test"]["cnn_hidden"]["mae"],
                             grad_ratio=h["val_grad_ratio"][b], grad_corr=h["val_grad_corr"][b],
                             phys_cloud_last=h["val_phys_cloud"][-1], phys_obs_last=h["val_phys_obs"][-1],
                             s_per_epoch=float(np.mean(h["epoch_s"])), peak_gb=c["peak_gpu_gb"], epochs=len(h["val"]),
                             n_val_hidden=m["val"]["cnn_hidden"]["n"], n_test_hidden=m["test"]["cnn_hidden"]["n"]))
    ax.set(xlabel="epoch", ylabel="val hidden-pixel MAE, ln CHL", title="Black Sea 3D stem (K=4, G=100): val MAE by epoch",
           xticks=range(0, 21, 5)); ax.grid(alpha=.3); ax.legend(fontsize=8)
    OUT.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(OUT / "blk_t3d_val_by_epoch.png", dpi=120); plt.close(fig)
    for r in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    print("wrote", OUT / "blk_t3d_val_by_epoch.png")


if __name__ == "__main__":
    main()
