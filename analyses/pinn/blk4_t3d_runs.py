"""Black Sea 4 km store (blk4): 3D temporal stem + gradient loss, with / without the semi-Lagrangian physics loss.

Table: val / test hidden-pixel MAE, RMSE, bias, grad ratio / corr, |r| (semi-Lagrangian residual) under clouds and on
observed pixels, references (per-pixel and DOY climatology), s/epoch, peak GPU.
Figure: figures/blk4_t3d_by_epoch.png (val hidden-pixel MAE and val |r| under clouds by epoch).
Usage: python analyses/pinn/blk4_t3d_runs.py   (from the project root)
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

RUNS = [("blk4_t3d_c4_gdl100_lam0", "no physics loss (lambda 0)"),
        ("blk4_t3d_c4_gdl100_sl_lam0p3", "semi-Lagrangian physics loss, lambda 0.3")]
OUT = Path(__file__).resolve().parent / "figures"


# --- data
def load(run):
    d = ROOT / "models" / "pinn" / "t3d" / run
    return (json.loads((d / f).read_text()) for f in ["config.json", "history.json", "metrics.json"])


# --- analysis
def table(runs):
    print(f"{'run':32s} {'split':5s} {'MAE':>7s} {'RMSE':>7s} {'bias':>7s} {'gratio':>7s} {'gcorr':>7s} {'|r|cld':>7s} {'|r|obs':>7s} {'n hidden':>9s}")
    for run, (c, h, m) in runs.items():
        for s in ["val", "test"]:
            x, p = m["scores"][s]["cnn_hidden"], m["scores"][s]["physics"]
            print(f"{run:32s} {s:5s} {x['mae']:7.4f} {x['rmse']:7.4f} {x['bias']:+7.4f} {p['grad_ratio']:7.3f} {p['grad_corr']:7.3f} "
                  f"{p['abs_r_cloud']:7.4f} {p['abs_r_obs']:7.4f} {x['n']:9d}")
        print(f"{'':32s} best epoch {c['best_epoch']}, {np.mean(h['epoch_s']):.0f} s/epoch, peak GPU {c['peak_gpu_gb']:.2f} GB, "
              f"legacy_resume {c.get('legacy_resume')}, NaN in val history {bool(np.isnan(np.array(h['val'], float)).any())}")
    m = next(iter(runs.values()))[2]["scores"]
    for s in ["val", "test"]:
        print(f"references {s}: per-pixel climatology {m[s]['climatology_hidden']['mae']:.4f}, "
              f"DOY climatology {m[s]['doy_climatology_hidden']['mae']:.4f} (n {m[s]['doy_climatology_hidden']['n']})")


# --- report
def figure(runs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (run, (c, h, m)) in enumerate(runs.items()):
        lab = dict(RUNS)[run]
        ep = np.arange(1, len(h["val"]) + 1)
        ax[0].plot(ep, h["val"], "o-", ms=3, color=OKABE_ITO[i], label=lab)
        ax[1].plot(ep, h["val_phys_cloud"], "o-", ms=3, color=OKABE_ITO[i], label=lab)
    n = next(iter(runs.values()))[2]["scores"]["val"]["cnn_hidden"]["n"]
    ax[0].set(xlabel="epoch", ylabel="val MAE on fake-cloud-hidden pixels (ln CHL)",
              title=f"BLK 4 km: val hidden-pixel MAE\n(n={n:,} px)")
    ax[1].set(xlabel="epoch", ylabel="val mean |r| under real clouds (ln CHL / day)",
              title="BLK 4 km: val semi-Lagrangian residual\nunder real clouds")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3); a.set_xticks(range(2, 21, 2))
    plt.tight_layout(); OUT.mkdir(exist_ok=True)
    f = OUT / "blk4_t3d_by_epoch.png"; plt.savefig(f, dpi=120); print("figure", f)


if __name__ == "__main__":
    runs = {r: tuple(load(r)) for r, _ in RUNS}
    table(runs)
    figure(runs)
