"""Black Sea: how often the semi-Lagrangian displacement k * u * dt (cells) of train_spatial.py --lagrangian would be
clipped at --lag-max 5 and 15, from the store currents on every 20th day (ocean pixels, finite currents).
Run: python analyses/spatial_node/blk_lag_clip.py  -> figures/blk_lag_displacement.png"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chl_cnn import store_blk as SB
from chl_cnn.physics import grid_metrics
from chl_cnn.plotting import OKABE_ITO

d = SB.load_store(); ocean = d["ocean"]; lat, lon = d["lat"], d["lon"]
dx, dy, _, _ = grid_metrics(lat, lon)
cpdx, cpdy = 86400.0 / np.abs(dx), 86400.0 / np.abs(dy)
idx = np.arange(0, d["u_curr"].shape[0], 20)
u = np.asarray(d["u_curr"][idx]).astype("float32"); v = np.asarray(d["v_curr"][idx]).astype("float32")
ok = np.isfinite(u) & np.isfinite(v) & ocean[None]
ux, vy = np.abs(u) * cpdx[None], np.abs(v) * cpdy[None]
disp = np.maximum(ux, vy)[ok]                                   # cells per day, per-axis max (clipping is per axis)
print(f"days {len(idx)}  pixel-days {disp.size}  cell size dx {np.abs(dx).mean()/1e3:.2f} km dy {np.abs(dy).mean()/1e3:.2f} km")
print(f"|displacement| per day (cells): median {np.median(disp):.2f}  p90 {np.percentile(disp, 90):.2f}  p99 {np.percentile(disp, 99):.2f}  max {disp.max():.1f}")
for k in (1, 2, 4):
    print(f"k={k}: fraction clipped at lag-max 5: {(k * disp > 5).mean():.3f}   at 15: {(k * disp > 15).mean():.4f}")
fig, ax = plt.subplots(figsize=(7, 4))
for k, c in zip((1, 2), OKABE_ITO):
    ax.hist(np.minimum(k * disp, 40), bins=80, range=(0, 40), histtype="step", color=c, label=f"k = {k} day(s)", density=True)
for lm, ls in ((5, "--"), (15, ":")):
    ax.axvline(lm, color="k", ls=ls, lw=1, label=f"--lag-max {lm}")
ax.set_xlabel("|k u dt| (cells, per-axis max; values > 40 binned at 40)"); ax.set_ylabel("density of ocean pixel-days")
ax.set_title("Black Sea: semi-Lagrangian displacement of neighbour day t+-k, every 20th day"); ax.legend(fontsize=8)
fig.savefig(Path(__file__).resolve().parent / "figures" / "blk_lag_displacement.png", dpi=120, bbox_inches="tight")
