"""Black Sea summary figure for analyses/BLACK_SEA.md: (a) best val MAE per model family at 2 km, (b) val MAE by
epoch of the 4 km 3D-stem runs, with the warm-restart stages placed end to end.
Run: python analyses/blk4/blk_summary.py  ->  analyses/blk4/figures/blk_summary.png"""
import json, sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from chl_cnn.plotting import OKABE_ITO
from chl_cnn.runs import find_run

# ---- data
TWO_KM = [  # (label, val MAE, source) on the same 2 km val days (every 4th day of 2017 + 2021, 4.20 M hidden px)
    ("DOY climatology", 0.3604, "FINDINGS"),
    ("dilated CNN, 12 ep", 0.1975, "cnn_blk_lam0"),
    ("spatial nODE, 6 ep", 0.1772, "blk_final_conv_ctx4_cf"),
    ("3D stem, λ 0, 20 ep", 0.1627, "blk_t3d_c4_gdl100_lam0"),
    ("3D stem, λ 0.3, 20 ep", 0.1606, "blk_t3d_c4_gdl100_sl_lam0p3"),
]
hist = lambda r: json.load(open(find_run(r) / "history.json"))["val"]
S = "blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep"
FOUR_KM = [  # (label, [(first epoch, val curve)...], colour)
    ("λ 0.3, batch 16: 50 ep, then restarts at lr 1e-3 and 1e-4",
     [(1, hist(S)), (51, hist(S + "_cont50")), (101, hist(S + "_cont50_lr1e-4_cont50"))], OKABE_ITO[0]),
    ("λ 0, batch 16, 50 ep", [(1, hist("blk4_t3d_c4_gdl100_lam0_b16_50ep"))], OKABE_ITO[1]),
    ("λ 1.0, batch 16, 50-ep schedule stopped at 12", [(1, hist("blk4_t3d_c4_gdl100_sl_lam1_b16_stop12"))], OKABE_ITO[2]),
    ("λ 0.3, batch 8, 20 ep", [(1, hist("blk4_t3d_c4_gdl100_sl_lam0p3"))], OKABE_ITO[3]),
    ("λ 0, batch 8, 20 ep", [(1, hist("blk4_t3d_c4_gdl100_lam0"))], OKABE_ITO[4]),
]

# ---- figure
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1, 1.5]))
labels, vals = [t[0] for t in TWO_KM], [t[1] for t in TWO_KM]
b = ax[0].barh(labels, vals, color=[OKABE_ITO[7]] + [OKABE_ITO[5]] * 2 + [OKABE_ITO[0]] * 2)
ax[0].bar_label(b, fmt="%.4f", padding=3); ax[0].invert_yaxis(); ax[0].set_xlim(0, 0.42)
ax[0].set(title="Black Sea 2 km: best val MAE per model (same val days)", xlabel="val MAE on fake-cloud hidden pixels, ln CHL")
for lab, segs, col in FOUR_KM:
    for k, (e0, v) in enumerate(segs):
        ax[1].plot(range(e0, e0 + len(v)), v, color=col, lw=1.4, label=lab if k == 0 else None)
for x in (50.5, 100.5):
    ax[1].axvline(x, color="0.6", ls=":", lw=1)
ax[1].set_ylim(0.14, 0.24)
ax[1].text(0.99, 0.02, "DOY climatology on these days: 0.358 (off scale)", transform=ax[1].transAxes, ha="right", va="bottom", fontsize=9)
ax[1].set(title="Black Sea 4 km, 3D stem + gradient loss: val MAE by epoch", xlabel="epoch (dotted lines: warm restarts)",
          ylabel="val MAE on fake-cloud hidden pixels, ln CHL")
ax[1].legend(fontsize=8, loc="upper center"); ax[1].grid(alpha=.3)
fig.tight_layout()
out = ROOT / "analyses/blk4/figures/blk_summary.png"; fig.savefig(out, dpi=130); print(out)
