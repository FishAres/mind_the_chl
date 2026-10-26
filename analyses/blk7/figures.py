"""blk7 overview figure (Black Sea 4 km), redrawn by analyses/io7/postrun.py after every blk4 postrun. Reporting only: reads the
results JSONs in analyses/blk7/results/ and draws analyses/blk7/figures/blk7_runs.png:
  (a) hidden-px MAE (ln CHL) val / test per run, (b) real-cloud reweighted MAE val / test per run, (c) val hidden-px MAE by distance
  to the nearest visible px of day t (px bins), one line per run. Sample sizes (hidden px) in the panel (a) labels.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chl_cnn.plotting import OKABE_ITO

FIG = Path(__file__).resolve().parent / "figures"


def short(run):
    return run.replace("blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50", "conv b16 λ0.3 +cont (ref)") \
              .replace("blk4_t3d_c4_gdl100_lam0_b16_50ep", "conv b16 λ0 (ref)").replace("blk4_", "")


def runs_figure(res_dir, runs, out=None):
    res_dir = Path(res_dir)
    R = {r: json.loads((res_dir / f"{r}.json").read_text()) for r in dict.fromkeys(runs) if (res_dir / f"{r}.json").exists()}
    names = list(R)
    if not names:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    x = np.arange(len(names)); w = 0.38
    for k, (split, col) in enumerate((("val", OKABE_ITO[0]), ("test", OKABE_ITO[1]))):
        ax[0].bar(x + (k - 0.5) * w, [R[n][split]["mae"] for n in names], w, color=col, label=split)
        ax[1].bar(x + (k - 0.5) * w, [R[n][split]["rw"]["mae"] for n in names], w, color=col, label=split)
    labels = [f"{short(n)}\nval n {R[n]['val']['n_hidden']:,} px" for n in names]
    for a_, t in zip(ax[:2], ("Hidden-px MAE by run (Black Sea 4 km)", "Real-cloud reweighted MAE by run (Black Sea 4 km)")):
        a_.set_xticks(x); a_.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        a_.set_ylabel("MAE, ln CHL"); a_.set_title(t); a_.legend(title="split")
        v = [R[n][s]["mae"] if a_ is ax[0] else R[n][s]["rw"]["mae"] for n in names for s in ("val", "test")]
        a_.set_ylim(min(v) * 0.95, max(v) * 1.02)
    bins = list(R[names[0]]["val"]["by_dist"])
    for i, n in enumerate(names):
        ax[2].plot(bins, [R[n]["val"]["by_dist"][b]["mae"] for b in bins], marker="o", color=OKABE_ITO[i % len(OKABE_ITO)], label=short(n))
    ax[2].set_xlabel("distance to nearest visible px of day t (px, 4.5 km)"); ax[2].set_ylabel("val hidden-px MAE, ln CHL")
    ax[2].set_title("Val MAE by distance to visible px (Black Sea 4 km)"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = Path(out or FIG / "blk7_runs.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    return str(out)
