"""Phase 1 summary figure (bridge #25): io7 runs vs the three io6_tokdec_blend_spatial seeds.

Reads analyses/io7/results/<run>.json (written by postrun.py). Run: python -m analyses.io7.report_phase1
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from chl_cnn.plotting import OKABE_ITO

RES = Path("analyses/io7/results")
FIG = Path("analyses/io7/figures")
SEEDS = ["io6_tokdec_blend_spatial", "io6_tokdec_blend_spatial_s1", "io6_tokdec_blend_spatial_s2"]
RUNS = ["io7_physmem", "io7_wmattn", "io7_analog"]
LABELS = {"io7_physmem": "physics memory", "io7_wmattn": "water-mass attn", "io7_analog": "analog tokens"}
PANELS = [("all hidden px", lambda s: s["mae"]),
          ("hidden px with ≥ 1 key", lambda s: s["by_key"][">=1 key"]["mae"]),
          ("hidden px with no key", lambda s: s["by_key"]["no key"]["mae"]),
          ("real-cloud reweighted", lambda s: s["rw"]["mae"])]


def load(run):
    return json.loads((RES / f"{run}.json").read_text())


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = {r: load(r) for r in SEEDS + RUNS}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (title, get) in zip(axes, PANELS):
        for i, split in enumerate(["val", "test"]):
            x0 = i * (len(RUNS) + 2)
            seeds = [get(data[r][split]) for r in SEEDS]
            ax.scatter([x0] * 3, seeds, color="grey", marker="o", s=25, label="io6 spatial seeds 0-2" if i == 0 else None)
            for j, r in enumerate(RUNS):
                ax.scatter(x0 + 1 + j, get(data[r][split]), color=OKABE_ITO[j], marker="D", s=40,
                           label=LABELS[r] if i == 0 else None)
        ax.set_xticks([1.5, 1.5 + len(RUNS) + 2], ["val", "test"])
        ax.set_title(f"MAE, {title}")
        ax.set_ylabel("MAE (ln CHL)")
    axes[0].legend(fontsize=8)
    n = data[SEEDS[0]]["val"]
    fig.suptitle(f"io7 Phase 1 runs vs io6 spatial seeds (val n {n['n_hidden']:,} hidden px, {n['n_days']} days)")
    fig.tight_layout()
    out = FIG / "p1_runs_vs_seeds.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
