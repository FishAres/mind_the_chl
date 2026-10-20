"""Tracer project summary figure (analyses/tracer/TRACER.md): val MAE by lead for every tracer stage, on the test-1
common set, read from results/band_test2.json (written by band_report.py). No new scoring.

    python -m analyses.tracer.summary_fig
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chl_cnn.plotting import OKABE_ITO

HERE = Path(__file__).resolve().parent
J = json.load(open(HERE / "results/band_test2.json"))["runs"]
LEADS = np.arange(1, 6)


def series(kind):
    """Reference predictors from trb_const_s0's best-epoch px (identical to test-1 mask px to within 4.3 k of 1.71 M)."""
    b = J["trb_const_s0"]["best"]
    out = {
        "persistence": (b[kind]["persistence"], OKABE_ITO[7], ":"),
        "currents x0.25": (b[kind]["x0.25"], OKABE_ITO[6], "-."),
        "step 1: learned velocity": (b[kind]["step1"], OKABE_ITO[0], "-"),
        "step 2: velocity + source (tr_src, best epoch)": (b[f"{kind}_tr_src_same_px"], OKABE_ITO[4], "-"),
        "band model, no training (test 1)": (b[kind]["test1"], OKABE_ITO[1], "--"),
    }
    for run, lab, c in (("trb_const_s0", "band model, constant fade (best epoch)", OKABE_ITO[2]),
                        ("trb_phys_s0", "band model, physics fade (best epoch)", OKABE_ITO[3])):
        out[lab] = (J[run]["best"][kind]["model"], c, "-")
    last5 = np.mean([J[r]["last5"][kind]["model"] for r in J], 0)
    out["band models, mean of last 5 epochs (4 runs)"] = (last5, OKABE_ITO[5], "--")
    return out


def main():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, (kind, ttl, yl) in enumerate((("mae", "Val pixel MAE by lead, tracer stages", "pixel MAE (ln CHL)"),
                                          ("band", "Val 20-70 km band MAE by lead, tracer stages", "20-70 km band-passed MAE (ln CHL)"))):
        for lab, (y, c, ls) in series(kind).items():
            ax[k].plot(LEADS, y, ls, color=c, marker="o", ms=4, lw=1.8, label=lab)
        ax[k].set(title=ttl, xlabel="forecast lead (days)", ylabel=yl, xticks=LEADS)
        ax[k].grid(alpha=.3)
    ax[0].legend(fontsize=8, loc="upper left")
    n = J["trb_const_s0"]["best"]["n"]
    fig.suptitle(f"Black Sea 4 km, val 2017 + 2021, 351 start days, common pixel set (n {int(n[0]):,} px at lead 1, {int(n[-1]):,} at lead 5)", fontsize=10)
    fig.tight_layout()
    out = HERE / "figures/tracer_summary.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
