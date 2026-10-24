"""Static weight-map figures of the io4 token-decoder runs, the counterpart of analyses/io3_attn/figures/b_weight_maps_*.png (2026-10-24).
Figures only; data from the explorer caches (analyses/attn_viz/explorer_cache/<run>.npz, analyses/attn_viz/build_explorer.py), which
hold the 3 io3 figure dates among their preset val days.

Per run and val day (2001-03-02, 2001-10-29, 2007-06-27), columns = day offsets t-4, t-2, t-1, t, t+1, t+2, t+4 (+ "no day" for the
blend weights); rows (chl_cnn/attn_viz.draw_maps):
  patch pooling weight   tokdec pooling weight of each ocean 4 x 4 px patch (mean over heads; day t included), drawn on the patch's
                         pixels with nearest-neighbour upsampling, so the patch blocks are as the model has them
  blend day weight       io4_tokdec_blend only: per-pixel softmax over the context days visible at the pixel + "no day"
  visible in input       the day's visible mask in the stem input (donor-masked)
  patch score (interp.), pixel term   io4_tokdec_blend only: the two logit terms of the blend day weight

Usage (project root): python -m analyses.io4_tokdec.weight_maps
Writes figures/b_weight_maps_<run>_<date>.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chl_cnn import attn_explorer as AE, attn_viz as AV

FIG = Path(__file__).resolve().parent / "figures"
RUNS = ("io4_tokdec", "io4_tokdec_blend")
DATES = ("2001-03-02", "2001-10-29", "2007-06-27")
OFFSETS = (-4, -2, -1, 1, 2, 4)


def main():
    for run in RUNS:
        z = AE.load(run)
        assert z is not None, f"{run}: no explorer cache; {AE.build_cmd(run)}"
        for d in DATES:
            i = int(np.flatnonzero((z["dates"] == d) & (z["split"] == "val"))[0])
            fig = AV.draw_maps(z, i, OFFSETS, with_t=True, with_none=True)
            fig.suptitle(f"{run} ({z['kind']}): weights by day offset and input visibility, val day {d} "
                         f"(observed {AE._obs_frac(z, i):.0%} of ocean, {int(z['n_hid'][i]):,} hidden px)", fontsize=11)
            out = FIG / f"b_weight_maps_{run}_{d}.png"
            fig.savefig(out, dpi=110); plt.close(fig)
            print("wrote", out)


if __name__ == "__main__":
    main()
