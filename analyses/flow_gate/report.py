"""Merge Check A and Check B outputs into results/flow_gate.md and results/flow_gate.json (no new computation).
Run after check_a.py and check_b.py: python -m analyses.flow_gate.report"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
ASSUME = """## Assumptions
- Val years 2017 and 2021 only; no train or test years added. ln CHL in MAE. Grid = train_pinn --crop (177 x 273), 1 px ~ 4.5 km, row index increases northward.
- Check A: accepted lead-1 patches = spectral_motion cache rows with k = 1, not null, peak exists, not on search edge, peak r >= 0.6, day t in val years. ln CHL as spectral_motion (CHL <= 0 or > 100 -> missing). Scored px = day t+1 observed inside the 32 x 32 patch AND day t observed at x AND x0.25 / oracle / current-shift all valid (common set). Overlapping patches: a px counts once per patch. Oracle and current-shift sample the full day-t field (not only the patch) bilinearly, valid if the observed bilinear weight >= 0.99 (the depart_sample rule). Current shift = 0.46 x trapezoid of the cached Hann-weighted patch-mean reanalysis velocity on days t, t+1, same factor for both components. Region = per-pixel label (regions_4km), season = month of day t. CIs: 2000 day-bootstrap resamples of the ratio of summed errors.
- Check B: ln CHL as train_pinn (no upper cap). Context visibility follows train_pinn.nb_masks (calendar day t+k; donor's calendar day + k, or the donor day itself if missing); viz.draw_day uses store-index offsets, which differ only where the store skips days (count in data quality). (b) / (c) choose the source day by validity of the advected value (trajectory valid and departure point visible with weight >= 0.99), not by visibility at x. (c) tie rule: past day first at equal |k|. Advection needs every calendar day on the path in the store. Gradient pairs: lag 1 px. Gap-filler pred is float16 (preds.npz).
"""


def main():
    a = (RES / "check_a.md").read_text(); b = (RES / "check_b.md").read_text()
    md = "# Flow gate: oracle motion ceiling and motion-only predictors vs the gap-filler (2026-10-20)\n\n" + ASSUME + "\n" + a + "\n" + b
    (RES / "flow_gate.md").write_text(md)
    out = dict(check_a=json.load(open(RES / "check_a.json")), check_b=json.load(open(RES / "check_b.json")), assumptions=ASSUME)
    json.dump(out, open(RES / "flow_gate.json", "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main()
