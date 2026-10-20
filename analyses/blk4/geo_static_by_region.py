"""Val hidden-pixel MAE by region: the two blk4 b16 50-epoch runs with and without --geo-static (bathymetry,
river distances, physics-availability mask). Run: python -m analyses.blk4.geo_static_by_region
-> analyses/blk4/results/geo_static_by_region.json, analyses/blk4/figures/geo_static_by_region.png"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from chl_cnn.viz import context_for
from chl_cnn.plotting import OKABE_ITO
from analyses.blk4.error_breakdown import regions_4km

ROOT = Path(__file__).resolve().parents[2]
T3D = ROOT / "models/pinn/t3d"
PAIRS = [("λ 0", "blk4_t3d_c4_gdl100_lam0_b16_50ep"), ("λ 0.3", "blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep")]
NAMES = json.load(open(ROOT / "analyses/blk/region_names.json"))

# ---- data: per-region sums of |error| over val hidden pixels
ctx = context_for(T3D / PAIRS[0][1])
reg, _ = regions_4km(ctx["ocean"])
runs = [r for _, base in PAIRS for r in (base, base + "_geo")]
P = {r: np.load(T3D / r / "preds.npz") for r in runs}
pos = {int(i): k for k, i in enumerate(P[runs[0]]["idx"])}
assert all((P[r]["idx"] == P[runs[0]]["idx"]).all() for r in runs)
days = [d for d in ctx["va"] if int(d) in pos]
S = {r: np.zeros(8) for r in runs}; N = np.zeros(8)
for d in days:
    k = pos[int(d)]; hid = P[runs[0]]["hidden"][k] & ctx["ocean"]; t = ctx["y"](d)
    g = reg[hid]; N += np.bincount(g, minlength=8)
    for r in runs:
        S[r] += np.bincount(g, weights=np.abs(P[r]["pred"][k][hid].astype("float64") - t[hid]), minlength=8)

# ---- analysis
labels = [NAMES[str(i)] if isinstance(NAMES, dict) else NAMES[i - 1] for i in range(1, 8)]
res = {lab: dict(n_px=int(N[i])) | {r: S[r][i] / N[i] for r in runs} for i, lab in enumerate(labels, 1)}
res["all"] = dict(n_px=int(N.sum())) | {r: S[r].sum() / N.sum() for r in runs}
(ROOT / "analyses/blk4/results").mkdir(exist_ok=True)
json.dump(dict(n_days=len(days), by_region=res), open(ROOT / "analyses/blk4/results/geo_static_by_region.json", "w"), indent=1)

# ---- report
print(f"val days {len(days)}")
print("| region | Δ MAE λ 0 (geo − base) | Δ MAE λ 0.3 (geo − base) | MAE λ 0.3 base | n px |\n|---|---|---|---|---|")
for lab, v in res.items():
    print(f"| {lab} | {v[runs[1]] - v[runs[0]]:+.4f} | {v[runs[3]] - v[runs[2]]:+.4f} | {v[runs[2]]:.4f} | {v['n_px']:,} |")
fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(res)); w = 0.38
for j, (lab, base) in enumerate(PAIRS):
    ax.bar(x + (j - 0.5) * w, [v[base + "_geo"] - v[base] for v in res.values()], w, color=OKABE_ITO[j], label=lab)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x, [f"{k}\n(n px {v['n_px']:,})" for k, v in res.items()], fontsize=7, rotation=20, ha="right")
ax.set(title="Black Sea 4 km: change in val hidden-pixel MAE with --geo-static, by region",
       ylabel="MAE with geo-static − MAE without (ln CHL)")
ax.legend(title="physics loss weight"); ax.grid(alpha=.3, axis="y"); fig.tight_layout()
fig.savefig(ROOT / "analyses/blk4/figures/geo_static_by_region.png", dpi=130)
