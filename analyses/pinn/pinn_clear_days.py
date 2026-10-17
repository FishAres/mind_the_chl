"""Fake clouds on clear days: does the physics term change what is predicted under a fake cloud when the truth
is known and the surroundings are observed?

Uses the saved val predictions of the lambda runs. (1) per-day hidden-pixel MAE vs the day's observed fraction;
(2) on clear val days (observed fraction >= CLEAR), structure of the prediction on hidden pixels whose 4 neighbours
are observed: gradient magnitude ratio pred/truth and correlation of the pred and truth gradient vectors;
(3) maps of the clearest val days, hidden region only.

  python analyses/pinn/pinn_clear_days.py [run_lam0 run_lam0p3 run_lam1p0]   -> analyses/pinn/figures/pinn_clear_*.png
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chl_cnn import store as S, data as D
from chl_cnn.physics import grid_metrics
from chl_cnn.plotting import OKABE_ITO, MAP_CMAP, ERR_CMAP
from chl_cnn.runs import find_run

ROOT = Path(__file__).resolve().parents[2]; FIG = Path(__file__).resolve().parent / "figures"
names = sys.argv[1:] if len(sys.argv) > 1 else ["pinn_lam0", "pinn_lam0p3", "pinn_lam1p0"]
tag = "" if len(sys.argv) == 1 else "_" + names[0].split("_")[1]
CLEAR = 0.6
d = S.load_store(); lat, lon, ocean = d["lat"].astype("float64"), d["lon"].astype("float64"), d["ocean"]
times = pd.to_datetime(d["time"]); tr, va, te = D.split_by_years(d["time"])
y = D.log_chl(np.asarray(d["chl_l3"][va])); obs = np.isfinite(y) & ocean
P, LAM, hidden, obs_frac, per_day = {}, {}, None, None, {}
for r in names:
    D = find_run(r); cfg = json.load(open(D / "config.json")); LAM[r] = cfg["lam"]
    z = np.load(D / "preds.npz"); assert np.array_equal(z["val_idx"], va)
    P[r] = z["pred"][:len(va)]; hidden = z["hidden"][:len(va)]
    m = json.load(open(D / "metrics.json")); obs_frac = np.array(m["obs_frac"]["val"]); per_day[r] = np.array([q["mae"] for q in m["per_day_val"]])
tv = times[va]
lab = lambda r: f"lambda {LAM[r]:g}"

# (1) per-day hidden MAE vs observed fraction
edges = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]); which = np.digitize(obs_frac, edges) - 1
print(f"val days {len(va)}; days with obs frac >= {CLEAR}: {(obs_frac >= CLEAR).sum()}")
print(f"{'obs frac bin':14s} {'n days':>6s} " + " ".join(f"{lab(r):>11s}" for r in names))
tab = {r: [] for r in names}
for b in range(len(edges) - 1):
    s = which == b
    print(f"{edges[b]:.1f}-{edges[b+1]:.1f}        {s.sum():6d} " + " ".join(f"{np.nanmean(per_day[r][s]):11.4f}" if s.sum() >= 5 else f"{'<5':>11s}" for r in names))
    for r in names: tab[r].append(np.nanmean(per_day[r][s]) if s.sum() >= 5 else np.nan)

# (2) structure on hidden pixels with 4 observed neighbours, clear days
dx, dy, _, _ = grid_metrics(lat, lon)
def grads(c):
    gx = (np.roll(c, -1, -1) - np.roll(c, 1, -1)) / (2 * np.abs(dx)) * 1e5
    gy = (np.roll(c, -1, -2) - np.roll(c, 1, -2)) / (2 * np.abs(dy)) * 1e5
    return gx, gy
clear_days = np.flatnonzero(obs_frac >= CLEAR)
nb_obs = obs & np.roll(obs, 1, 1) & np.roll(obs, -1, 1) & np.roll(obs, 1, 2) & np.roll(obs, -1, 2)
sel_all = hidden & nb_obs & np.roll(ocean, 1, 0)[None] & np.roll(ocean, -1, 0)[None] & np.roll(ocean, 1, 1)[None] & np.roll(ocean, -1, 1)[None]
yt = np.where(obs, y, np.nan)
gtx, gty = grads(np.nan_to_num(yt))
rows = {}
for r in names:
    px, py = grads(np.nan_to_num(P[r]))
    s = sel_all[clear_days]; tx, ty = gtx[clear_days][s], gty[clear_days][s]; qx, qy = px[clear_days][s], py[clear_days][s]
    gm_t, gm_p = np.hypot(tx, ty), np.hypot(qx, qy)
    cos = (tx * qx + ty * qy) / (gm_t * gm_p + 1e-9)
    err = (P[r] - y)[clear_days][hidden[clear_days]]
    rows[r] = dict(mae=float(np.abs(err).mean()), bias=float(err.mean()), grad_ratio=float(gm_p.mean() / gm_t.mean()),
                   grad_corr=float(np.corrcoef(np.r_[tx, ty], np.r_[qx, qy])[0, 1]), cos_mean=float(cos.mean()), n=int(s.sum()), n_hidden=int(hidden[clear_days].sum()))
print(f"\nclear val days (obs frac >= {CLEAR}): {len(clear_days)}; hidden px with 4 observed neighbours: {rows[names[0]]['n']}")
print(f"{'run':12s} {'hidden MAE':>10s} {'bias':>8s} {'|grad| pred/truth':>18s} {'grad corr':>10s} {'mean cos(angle)':>16s} {'n hidden px':>11s}")
for r in names:
    q = rows[r]; print(f"{lab(r):12s} {q['mae']:10.4f} {q['bias']:8.4f} {q['grad_ratio']:18.3f} {q['grad_corr']:10.3f} {q['cos_mean']:16.3f} {q['n_hidden']:11d}")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
cen = 0.5 * (edges[:-1] + edges[1:])
for k, r in enumerate(names):
    ax[0].plot(cen, tab[r], marker="o", color=OKABE_ITO[k], label=lab(r))
ax[0].set(xlabel="observed fraction of ocean pixels on the day", ylabel="mean per-day hidden-pixel MAE, ln CHL", title="Val hidden-pixel MAE by the day's cloudiness"); ax[0].legend(); ax[0].grid(alpha=.3)
xi = np.arange(len(names))
ax[1].bar(xi - 0.2, [rows[r]["grad_ratio"] for r in names], 0.4, color=OKABE_ITO[0], label="|∇pred| / |∇truth|")
ax[1].bar(xi + 0.2, [rows[r]["grad_corr"] for r in names], 0.4, color=OKABE_ITO[1], label="corr(∇pred, ∇truth)")
ax[1].set(xticks=xi, xticklabels=[lab(r) for r in names], ylim=(0, 1.05), title=f"Gradient structure on hidden pixels, clear val days (obs frac ≥ {CLEAR}, n={len(clear_days)})"); ax[1].legend(); ax[1].grid(alpha=.3, axis="y")
ax[2].bar(xi, [rows[r]["mae"] for r in names], color=[OKABE_ITO[k] for k in xi])
ax[2].set(xticks=xi, xticklabels=[lab(r) for r in names], ylabel="hidden-pixel MAE, ln CHL", title=f"Hidden-pixel MAE on clear val days (obs frac ≥ {CLEAR})",
          ylim=(min(rows[r]["mae"] for r in names) * 0.97, max(rows[r]["mae"] for r in names) * 1.03)); ax[2].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(FIG / f"pinn_clear_structure{tag}.png", dpi=110); plt.close(fig)

# (3) maps on the two clearest val days, hidden region
top = clear_days[np.argsort(-obs_frac[clear_days])[:2]]
fig, ax = plt.subplots(len(top), 2 + len(names) + 1, figsize=(4.3 * (3 + len(names)), 4.2 * len(top)))
ext = [lon[0], lon[-1], lat[-1], lat[0]]
for r_, k in enumerate(top):
    date = tv[k].strftime("%Y-%m-%d"); hm = hidden[k]
    truth = np.where(obs[k], y[k], np.nan); vmin, vmax = np.nanpercentile(truth, 1), np.nanpercentile(truth, 99)
    panels = [("True ln CHL", truth), ("True ln CHL on hidden (fake-cloud) pixels", np.where(hm, truth, np.nan))] + [(f"Pred on hidden pixels, {lab(r)}", np.where(hm, P[r][k], np.nan)) for r in names]
    for c, (ttl, img) in enumerate(panels):
        im = ax[r_, c].imshow(img, extent=ext, cmap=MAP_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax[r_, c].set(title=f"{ttl}\n{date}, obs frac {obs_frac[k]:.2f}", xlabel="lon (°E)", ylabel="lat (°N)"); fig.colorbar(im, ax=ax[r_, c], label="ln CHL", shrink=.8)
    dif = np.where(hm, P[names[-1]][k] - P[names[0]][k], np.nan); lim = np.nanpercentile(np.abs(dif), 99)
    im = ax[r_, -1].imshow(dif, extent=ext, cmap=ERR_CMAP, vmin=-lim, vmax=lim, aspect="auto")
    ax[r_, -1].set(title=f"pred({lab(names[-1])}) − pred({lab(names[0])}) on hidden pixels\n{date}", xlabel="lon (°E)", ylabel="lat (°N)"); fig.colorbar(im, ax=ax[r_, -1], label="ln CHL", shrink=.8)
fig.tight_layout(); fig.savefig(FIG / f"pinn_clear_maps{tag}.png", dpi=100); plt.close(fig)
print("figures ->", FIG)
