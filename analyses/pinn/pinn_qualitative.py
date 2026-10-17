"""Qualitative comparison of PINN predictions across lambda on the val days, from the saved preds.npz
(full fields incl. under real clouds). No retraining.

  python analyses/pinn/pinn_qualitative.py   -> analyses/pinn/figures/pinn_qual_*.png + tables on stdout

Checks
  1 where predictions differ: per-pixel mean |pred_lam - pred_0|, split under real clouds / observed
  2 smoothness: mean gradient magnitude per lambda (cloud / observed), radial power spectrum in a land-free box
  3 temporal coherence: |pred_t+1 - pred_t| on pixels cloudy on both consecutive days (truth on observed pairs as reference)
  4 persistence vs advection: corr of the predicted day-to-day change under clouds with the advective change
    (yesterday's field back-tracked one day along the currents) and with the climatological change
  5 case days: largest real-cloud cover over the Somali (July) and Oman (August) upwelling boxes
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chl_cnn import store as S, data as D, train2 as T2
from chl_cnn.physics import grid_metrics
from chl_cnn.plotting import OKABE_ITO, MAP_CMAP, ERR_CMAP
from chl_cnn.runs import find_run

ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent / "figures"
RUNS = {0.0: "pinn_lam0", 0.3: "pinn_lam0p3", 1.0: "pinn_lam1p0"}
DT = 86400.0

d = S.load_store()
times = pd.to_datetime(d["time"]); T2.TIMES = d["time"]
lat, lon, ocean = d["lat"].astype("float64"), d["lon"].astype("float64"), d["ocean"]
H, W = ocean.shape
tr, va, te = D.split_by_years(d["time"])
y = D.log_chl(np.asarray(d["chl_l3"]))
obs = np.isfinite(y) & ocean
clim, dbin = T2.doy_climatology(y, obs, tr)
P = {}
for lam, r in RUNS.items():
    z = np.load(find_run(r) / "preds.npz")
    assert np.array_equal(z["val_idx"], va)
    P[lam] = z["pred"][:len(va)]
yv, obsv = y[va], obs[va]
cloud = ocean[None] & ~obsv                     # under real clouds (no truth)
tv = times[va]
ext = [lon[0], lon[-1], lat[-1], lat[0]]
print(f"val days {len(va)}; ocean px {ocean.sum()}; mean real-cloud fraction {cloud[:, ocean].mean():.3f}")


def stat(name, vals):
    print(f"{name:60s} " + "  ".join(f"lam {k:g}: {v:.4f}" for k, v in vals.items()))


# ---------- 1. where do predictions differ
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
diff_tab = {}
for k, lam in enumerate([0.3, 1.0]):
    dif = np.abs(P[lam] - P[0.0])
    m = np.nanmean(dif, 0); m[~ocean] = np.nan
    im = ax[k].imshow(m, extent=ext, cmap="viridis", vmin=0, vmax=np.nanpercentile(m, 99), aspect="auto")
    ax[k].set(title=f"Mean |pred(lambda={lam:g}) − pred(lambda=0)| over val days", xlabel="lon (°E)", ylabel="lat (°N)")
    fig.colorbar(im, ax=ax[k], label="ln CHL", shrink=.85)
    diff_tab[lam] = dict(cloud=float(np.nanmean(dif[cloud])), observed=float(np.nanmean(dif[obsv])))
    ax[2].hist(dif[cloud], bins=np.linspace(0, 0.6, 61), density=True, histtype="step", lw=1.5, color=OKABE_ITO[k], label=f"lambda {lam:g}, under real clouds")
    ax[2].hist(dif[obsv], bins=np.linspace(0, 0.6, 61), density=True, histtype="step", lw=1.5, ls="--", color=OKABE_ITO[k], label=f"lambda {lam:g}, observed pixels")
ax[2].set(title="Distribution of |pred(lambda) − pred(lambda=0)|, val days", xlabel="ln CHL", ylabel="density", yscale="log"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "pinn_qual_difference.png", dpi=110); plt.close(fig)
stat("1. mean |pred - pred_lam0| under real clouds", {l: v["cloud"] for l, v in diff_tab.items()})
stat("1. mean |pred - pred_lam0| on observed pixels", {l: v["observed"] for l, v in diff_tab.items()})

# ---------- 2. smoothness
dx, dy, _, _ = grid_metrics(lat, lon)
def grad_mag(c):                                  # per 100 km
    gx = (np.roll(c, -1, -1) - np.roll(c, 1, -1)) / (2 * np.abs(dx)) * 1e5
    gy = (np.roll(c, -1, -2) - np.roll(c, 1, -2)) / (2 * np.abs(dy)) * 1e5
    g = np.hypot(gx, gy); g[..., ~ocean] = np.nan
    return g
inner = ocean & np.roll(ocean, 1, 0) & np.roll(ocean, -1, 0) & np.roll(ocean, 1, 1) & np.roll(ocean, -1, 1)
gtab = {}
for lam in RUNS:
    g = grad_mag(P[lam])
    gtab[lam] = dict(cloud=float(np.nanmean(g[cloud & inner[None]])), observed=float(np.nanmean(g[obsv & inner[None]])))
gt = grad_mag(np.where(obsv, yv, np.nan))       # truth gradient where all 5 stencil pixels are observed
stat("2. mean |grad pred| (ln CHL per 100 km) under real clouds", {l: v["cloud"] for l, v in gtab.items()})
stat("2. mean |grad pred| (ln CHL per 100 km) on observed pixels", {l: v["observed"] for l, v in gtab.items()})
print(f"{'2. mean |grad truth| on fully observed stencils':60s} {np.nanmean(gt):.4f}")
# radial power spectrum in a land-free box
i0, i1 = np.searchsorted(-lat, [-19.25, -4.5]); j0, j1 = np.searchsorted(lon, [58, 72.75])   # largest land-free square: 4.5-19.25N, 58-72.75E
box = (slice(i0, i1 + 1), slice(j0, j1 + 1))
assert ocean[box].all(), "box has land"
def radial_psd(fields):
    n = min(fields.shape[-2:]); f = fields[..., :n, :n]
    f = f - f.mean(axis=(-2, -1), keepdims=True)
    F = np.abs(np.fft.fftshift(np.fft.fft2(f), axes=(-2, -1))) ** 2
    ky, kx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(n)), np.fft.fftshift(np.fft.fftfreq(n)), indexing="ij")
    kr = np.hypot(kx, ky); bins = np.arange(0.5, n // 2 + 1) / n
    which = np.digitize(kr, bins)
    ps = np.array([F[..., which == b].mean(-1) for b in range(1, len(bins))]).T   # (days, nb)
    return bins[:-1] + 0.5 / n, ps.mean(0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for k, lam in enumerate(RUNS):
    kk, ps = radial_psd(P[lam][(slice(None),) + box])
    ax[0].loglog(kk / 27.8, ps, color=OKABE_ITO[k], label=f"prediction, lambda {lam:g}")
yt = np.where(obsv, yv, np.nan)[(slice(None),) + box]
full = np.isfinite(yt).all(axis=(1, 2))
if full.sum() >= 5:
    kk, ps = radial_psd(yt[full]); ax[0].loglog(kk / 27.8, ps, color="k", ls="--", label=f"truth, fully observed box days (n={full.sum()})")
ax[0].set(xlabel="wavenumber (cycles per km)", ylabel="power (ln CHL²)", title="Radial power spectrum of ln CHL, box 4.5-19.25°N 58-72.75°E, val days"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
xi = np.arange(len(RUNS))
ax[1].bar(xi - 0.2, [gtab[l]["cloud"] for l in RUNS], 0.4, color=OKABE_ITO[0], label="under real clouds")
ax[1].bar(xi + 0.2, [gtab[l]["observed"] for l in RUNS], 0.4, color=OKABE_ITO[1], label="observed pixels")
ax[1].axhline(np.nanmean(gt), color="k", ls="--", label="truth, fully observed stencils")
ax[1].set(xticks=xi, xticklabels=[f"lambda {l:g}" for l in RUNS], ylabel="mean |∇ ln CHL| (per 100 km)", title="Gradient magnitude of predicted ln CHL, val days"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(FIG / "pinn_qual_smoothness.png", dpi=110); plt.close(fig)

# ---------- 3 + 4. temporal coherence and persistence vs advection on consecutive val days
pos = {int(i): k for k, i in enumerate(va)}
pairs = [(k, k + 1) for k in range(len(va) - 1) if (tv[k + 1] - tv[k]).days == 1]
uc = d["u_curr"]; vc = d["v_curr"]
jj, ii = np.meshgrid(np.arange(W), np.arange(H))
coh = {lam: [] for lam in RUNS}; coh_truth = []
corr_adv = {lam: [] for lam in RUNS}; corr_clim = {lam: [] for lam in RUNS}; corr_pers = {lam: [] for lam in RUNS}
sd_change = {lam: [] for lam in RUNS}
for k0, k1 in pairs:
    both_cloud = cloud[k0] & cloud[k1]
    both_obs = obsv[k0] & obsv[k1]
    coh_truth.append(np.abs(yv[k1] - yv[k0])[both_obs].mean() if both_obs.sum() > 100 else np.nan)
    u = np.asarray(uc[va[k1]]).astype("float32"); v = np.asarray(vc[va[k1]]).astype("float32")
    okc = np.isfinite(u) & np.isfinite(v)
    a_ = np.nan_to_num(u * DT / dx); b_ = np.nan_to_num(v * DT / dy)        # cells/day along index axes
    dclim = clim[dbin[va[k1]]] - clim[dbin[va[k0]]]
    sel = both_cloud & okc & inner
    for lam in RUNS:
        c0, c1 = P[lam][k0], P[lam][k1]
        coh[lam].append(np.abs(c1 - c0)[both_cloud].mean() if both_cloud.sum() > 100 else np.nan)
        if sel.sum() < 500:
            for dct in (corr_adv, corr_clim, corr_pers, sd_change): dct[lam].append(np.nan)
            continue
        fill = np.where(ocean, c0, np.nanmean(c0[ocean]))
        adv = map_coordinates(fill, [ii - b_, jj - a_], order=1, mode="nearest")   # back-tracked field
        change = (c1 - c0)[sel]; dadv = (adv - c0)[sel]; dcl = dclim[sel]
        corr_adv[lam].append(np.corrcoef(change, dadv)[0, 1]); corr_clim[lam].append(np.corrcoef(change, dcl)[0, 1])
        corr_pers[lam].append(float(np.mean(np.abs(change) < 0.02)))
        sd_change[lam].append(change.std())
print(f"consecutive val-day pairs: {len(pairs)}")
stat("3. mean |pred_t+1 - pred_t| on pixels cloudy both days", {l: np.nanmean(coh[l]) for l in RUNS})
print(f"{'3. mean |truth_t+1 - truth_t| on pixels observed both days':60s} {np.nanmean(coh_truth):.4f}")
stat("4. corr(pred change under clouds, advective change), median", {l: np.nanmedian(corr_adv[l]) for l in RUNS})
stat("4. corr(pred change under clouds, climatological change), median", {l: np.nanmedian(corr_clim[l]) for l in RUNS})
stat("4. sd of pred change under clouds (ln CHL/day), median", {l: np.nanmedian(sd_change[l]) for l in RUNS})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
bins = np.linspace(-0.4, 0.8, 49)
for k, lam in enumerate(RUNS):
    ax[0].hist(corr_adv[lam], bins=bins, histtype="step", lw=1.5, color=OKABE_ITO[k], label=f"lambda {lam:g}")
    ax[1].hist(corr_clim[lam], bins=bins, histtype="step", lw=1.5, color=OKABE_ITO[k], label=f"lambda {lam:g}")
ax[0].set(title="Corr of predicted change under clouds\nwith advective change, per val-day pair", xlabel="correlation (one value per consecutive val-day pair)", ylabel="days"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set(title="Corr of predicted change under clouds\nwith climatological change, per val-day pair", xlabel="correlation (one value per consecutive val-day pair)", ylabel="days"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[2].bar(xi, [np.nanmean(coh[l]) for l in RUNS], color=[OKABE_ITO[k] for k in xi])
ax[2].axhline(np.nanmean(coh_truth), color="k", ls="--", label="truth, pixels observed both days")
ax[2].set(xticks=xi, xticklabels=[f"lambda {l:g}" for l in RUNS], ylabel="mean |ln CHL(t+1) − ln CHL(t)|", title="Day-to-day change of predictions\nunder persistent real clouds"); ax[2].legend(); ax[2].grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(FIG / "pinn_qual_temporal.png", dpi=110); plt.close(fig)

# ---------- 5. case days
boxes = {"Somali coast (5-12N, 48-56E), July": ((5, 12, 48, 56), 7), "Oman coast (16-23N, 55-62E), August": ((16, 23, 55, 62), 8)}
fig, ax = plt.subplots(len(boxes), 5, figsize=(24, 4.3 * len(boxes)))
for r, (name, ((la0, la1, lo0, lo1), month)) in enumerate(boxes.items()):
    bi = (lat >= la0) & (lat <= la1); bj = (lon >= lo0) & (lon <= lo1)
    bm = ocean & bi[:, None] & bj[None, :]
    cand = np.flatnonzero(tv.month == month)
    cf = cloud[cand][:, bm].mean(1); of = obsv[cand][:, bm].mean(1)
    ok = (of > 0.1)                                               # keep some truth in the box
    k = cand[ok][np.argmax(cf[ok])]
    date = tv[k].strftime("%Y-%m-%d")
    sub = (slice(np.flatnonzero(bi).min() - 8, np.flatnonzero(bi).max() + 9), slice(np.flatnonzero(bj).min() - 8, np.flatnonzero(bj).max() + 9))
    e2 = [lon[sub[1]][0], lon[sub[1]][-1], lat[sub[0]][-1], lat[sub[0]][0]]
    truth = np.where(obsv[k], yv[k], np.nan)[sub]
    vmin, vmax = np.nanpercentile(P[0.0][k][sub], 1), np.nanpercentile(P[0.0][k][sub], 99)
    panels = [("True ln CHL (observed pixels)", truth)] + [(f"Predicted ln CHL, lambda {lam:g}", P[lam][k][sub]) for lam in RUNS]
    for c, (ttl, img) in enumerate(panels):
        im = ax[r, c].imshow(img, extent=e2, cmap=MAP_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax[r, c].set(title=f"{ttl}\n{name.split(',')[0]}, {date}", xlabel="lon (°E)", ylabel="lat (°N)"); fig.colorbar(im, ax=ax[r, c], label="ln CHL", shrink=.8)
        ax[r, c].contour(lon[sub[1]], lat[sub[0]], cloud[k][sub].astype(float), levels=[0.5], colors="w", linewidths=0.5)
    dif = (P[1.0][k] - P[0.0][k])[sub]; lim = np.nanpercentile(np.abs(dif), 99)
    im = ax[r, 4].imshow(dif, extent=e2, cmap=ERR_CMAP, vmin=-lim, vmax=lim, aspect="auto")
    ax[r, 4].set(title=f"pred(lambda 1) − pred(lambda 0)\n{name.split(',')[0]}, {date}", xlabel="lon (°E)", ylabel="lat (°N)"); fig.colorbar(im, ax=ax[r, 4], label="ln CHL", shrink=.8)
    ax[r, 4].contour(lon[sub[1]], lat[sub[0]], cloud[k][sub].astype(float), levels=[0.5], colors="k", linewidths=0.5)
    print(f"5. case day {name}: {date}, real-cloud fraction in box {cf[ok].max():.2f}, mean |pred1 - pred0| in box under clouds {np.nanmean(np.abs(P[1.0][k] - P[0.0][k])[bm & cloud[k]]):.4f}")
fig.suptitle("Case days: predictions by lambda under large real clouds (white/black contour = real cloud edge)"); fig.tight_layout()
fig.savefig(FIG / "pinn_qual_cases.png", dpi=100); plt.close(fig)
print("figures ->", FIG)
