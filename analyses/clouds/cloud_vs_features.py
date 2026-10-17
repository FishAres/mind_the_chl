"""Does real cloud cover (no L3 CHL observation) co-vary with the physical features within a day?

For every 5th day of the 1997-2020 store: z-score each feature over the ocean pixels of that day, then take
mean z under real clouds minus mean z on observed pixels. Also the cloud fraction by within-day feature decile.
"Real cloud" here = no L3 observation on an ocean pixel, which also includes sun glint, aerosol and swath gaps.

  python analyses/clouds/cloud_vs_features.py  -> analyses/clouds/figures/cloud_vs_features.png + table
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chl_cnn import store as S, data as D
from chl_cnn.plotting import OKABE_ITO

ROOT = Path(__file__).resolve().parents[2]
FEATS = ["sst", "air_temp", "u_wind", "v_wind", "wind_speed", "so", "sla", "mlotst", "u_curr", "v_curr", "clim_lnchl"]
d = S.load_store()
ocean = d["ocean"]; n = len(d["time"]); times = pd.to_datetime(d["time"])
days = np.arange(0, n, 5)
y = D.log_chl(np.asarray(d["chl_l3"][days]))
obs = np.isfinite(y) & ocean
cloud = ocean[None] & ~obs
# per-pixel all-year mean ln CHL as a 'where CHL is high' feature
clim = np.nanmean(y, 0)

dz = {f: [] for f in FEATS}; dec = {f: np.zeros((10, 2)) for f in FEATS}
for f in FEATS:
    if f == "wind_speed":
        x = np.hypot(np.asarray(d["u_wind"][days]).astype("float32"), np.asarray(d["v_wind"][days]).astype("float32"))
    elif f == "clim_lnchl":
        x = np.broadcast_to(clim, y.shape)
    else:
        x = np.asarray(d[f][days]).astype("float32")
    for i in range(len(days)):
        xi = x[i]; ok = np.isfinite(xi) & ocean
        c, o = cloud[i] & ok, obs[i] & ok
        if c.sum() < 200 or o.sum() < 200:
            dz[f].append(np.nan); continue
        mu, sd = xi[ok].mean(), xi[ok].std() + 1e-6
        dz[f].append((xi[c].mean() - mu) / sd - (xi[o].mean() - mu) / sd)
        q = np.quantile(xi[ok], np.linspace(0, 1, 11)); which = np.clip(np.digitize(xi, q[1:-1]), 0, 9)
        for b in range(10):
            m = ok & (which == b)
            dec[f][b, 0] += (m & cloud[i]).sum(); dec[f][b, 1] += m.sum()
    dz[f] = np.array(dz[f])

print(f"days used {len(days)}; mean real-cloud fraction {cloud[:, ocean].mean():.3f}")
print(f"{'feature':12s} {'mean dz (cloud - clear)':>24s} {'5th-95th pct over days':>24s} {'days dz>0':>10s} {'cloud frac decile1':>19s} {'decile10':>9s} {'n days':>7s}")
rows = []
for f in FEATS:
    v = dz[f]; ok = np.isfinite(v)
    cf = dec[f][:, 0] / np.maximum(dec[f][:, 1], 1)
    rows.append((f, np.nanmean(v), np.nanpercentile(v, 5), np.nanpercentile(v, 95), np.mean(v[ok] > 0), cf[0], cf[-1], ok.sum()))
    print(f"{f:12s} {np.nanmean(v):24.3f} {np.nanpercentile(v, 5):11.2f}..{np.nanpercentile(v, 95):10.2f} {np.mean(v[ok] > 0):10.2f} {cf[0]:19.3f} {cf[-1]:9.3f} {ok.sum():7d}")

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
xi = np.arange(len(FEATS))
ax[0].bar(xi, [r[1] for r in rows], color=OKABE_ITO[0])
ax[0].errorbar(xi, [r[1] for r in rows], yerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]], fmt="none", ecolor="k", capsize=3, lw=1)
ax[0].axhline(0, color="k", lw=.8)
ax[0].set(xticks=xi, xticklabels=FEATS, ylabel="mean z under real clouds − mean z on observed pixels",
          title="Within-day standardised difference of each feature, cloudy vs observed pixels\n(bar = mean over days, whiskers = 5th-95th percentile of days)")
ax[0].tick_params(axis="x", rotation=35); ax[0].grid(alpha=.3, axis="y")
for k, f in enumerate(FEATS):
    cf = dec[f][:, 0] / np.maximum(dec[f][:, 1], 1)
    ax[1].plot(np.arange(1, 11), cf, marker="o", ms=3, color=OKABE_ITO[k % 8], ls="-" if k < 8 else "--", label=f)
ax[1].axhline(cloud[:, ocean].mean(), color="k", ls=":", label="overall cloud fraction")
ax[1].set(xlabel="within-day decile of the feature (1 = lowest)", ylabel="fraction of pixels under real cloud", title="Real-cloud fraction by within-day feature decile, 1997-2020 (every 5th day)")
ax[1].legend(fontsize=8, ncol=2); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent / "figures" / "cloud_vs_features.png", dpi=110)
