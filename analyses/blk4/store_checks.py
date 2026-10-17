"""Sanity checks of the 4 km Black Sea store against the 2 km store; figure analyses/blk4/store_coverage.png."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from chl_cnn import store_blk as SB, store_blk4 as S4

# --- data
d2, d4 = SB.load_store(), S4.load_store()
o2, o4 = d2["ocean"].astype(bool), d4["ocean"].astype(bool)
n = len(d2["time"]); assert (d2["time"] == d4["time"]).all()
cov2, cov4, ml2, ml4 = (np.zeros(n) for _ in range(4))
for b in range(0, n, 100):
    for d, o, cov, ml in [(d2, o2, cov2, ml2), (d4, o4, cov4, ml4)]:
        c = np.asarray(d["chl_l3"][b:b + 100], dtype="float32")[:, o]
        with np.errstate(divide="ignore", invalid="ignore"):
            l = np.log(c)
        ok = np.isfinite(l); cov[b:b + 100] = ok.mean(1)
        with np.errstate(invalid="ignore"):
            ml[b:b + 100] = np.where(ok, l, 0).sum(1) / ok.sum(1)
# --- analysis
print(f"ocean px 2 km {o2.sum()}  4 km {o4.sum()}  grid 4 km {o4.shape}")
print(f"mean coverage (fraction of ocean px observed) 2 km {cov2.mean():.4f}  4 km {cov4.mean():.4f}  ratio {cov4.mean() / cov2.mean():.3f}")
print(f"days with >=1 obs: 2 km {(cov2 > 0).sum()}  4 km {(cov4 > 0).sum()} of {n}")
print(f"Pearson r(coverage 2 km, 4 km) over days = {np.corrcoef(cov2, cov4)[0, 1]:.4f}")
rng = np.random.default_rng(0); days = rng.choice(np.flatnonzero(cov4 > 0.05), 5, replace=False)
print("random days: date  mean ln CHL 2 km  4 km  diff")
for t in sorted(days):
    print(f"  {str(d2['time'][t])[:10]}  {ml2[t]:.4f}  {ml4[t]:.4f}  {ml4[t] - ml2[t]:+.4f}")
both = (cov4 > 0)
dd = ml4[both] - ml2[both]
print(f"all {both.sum()} days with 4 km obs: |diff| mean {np.abs(dd).mean():.4f} max {np.abs(dd).max():.4f}; days |diff|>0.05: {(np.abs(dd) > 0.05).sum()}")
for v in S4.FEATS:
    a2 = np.zeros(n, bool); a4 = np.zeros(n, bool); nn2 = np.zeros(n); nn4 = np.zeros(n)
    for b in range(0, n, 200):
        x2 = np.asarray(d2[v][b:b + 200])[:, o2]; x4 = np.asarray(d4[v][b:b + 200])[:, o4]
        a2[b:b + 200] = np.isnan(x2).all(1); a4[b:b + 200] = np.isnan(x4).all(1)
        nn2[b:b + 200] = np.isnan(x2).mean(1); nn4[b:b + 200] = np.isnan(x4).mean(1)
    print(f"{v:7s} all-NaN-on-ocean days 2 km {a2.sum()}  4 km {a4.sum()}  new at 4 km {(a4 & ~a2).sum()}  "
          f"mean NaN frac on ocean 2 km {nn2.mean():.4f} 4 km {nn4.mean():.4f}")
# --- report
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(cov2, cov4, s=3, alpha=0.4, color="#0072B2"); m = max(cov2.max(), cov4.max())
ax[0].plot([0, m], [0, m], "k--", lw=0.8, label="y = x"); ax[0].legend()
ax[0].set(xlabel="2 km: fraction of ocean pixels with CHL", ylabel="4 km: fraction of ocean pixels with CHL",
          title=f"Daily CHL coverage, 4 km vs 2 km store (n={n} days)")
ax[1].hist(dd, bins=80, color="#E69F00")
ax[1].set(xlabel="4 km minus 2 km mean ln CHL over observed ocean pixels", ylabel="days",
          title=f"Daily mean ln CHL difference (n={both.sum()} days)")
plt.tight_layout(); f = Path(__file__).with_name("store_coverage.png"); plt.savefig(f, dpi=120); print("figure", f)
