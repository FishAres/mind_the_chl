"""Climatology hidden-pixel MAE by region x season on val days (every 4th) and observed-pixel coverage by region x month."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

if __name__ == "__main__":
    d = S.load_store(); reg, names = load_regions(); times = d["time"]
    c = train_climatology(d)
    clim, b, va, y, obs, hid_all = c["clim"], c["bin"], c["va"], c["y"], c["obs"], c["hid"]
    print(f"days {len(times)}; train {len(c['tr'])}; val {len(va)}; val every 4th: {len(va[::4])}")
    print(f"ocean px {int(d['ocean'].sum())}; all-NaN-CHL days: {int((~obs.any((1, 2))).sum())}; chl_l3 values <=0 on ocean (share of finite): see below")
    # data-quality: non-positive chl, nan fraction on ocean px
    ch = np.asarray(d["chl_l3"][::40]); oc = d["ocean"]
    f = ch[:, oc]; print(f"chl_l3 (every 40th day) ocean px: NaN {np.isnan(f).mean():.3f}, <=0 {(f <= 0).sum()}, max {np.nanmax(f):.1f} mg/m3, p99.9 {np.nanpercentile(f, 99.9):.1f}")
    # val every 4th
    sub = np.arange(0, len(va), 4); vi = va[sub]; hid = hid_all[sub]
    err = np.abs(clim[b[vi]] - y[vi])
    res = score_by_region_season(err, hid, reg, times[vi], names)
    print(md_table(res, names, "Climatology hidden-pixel MAE (ln CHL), val days every 4th; cell = MAE (n hidden px)"))
    # coverage by region x month: fraction of ocean px observed, over all days (every 4th)
    allv = np.arange(0, len(times), 4); mon = months(times[allv]); cov = np.full((len(names), 12), np.nan); ncov = np.zeros((len(names), 12), int)
    ob = obs[allv]
    for ri, r in enumerate(names):
        rm = reg == r; per_day = ob[:, rm].mean(1)
        for m in range(1, 13):
            sel = mon == m; cov[ri, m - 1] = per_day[sel].mean(); ncov[ri, m - 1] = sel.sum()
    print("\n**Observed-pixel fraction by region x month (every 4th day, all years); n days per month = " + str(ncov[0].tolist()) + "**\n")
    print("| region | " + " | ".join(str(m) for m in range(1, 13)) + " |\n|---|" + "---|" * 12)
    for ri, r in enumerate(names):
        print(f"| {r} {names[r]} | " + " | ".join(f"{v:.2f}" for v in cov[ri]) + " |")
    # also: store keeps only days with >=1 obs; fraction of ocean obs per day
    print(f"\nall-zero-obs val days: {int((~obs[vi].any((1, 2))).sum())}; hidden px per val day mean {hid.sum((1, 2)).mean():.0f}")
    # figure
    mae = np.array([[res[(r, s)][0] for s in SEAS] for r in names]); nn = np.array([[res[(r, s)][1] for s in SEAS] for r in names])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw=dict(width_ratios=[1, 1.4]))
    im = a1.imshow(np.where(nn >= 5, mae, np.nan), cmap="viridis", aspect="auto")
    a1.set_xticks(range(4)); a1.set_xticklabels(SEAS); a1.set_yticks(range(len(names))); a1.set_yticklabels([f"{r} {names[r]}" for r in names], fontsize=8)
    for i in range(len(names)):
        for j in range(4):
            if nn[i, j] >= 5: a1.text(j, i, f"{mae[i, j]:.2f}\n(n={nn[i, j]})", ha="center", va="center", color="w", fontsize=7)
    plt.colorbar(im, ax=a1, label="Climatology MAE on hidden pixels (ln CHL)")
    a1.set_title("DOY-climatology hidden-pixel MAE by region and season (val days, every 4th)"); a1.set_xlabel("Season"); a1.set_ylabel("Region")
    for ri, r in enumerate(names):
        a2.plot(range(1, 13), cov[ri], marker="o", color=OKABE_ITO[ri], label=f"{r} {names[r]}")
    a2.set_xlabel("Month"); a2.set_ylabel("Fraction of region's ocean pixels with CHL observed"); a2.set_xticks(range(1, 13))
    a2.set_title("Observed-pixel fraction by month and region (every 4th day, 2014-2024)"); a2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "clim_by_region.png", dpi=130)
