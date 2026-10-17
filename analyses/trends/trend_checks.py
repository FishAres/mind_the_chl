"""Checks on the long-term rise in SST / air temp / SLA ocean means.

Question: are the rises in the ocean-mean series a data artefact?
Checks (no interpretation here):
  1. coverage: is the set of ocean pixels with data stable over the years?
  2. deseasonalised annual anomalies + OLS trend, residuals (look for steps)
  3. per-pixel trend maps (is the rise spatially widespread or localised?)
  4. SLA vs ADT (ADT = SLA + fixed mean dynamic topography -> trends should match)

Assumptions:
  - anomalies = monthly mean minus that calendar month's mean over the same period
    (removes the seasonal cycle, so ERA5's missing months don't bias annual means)
  - a year is used only if it has >= 9 months with data
  - OLS standard errors ignore autocorrelation (under-estimate uncertainty)

Run: python analyses/trends/trend_checks.py   -> figures in analyses/trends/figures/
"""
import sys
from pathlib import Path

import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import io_summaries as S  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
OKABE = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", blue="#0072B2", verm="#D55E00", purple="#CC79A7", black="#000000")
ERA5_SST_SWITCH = pd.Timestamp("2007-09-01")  # ERA5 SST boundary input: HadISST2 -> OSTIA
SERIES = [  # var, label, offset, periods
    ("sst", "SST (ERA5)", -273.15, [("1979", "2022"), ("1993", "2022")]),
    ("air_temp", "2 m air temp (ERA5)", -273.15, [("1979", "2022"), ("1993", "2022")]),
    ("sla", "Sea level anomaly (CMEMS)", 0, [("1993", "2022")]),
    ("adt", "Absolute dynamic topography (CMEMS)", 0, [("1993", "2022")]),
]
MIN_MONTHS = 9


# ---------------------------------------------------------------- data
def monthly_anoms(series, start, end):
    m = series.loc[start:end].resample("MS").mean()
    return m - m.groupby(m.index.month).transform("mean")


def annual(anom):
    g = anom.groupby(anom.index.year)
    a = g.mean()
    return a[g.count() >= MIN_MONTHS]


def ols(y):
    x = y.index.values.astype(float)
    X = np.c_[np.ones_like(x), x]
    beta, res, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    resid = y.values - X @ beta
    se = np.sqrt(resid @ resid / (len(y) - 2) / ((x - x.mean()) ** 2).sum())
    return beta[1], se, pd.Series(resid, index=y.index)


def pixel_trend(da, start, end):
    """Per-pixel OLS slope of annual-mean deseasonalised anomalies (lazy until compute)."""
    m = da.sel(time=slice(start, end)).resample(time="MS").mean()
    anom = m.groupby("time.month") - m.groupby("time.month").mean()
    cnt = anom.notnull().groupby("time.year").sum()
    ann = anom.groupby("time.year").mean().where(cnt >= MIN_MONTHS)
    fit = ann.polyfit("year", deg=1, skipna=True)
    return fit.polyfit_coefficients.sel(degree=1)


# ---------------------------------------------------------------- analysis
def main():
    daily = xr.open_zarr(S.OUT, group="daily").load()
    ds = S.open_io()
    ocean = S.ocean_mask(ds).compute()

    # 1. coverage (days with any data only; ERA5 missing months excluded)
    print("## 1. Ocean coverage per year (fraction of ocean pixels with data, days with data only)")
    rows = []
    for v, lab, _, periods in SERIES:
        val = daily[f"{v}__valid"].to_series()
        val = val[val > 0].loc[periods[0][0]:periods[0][1]]
        yr = val.groupby(val.index.year).mean()
        rows.append(dict(variable=v, min_year_frac=yr.min(), max_year_frac=yr.max(), n_years=len(yr), n_days=len(val)))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # 2. annual anomalies + trends
    print("\n## 2. OLS trend of annual-mean deseasonalised ocean-mean anomalies")
    anoms, rows, resids = {}, [], {}
    for v, lab, off, periods in SERIES:
        s = daily[f"{v}__mean"].to_series() + off
        for a, b in periods:
            ann = annual(monthly_anoms(s, a, b))
            slope, se, r = ols(ann)
            unit = "°C" if off else "mm"
            k = 10 if off else 1000  # °C/decade or mm/yr
            rows.append(dict(variable=v, period=f"{a}-{b}", trend=round(slope * k, 3),
                             se=round(se * k, 3), unit=f"{unit}/decade" if off else "mm/yr",
                             total_change=round(slope * (int(b) - int(a)) * (1 if off else 1000), 3),
                             n_years=len(ann), years_dropped=(int(b) - int(a) + 1) - len(ann)))
            anoms[(v, a)] = ann
            resids[(v, a)] = r
    trends = pd.DataFrame(rows)
    print(trends.to_string(index=False))
    sla = daily["sla__mean"].to_series()
    print(f"\nSLA mean over 1993-2012 (CMEMS reference period): {sla.loc['1993':'2012'].mean()*1000:.1f} mm "
          f"(n = {sla.loc['1993':'2012'].notna().sum()} days)")

    # 3. per-pixel trend maps (one read of each variable)
    print("\n## 3. per-pixel trends (computing...)")
    maps = dict(
        sst_79=pixel_trend(ds.sst, "1979", "2022") * 10,
        sst_93=pixel_trend(ds.sst, "1993", "2022") * 10,
        sla_93=pixel_trend(ds.sla, "1993", "2022") * 1000,
    )
    maps = dict(zip(maps, dask.compute(*maps.values())))
    maps = {k: m.where(ocean) for k, m in maps.items()}
    for k, m in maps.items():
        vals = m.values[np.isfinite(m.values)]
        print(f"{k}: n pixels {vals.size}, frac > 0 {np.mean(vals > 0):.3f}, "
              f"median {np.median(vals):.3f}, 5-95% [{np.percentile(vals, 5):.3f}, {np.percentile(vals, 95):.3f}]")

    # ------------------------------------------------------------ reporting
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (v, lab, off, periods) in zip(axs, SERIES[:3]):
        a0 = periods[0][0]
        ann = anoms[(v, a0)]
        slope, se, _ = ols(ann)
        unit = "°C" if off else "m"
        ax.plot(ann.index, ann, "o-", color=OKABE["blue"], ms=4, label="annual mean anomaly")
        ax.plot(ann.index, ann.index * slope + (ann.mean() - slope * ann.index.values.mean()), color=OKABE["verm"],
                label=f"OLS {a0}–2022")
        if off:
            ax.axvline(ERA5_SST_SWITCH.year + 8 / 12, color="0.4", ls=":", label="ERA5 SST input change (Sep 2007)")
        ax.set(ylabel=f"anomaly ({unit})", title=f"{lab}: ocean-mean annual anomaly (n = {len(ann)} years, ≥{MIN_MONTHS} months each)")
        ax.legend(fontsize=8, loc="upper left")
    axs[-1].set_xlabel("year")
    plt.tight_layout(); fig.savefig(FIG / "annual_anomalies.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    for ax, key, lab in zip(axs, [("sst", "1979"), ("air_temp", "1979"), ("sla", "1993")],
                            ["SST (°C)", "2 m air temp (°C)", "SLA (m)"]):
        r = resids[key]
        ax.bar(r.index, r, color=[OKABE["verm"] if x > 0 else OKABE["blue"] for x in r])
        if key[0] != "sla":
            ax.axvline(2007.67, color="0.4", ls=":")
        ax.set(title=f"{lab}: residual from linear trend (n = {len(r)} years)", xlabel="year", ylabel=f"residual {lab.split()[-1]}")
        ax.title.set_fontsize(9)
    plt.tight_layout(); fig.savefig(FIG / "trend_residuals.png", dpi=110); plt.close(fig)

    land = (ds["CHL_cmes-land"] == 2).load()
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (k, title, unit, lim) in zip(axs, [
        ("sst_79", "SST trend per pixel, 1979–2022", "°C / decade", 0.4),
        ("sst_93", "SST trend per pixel, 1993–2022", "°C / decade", 0.4),
        ("sla_93", "SLA trend per pixel, 1993–2022", "mm / yr", 8),
    ]):
        im = maps[k].plot(ax=ax, cmap="RdBu_r", vmin=-lim, vmax=lim, add_colorbar=False)
        plt.colorbar(im, ax=ax, label=unit, shrink=0.85)
        ax.contour(land.lon, land.lat, land.astype(float), levels=[0.5], colors="k", linewidths=0.5)
        n = int(np.isfinite(maps[k]).sum())
        ax.set(title=f"{title} (n = {n} ocean pixels)", xlabel="longitude (°E)", ylabel="latitude (°N)", aspect="equal")
        ax.set_facecolor("0.85"); ax.title.set_fontsize(10)
    plt.tight_layout(); fig.savefig(FIG / "trend_maps.png", dpi=110); plt.close(fig)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
