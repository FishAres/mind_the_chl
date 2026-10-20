"""Checks of rsntds.npy and friver.npy in the 4 km Black Sea store (chl_cnn/store_blk4_extra.py).
Figure analyses/blk4/figures/extra_vars_check.png; numbers printed.
Assumptions: basin mean = mean over store ocean px with finite values; friver pixels are attributed to the named mouth
(river_dist.npy / river_names.json) whose snapped mouth pixel is nearest, 'other' if > 40 km from all of them;
volume flux = mass flux / 1000 kg m-3."""
import json
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from chl_cnn import plotting as P                      # noqa: E402  (Okabe-Ito cycle, viridis maps)
from chl_cnn import store_blk4 as S4                    # noqa: E402
from chl_cnn.store_blk4_extra import cell_area          # noqa: E402

FIG = ROOT / "analyses/blk4/figures/extra_vars_check.png"
DIAG = ROOT / "analyses/blk4/results/extra_vars_friver_totals.npz"
NAMED_RADIUS_KM = 40


def load():
    p = S4.store_path()
    d = {k: np.load(p / f"{k}.npy") for k in ["time", "lat", "lon", "ocean", "river_dist"]}
    for k in ["rsntds", "friver", "mlotst"]:
        d[k] = np.load(p / f"{k}.npy", mmap_mode="r")
    d["names"] = json.load(open(p / "river_names.json"))
    return d


def analyse(d):
    t = pd.DatetimeIndex(d["time"]); oc = d["ocean"]; lat, lon = d["lat"], d["lon"]
    r = {}
    for v in ["rsntds", "friver", "mlotst"]:
        x = np.asarray(d[v][:, oc], dtype="float32")
        r[v] = dict(dtype=str(d[v].dtype), shape=d[v].shape, nan_frac_ocean=float(np.isnan(x).mean()),
                    all_nan_days=int((~np.isfinite(x).any(1)).sum()), min=float(np.nanmin(x)), max=float(np.nanmax(x)),
                    nan_px_always=int((~np.isfinite(x).any(0)).sum()), n_ocean_px=int(oc.sum()))
    rs = np.asarray(d["rsntds"], dtype="float32")
    r["rs_basin"] = np.nanmean(np.where(oc, rs, np.nan), (1, 2))
    r["rs_doy"] = pd.Series(r["rs_basin"], index=t).groupby(t.dayofyear).agg(["mean", "std", "count"])
    r["rs_map"] = np.where(oc, np.nanmean(rs, 0), np.nan); del rs
    fr = np.asarray(d["friver"], dtype="float64")
    a4 = cell_area(lat, abs(float(np.diff(lat).mean())), abs(float(np.diff(lon).mean())))[:, None] * np.ones((1, len(lon)))
    r["fr_map"] = np.nanmean(fr, 0)
    nz = (np.nan_to_num(fr) > 0).any(0)
    r["fr_nz"] = nz; r["fr_nz_off_ocean"] = int((nz & ~oc).sum())
    mass = np.nan_to_num(fr[:, nz]) * a4[nz]                          # (t, k) kg s-1
    mouth_ij = [np.unravel_index(np.nanargmin(m), m.shape) for m in d["river_dist"]]
    ki, kj = np.nonzero(nz)
    dist = np.stack([d["river_dist"][m][ki, kj] for m in range(len(d["names"]))])   # km from each snapped mouth px
    lab = np.where(dist.min(0) <= NAMED_RADIUS_KM, dist.argmin(0), -1)
    names = d["names"] + ["other"]
    r["mouth_ij"] = mouth_ij
    r["groups"] = pd.DataFrame({names[g]: mass[:, lab == g].sum(1) / 1e3 for g in list(range(len(d["names"]))) + [-1]}, index=t)  # m3 s-1
    r["group_npx"] = {names[g]: int((lab == g).sum()) for g in list(range(len(d["names"]))) + [-1]}
    r["pix"] = pd.DataFrame(dict(lat=lat[ki], lon=lon[kj], mean_m3s=mass.mean(0) / 1e3, group=[names[g] for g in lab],
                                 dist_named_km=dist.min(0))).sort_values("mean_m3s", ascending=False)
    r["diag"] = dict(np.load(DIAG)) if DIAG.exists() else None
    return r


def report(d, r):
    t = pd.DatetimeIndex(d["time"])
    print(f"store days {len(t)} ({t[0].date()}..{t[-1].date()}), ocean px {int(d['ocean'].sum())}")
    print(pd.DataFrame({v: r[v] for v in ["rsntds", "friver", "mlotst"]}).T.to_string())
    print(f"friver: {int(r['fr_nz'].sum())} pixels ever > 0 ({r['fr_nz_off_ocean']} off the store ocean mask)")
    g = r["groups"]
    print(pd.DataFrame({"mean_m3s": g.mean(), "min": g.min(), "max": g.max(), "n_px": pd.Series(r["group_npx"])}).sort_values("mean_m3s", ascending=False).round(1).to_string())
    print(r["pix"].head(25).round(3).to_string(index=False))
    if r["diag"] is not None:
        q = r["diag"]
        for k in ["total_conservative", "total_linear"]:
            rel = q[k] / q["total_src"] - 1
            print(f"{k}: mean {np.nanmean(q[k]) / 1e3:.1f} m3 s-1 vs source {np.nanmean(q['total_src']) / 1e3:.1f}; daily rel. diff "
                  f"min {np.nanmin(rel):.2e} median {np.nanmedian(rel):.2e} max {np.nanmax(rel):.2e} (n={np.isfinite(rel).sum()})")
        print(f"source cells > 0: {len(q['snap_km'])}, -> {len(np.unique(q['target_flat']))} pixels, snap distance km: median "
              f"{np.median(q['snap_km']):.2f} max {q['snap_km'].max():.2f}; missing days rsntds {list(q['missing_rsntds'])} friver {list(q['missing_friver'])}")


def plot(d, r):
    lat, lon = d["lat"], d["lon"]; ext = [lon[0], lon[-1], lat[0], lat[-1]]
    fig, ax = plt.subplots(3, 2, figsize=(14, 14))
    s = r["rs_doy"]
    a = ax[0, 0]; a.plot(s.index, s["mean"], color=P.OKABE_ITO[0], label="mean over years")
    a.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=P.OKABE_ITO[0], alpha=.25, label="± 1 SD across years")
    a.set(xlabel="day of year", ylabel="basin-mean rsntds (W m$^{-2}$)",
          title=f"rsntds basin mean (ocean px) by day of year, {s['count'].min()}-{s['count'].max()} years per day"); a.legend(); a.grid(alpha=.3)
    a = ax[0, 1]; im = a.imshow(r["rs_map"], origin="lower", extent=ext, cmap=P.MAP_CMAP, aspect="auto")
    fig.colorbar(im, ax=a, label="rsntds (W m$^{-2}$)"); a.set(xlabel="lon (°E)", ylabel="lat (°N)", title=f"rsntds time mean 2014-2024 ({len(d['time'])} days), ocean px")
    a = ax[1, 0]; m = r["fr_map"]
    a.imshow(np.where(d["ocean"], 1.0, np.nan), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=4, aspect="auto", zorder=0)
    ii, jj = np.nonzero(r["fr_nz"])                     # single 4 km pixels: drawn as markers so they are visible
    im = a.scatter(lon[jj], lat[ii], c=np.log10(m[ii, jj]), s=22, cmap=P.MAP_CMAP, edgecolors="k", linewidths=.3, zorder=3)
    fig.colorbar(im, ax=a, label="log10 time-mean friver (kg m$^{-2}$ s$^{-1}$)")
    for n, (i, j) in zip(d["names"], r["mouth_ij"]):
        a.plot(lon[j], lat[i], "^", mfc="none", mec=P.OKABE_ITO[3], ms=9); a.text(lon[j] + .15, lat[i] + .12, n, fontsize=8, color=P.OKABE_ITO[3])
    a.set(xlabel="lon (°E)", ylabel="lat (°N)", title=f"friver time mean, {int(r['fr_nz'].sum())} px ever > 0 (grey: ocean; triangles: river_dist mouths)")
    a = ax[1, 1]; g = r["groups"]; top = g.mean().sort_values(ascending=False).index[:6]
    for k, c in enumerate(top):
        a.plot(g.index, g[c], lw=.7, color=P.OKABE_ITO[k], label=f"{c} ({r['group_npx'][c]} px)")
    a.set(yscale="log", xlabel="date", ylabel="river inflow (m$^3$ s$^{-1}$, sum over px)",
          title=f"friver daily inflow by mouth (px within {NAMED_RADIUS_KM} km of a named mouth)"); a.legend(fontsize=8); a.grid(alpha=.3)
    a = ax[2, 0]
    if r["diag"] is not None:
        q = r["diag"]; tt = pd.DatetimeIndex(q["time"])
        a.plot(tt, q["total_src"] / 1e3, lw=1.6, color=P.OKABE_ITO[7], label="source 2.5 km grid")
        a.plot(tt, q["total_conservative"] / 1e3, lw=.8, ls="--", color=P.OKABE_ITO[1], label="store friver.npy (nearest ocean px, conservative)")
        a.plot(tt, q["total_linear"] / 1e3, lw=.8, color=P.OKABE_ITO[0], label="linear to 2 km + 2x2 mean (not stored)")
        a.legend(fontsize=8)
    a.set(xlabel="date", ylabel="basin total river inflow (m$^3$ s$^{-1}$)", title="friver basin total by regridding method"); a.grid(alpha=.3)
    a = ax[2, 1]; x = np.asarray(d["rsntds"][::10][:, d["ocean"]], dtype="float32").ravel(); x = x[np.isfinite(x)]
    a.hist(x, bins=80, color=P.OKABE_ITO[0]); a.set(xlabel="rsntds (W m$^{-2}$)", ylabel="pixel-days (every 10th day)", title=f"rsntds value distribution, ocean px (n={x.size:,})")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=110); print("wrote", FIG)


if __name__ == "__main__":
    d = load(); r = analyse(d); report(d, r); plot(d, r)
