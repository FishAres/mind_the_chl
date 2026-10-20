"""Test 2 of the static-base hypothesis: where are the zero-shift tiles / patches, and is their day-t structure more
similar to the long-term ln CHL pattern than that of moving tiles?  CPU only; reads the existing caches.

Method
- Block matching (analyses/blk4/results/chl_motion_tiles.npz, chl_motion.build_table): lag 1, 16 px tiles, search +-6;
  groups: "zero" = accepted with integer best shift (0, 0); "moving" = accepted, non-zero best shift (significant vs the
  zero shift); "ambiguous" = peak r >= 0.6, not on edge, non-zero best shift NOT significant vs zero (rejected by
  chl_motion; listed for reference).
- Masked NCC (analyses/blk4/results/spectral_motion_patches.npz): lead 1, accepted patch-pairs (peak r >= 0.6, not on
  edge); "zero" = integer peak (0, 0) (rint of the sub-pixel shift), "moving" = otherwise.
- Covariates per tile / patch: region (majority label, chl_motion.regions), season (month of day t), distance to coast
  (mean of -topo.npy over ocean px, km), depth (median of depth.npy over ocean px, m), cloud fraction (1 - observed share
  of the tile / patch on day t and on day t+1, mean of the two; land counted as unobserved for patches, as the
  estimators), |current| (reanalysis current over the lag, px/day: chl_motion tile current / spectral patch current).
- Climatological pattern correlation: Pearson r over the px observed on day t inside the tile / patch between ln C(t)
  and (i) the long-term per-pixel mean ln C of TRAIN days (2014-2024 minus 2017, 2019, 2021, 2023), (ii) the train-days
  doy climatology at doy(t) (tracer_static.base_doy, sigma 15 d). Values of day t enter (i) / (ii) when t is a train
  day (one of ~2,500 days per pixel); a val-years-only row is reported as a check.
- Statistics: per group n, medians; zero share per covariate bin with n (bins with n < 5 flagged); r distributions.

Run from the repo root: python -m analyses.tracer.static_zero_tiles
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from analyses.blk4 import chl_motion as CM
from analyses.blk4 import spectral_motion as SM
from chl_cnn import tracer_static as TS
from chl_cnn import data as D
from chl_cnn.plotting import OKABE_ITO as C

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
OUT = RES / "static_zero_tiles.json"
MIN_N = 5
SEAS = ["DJF", "MAM", "JJA", "SON"]


def pearson_masked(X, Y, M):
    """Row-wise Pearson r of X, Y (n, m) over M (n, m) bool; NaN if < 10 px or zero variance."""
    n = M.sum(1).astype(float)
    x = np.where(M, X, 0.0); y = np.where(M, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = x.sum(1) / n, y.sum(1) / n
        cxy = (x * y).sum(1) / n - mx * my; vx = (x * x).sum(1) / n - mx ** 2; vy = (y * y).sum(1) / n - my ** 2
        r = cxy / np.sqrt(vx * vy)
    r[(n < 10) | ~(vx > 1e-12) | ~(vy > 1e-12)] = np.nan
    return r


def main():
    t0 = time.time()
    g = CM.load()
    ocean = g["ocean"]; H, W = ocean.shape
    times = g["time"]; nd = len(times)
    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    seas = np.array([CM.SEASONS[m] for m in months])
    tr, va, te = D.split_by_years(times, (2017, 2021), (2019, 2023))
    isval = np.isin(np.arange(nd), va)
    # ln C full grid
    y = np.empty((nd, H, W), "float32")
    for b0 in range(0, nd, 250):
        y[b0:b0 + 250] = CM.ln_chl(g["chl_l3"][b0:b0 + 250])
    obs = np.isfinite(y) & ocean[None]
    y0 = np.where(obs, y, 0.0).astype("float32")
    doy = TS.doy_of(times)
    Bdoy, _ = TS.base_doy(y0, obs, tr, doy)
    s = np.zeros((H, W)); n = np.zeros((H, W))
    for b0 in range(0, len(tr), 250):
        ii = tr[b0:b0 + 250]; s += y0[ii].sum(0); n += obs[ii].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ltm = np.where(n > 0, s / n, np.nan)
    print(f"fields {time.time() - t0:.0f}s", flush=True)
    dist = -np.load(CM.STORE / "topo.npy").astype("float64"); depth = np.load(CM.STORE / "depth.npy").astype("float64")
    reg = CM.regions(ocean)
    res = dict(assumptions=__doc__.split("Method")[1].split("Run from")[0].strip(),
               data_quality=dict(ltm_nan_ocean_px=int((~np.isfinite(ltm) & ocean).sum()), doy_nan_ocean_px_any_bin=int((~np.isfinite(Bdoy) & ocean[None]).any(0).sum()),
                                 dist_nan_ocean_px=int((~np.isfinite(dist) & ocean).sum()), depth_nan_ocean_px=int((~np.isfinite(depth) & ocean).sum())))

    # ---------------- block matching tiles
    z = dict(np.load(CM.CACHE))
    d = CM.build_table(z, 1, 16, 6)
    T = 16; r0g, c0g, nty, ntx = CM.tile_geom(H, W, T)
    rr = (r0g + T * d["ty"])[:, None] + np.arange(T)[None]; cc = (c0g + T * d["tx"])[:, None] + np.arange(T)[None]
    R_, C_ = rr[:, :, None], cc[:, None, :]
    day = d["day"]
    ot = obs[day[:, None, None], R_, C_].reshape(len(day), -1); ot1 = obs[day[:, None, None] + 1, R_, C_].reshape(len(day), -1)
    yt = y0[day[:, None, None], R_, C_].reshape(len(day), -1)
    oc = ocean[R_, C_].reshape(len(day), -1)
    tile = dict(
        group=np.where(d["accept"] & d["zero"], "zero", np.where(d["accept"], "moving", np.where(d["ambig"], "ambiguous", "other"))),
        region=np.array([[np.bincount(reg[r0g + T * a:r0g + T * (a + 1), c0g + T * b:c0g + T * (b + 1)].ravel()[reg[r0g + T * a:r0g + T * (a + 1), c0g + T * b:c0g + T * (b + 1)].ravel() > 0], minlength=8)[1:].argmax() + 1
                            if (reg[r0g + T * a:r0g + T * (a + 1), c0g + T * b:c0g + T * (b + 1)] > 0).any() else 0 for b in range(ntx)] for a in range(nty)])[d["ty"], d["tx"]],
        season=seas[day],
        dist=np.nanmean(np.where(oc, dist[R_, C_].reshape(len(day), -1), np.nan), 1),
        depth=np.nanmedian(np.where(oc, depth[R_, C_].reshape(len(day), -1), np.nan), 1),
        cloud=1 - 0.5 * (ot.mean(1) + ot1.mean(1)),
        cur=np.hypot(*d["dc"]),
        r_ltm=pearson_masked(yt, ltm[R_, C_].reshape(len(day), -1), ot & np.isfinite(ltm[R_, C_].reshape(len(day), -1))),
        r_doy=pearson_masked(yt, Bdoy[doy[day][:, None, None], R_, C_].reshape(len(day), -1), ot & np.isfinite(Bdoy[doy[day][:, None, None], R_, C_].reshape(len(day), -1))),
        val=isval[day], ty=d["ty"], tx=d["tx"])
    print(f"tiles {time.time() - t0:.0f}s", flush=True)

    # ---------------- NCC patches
    zs = np.load(SM.CACHE)
    m = (zs["k"] == 1) & (zs["null"] == 0)
    acc = zs["okpk"][m] & ~zs["edge"][m] & (zs["rpk"][m] >= SM.R_MIN)
    i_, p_ = zs["i"][m][acc], zs["p"][m][acc]
    fy, fx = zs["fy"][m][acc], zs["fx"][m][acc]
    pos_all, _ = SM.patch_grid(ocean)
    pos = pos_all[p_]
    pc = np.searchsorted(zs["pidx"], p_)
    cur = np.hypot(*np.stack([SM.lag_disp(zs["vel_cu"], i_, pc, 1), SM.lag_disp(zs["vel_cv"], i_, pc, 1)]))
    P = SM.P
    rr = pos[:, 0][:, None] + np.arange(P)[None]; cc = pos[:, 1][:, None] + np.arange(P)[None]
    R_, C_ = rr[:, :, None], cc[:, None, :]
    ot = obs[i_[:, None, None], R_, C_].reshape(len(i_), -1); ot1 = obs[zs["j"][m][acc][:, None, None], R_, C_].reshape(len(i_), -1)
    yt = y0[i_[:, None, None], R_, C_].reshape(len(i_), -1); oc = ocean[R_, C_].reshape(len(i_), -1)
    preg = np.array([np.bincount(reg[r:r + P, c:c + P][reg[r:r + P, c:c + P] > 0], minlength=8)[1:].argmax() + 1 if (reg[r:r + P, c:c + P] > 0).any() else 0 for r, c in pos_all])
    lt = ltm[R_, C_].reshape(len(i_), -1); bd = Bdoy[doy[i_][:, None, None], R_, C_].reshape(len(i_), -1)
    patch = dict(group=np.where((np.rint(fy) == 0) & (np.rint(fx) == 0), "zero", "moving"), region=preg[p_], season=seas[i_],
                 dist=np.nanmean(np.where(oc, dist[R_, C_].reshape(len(i_), -1), np.nan), 1),
                 depth=np.nanmedian(np.where(oc, depth[R_, C_].reshape(len(i_), -1), np.nan), 1),
                 cloud=1 - 0.5 * (ot.mean(1) + ot1.mean(1)), cur=cur,
                 r_ltm=pearson_masked(yt, lt, ot & np.isfinite(lt)), r_doy=pearson_masked(yt, bd, ot & np.isfinite(bd)), val=isval[i_])
    print(f"patches {time.time() - t0:.0f}s", flush=True)

    bins = dict(dist=[0, 10, 25, 50, 100, 400], depth=[0, 50, 200, 1000, 2000, 3000], cloud=[0, 0.05, 0.1, 0.2, 0.5, 1.01],
                cur=[0, 0.5, 1, 2, 3, 5, 50])
    NAMES = {1: "NW shelf", 2: "W open", 3: "E open", 4: "Anatolian", 5: "Crimea/Caucasus", 6: "Azov", 7: "Marmara"}

    def summarise(tb, groups, zero_vs):
        out = {}
        G = tb["group"]
        out["n_by_group"] = {gname: int((G == gname).sum()) for gname in groups}
        q = lambda x: [round(float(v), 3) for v in np.nanpercentile(x, [25, 50, 75])] if np.isfinite(x).sum() >= MIN_N else None
        out["quartiles_by_group"] = {gname: {c: q(tb[c][G == gname]) for c in ("dist", "depth", "cloud", "cur", "r_ltm", "r_doy")} for gname in groups}
        out["nan_by_group"] = {gname: {c: int(np.isnan(tb[c][G == gname]).sum()) for c in ("dist", "depth", "cloud", "cur", "r_ltm", "r_doy")} for gname in groups}
        out["quartiles_by_group_val_years"] = {gname: {c: q(tb[c][(G == gname) & tb["val"]]) for c in ("r_ltm", "r_doy")} for gname in groups}
        acc_ = np.isin(G, zero_vs)
        zero = G == "zero"

        def share(mask):
            nn = int((acc_ & mask).sum())
            return dict(zero_share=round(float(zero[acc_ & mask].mean()), 3) if nn >= MIN_N else None, n=nn, **({"flag": "n<5"} if nn < MIN_N else {}))
        out["zero_share_by_region"] = {NAMES[k]: share(tb["region"] == k) for k in range(1, 8)}
        out["zero_share_by_season"] = {s_: share(tb["season"] == s_) for s_ in SEAS}
        for c, e in bins.items():
            out[f"zero_share_by_{c}"] = {f"[{e[j]}, {e[j + 1]})": share((tb[c] >= e[j]) & (tb[c] < e[j + 1])) for j in range(len(e) - 1)}
        rb = [-1, 0, 0.25, 0.5, 0.75, 1.01]
        out["zero_share_by_r_ltm"] = {f"[{rb[j]}, {rb[j + 1]})": share((tb["r_ltm"] >= rb[j]) & (tb["r_ltm"] < rb[j + 1])) for j in range(len(rb) - 1)}
        out["zero_share_by_r_doy"] = {f"[{rb[j]}, {rb[j + 1]})": share((tb["r_doy"] >= rb[j]) & (tb["r_doy"] < rb[j + 1])) for j in range(len(rb) - 1)}
        # within-region check of the r_ltm difference (zero minus moving medians)
        out["median_r_ltm_zero_minus_moving_by_region"] = {}
        for k in range(1, 8):
            a_, b_ = tb["r_ltm"][(G == "zero") & (tb["region"] == k)], tb["r_ltm"][(G == "moving") & (tb["region"] == k)]
            if np.isfinite(a_).sum() >= MIN_N and np.isfinite(b_).sum() >= MIN_N:
                out["median_r_ltm_zero_minus_moving_by_region"][NAMES[k]] = dict(diff=round(float(np.nanmedian(a_) - np.nanmedian(b_)), 3), n_zero=int(np.isfinite(a_).sum()), n_moving=int(np.isfinite(b_).sum()))
        return out

    res["block_matching_lag1_T16_S6"] = summarise(tile, ["zero", "moving", "ambiguous"], ["zero", "moving"])
    res["ncc_lead1"] = summarise(patch, ["zero", "moving"], ["zero", "moving"])
    # tile map of zero share
    A = np.isin(tile["group"], ["zero", "moving"])
    zmap = np.full((nty, ntx), np.nan); nmap = np.zeros((nty, ntx))
    for a in range(nty):
        for b in range(ntx):
            mm = A & (tile["ty"] == a) & (tile["tx"] == b)
            nmap[a, b] = mm.sum()
            if mm.sum() >= 20:
                zmap[a, b] = (tile["group"][mm] == "zero").mean()
    res["tile_map"] = dict(zero_share=np.round(zmap, 3).tolist(), n=nmap.astype(int).tolist(), min_n_shown=20)
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    plots(res, tile, patch, zmap, nmap, g, (r0g, c0g, nty, ntx), bins)
    print(f"done {time.time() - t0:.0f}s", flush=True)


def plots(res, tile, patch, zmap, nmap, g, geom, bins):
    r0g, c0g, nty, ntx = geom
    lat, lon = g["lat"].astype(float), g["lon"].astype(float)
    T = 16
    ext = [lon[c0g], lon[c0g + ntx * T - 1], lat[r0g], lat[r0g + nty * T - 1]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    im = ax[0].imshow(zmap, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax[0].contour(lon, lat, g["ocean"].astype(float), [0.5], colors="k", linewidths=.6)
    fig.colorbar(im, ax=ax[0], label="zero-shift share of accepted tiles")
    ax[0].set(title="Block matching lag 1: zero-shift share per 16 px tile (n >= 20)", xlabel="lon (°E)", ylabel="lat (°N)")
    im = ax[1].imshow(np.where(nmap > 0, nmap, np.nan), origin="lower", extent=ext, cmap="viridis", aspect="auto")
    ax[1].contour(lon, lat, g["ocean"].astype(float), [0.5], colors="k", linewidths=.6)
    fig.colorbar(im, ax=ax[1], label="accepted tile-days")
    ax[1].set(title="Block matching lag 1: accepted tile-days per tile", xlabel="lon (°E)", ylabel="lat (°N)")
    fig.tight_layout(); fig.savefig(FIG / "static_zero_tiles_map.png", dpi=110); plt.close(fig)
    # zero share by covariates
    covs = [("region", "region"), ("season", "season"), ("dist", "distance to coast (km)"), ("depth", "depth (m)"),
            ("cloud", "cloud fraction, days t and t+1"), ("cur", "|reanalysis current| (px/day)"), ("r_ltm", "r(ln C day t, long-term mean pattern)")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5)); axes = axes.ravel()
    for q, (c, lab) in enumerate(covs):
        ax = axes[q]
        for u, (nm, key) in enumerate([("block matching (16 px tiles)", "block_matching_lag1_T16_S6"), ("masked NCC (32 px patches)", "ncc_lead1")]):
            tab = res[key][f"zero_share_by_{c}"]
            labs = list(tab.keys()); v = [tab[l_]["zero_share"] if tab[l_]["zero_share"] is not None else np.nan for l_ in labs]
            x = np.arange(len(labs))
            ax.bar(x + (u - 0.5) * 0.4, v, 0.4, color=C[u], label=nm)
            for xx, l_ in zip(x, labs):
                ax.text(xx + (u - 0.5) * 0.4, 0.01, f"{tab[l_]['n']}", rotation=90, fontsize=6, ha="center", va="bottom")
            ax.set_xticks(x); ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=7)
        ax.set(title=f"Zero-shift share by {lab}", ylabel="zero-shift share of accepted (n in bars)", ylim=(0, 1))
        ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=7)
    ax = axes[7]
    for u, (tb, nm) in enumerate([(tile, "tiles"), (patch, "patches")]):
        for v_, (grp, ls) in enumerate([("zero", "-"), ("moving", "--")]):
            x = tb["r_ltm"][tb["group"] == grp]; x = x[np.isfinite(x)]
            ax.hist(x, bins=np.linspace(-1, 1, 41), density=True, histtype="step", color=C[u], ls=ls, label=f"{nm} {grp} (n {len(x):,})")
    ax.set(title="Corr. of day-t ln C with long-term mean pattern", xlabel="Pearson r over px observed on day t", ylabel="density")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "static_zero_tiles_covariates.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
