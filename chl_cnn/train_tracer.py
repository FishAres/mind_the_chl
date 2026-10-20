"""Learned Lagrangian tracer of ln CHL, step 1: velocity terms only; step 2 (--source): plus a pointwise source term. No
latent channels. Black Sea 4 km store.

    ln C(x, t+k) = ln C_obs(X_k(x), t) [+ sum over sub-steps of S(features at the trajectory position and time, c) h]
    X_k(x) = departure point at day t of the backward trajectory from x at day t+k
    u_eff(x, t)  = a(x, s) u_rea + b(x, s) u_geo + c(s) R(theta(s)) tau + du_coarse(x, t)                   (chl_cnn/tracer.py)

Terms (--terms, comma list of rea, geo, wind, corr)
  rea   reanalysis current (store u_curr, v_curr, m/s) x a(x, s); a = 1.5 sigmoid(MLP) in (0, 1.5), init --a-init (0.25).
  geo   geostrophic velocity from zos (store "sla"): u_g = -g/f d(eta)/dy, v_g = g/f d(eta)/dx, centred differences in m;
        0 within --geo-margin px of land / non-finite eta; x b(x, s) in (--b-min, 1.5), init --b-init (0).
  wind  wind stress tau (store u_wind, v_wind, Pa; NaN -> 0) times the complex coefficient w(s) = p(s) + i q(s) (first-
        order doy harmonics, m/s per Pa, init 0): c(s) = |w|, theta(s) = arg w (counter-clockwise from tau).
  corr  du_coarse: 3x3-conv net on the day's physics fields [u, v, u_g, v_g, tau_x, tau_y, doy sin/cos], average-pooled
        to --corr-cell px cells, bilinearly upsampled (m/s); loss + --corr-mag mean(du_c^2) + --corr-grad mean |grad du_c|^2
        on the coarse grid. Off unless listed.
  a, b: 1x1-conv MLPs (--hidden) of the static geography [log10 depth, log10(1 + distance to coast km), lat, lon]
  (z-scored over ocean) and doy sin / cos. No CHL input to the velocity.
Trajectories: backward RK2 (midpoint), --n-sub sub-steps per day, daily fields at integer times and linear in time
between them (days t..t+k used for lead k), velocity components clipped to +-(--vmax) px/day, bilinear grid_sample.
A trajectory is invalid if the sampled (ocean & finite reanalysis current) mask is < 0.99 at any evaluation (same rule
for every term set, so valid sets differ only through the velocities). Prediction valid only where the sampled observed
mask of day t at the departure point is >= 0.99. m/s -> px/day with physics.grid_metrics (lat ascending: +row = north).
Loss: L1 over pixels observed at t+k with a valid prediction, mean per lead, leads 1..--leads weighted by
--lead-weights (default equal). No filling.
Samples: start day t with days t..t+leads all present in the store (consecutive calendar days) and in the same split.
Train: train years (all t, --train-stride); val: val years, every --val-stride-th t (default 2). Test years not scored.
Scored each epoch (val): model, persistence (pixel value at t; needs obs at t), persistence along reanalysis currents
x1.0 and x0.25 (same integrator / validity rule), DOY climatology (train2.doy_climatology, train days). Per lead:
MAE / RMSE / n on (i) the common set (target observed and every predictor valid) and (ii) each predictor's own set;
grad ratio / corr at lags 1, 4, 16 px on common-set pixel pairs; lead-1 MAE per region (error_breakdown.regions_4km).
Source (--source; off by default, then step-1 behaviour is unchanged)
  S = --g-max tanh(MLP) (tracer.SourceModel, 1x1 convs, --src-hidden, zero output at the start, lr --lr-src), integrated
  forward from the departure point along the backward trajectory (tracer.py docstring). Inputs, sampled at the sub-step
  midpoint position (bilinear, grid_sample) and time (linear between days): sst32, rsntds, log10 mlotst, dilution
  max(MLD(t) - MLD(t-1), 0) / MLD(t) per day, |tau|, river influence log10(1 + sum_r Q_r(t) exp(-d_r / --river-L km))
  (d_r: river_dist.npy, 8 mouths; Q_r(t) in m3/s: friver x cell area / 1000 summed over the friver pixels whose nearest
  mouth is r, if within 40 km, as analyses/blk4/extra_vars_check.py; other friver pixels not used), so, so_ok (finite
  physics flag, 0/1), log10 depth, log10(1 + distance to coast) (the static channels), doy sin / cos (linear in time), and
  the carried value c (ln C along the trajectory; dropped with --src-no-c). Continuous channels z-scored over ocean
  pixel-days of every 5th train day; NaN -> 0 after z-scoring. rsntds all-NaN days are filled with the per-pixel mean
  over the other years of the same day of year. c is z-scored with the train observed ln C mean / sd.
  Extra outputs: source_maps.npz (season-mean lead-1 source (pred - departure value) per pixel, val days, best model)
  and source_pd.npz (subsample of S inputs / outputs at lead-1 sub-steps, val days, best model).
Step 1b (both off by default; then behaviour is unchanged)
  --vel-param decomp  u_eff = sum_j alpha_j(region, season) u_j (+ wind if --terms wind): u_j = geostrophic and
        ageostrophic (u_rea - u_geo; = u_rea where u_geo is invalid) velocities, each split into its train-years
        day-of-year climatology (366 doy bins, circular Gaussian smoothing --clim-sigma days, tracer.decomp_clim) and the
        daily anomaly -> 4 components (tracer.DECOMP_COMPS). alpha: free scalar per (component, region
        0..7 of error_breakdown.regions_4km, season DJF/MAM/JJA/SON) (--alpha-time season) or c0 + c1 sin + c2 cos(doy) per
        (component, region) (--alpha-time harm); init --alpha-init (0.25) -> at init u_eff = 0.25 u_rea (the x0.25 baseline).
        --terms may only contain wind. alpha.json (best model) lists the alphas.
  --loss-band lo,hi  loss = L1 of band-passed ln CHL (tracer.BandPass: normalised-convolution DoG, sigmas for 0.5 response
        at lo and hi km, 4.5 km px) of prediction and target, both filtered with the same mask (target observed & valid
        prediction), at band-valid pixels; mean per lead. Model selection then uses the mean-lead val band L1 (own set)
        instead of the pixel MAE.
Output models/tracer/<run>/{config,history,metrics}.json, model.pt (best mean-lead val MAE, own set), factor_maps.npz.
metrics.json is first written right after the initial evaluation (dict(init=...): baselines visible during training)
and overwritten at the end; factor_maps.npz is rewritten at every new best model.pt.

Flags: --run, --terms, --leads, --lead-weights, --n-sub, --vmax, --batch, --epochs, --train-stride, --val-stride, --lr,
  --lr-wind, --hidden, --a-init, --b-init, --b-min, --geo-margin, --corr-cell, --corr-hidden, --corr-mag, --corr-grad,
  --chl-max, --max-batches, --eval-only, --init-from, --seed, --source, --src-no-c, --src-hidden, --g-max, --lr-src,
  --river-L, --vel-param, --alpha-time, --alpha-init, --clim-sigma, --loss-band, --band-min-frac.
Example
  python -m chl_cnn.train_tracer --run tr_rea --terms rea --epochs 12
  python -m chl_cnn.train_tracer --run tr_rea_geo_wind --terms rea,geo,wind --epochs 12
  python -m chl_cnn.train_tracer --run tr_src --terms rea,geo,wind --source --epochs 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import train2 as T2
from . import tracer as TR
from . import store_blk4 as S4
from .physics import grid_metrics
from .runs import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
PRED0 = ["model", "persistence", "adv_x1.0", "adv_x0.25", "doy_clim"]
LAGS = (1, 4, 16)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--terms", default="rea", help="comma list of rea, geo, wind, corr")
    p.add_argument("--leads", type=int, default=5)
    p.add_argument("--lead-weights", default=None, help="comma list of --leads weights (default equal)")
    p.add_argument("--n-sub", type=int, default=4)
    p.add_argument("--vmax", type=float, default=10.0, help="clip velocity components to +-vmax px/day")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--val-stride", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3, help="Adam lr of the a / b / corr nets")
    p.add_argument("--lr-wind", type=float, default=1e-2, help="Adam lr of the wind coefficients")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--a-init", type=float, default=0.25)
    p.add_argument("--b-init", type=float, default=0.0)
    p.add_argument("--b-min", type=float, default=-0.5)
    p.add_argument("--geo-margin", type=int, default=2)
    p.add_argument("--corr-cell", type=int, default=8)
    p.add_argument("--corr-hidden", type=int, default=16)
    p.add_argument("--corr-mag", type=float, default=1.0)
    p.add_argument("--corr-grad", type=float, default=10.0)
    p.add_argument("--chl-max", type=float, default=100.0)
    p.add_argument("--max-batches", type=int, default=0, help="debug: train only the first N batches per epoch")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--init-from", default=None, help="tracer run whose model.pt initialises the model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", action="store_true", help="step 2: add the pointwise source term")
    p.add_argument("--src-no-c", action="store_true", help="source without the carried ln C input")
    p.add_argument("--src-hidden", type=int, default=32)
    p.add_argument("--g-max", type=float, default=0.5, help="source bound, ln CHL per day")
    p.add_argument("--lr-src", type=float, default=3e-3)
    p.add_argument("--river-L", type=float, default=50.0, help="river influence e-folding distance, km (fixed)")
    p.add_argument("--ref-run", default=None, help="--source: velocity-only run scored as predictor 'ref' in the final evaluation (metrics['final'])")
    p.add_argument("--vel-param", default="factor", choices=["factor", "decomp"], help="step 1b: decomp = alpha-weighted velocity components")
    p.add_argument("--alpha-time", default="season", choices=["season", "harm"])
    p.add_argument("--alpha-init", type=float, default=0.25)
    p.add_argument("--clim-sigma", type=float, default=15.0, help="decomp: doy smoothing of the velocity climatology, days")
    p.add_argument("--loss-band", default=None, help="step 1b: lo,hi km -> band-passed L1 loss")
    p.add_argument("--band-min-frac", type=float, default=0.5)
    a = p.parse_args(argv)
    a.terms = [t for t in a.terms.split(",") if t]
    assert set(a.terms) <= {"rea", "geo", "wind", "corr"}, a.terms
    if a.vel_param == "decomp":
        assert set(a.terms) <= {"wind"} and not a.source, "decomp: --terms may only contain wind; no --source"
    a.loss_band = None if not a.loss_band else [float(x) for x in a.loss_band.split(",")]
    a.lead_weights = [1.0] * a.leads if a.lead_weights is None else [float(w) for w in a.lead_weights.split(",")]
    assert len(a.lead_weights) == a.leads
    return a


def load(a, dev):
    """Cropped blk4 fields on the GPU. Returns a dict of tensors / arrays."""
    sp = S4.store_path()
    ld = lambda nm, mm=True: np.load(sp / f"{nm}.npy", mmap_mode="r" if mm else None)
    times = ld("time", False); oc = ld("ocean", False)
    rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
    ys, xs = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0])), slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
    ocean = oc[ys, xs]; H, W = ocean.shape; n = len(times)
    lat, lon = ld("lat", False)[ys].astype("float64"), ld("lon", False)[xs].astype("float64")
    assert lat[1] > lat[0]
    rd = lambda nm, t=slice(None): np.asarray(ld(nm)[t, ys, xs]).astype("float32")
    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)
    dx, dy, f, _ = grid_metrics(lat, lon)
    info = dict(grid=[H, W], ocean_px=int(ocean.sum()), crop=[int(ys.start), int(ys.stop), int(xs.start), int(xs.stop)], n_days=n)

    chl = rd("chl_l3")
    info["chl_gt_max"] = int(np.sum(chl > a.chl_max)); chl[chl > a.chl_max] = np.nan
    y = D.log_chl(chl); del chl
    obs = np.isfinite(y) & ocean[None]
    y = np.nan_to_num(y, nan=0.0)
    T2.TIMES = times
    tr, va, te = D.split_by_years(times, (2017, 2021), (2019, 2023))
    clim, dbin = T2.doy_climatology(y, obs, tr)
    out = dict(times=times, ocean=ocean, lat=lat, lon=lon, tr=tr, va=va, te=te, info=info,
               Y=T(y), OBS=T(obs), CLIM=T(clim), BIN=T(dbin.astype("int64")), OCEAN=T(ocean), OCf=T(ocean.astype("float32")),
               CPDX=T((86400.0 / dx).astype("float32")), CPDY=T((86400.0 / dy).astype("float32")))
    del y, obs
    # currents (m/s), validity; wind stress (Pa); geostrophic from zos
    nanc = {}
    U = np.empty((n, H, W), np.float16); V = np.empty_like(U); OK = np.empty((n, H, W), bool)
    TX = np.empty_like(U); TY = np.empty_like(U); UG = np.empty_like(U); VG = np.empty_like(U); GOK = np.empty_like(OK)
    for b0 in range(0, n, 250):
        s = slice(b0, min(b0 + 250, n))
        u, v = rd("u_curr", s), rd("v_curr", s)
        ok = np.isfinite(u) & np.isfinite(v) & ocean[None]
        U[s], V[s], OK[s] = np.where(ok, u, 0), np.where(ok, v, 0), ok
        tx, ty = rd("u_wind", s), rd("v_wind", s)
        okw = np.isfinite(tx) & np.isfinite(ty) & ocean[None]
        nanc["wind_nan_ocean_px"] = nanc.get("wind_nan_ocean_px", 0) + int((~okw & ocean[None]).sum())
        TX[s], TY[s] = np.where(okw, tx, 0), np.where(okw, ty, 0)
        z = rd("sla", s)
        nanc["zos_nan_ocean_px"] = nanc.get("zos_nan_ocean_px", 0) + int((~np.isfinite(z) & ocean[None]).sum())
        ug, vg, gok = TR.geostrophic(z, dx, dy, f, ocean, a.geo_margin)
        UG[s], VG[s], GOK[s] = ug, vg, gok
    info.update(nanc, cur_valid_frac_ocean=float(OK[:, ocean].mean()), geo_valid_frac_ocean=float(GOK[:, ocean].mean()))
    out.update(U=T(U), V=T(V), CUROK=T(OK), TX=T(TX), TY=T(TY), UG=T(UG), VG=T(VG), GEOOK=T(GOK))
    if getattr(a, "vel_param", "factor") == "decomp":
        doy0 = (np.asarray(times) - np.asarray(times).astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
        clim, ccnt = TR.decomp_clim(U, V, UG, VG, OK, doy0, tr, a.clim_sigma)
        out.update(CLIMD=T(clim), DOYI=T(doy0.astype("int64")))
        info["decomp"] = dict(clim_sigma_days=a.clim_sigma, train_days=int(len(tr)), doy_bins=366,
                              ocean_px_with_no_train_current=int((ccnt.sum(0)[ocean] == 0).sum()),
                              train_count_per_doy_bin_ocean_median=float(np.median(ccnt[:, ocean].mean(1))))
        del clim, ccnt
    # static geography
    dep = np.load(sp / "depth.npy")[ys, xs].astype("float32")
    info["depth_nan_ocean_px"] = int((~np.isfinite(dep) & ocean).sum())
    ldep = np.log10(np.maximum(np.nan_to_num(np.abs(dep), nan=5.0), 5.0))
    dist = -np.load(sp / "topo.npy")[ys, xs].astype("float32")
    ldist = np.log10(1.0 + np.maximum(np.nan_to_num(dist, nan=0.0), 0.0))
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    st = []
    for x in (ldep, ldist, la, lo):
        x = x.astype("float32"); st.append(np.where(ocean, (x - x[ocean].mean()) / x[ocean].std(), 0.0))
    out["STATIC"] = T(np.stack(st).astype("float32"))
    info["static_channels"] = ["log10 depth (|depth.npy|, min 5 m)", "log10(1 + distance to coast km)", "lat", "lon"]
    doy = (np.asarray(times) - np.asarray(times).astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
    ang = 2 * np.pi * doy / 365.25
    out["DS"], out["DC"] = T(np.sin(ang).astype("float32")), T(np.cos(ang).astype("float32"))
    out["doy"] = doy
    reg_path = ROOT / "analyses" / "blk4"
    sys.path.insert(0, str(ROOT))
    from analyses.blk4.error_breakdown import regions_4km
    reg, ties = regions_4km(ocean)
    out["REG"] = T(reg.astype("int64")); info["region_ties"] = int(ties)
    if a.source:
        feat, sinfo = source_fields(a, sp, ys, xs, ocean, lat, lon, times, tr)
        out["FEAT"] = T(feat); del feat
        out["SSTATIC"] = out["STATIC"][:2].clone()
        yv = out["Y"][torch.as_tensor(tr, device=dev)][out["OBS"][torch.as_tensor(tr, device=dev)]]
        out["CMU"], out["CSD"] = float(yv.mean()), float(yv.std()); del yv
        sinfo.update(c_mean=out["CMU"], c_sd=out["CSD"])
        info["source"] = sinfo
    return out


SRC_CH = ["sst32", "rsntds", "log10 mlotst", "dilution max(dMLD,0)/MLD per day", "|tau|", "log10(1 + river influence m3/s)", "so", "so_ok"]


def source_fields(a, sp, ys, xs, ocean, lat, lon, times, tr, chunk=250):
    """Daily source input channels SRC_CH (z-scored over ocean, NaN -> 0; so_ok 0/1), float16 (n, 8, H, W), and info."""
    from .store_blk4_extra import cell_area
    ld = lambda nm: np.load(sp / f"{nm}.npy", mmap_mode="r")
    rd = lambda nm, t: np.asarray(ld(nm)[t, ys, xs]).astype("float32")
    n = len(times); H, W = ocean.shape; info = {}
    doy = (np.asarray(times) - np.asarray(times).astype("datetime64[Y]")).astype("timedelta64[D]").astype(int)
    dnum = np.asarray(times).astype("datetime64[D]").astype(int)
    # rsntds all-NaN days -> per-pixel mean of the same doy over the other years
    rsm = ld("rsntds")
    alln = [t for t in range(n) if not np.isfinite(np.asarray(rsm[t, ys, xs], dtype="float32")[ocean]).any()]
    fill = {}
    for t in alln:
        same = [u for u in np.flatnonzero(doy == doy[t]) if u not in alln]
        fill[t] = np.nanmean(np.stack([rd("rsntds", u) for u in same]), 0)
        info.setdefault("rsntds_filled", []).append(dict(day=str(times[t])[:10], n_years=len(same)))
    # river discharge per mouth (m3/s) and influence weights
    rdist = np.asarray(np.load(sp / "river_dist.npy")[:, ys, xs], dtype="float64")       # (8, H, W) km
    names = json.load(open(sp / "river_names.json"))
    fr = ld("friver")
    nz = np.zeros((H, W), bool)
    for b0 in range(0, n, chunk):
        nz |= (np.nan_to_num(np.asarray(fr[b0:b0 + chunk, ys, xs], dtype="float32")) > 0).any(0)
    ki, kj = np.nonzero(nz)
    dk = rdist[:, ki, kj]                                                                # (8, npx)
    grp = np.where(dk.min(0) <= 40.0, dk.argmin(0), -1)
    area = cell_area(lat, abs(float(np.diff(lat).mean())), abs(float(np.diff(lon).mean())))[ki]
    Q = np.zeros((n, len(names)))
    q_other = 0.0
    for b0 in range(0, n, chunk):
        v = np.nan_to_num(np.asarray(fr[b0:b0 + chunk, ys, xs], dtype="float64")[:, ki, kj]) * area / 1e3      # m3/s per px
        for r in range(len(names)):
            Q[b0:b0 + chunk, r] = v[:, grp == r].sum(1)
        q_other += v[:, grp < 0].sum()
    info["river_groups"] = {nm: dict(n_px=int((grp == r).sum()), mean_m3s=float(Q[:, r].mean())) for r, nm in enumerate(names)}
    info["river_unassigned"] = dict(n_px=int((grp < 0).sum()), mean_m3s=float(q_other / n))
    info["river_L_km"] = a.river_L
    wr = np.exp(-np.nan_to_num(rdist, nan=1e4) / a.river_L).astype("float32")             # (8, H, W)

    def chunk_raw(s):
        idx = np.arange(s.start, s.stop)
        sst = rd("sst32", s); so = rd("so", s)
        rs = rd("rsntds", s)
        for t in alln:
            if s.start <= t < s.stop:
                rs[t - s.start] = fill[t]
        mld = rd("mlotst", s)
        if s.start > 0:
            prev = rd("mlotst", slice(s.start - 1, s.stop - 1)); dd = (dnum[idx] - dnum[idx - 1]).astype("float32")
        else:
            prev = np.concatenate([mld[:1], mld[:-1]]); dd = np.concatenate([[1], dnum[idx[1:]] - dnum[idx[:-1]]]).astype("float32")
        dil = np.maximum(mld - prev, 0) / mld / dd[:, None, None]
        if s.start == 0:
            dil[0] = 0.0
        tx, ty = rd("u_wind", s), rd("v_wind", s)
        tau = np.hypot(tx, ty)
        riv = np.log10(1.0 + np.einsum("tr,rhw->thw", Q[idx].astype("float32"), wr))
        ok = (np.isfinite(so) & ocean[None]).astype("float32")
        return [sst, rs, np.log10(mld), dil, tau, riv, so], ok

    sub = tr[::5]
    stats = []
    acc = [[] for _ in range(7)]
    for b0 in range(0, len(sub), 60):
        for t in sub[b0:b0 + 60]:
            r_, _ = chunk_raw(slice(int(t), int(t) + 1))
            for c in range(7):
                x = r_[c][0][ocean]; acc[c].append(x[np.isfinite(x)])
    for c in range(7):
        x = np.concatenate(acc[c]); stats.append((float(x.mean()), float(x.std())))
    info["z_stats"] = {SRC_CH[c]: dict(mean=stats[c][0], sd=stats[c][1]) for c in range(7)}
    info["z_stats_days"] = int(len(sub))
    feat = np.empty((n, 8, H, W), np.float16)
    nanc = np.zeros(7, np.int64)
    for b0 in range(0, n, chunk):
        s = slice(b0, min(b0 + chunk, n))
        r_, ok = chunk_raw(s)
        for c in range(7):
            x = (r_[c] - stats[c][0]) / stats[c][1]
            fin = np.isfinite(x) & ocean[None]
            nanc[c] += int((~np.isfinite(r_[c]) & ocean[None]).sum())
            feat[s, c] = np.where(fin, x, 0.0)
        feat[s, 7] = ok
    info["nan_ocean_px_days"] = {SRC_CH[c]: int(nanc[c]) for c in range(7)}
    info["channels"] = SRC_CH + ["log10 depth (static)", "log10(1 + distance to coast) (static)", "doy sin", "doy cos"] + ([] if a.src_no_c else ["c = carried ln C (z)"])
    return feat, info


def starts(times, idx, K):
    """Days t in idx with t..t+K consecutive calendar days, all in idx."""
    days = np.asarray(times).astype("datetime64[D]").astype(int)
    s = set(idx.tolist()); res = []
    for t in idx:
        if t + K < len(days) and days[t + K] - days[t] == K and all((t + k) in s for k in range(1, K + 1)):
            res.append(t)
    return np.array(res, dtype=np.int64)


def main(argv=None):
    a = parse(argv)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    out = CATEGORIES["tracer"] / a.run; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = load(a, dev)
    K = a.leads
    H, W = d["ocean"].shape
    tr_s = starts(d["times"], d["tr"], K)[::a.train_stride]
    va_s = starts(d["times"], d["va"], K)[::a.val_stride]
    print(f"load {time.time() - t0:.0f}s  grid {H}x{W}  train starts {len(tr_s)}  val starts {len(va_s)}  "
          f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB  {d['info']}", flush=True)
    it = TR.Integrator(H, W, dev, n_sub=a.n_sub, vmax=a.vmax)
    if a.vel_param == "decomp":
        model = TR.DecompVelocity(len(TR.DECOMP_COMPS), int(d["REG"].max()) + 1, a.alpha_time, a.alpha_init, wind="wind" in a.terms).to(dev)
    else:
        model = TR.VelocityModel(a.terms, n_static=d["STATIC"].shape[0], hidden=a.hidden, a0=a.a_init, b0=a.b_init,
                                 b_range=(a.b_min, 1.5), cell=a.corr_cell, corr_hidden=a.corr_hidden).to(dev)
    bp = TR.BandPass(*a.loss_band, min_frac=a.band_min_frac, device=dev) if a.loss_band else None
    src = TR.SourceModel(len(SRC_CH) + 4 + (0 if a.src_no_c else 1), a.src_hidden, a.g_max).to(dev) if a.source else None
    if a.init_from:
        ck0 = torch.load(CATEGORIES["tracer"] / a.init_from / "model.pt", map_location=dev)
        model.load_state_dict(ck0["model"])
        if src is not None and "source" in ck0:
            src.load_state_dict(ck0["source"])
    groups = [dict(params=[p for n_, p in model.named_parameters() if not n_.startswith("w")], lr=a.lr)]
    if "wind" in a.terms:
        groups.append(dict(params=[model.wp, model.wq], lr=a.lr_wind))
    if src is not None:
        groups.append(dict(params=list(src.parameters()), lr=a.lr_src))
    opt = torch.optim.Adam([g for g in groups if g["params"]])
    lw = torch.tensor(a.lead_weights, device=dev); lw = lw / lw.sum()
    OCf = d["OCf"]

    def velocity(days, collect=None, model_=None):
        """days (B, K+1) int64 tensor -> (U (B, K+1, 2, H, W) px/day, ok (B, K+1, H, W) float, ue, ve m/s)."""
        B = days.shape[0]; fl = days.reshape(-1)
        g = lambda X: X[fl].float()
        if a.vel_param == "decomp" and model_ is None:
            comps = TR.decomp_components(g(d["U"]), g(d["V"]), g(d["UG"]), g(d["VG"]), g(d["CUROK"]), d["CLIMD"][d["DOYI"][fl]])
            ue, ve = model(comps, d["REG"], SEASON[fl], d["DS"][fl], d["DC"][fl], g(d["TX"]), g(d["TY"]), OCf, collect=collect)
            Upx = torch.stack([ue * d["CPDX"], ve * d["CPDY"]], 1).view(B, K + 1, 2, H, W)
            return Upx, d["CUROK"][fl].float().view(B, K + 1, H, W), ue, ve
        ue, ve = (model if model_ is None else model_)(d["STATIC"], d["DS"][fl], d["DC"][fl], g(d["U"]), g(d["V"]), g(d["UG"]), g(d["VG"]), g(d["TX"]), g(d["TY"]), OCf, collect=collect)
        Upx = torch.stack([ue * d["CPDX"], ve * d["CPDY"]], 1).view(B, K + 1, 2, H, W)
        return Upx, d["CUROK"][fl].float().view(B, K + 1, H, W), ue, ve

    def rea_px(days, alpha):
        B = days.shape[0]; fl = days.reshape(-1)
        Upx = torch.stack([d["U"][fl].float() * d["CPDX"], d["V"][fl].float() * d["CPDY"]], 1).view(B, K + 1, 2, H, W) * alpha
        return Upx, d["CUROK"][fl].float().view(B, K + 1, H, W)

    def batch_days(bt):
        return torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev)

    def forecast(Upx, ok, days, k):
        t = days[:, 0]
        pos, val = it.departure(Upx, ok, k)
        pr, pv = it.depart_sample(d["Y"][t], d["OBS"][t], pos)
        return pr, val & pv

    NF = len(SRC_CH)

    def src_inputs(F, days, p, j, w, c):
        """Source inputs at sub-step midpoint positions p (B, H, W, 2), interval j, weight w of day j + 1; c carried ln C."""
        B = days.shape[0]
        smp = it.sample(torch.cat([F[:, j], F[:, j + 1], d["SSTATIC"][None].expand(B, -1, -1, -1)], 1), p)
        dj, dj1 = days[:, j], days[:, j + 1]
        ds = ((1 - w) * d["DS"][dj] + w * d["DS"][dj1]).view(B, 1, 1, 1).expand(B, 1, H, W)
        dc = ((1 - w) * d["DC"][dj] + w * d["DC"][dj1]).view(B, 1, 1, 1).expand(B, 1, H, W)
        parts = [(1 - w) * smp[:, :NF] + w * smp[:, NF:2 * NF], smp[:, 2 * NF:], ds, dc]
        if not a.src_no_c:
            parts.append(((c - d["CMU"]) / d["CSD"])[:, None])
        return torch.cat(parts, 1)

    def forecast_src(Upx, ok, days, k, rec=None):
        """Velocity transport + source along the trajectory. Returns (prediction, valid, departure value c0). rec: list
        that receives (inputs, S) of every sub-step (no-grad evaluation only)."""
        from torch.utils.checkpoint import checkpoint
        t = days[:, 0]; B = days.shape[0]
        path = []
        pos, val = it.departure(Upx, ok, k, path=path)
        c0, pv = it.depart_sample(d["Y"][t], d["OBS"][t], pos)
        F = d["FEAT"][days.reshape(-1)].float().view(B, K + 1, NF, H, W)
        h = 1.0 / a.n_sub
        c = c0

        def step(p, c, j, w):
            return src(src_inputs(F, days, p, j, w, c))
        for p, j, w in reversed(path):
            if torch.is_grad_enabled():
                S = checkpoint(step, p, c, j, w, use_reentrant=False)
            else:
                S = step(p, c, j, w)
                if rec is not None:
                    rec.append((src_inputs(F, days, p, j, w, c), S))
            c = c + h * S
        return c, val & pv, c0

    def model_forecast(Upx, ok, days, k):
        return forecast_src(Upx, ok, days, k)[:2] if src is not None else forecast(Upx, ok, days, k)

    SEASON = torch.as_tensor((((np.asarray(d["times"]).astype("datetime64[M]").astype(int) % 12) + 1) % 12) // 3, device=dev)   # 0 DJF, 1 MAM, 2 JJA, 3 SON
    src_maps = {}

    # ---------------- evaluation
    @torch.no_grad()
    def evaluate(sidx, eval_batch=16, collect_factors=False, collect_pd=False, pd_every=50, ref=None):
        """ref: optional velocity-only VelocityModel scored as the extra predictor "ref" (its validity enters the common set)."""
        PRED = PRED0 + (["ref"] if ref is not None else [])
        model.eval()
        if src is not None:
            src.eval()
            sm_sum = torch.zeros(4, H, W, dtype=torch.float64, device=dev); sm_n = torch.zeros_like(sm_sum)
            pd_x, pd_s = [], []
            gen = torch.Generator(device=dev); gen.manual_seed(0)
        z = lambda: torch.zeros(3, dtype=torch.float64, device=dev)           # sum|e|, sum e^2, n
        acc = {s: {k: {p: z() for p in PRED} for k in range(1, K + 1)} for s in ("common", "own")}
        gacc = {k: {p: {L: torch.zeros(6, dtype=torch.float64, device=dev) for L in LAGS} for p in PRED} for k in range(1, K + 1)}
        nreg = int(d["REG"].max()) + 1
        racc = {p: torch.zeros(nreg, 2, dtype=torch.float64, device=dev) for p in PRED}
        bacc = {k: torch.zeros(2, dtype=torch.float64, device=dev) for k in range(1, K + 1)}
        fac = dict(a=[], b=[], speed_model=[], speed_rea=[]) if collect_factors or True else None
        for b0 in range(0, len(sidx), eval_batch):
            bt = sidx[b0:b0 + eval_batch]; days = batch_days(bt); t = days[:, 0]
            col = {}
            Um, okm, ue, ve = velocity(days, col)
            U1, ok1 = rea_px(days, 1.0); U25, ok25 = rea_px(days, 0.25)
            if ref is not None:
                Uref, okref, _, _ = velocity(days, model_=ref)
            oc = d["OCEAN"]
            for nm in ("a", "b"):
                if nm in col:
                    fac[nm].append(col[nm].view(len(bt), K + 1, H, W)[:, 0][:, oc].mean(1))
            spd = torch.hypot(Um[:, 0, 0], Um[:, 0, 1])[:, oc]; fac["speed_model"].append(spd.mean(1))
            fac["speed_rea"].append(torch.hypot(U1[:, 0, 0], U1[:, 0, 1])[:, oc].mean(1))
            for k in range(1, K + 1):
                tk = days[:, k]; truth = d["Y"][tk]; tobs = d["OBS"][tk]
                preds, valids = {}, {}
                if src is not None:
                    rec = [] if (collect_pd and k == 1) else None
                    preds["model"], valids["model"], c0 = forecast_src(Um, okm, days, k, rec)
                    if k == 1:
                        vm_ = valids["model"] & oc[None]
                        sbar = (preds["model"] - c0).double()               # mean source over the lead-1 trajectory, ln CHL / day
                        sea = SEASON[t]
                        for q in range(4):
                            mq = vm_ & (sea == q)[:, None, None]
                            sm_sum[q] += torch.where(mq, sbar, 0.0).sum(0); sm_n[q] += mq.double().sum(0)
                        if rec:
                            sel = vm_ & (torch.rand(vm_.shape, generator=gen, device=dev) < 1.0 / pd_every)
                            for xr, Sr in rec:
                                pd_x.append(xr.permute(1, 0, 2, 3)[:, sel].T.cpu()); pd_s.append(Sr[sel].cpu())
                else:
                    preds["model"], valids["model"] = forecast(Um, okm, days, k)
                preds["adv_x1.0"], valids["adv_x1.0"] = forecast(U1, ok1, days, k)
                preds["adv_x0.25"], valids["adv_x0.25"] = forecast(U25, ok25, days, k)
                if ref is not None:
                    preds["ref"], valids["ref"] = forecast(Uref, okref, days, k)
                if bp is not None:
                    mb = tobs & valids["model"]
                    bpp, vb = bp(preds["model"], mb); bpt, _ = bp(truth, mb)
                    bacc[k] += torch.stack([((bpp - bpt).abs().double() * vb).sum(), vb.double().sum()])
                preds["persistence"], valids["persistence"] = d["Y"][t], d["OBS"][t]
                preds["doy_clim"] = d["CLIM"][d["BIN"][tk]]; valids["doy_clim"] = oc[None].expand_as(tobs)
                common = tobs.clone()
                for p in PRED:
                    common &= valids[p]
                for p in PRED:
                    e = preds[p] - truth
                    for s, m in (("common", common), ("own", tobs & valids[p])):
                        mf = m.double()
                        acc[s][k][p] += torch.stack([(e.abs().double() * mf).sum(), (e.double() ** 2 * mf).sum(), mf.sum()])
                    ga = TR.grad_acc(preds[p], truth, common, LAGS)
                    for L in LAGS:
                        gacc[k][p][L] += ga[L]
                    if k == 1:
                        r = d["REG"][None].expand_as(common)[common]
                        racc[p][:, 0] += torch.bincount(r, weights=e[common].abs().double(), minlength=nreg)
                        racc[p][:, 1] += torch.bincount(r, minlength=nreg).double()
        res = {}
        for s in ("common", "own"):
            res[s] = {p: {str(k): dict(mae=float(acc[s][k][p][0] / acc[s][k][p][2].clamp(min=1)), rmse=float((acc[s][k][p][1] / acc[s][k][p][2].clamp(min=1)) ** 0.5),
                                       n=int(acc[s][k][p][2])) for k in range(1, K + 1)} for p in PRED}
        res["grad"] = {p: {str(k): {str(L): dict(ratio=float(gacc[k][p][L][0] / gacc[k][p][L][1].clamp(min=1e-9)),
                                                  corr=float(gacc[k][p][L][2] / (gacc[k][p][L][3] * gacc[k][p][L][4]).clamp(min=1e-18).sqrt()),
                                                  n_pairs=int(gacc[k][p][L][5])) for L in LAGS} for k in range(1, K + 1)} for p in PRED}
        res["region_lead1_common"] = {p: {str(r): dict(mae=float(racc[p][r, 0] / racc[p][r, 1].clamp(min=1)), n=int(racc[p][r, 1])) for r in range(1, nreg)} for p in PRED}
        res["mean_lead_mae_model_own"] = float(np.mean([res["own"]["model"][str(k)]["mae"] for k in range(1, K + 1)]))
        if bp is not None:
            res["band_own"] = {str(k): dict(l1=float(bacc[k][0] / bacc[k][1].clamp(min=1)), n=int(bacc[k][1])) for k in range(1, K + 1)}
            res["mean_lead_band_l1_model_own"] = float(np.mean([res["band_own"][str(k)]["l1"] for k in range(1, K + 1)]))
        res["mean_lead_mae_model_common"] = float(np.mean([res["common"]["model"][str(k)]["mae"] for k in range(1, K + 1)]))
        q = lambda xs: (lambda v: dict(mean=float(v.mean()), sd=float(v.std()), p10=float(v.quantile(.1)), p90=float(v.quantile(.9))))(torch.cat(xs))
        res["factors_day_mean_over_ocean"] = {k_: q(v) for k_, v in fac.items() if v}
        res["per_day_factor_means"] = {k_: torch.cat(v).cpu().tolist() for k_, v in fac.items() if v}
        res["val_days"] = [str(x)[:10] for x in d["times"][sidx]]
        if src is not None:
            tot = sm_sum.sum(0); nn_ = sm_n.sum(0)
            reg = d["REG"]
            res["source_lead1"] = dict(mean=float(tot.sum() / nn_.sum().clamp(min=1)), n=int(nn_.sum()),
                                       by_season={nm: dict(mean=float(sm_sum[q].sum() / sm_n[q].sum().clamp(min=1)), n=int(sm_n[q].sum())) for q, nm in enumerate(["DJF", "MAM", "JJA", "SON"])},
                                       by_region={str(r): dict(mean=float(tot[reg == r].sum() / nn_[reg == r].sum().clamp(min=1)), n=int(nn_[reg == r].sum())) for r in range(1, nreg)})
            src_maps.update(sum=sm_sum.cpu().numpy(), n=sm_n.cpu().numpy())
            if collect_pd:
                src_maps.update(pd_x=torch.cat(pd_x).numpy(), pd_s=torch.cat(pd_s).numpy())
            src.train()
        model.train()
        return res

    def wind_summary():
        if "wind" not in a.terms:
            return None
        ang = torch.linspace(0, 2 * np.pi, 13, device=dev)[:-1]
        p, q = model.wind_coef(torch.sin(ang), torch.cos(ang))
        return dict(doy=[int(round(float(x) / (2 * np.pi) * 365.25)) for x in ang], c_m_s_per_Pa=torch.hypot(p, q).tolist(),
                    theta_deg=torch.rad2deg(torch.atan2(q, p)).tolist(), wp=model.wp.tolist(), wq=model.wq.tolist())

    def save_factor_maps():
        """a / b (12, H, W) of the current model at 12 mid-month doys -> factor_maps.npz."""
        maps = {}
        doys = np.array([15 + 30.4 * m for m in range(12)])
        with torch.no_grad():
            ang = torch.as_tensor(2 * np.pi * doys / 365.25, device=dev, dtype=torch.float32)
            x = torch.cat([d["STATIC"][None].expand(12, -1, -1, -1), torch.sin(ang).view(12, 1, 1, 1).expand(12, 1, H, W),
                           torch.cos(ang).view(12, 1, 1, 1).expand(12, 1, H, W)], 1)
            if "rea" in a.terms:
                maps["a"] = (model.a(x) * OCf).cpu().numpy()
            if "geo" in a.terms:
                maps["b"] = (model.b(x) * OCf).cpu().numpy()
        np.savez_compressed(out / "factor_maps.npz", doy=doys, ocean=d["ocean"], lat=d["lat"], lon=d["lon"], **maps)

    def save_source_maps():
        """Season-mean lead-1 source (pred - departure value, ln CHL / day) per pixel over the evaluated val days."""
        if src is None or "sum" not in src_maps:
            return
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(src_maps["n"] > 0, src_maps["sum"] / src_maps["n"], np.nan).astype("float32")
        np.savez_compressed(out / "source_maps.npz", seasons=np.array(["DJF", "MAM", "JJA", "SON"]), mean=mean, n=src_maps["n"].astype("int32"),
                            ocean=d["ocean"], lat=d["lat"], lon=d["lon"], region=d["REG"].cpu().numpy())
        if "pd_x" in src_maps:
            zs = d["info"]["source"]["z_stats"]
            np.savez_compressed(out / "source_pd.npz", x=src_maps["pd_x"].astype("float32"), s=src_maps["pd_s"].astype("float32"),
                                channels=np.array(d["info"]["source"]["channels"]), z_mean=np.array([zs[c]["mean"] for c in SRC_CH[:7]]),
                                z_sd=np.array([zs[c]["sd"] for c in SRC_CH[:7]]), c_mean=d["CMU"], c_sd=d["CSD"])

    def alpha_mean():
        al = model.alpha.detach()
        al = al[..., 0] if a.alpha_time == "harm" else al
        return [round(float(x), 3) for x in al.mean((1, 2) if al.dim() == 3 else (1,))]

    hist = dict(epoch=[], train_l1=[], train_l1_lead=[], val_model_own=[], val_model_common=[], val_n_own=[], val_mean_lead=[],
                factors=[], wind=[], corr_pen=[], epoch_s=[], steps=[], source=[])
    cfg = dict(vars(a)); cfg.update(store="blk4", data=d["info"], n_train_starts=len(tr_s), n_val_starts=len(va_s), val_years=[2017, 2021],
                                    test_years=[2019, 2023], n_params=sum(p.numel() for p in model.parameters()))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    best, best_ep, steps = np.inf, -1, 0
    torch.cuda.reset_peak_memory_stats()
    ev0 = evaluate(va_s)
    print(f"init  val model own {[round(ev0['own']['model'][str(k)]['mae'], 4) for k in range(1, K + 1)]}  "
          f"common: model {[round(ev0['common']['model'][str(k)]['mae'], 4) for k in range(1, K + 1)]} "
          f"pers {[round(ev0['common']['persistence'][str(k)]['mae'], 4) for k in range(1, K + 1)]} "
          f"x0.25 {[round(ev0['common']['adv_x0.25'][str(k)]['mae'], 4) for k in range(1, K + 1)]}  ({time.time() - t0:.0f}s)", flush=True)
    metrics = dict(init=ev0)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))      # baselines visible during training
    for ep in range(0 if a.eval_only else a.epochs):
        te0 = time.time()
        perm = rng.permutation(len(tr_s)); nb = len(perm) // a.batch
        if a.max_batches:
            nb = min(nb, a.max_batches)
        lsum = torch.zeros(K, device=dev); cpen = 0.0
        for bi in range(nb):
            bt = tr_s[perm[bi * a.batch:(bi + 1) * a.batch]]; days = batch_days(bt); t = days[:, 0]
            col = {}
            Um, okm, _, _ = velocity(days, col)
            ls = []
            for k in range(1, K + 1):
                pr, val = model_forecast(Um, okm, days, k)
                m = val & d["OBS"][days[:, k]]
                if bp is not None:
                    bpp, vb = bp(pr, m); bpt, _ = bp(d["Y"][days[:, k]], m)
                    ls.append(((bpp - bpt).abs() * vb).sum() / vb.sum().clamp(min=1))     # 0 (not NaN) if no band-valid px
                else:
                    ls.append((pr - d["Y"][days[:, k]]).abs()[m].mean())
            ls = torch.stack(ls)
            loss = (ls * lw).sum()
            if "corr" in a.terms:
                pen = a.corr_mag * col["corr_mag"] + a.corr_grad * col["corr_grad"]; loss = loss + pen; cpen += float(pen.detach())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); steps += 1
            lsum += ls.detach()
        ev = evaluate(va_s)
        ltr = (lsum / max(nb, 1)).tolist()
        hist["epoch"].append(ep + 1); hist["train_l1"].append(float(np.dot(ltr, a.lead_weights) / sum(a.lead_weights))); hist["train_l1_lead"].append(ltr)
        hist["val_model_own"].append([ev["own"]["model"][str(k)]["mae"] for k in range(1, K + 1)])
        hist["val_model_common"].append([ev["common"]["model"][str(k)]["mae"] for k in range(1, K + 1)])
        hist["val_n_own"].append([ev["own"]["model"][str(k)]["n"] for k in range(1, K + 1)])
        hist["val_mean_lead"].append(ev["mean_lead_mae_model_own"]); hist["factors"].append(ev["factors_day_mean_over_ocean"])
        hist["source"].append(ev.get("source_lead1")); hist["wind"].append(wind_summary()); hist["corr_pen"].append(cpen / max(nb, 1)); hist["epoch_s"].append(time.time() - te0); hist["steps"].append(steps)
        if a.vel_param == "decomp":
            hist.setdefault("alpha", []).append(model.alpha.detach().cpu().tolist())
        (out / "history.json").write_text(json.dumps(hist, indent=1))
        crit = ev["mean_lead_band_l1_model_own"] if bp is not None else ev["mean_lead_mae_model_own"]
        if crit < best:
            best, best_ep = crit, ep + 1
            torch.save(dict(model=model.state_dict(), epoch=ep + 1, val=best, terms=a.terms, **({"source": src.state_dict()} if src is not None else {})), out / "model.pt")
            save_factor_maps(); save_source_maps()
            if a.vel_param == "decomp":
                (out / "alpha.json").write_text(json.dumps(dict(alpha=model.alpha.detach().cpu().tolist(), dims=["component", "region", "season" if a.alpha_time == "season" else "harmonic [c0, sin, cos]"],
                                                                components=TR.DECOMP_COMPS, seasons=["DJF", "MAM", "JJA", "SON"], epoch=ep + 1), indent=1))
            metrics["best"] = ev
        fs = ev["factors_day_mean_over_ocean"]
        print(f"epoch {ep + 1:3d}  train {hist['train_l1'][-1]:.4f}  val own {[round(x, 4) for x in hist['val_model_own'][-1]]}  "
              f"common {[round(x, 4) for x in hist['val_model_common'][-1]]}  n own l1 {hist['val_n_own'][-1][0]}  "
              + "  ".join(f"{k_} {v['mean']:.3f}" for k_, v in fs.items())
              + (f"  S l1 {ev['source_lead1']['mean']:+.4f}" if src is not None else "")
              + (f"  band l1 {ev['mean_lead_band_l1_model_own']:.4f}" if bp is not None else "")
              + (f"  alpha mean by comp {alpha_mean()}" if a.vel_param == "decomp" else "")
              + (f"  wind c {np.mean(hist['wind'][-1]['c_m_s_per_Pa']):.3f} theta {np.mean(hist['wind'][-1]['theta_deg']):.0f}" if hist['wind'][-1] else "")
              + f"  ({hist['epoch_s'][-1]:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
    if a.eval_only:
        metrics["best"] = ev0; best_ep = 0
        torch.save(dict(model=model.state_dict(), epoch=0, val=ev0["mean_lead_mae_model_own"], terms=a.terms, **({"source": src.state_dict()} if src is not None else {})), out / "model.pt")
    # final: reload best, factor maps for 12 mid-month doys
    ck = torch.load(out / "model.pt", map_location=dev); model.load_state_dict(ck["model"])
    save_factor_maps()
    if src is not None:                                       # best model: source maps + partial-dependence samples
        src.load_state_dict(ck["source"])
        refm = None
        if a.ref_run:                                         # step-1 velocity-only run scored on the same pixels
            ckr = torch.load(CATEGORIES["tracer"] / a.ref_run / "model.pt", map_location=dev)
            cr = json.load(open(CATEGORIES["tracer"] / a.ref_run / "config.json"))
            refm = TR.VelocityModel(cr["terms"], n_static=d["STATIC"].shape[0], hidden=cr["hidden"], a0=cr["a_init"], b0=cr["b_init"],
                                    b_range=(cr["b_min"], 1.5), cell=cr["corr_cell"], corr_hidden=cr["corr_hidden"]).to(dev)
            refm.load_state_dict(ckr["model"]); refm.eval()
        evf = evaluate(va_s, collect_pd=True, ref=refm); save_source_maps()
        metrics["final"] = evf; metrics["ref_run"] = a.ref_run
    metrics.update(best_epoch=best_ep, best_val_mae_mean_leads=best, wind=wind_summary(), wall_s=time.time() - t0,
                   peak_gb=torch.cuda.max_memory_allocated() / 2**30, steps=steps)
    if bp is not None or a.vel_param == "decomp":
        metrics["selection"] = "mean-lead val band L1 (own set)" if bp is not None else "mean-lead val pixel MAE (own set)"
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"done  best epoch {best_ep}  mean-lead val {best:.4f}  wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
