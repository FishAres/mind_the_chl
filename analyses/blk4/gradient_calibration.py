"""Gradient calibration of the best Black Sea 4 km run, and a refined screening of candidate input fields.

Part A (VAL days of preds.npz; post-hoc gain also on TEST days of preds.npz)
  A1. OLS of true pair differences on predicted ones, dt = b dp (+ a), on the trainer's pairs (lag L along x and y,
      truth observed at both ends, >= 1 hidden end). Through the origin and with intercept; reverse regression
      dp = b' dt for reference. Overall per lag (1, 2, 4, 8, 16 px); at lag 1 per distance-to-visible bin and per
      region (rules of gradient_diagnosis). 95 % CIs: day-block bootstrap (lag 1 only).
  A2. Post-hoc gain on hidden pixels: pred' = S + g (pred - S), S = Gaussian blur of pred normalised over ocean px
      (sigma 1, 2 px). Hidden-pixel MAE (ln CHL, pooled over pixels) for g = 0.8 .. 2.0; best g per sigma chosen on
      VAL, then TEST MAE at that g vs g = 1.

Part B (observed CHL pixels; VAL days of preds.npz + every 8th TRAIN day)
  Fields: SST (sst32), SLA, current speed, relative vorticity, strain rate (from u_curr, v_curr), ln CHL t-1.
  B1. Pearson / Spearman of |grad field| (and of |field| for speed / vorticity / strain) with |grad ln CHL|.
  B2. mean |cos| between grad field and grad ln CHL where both magnitudes exceed their median (random: 2/pi).
  B3. coastal band excluded: ocean px within 0 (none), 2, 4, 8 px of land.
  B4. +-1 px tolerance for B1: max |grad field| over the 3x3 neighbourhood.
  B5. gradient_diagnosis 4a (forward differences on s x s block means, SST) repeated with the coastal bands.

Run from the repo root: python -m analyses.blk4.gradient_calibration   (CPU only)
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
from scipy.stats import spearmanr
from chl_cnn.viz import context_for
from chl_cnn.plotting import OKABE_ITO
from analyses.blk4.error_breakdown import regions_4km
from analyses.blk4.gradient_diagnosis import RUNS, LAGS, BLOCKS, MIN_N, pairs, blockmean, gradmag

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RUN = RUNS["best (sl lam0.3 cont)"]
SIGMAS = [1, 2]
GAINS = np.round(np.arange(0.8, 2.0001, 0.1), 2)
BANDS = [0, 2, 4, 8]                       # exclude ocean px with distance to land <= band (px); 0 = nothing excluded
TRAIN_STRIDE = 8
NBOOT = 1000
DIST_LABELS = ["1 (<1.5)", "2-3", "4-7", "8-15", "16+"]
FIELDS = ["SST", "SLA", "current speed", "rel. vorticity", "strain rate", "ln CHL t-1"]
VALUE_FIELDS = ["current speed", "rel. vorticity", "strain rate"]
R_EARTH_KM = 6371.0


# ============================== shared ==============================
def setup():
    ctx = context_for(RUN)
    d, ocean = ctx["d"], ctx["ocean"]
    oc = np.asarray(d["ocean"]); rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
    ys = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0])); xs = slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
    assert (oc[ys, xs] == ocean).all()
    rd = lambda name, i: np.asarray(d[name][i, ys, xs]).astype("float32")
    return ctx, ocean, rd


def dist_bin(dist):
    """gradient_diagnosis distance bins (Euclidean px): 1, 2-3, 4-7, 8-15, 16+; -1 = unbinned (< 0.5)."""
    edges = np.array([0.5, 1.5, 3.5, 7.5, 15.5, np.inf])
    g = np.searchsorted(edges, dist, side="right") - 1
    g[dist < 0.5] = -1
    return g


# ============================== Part A: data ==============================
SUMS = ["n", "x", "y", "xx", "yy", "xy"]                   # x = dpred, y = dtruth


def sums(dp, dt):
    dp = dp.astype("float64"); dt = dt.astype("float64")
    return np.array([dp.size, dp.sum(), dt.sum(), (dp * dp).sum(), (dt * dt).sum(), (dp * dt).sum()])


def collect_A(ctx, ocean, rd):
    z = np.load(RUN / "preds.npz")
    pos = {int(i): k for k, i in enumerate(z["idx"])}
    reg, _ = regions_4km(ocean)
    lagsum = {L: np.zeros(6) for L in LAGS}
    tab = {c: [] for c in ["dp", "dt", "dist", "region", "day"]}
    gainsum = {(sp, s): np.zeros((len(GAINS), 2)) for sp in ("val", "test") for s in SIGMAS}   # sum|err|, n
    perday_test = {s: [] for s in SIGMAS}                       # (len(GAINS),) per-day test MAE
    dq = dict(n_val_days=0, n_test_days=0, days_no_visible=0, hidden_off_obs=0, pred_nan_ocean=0, pred_all_zero_days=0,
              truth_nan_hidden=0)
    norm = {s: gaussian_filter(ocean.astype("float64"), s, mode="constant") for s in SIGMAS}
    for split, key in (("val", "val_idx"), ("test", "test_idx")):
        for i in (int(v) for v in z[key]):
            k = pos[i]; dq[f"n_{split}_days"] += 1
            y = ctx["y"](i); obs = np.isfinite(y) & ocean
            hid = z["hidden"][k] & ocean
            dq["hidden_off_obs"] += int((hid & ~obs).sum()); hid &= obs
            P = z["pred"][k].astype("float32")
            dq["pred_nan_ocean"] += int((~np.isfinite(P) & ocean).sum())
            if np.all(P[ocean] == 0):
                dq["pred_all_zero_days"] += 1; continue
            # --- A2 gain test
            for s in SIGMAS:
                S = (gaussian_filter(np.where(ocean, P, 0).astype("float64"), s, mode="constant") / np.maximum(norm[s], 1e-12)).astype("float32")
                h = (P - S)[hid]; sb = S[hid]; t = y[hid]
                e = np.abs(sb[None] + GAINS[:, None].astype("float32") * h[None] - t[None])     # (G, n)
                gainsum[(split, s)][:, 0] += e.sum(1, dtype="float64"); gainsum[(split, s)][:, 1] += h.size
                if split == "test":
                    perday_test[s].append((e.sum(1, dtype="float64"), h.size))
            if split != "val":
                continue
            # --- A1 pairs (val only)
            vis = obs & ~hid
            if vis.any():
                dist = distance_transform_edt(~vis).astype("float32")
            else:
                dq["days_no_visible"] += 1; dist = np.full(y.shape, np.inf, "float32")
            for L in LAGS:
                for ax in (0, 1):
                    o1, o2 = pairs(obs, L, ax); h1, h2 = pairs(hid, L, ax)
                    t1, t2 = pairs(y, L, ax); p1, p2 = pairs(P, L, ax)
                    m = o1 & o2 & (h1 | h2)
                    lagsum[L] += sums((p2 - p1)[m], (t2 - t1)[m])
                    if L == 1:
                        d1, d2 = pairs(dist, 1, ax); r1, r2 = pairs(reg, 1, ax)
                        dd1, dd2 = np.where(h1, d1, -1), np.where(h2, d2, -1)
                        tab["dp"].append((p2 - p1)[m]); tab["dt"].append((t2 - t1)[m])
                        tab["dist"].append(np.maximum(dd1, dd2)[m]); tab["region"].append(np.where(dd1 >= dd2, r1, r2)[m])
                        tab["day"].append(np.full(int(m.sum()), i, np.int32))
    tab = {c: np.concatenate(v) for c, v in tab.items()}
    dq["table_nan"] = dict(dp=int(np.isnan(tab["dp"]).sum()), dt=int(np.isnan(tab["dt"]).sum()), dist_inf=int(np.isinf(tab["dist"]).sum()))
    return dict(lagsum=lagsum, tab=tab, gainsum=gainsum, perday_test=perday_test, dq=dq)


# ============================== Part A: analysis ==============================
def reg_from_sums(s):
    n, x, y, xx, yy, xy = s
    if n < 2 or xx == 0:
        return dict(n_pairs=int(n))
    sxx, syy, sxy = xx - x * x / n, yy - y * y / n, xy - x * y / n
    b = sxy / sxx
    return dict(slope_origin=xy / xx, slope=b, intercept=(y - b * x) / n, rev_slope_origin=xy / yy, rev_slope=sxy / syy,
                r=sxy / np.sqrt(sxx * syy), n_pairs=int(n))


def grouped_reg(tab, gid, labels, rng):
    """Regression per group, with a day-block bootstrap 95 % CI of slope_origin and slope."""
    days, dinv = np.unique(tab["day"], return_inverse=True)
    G, D = len(labels), len(days)
    ok = (gid >= 0) & (gid < G)                 # dist = inf (days with no visible px) falls outside the bins
    cols = dict(n=np.ones_like(tab["dp"], "float64"), x=tab["dp"], y=tab["dt"], xx=tab["dp"] ** 2, yy=tab["dt"] ** 2, xy=tab["dp"] * tab["dt"])
    idx = gid[ok] * D + dinv[ok]
    S = np.stack([np.bincount(idx, weights=cols[c][ok].astype("float64"), minlength=G * D).reshape(G, D) for c in SUMS], -1)  # (G, D, 6)
    bi = rng.integers(0, D, (NBOOT, D))
    cnt = np.stack([np.bincount(b, minlength=D) for b in bi])                    # (B, D) multiplicity
    rows = []
    for g, lab in enumerate(labels):
        r = dict(group=lab, **reg_from_sums(S[g].sum(0)), n_days=int((S[g, :, 0] > 0).sum()))
        bs = cnt @ S[g]                                                          # (B, 6)
        n, x, y, xx, yy, xy = bs.T
        with np.errstate(invalid="ignore", divide="ignore"):
            so = xy / xx; sl = (xy - x * y / n) / (xx - x * x / n)
        r["slope_origin_ci95"] = [float(np.nanpercentile(so, 2.5)), float(np.nanpercentile(so, 97.5))]
        r["slope_ci95"] = [float(np.nanpercentile(sl, 2.5)), float(np.nanpercentile(sl, 97.5))]
        r["flag_n_lt5"] = r["n_pairs"] < MIN_N or r["n_days"] < MIN_N
        rows.append(r)
    return rows


def analyse_A(a, names):
    rng = np.random.default_rng(0)
    tab = a["tab"]; res = {}
    res["lag"] = {L: reg_from_sums(a["lagsum"][L]) for L in LAGS}
    res["overall_lag1"] = grouped_reg(tab, np.zeros(len(tab["dp"]), int), ["all"], rng)[0]
    gid = dist_bin(tab["dist"]); res["dist_unbinned"] = int(((gid < 0) | (gid >= len(DIST_LABELS))).sum())
    res["distance"] = grouped_reg(tab, gid, DIST_LABELS, rng)
    res["region"] = grouped_reg(tab, tab["region"].astype(int) - 1, [names[r] for r in range(1, 8)], rng)
    gain = {}
    for s in SIGMAS:
        v = a["gainsum"][("val", s)]; t = a["gainsum"][("test", s)]
        mv, mt = v[:, 0] / v[:, 1], t[:, 0] / t[:, 1]
        jb, j1 = int(np.argmin(mv)), int(np.argmin(np.abs(GAINS - 1.0)))
        pd = np.array([e / n for e, n in a["perday_test"][s]])                  # (days, G)
        dd = pd[:, jb] - pd[:, j1]
        bi = rng.integers(0, len(dd), (NBOOT, len(dd)))
        tot = np.array([[e[jb], e[j1], n] for e, n in a["perday_test"][s]])     # pooled-pixel bootstrap over days
        bd = (tot[bi, 0].sum(1) - tot[bi, 1].sum(1)) / tot[bi, 2].sum(1)
        gain[f"sigma {s}"] = dict(gains=GAINS.tolist(), val_mae=mv.tolist(), test_mae=mt.tolist(), n_val_px=int(v[0, 1]), n_test_px=int(t[0, 1]),
                                  best_g_val=float(GAINS[jb]), val_mae_g1=float(mv[j1]), val_mae_best=float(mv[jb]),
                                  test_mae_g1=float(mt[j1]), test_mae_at_best_g=float(mt[jb]), test_delta=float(mt[jb] - mt[j1]),
                                  test_delta_ci95_day_bootstrap=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                                  test_days_improved=int((dd < 0).sum()), test_days_worse=int((dd > 0).sum()), n_test_days=int(len(dd)))
    res["gain"] = gain
    return res


# ============================== Part B: data ==============================
def metrics(lat, lon, shape):
    dy = np.abs(np.diff(lat)).mean() * np.pi / 180 * R_EARTH_KM                  # km per row (lat ascending -> north)
    dx = np.abs(np.diff(lon)).mean() * np.pi / 180 * R_EARTH_KM * np.cos(np.deg2rad(lat))   # km per column, per row
    assert len(lat) == shape[0]
    return float(dy), dx[:, None].astype("float64")


def grad(a, ok, dy, dx):
    """Centred-difference gradient (per km); NaN unless the pixel and its 4 neighbours are valid."""
    an = np.where(ok, a, np.nan).astype("float64")
    gy, gx = np.gradient(an)
    gy, gx = gy / dy, gx / dx
    bad = ~ok
    gy[bad] = np.nan; gx[bad] = np.nan
    return gx, gy


def max3(m):
    """3x3 max of a magnitude (NaN neighbours ignored); NaN where the centre is NaN."""
    f = maximum_filter(np.where(np.isfinite(m), m, -np.inf), size=3, mode="constant", cval=-np.inf)
    return np.where(np.isfinite(m), f, np.nan)


def collect_B(ctx, ocean, rd):
    z = np.load(RUN / "preds.npz")
    days = [("val", int(i)) for i in z["val_idx"]] + [("train", int(i)) for i in ctx["tr"][::TRAIN_STRIDE]]
    dy, dx = metrics(ctx["lat"], ctx["lon"], ocean.shape)
    dland = distance_transform_edt(ocean).astype("float32")                     # 1 = adjacent to land
    cols = ["split", "dland", "gc"] + [f"{p}_{f}" for f in FIELDS for p in ("g", "g3", "cos")] + [f"v_{f}" for f in VALUE_FIELDS]
    tab = {c: [] for c in cols}
    blk = {(sp, s, b): [] for sp in ("val", "train") for s in BLOCKS for b in BANDS}
    dq = dict(n_days=dict(val=0, train=0), field_nan_ocean_px_day={f: 0 for f in ["sst32", "sla", "u_curr", "v_curr"]},
              field_all_zero_days={f: 0 for f in ["sst32", "sla", "u_curr", "v_curr", "chl"]}, chl_obs_px=0, chl_grad_px=0,
              tm1_day_missing=0, ocean_px=int(ocean.sum()))
    for sp, i in days:
        dq["n_days"][sp] += 1
        y = ctx["y"](i); obs = np.isfinite(y) & ocean
        raw = {f: rd(f, i) for f in ["sst32", "sla", "u_curr", "v_curr"]}
        ok = {}
        for f, a in raw.items():
            ok[f] = ocean & np.isfinite(a)
            dq["field_nan_ocean_px_day"][f] += int((ocean & ~ok[f]).sum())
            if ok[f].any() and np.all(a[ok[f]] == 0):
                dq["field_all_zero_days"][f] += 1; ok[f][:] = False      # discard the day for that field
        if obs.any() and np.all(y[obs] == 0):
            dq["field_all_zero_days"]["chl"] += 1; continue
        dq["chl_obs_px"] += int(obs.sum())
        gcx, gcy = grad(y, obs, dy, dx); gc = np.hypot(gcx, gcy)
        # currents: m/s, derivatives per km -> 1/day
        u, v = raw["u_curr"], raw["v_curr"]; uok = ok["u_curr"] & ok["v_curr"]
        ux, uy = grad(u, uok, dy, dx); vx, vy = grad(v, uok, dy, dx)
        c = 86400.0 / 1000.0
        vort = (vx - uy) * c; strain = np.hypot(ux - vy, vx + uy) * c
        spd = np.where(uok, np.hypot(u, v), np.nan)
        if i - 1 >= 0:
            yp = ctx["y"](i - 1); okp = np.isfinite(yp) & ocean
        else:
            dq["tm1_day_missing"] += 1; yp = np.full_like(y, np.nan); okp = np.zeros_like(obs)
        F = {"SST": grad(raw["sst32"], ok["sst32"], dy, dx), "SLA": grad(raw["sla"], ok["sla"], dy, dx),
             "current speed": grad(spd, np.isfinite(spd), dy, dx),
             "rel. vorticity": grad(vort, np.isfinite(vort), dy, dx), "strain rate": grad(strain, np.isfinite(strain), dy, dx),
             "ln CHL t-1": grad(yp, okp, dy, dx)}
        V = {"current speed": spd, "rel. vorticity": np.abs(vort), "strain rate": strain}
        m = np.isfinite(gc); dq["chl_grad_px"] += int(m.sum())
        tab["split"].append(np.full(int(m.sum()), sp == "val", bool)); tab["dland"].append(dland[m]); tab["gc"].append(gc[m].astype("float32"))
        for f, (gx, gy) in F.items():
            g = np.hypot(gx, gy)
            with np.errstate(invalid="ignore", divide="ignore"):
                cs = np.abs(gx * gcx + gy * gcy) / (g * gc)
            tab[f"g_{f}"].append(g[m].astype("float32")); tab[f"g3_{f}"].append(max3(g)[m].astype("float32"))
            tab[f"cos_{f}"].append(cs[m].astype("float32"))
        for f, a in V.items():
            tab[f"v_{f}"].append(a[m].astype("float32"))
        # B5: gradient_diagnosis 4a (forward differences on block means) with coastal bands
        for b in BANDS:
            far = dland > b
            for s in BLOCKS:
                gcb = gradmag(blockmean(y, obs & far, s)) / s
                gsb = gradmag(blockmean(raw["sst32"], ok["sst32"] & far, s)) / s
                mm = np.isfinite(gcb) & np.isfinite(gsb)
                blk[(sp, s, b)].append(np.stack([gcb[mm], gsb[mm]], 1).astype("float32"))
    tab = {c: np.concatenate(v) for c, v in tab.items()}
    blk = {k: np.concatenate(v) for k, v in blk.items()}
    dq["tab_nan"] = {c: int(np.isnan(tab[c]).sum()) for c in tab if tab[c].dtype.kind == "f"}
    dq["grid_km"] = dict(dy=dy, dx_min=float(dx.min()), dx_max=float(dx.max()))
    return dict(tab=tab, blk=blk, dq=dq)


# ============================== Part B: analysis ==============================
def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b); n = int(m.sum())
    if n < MIN_N:
        return dict(pearson=None, spearman=None, n_px=n, flag_n_lt5=True)
    return dict(pearson=float(np.corrcoef(a[m], b[m])[0, 1]), spearman=float(spearmanr(a[m], b[m]).statistic), n_px=n)


def align(gc, g, cs):
    m = np.isfinite(gc) & np.isfinite(g) & np.isfinite(cs)
    if m.sum() < MIN_N:
        return dict(mean_abs_cos=None, n_px=int(m.sum()), flag_n_lt5=True)
    top = m & (gc > np.median(gc[m])) & (g > np.median(g[m]))
    zero = int((m & ((gc == 0) | (g == 0))).sum())
    return dict(mean_abs_cos=float(cs[top].mean()), n_px=int(top.sum()), n_px_both_finite=int(m.sum()),
                mean_abs_cos_all_finite=float(cs[m].mean()), n_zero_magnitude=zero)


def analyse_B(b):
    t = b["tab"]; res = {"corr": {}, "corr_tol3x3": {}, "corr_value": {}, "align": {}}
    for band in BANDS:
        keep = t["dland"] > band
        for f in FIELDS:
            res["corr"][f"{f} | band {band}"] = corr(t["gc"][keep], t[f"g_{f}"][keep])
            res["corr_tol3x3"][f"{f} | band {band}"] = corr(t["gc"][keep], t[f"g3_{f}"][keep])
            res["align"][f"{f} | band {band}"] = align(t["gc"][keep], t[f"g_{f}"][keep], t[f"cos_{f}"][keep])
        for f in VALUE_FIELDS:
            res["corr_value"][f"{f} | band {band}"] = corr(t["gc"][keep], t[f"v_{f}"][keep])
    res["corr_by_split_band0"] = {sp: {f: corr(t["gc"][t["split"] == (sp == "val")], t[f"g_{f}"][t["split"] == (sp == "val")]) for f in FIELDS}
                                  for sp in ("val", "train")}
    blk = {}
    for sub in ("val", "val+train"):
        for band in BANDS:
            for s in BLOCKS:
                a = b["blk"][("val", s, band)] if sub == "val" else np.concatenate([b["blk"][("val", s, band)], b["blk"][("train", s, band)]])
                blk[f"{sub} | band {band} | {s}x{s}"] = corr(a[:, 0], a[:, 1])
    res["sst_block"] = blk
    return res


# ============================== reporting ==============================
def f3(v):
    return "-" if v is None else f"{v:.3f}"


def tables(A, B):
    o = ["\n## A1. dtruth ~ dpred, VAL, pairs with truth at both ends and >= 1 hidden end",
         "| lag px | slope (origin) | slope (intercept) | intercept | reverse slope dp~dt (origin) | r | n pairs |", "|---|---|---|---|---|---|---|"]
    for L, r in A["lag"].items():
        o.append(f"| {L} | {f3(r['slope_origin'])} | {f3(r['slope'])} | {r['intercept']:.4f} | {f3(r['rev_slope_origin'])} | {f3(r['r'])} | {r['n_pairs']:,} |")
    for key, title in (("distance", "distance of hidden end to nearest visible px"), ("region", "region of hidden end")):
        o += [f"\n### lag 1 by {title} (95 % CI: day-block bootstrap)",
              "| group | slope (origin) [CI] | slope (intercept) [CI] | reverse slope (origin) | r | n pairs | n days |", "|---|---|---|---|---|---|---|"]
        for r in [A["overall_lag1"]] + A[key]:
            fl = " (n<5)" if r.get("flag_n_lt5") else ""
            o.append(f"| {r['group']}{fl} | {f3(r['slope_origin'])} [{r['slope_origin_ci95'][0]:.3f}, {r['slope_origin_ci95'][1]:.3f}] | "
                     f"{f3(r['slope'])} [{r['slope_ci95'][0]:.3f}, {r['slope_ci95'][1]:.3f}] | {f3(r['rev_slope_origin'])} | {f3(r['r'])} | {r['n_pairs']:,} | {r['n_days']} |")
    o += ["\n## A2. post-hoc gain, hidden-pixel MAE (ln CHL)", "| sigma | best g (val) | val MAE g=1 | val MAE best g | test MAE g=1 | test MAE best g | test delta [95% CI] | test days better / worse | n val px | n test px | n test days |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, g in A["gain"].items():
        o.append(f"| {s} | {g['best_g_val']:.1f} | {g['val_mae_g1']:.5f} | {g['val_mae_best']:.5f} | {g['test_mae_g1']:.5f} | {g['test_mae_at_best_g']:.5f} | "
                 f"{g['test_delta']:+.5f} [{g['test_delta_ci95_day_bootstrap'][0]:+.5f}, {g['test_delta_ci95_day_bootstrap'][1]:+.5f}] | "
                 f"{g['test_days_improved']} / {g['test_days_worse']} | {g['n_val_px']:,} | {g['n_test_px']:,} | {g['n_test_days']} |")
    o += ["\n| g | " + " | ".join(f"val s{s}" for s in SIGMAS) + " | " + " | ".join(f"test s{s}" for s in SIGMAS) + " |", "|---" * (1 + 2 * len(SIGMAS)) + "|"]
    for j, g in enumerate(GAINS):
        o.append(f"| {g:.1f} | " + " | ".join(f"{A['gain'][f'sigma {s}']['val_mae'][j]:.5f}" for s in SIGMAS) + " | "
                 + " | ".join(f"{A['gain'][f'sigma {s}']['test_mae'][j]:.5f}" for s in SIGMAS) + " |")
    o += ["\n## B1/B3. Pearson | Spearman of |grad field| with |grad ln CHL| (n px), by coastal band (px from land excluded)",
          "| field | " + " | ".join(f"band {b}" for b in BANDS) + " |", "|---" * (1 + len(BANDS)) + "|"]
    for f in FIELDS:
        o.append(f"| {f} | " + " | ".join((lambda r: f"{f3(r['pearson'])} / {f3(r['spearman'])} ({r['n_px']:,})")(B['corr'][f'{f} | band {b}']) for b in BANDS) + " |")
    o += ["\n### |field| (not its gradient) vs |grad ln CHL|", "| field | " + " | ".join(f"band {b}" for b in BANDS) + " |", "|---" * (1 + len(BANDS)) + "|"]
    for f in VALUE_FIELDS:
        o.append(f"| |{f}| | " + " | ".join((lambda r: f"{f3(r['pearson'])} / {f3(r['spearman'])} ({r['n_px']:,})")(B['corr_value'][f'{f} | band {b}']) for b in BANDS) + " |")
    o += ["\n## B4. with +-1 px tolerance (3x3 max of |grad field|)", "| field | " + " | ".join(f"band {b}" for b in BANDS) + " |", "|---" * (1 + len(BANDS)) + "|"]
    for f in FIELDS:
        o.append(f"| {f} | " + " | ".join((lambda r: f"{f3(r['pearson'])} / {f3(r['spearman'])} ({r['n_px']:,})")(B['corr_tol3x3'][f'{f} | band {b}']) for b in BANDS) + " |")
    o += ["\n## B2. mean |cos| (both magnitudes > median) ; random = 0.637", "| field | " + " | ".join(f"band {b}" for b in BANDS) + " |", "|---" * (1 + len(BANDS)) + "|"]
    for f in FIELDS:
        o.append(f"| {f} | " + " | ".join((lambda r: f"{f3(r['mean_abs_cos'])} ({r['n_px']:,})")(B['align'][f'{f} | band {b}']) for b in BANDS) + " |")
    o += ["\n## B1 split check (band 0): Pearson / Spearman (n)", "| field | val | train |", "|---|---|---|"]
    for f in FIELDS:
        o.append(f"| {f} | " + " | ".join((lambda r: f"{f3(r['pearson'])} / {f3(r['spearman'])} ({r['n_px']:,})")(B['corr_by_split_band0'][sp][f]) for sp in ("val", "train")) + " |")
    o += ["\n## B5. 4a repeated: |grad ln CHL| vs |grad SST| on block means, Pearson / Spearman (n)", "| days | band | 1x1 | 2x2 | 4x4 |", "|---|---|---|---|---|"]
    for sub in ("val", "val+train"):
        for b in BANDS:
            o.append(f"| {sub} | {b} | " + " | ".join((lambda r: f"{f3(r['pearson'])} / {f3(r['spearman'])} ({r['n_px']:,})")(B['sst_block'][f'{sub} | band {b} | {s}x{s}']) for s in BLOCKS) + " |")
    return "\n".join(o)


def plot_A(A, tab, fn):
    fig, ax = plt.subplots(2, 3, figsize=(20, 11)); ax = ax.ravel()
    C = dict(so=OKABE_ITO[0], sl=OKABE_ITO[1], rv=OKABE_ITO[2])
    a = ax[0]
    h = a.hexbin(tab["dp"], tab["dt"], gridsize=120, extent=(-1, 1, -1, 1), bins="log", cmap="cividis", mincnt=1)
    xx = np.array([-1, 1]); r = A["overall_lag1"]
    a.plot(xx, xx, color="grey", lw=1, ls=":", label="1:1")
    a.plot(xx, r["slope_origin"] * xx, color=C["so"], label=f"OLS dt~dp, origin: {r['slope_origin']:.2f}")
    a.plot(xx, xx / r["rev_slope_origin"], color=C["rv"], ls="--", label=f"reverse OLS dp~dt, origin (1/{r['rev_slope_origin']:.2f})")
    a.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="Δ pred ln CHL (lag 1)", ylabel="Δ truth ln CHL (lag 1)",
          title=f"Δtruth vs Δpred, val pairs with ≥1 hidden end (n {r['n_pairs']:,})")
    plt.colorbar(h, ax=a, label="pairs per bin (log)"); a.legend(fontsize=7, loc="upper left")

    def slopes(a, rows, labels, title, xlabel):
        x = np.arange(len(rows)); w = 0.27
        for j, (k, lab) in enumerate((("slope_origin", "dt~dp, origin"), ("slope", "dt~dp, intercept"), ("rev_slope_origin", "reverse dp~dt, origin"))):
            vals = [rr[k] for rr in rows]
            err = None
            if k + "_ci95" in rows[0]:
                err = np.array([[v - rr[k + "_ci95"][0], rr[k + "_ci95"][1] - v] for v, rr in zip(vals, rows)]).T
            a.bar(x + (j - 1) * w, vals, w, yerr=err, capsize=2, color=list(C.values())[j], label=lab)
        a.axhline(1, color="grey", lw=0.8)
        a.set_xticks(x); a.set_xticklabels([f"{l}\n(n {rr['n_pairs']:,})" for l, rr in zip(labels, rows)], rotation=30, ha="right", fontsize=7)
        a.set(title=title, xlabel=xlabel, ylabel="OLS slope"); a.legend(fontsize=7); a.grid(axis="y", alpha=0.3)

    km = 4.52
    lr = [A["lag"][L] for L in LAGS]
    slopes(ax[1], lr, [f"{L} px ({L * km:.0f} km)" for L in LAGS], "Regression slopes of Δtruth on Δpred by pair lag (val)", "pair lag")
    dr = A["distance"]
    slopes(ax[2], dr, [r_["group"] + (" (n<5)" if r_["flag_n_lt5"] else "") for r_ in dr],
           "Slopes by distance of hidden end to visible px (lag 1, val; 95% day-bootstrap CI)", "distance (px)")
    rr = A["region"]
    slopes(ax[3], rr, [r_["group"] + (" (n<5)" if r_["flag_n_lt5"] else "") for r_ in rr],
           "Slopes by region of hidden end (lag 1, val; 95% day-bootstrap CI)", "region")
    for j, s in enumerate(SIGMAS):
        g = A["gain"][f"sigma {s}"]
        ax[4].plot(GAINS, g["val_mae"], "o-", color=OKABE_ITO[j], label=f"val, sigma {s} px (n {g['n_val_px']:,} px)")
        ax[4].plot(GAINS, g["test_mae"], "s--", color=OKABE_ITO[j], alpha=0.6, label=f"test, sigma {s} px (n {g['n_test_px']:,} px)")
    ax[4].axvline(1, color="grey", lw=0.8)
    ax[4].set(title="Hidden-pixel MAE vs high-pass gain g: pred' = S + g (pred − S)", xlabel="gain g", ylabel="hidden-pixel MAE (ln CHL)")
    ax[4].legend(fontsize=7); ax[4].grid(alpha=0.3)
    a = ax[5]
    for j, s in enumerate(SIGMAS):
        g = A["gain"][f"sigma {s}"]
        a.plot(GAINS, np.array(g["val_mae"]) - g["val_mae_g1"], "o-", color=OKABE_ITO[j], label=f"val, sigma {s}")
        a.plot(GAINS, np.array(g["test_mae"]) - g["test_mae_g1"], "s--", color=OKABE_ITO[j], alpha=0.6, label=f"test, sigma {s}")
    a.axhline(0, color="grey", lw=0.8)
    a.set(title="Change in hidden-pixel MAE relative to g = 1", xlabel="gain g", ylabel="MAE(g) − MAE(1) (ln CHL)"); a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.suptitle("Black Sea 4 km, best run: calibration of predicted ln CHL pair differences (val 182 days; gain test val 182 / test 183 days)")
    plt.tight_layout(); plt.savefig(fn, dpi=110); plt.close(fig)


def plot_B(B, ndays, fn):
    fig, ax = plt.subplots(2, 3, figsize=(21, 11)); ax = ax.ravel()
    x = np.arange(len(FIELDS)); w = 0.2
    sub = f"(val {ndays['val']} + train {ndays['train']} days)"
    for a, key, stat, title, yl in ((ax[0], "corr", "pearson", f"Pearson |∇field| vs |∇ ln CHL| by coastal band {sub}", "Pearson r"),
                                    (ax[1], "corr", "spearman", f"Spearman |∇field| vs |∇ ln CHL| by coastal band {sub}", "Spearman rho"),
                                    (ax[3], "corr_tol3x3", "pearson", "Pearson, 3x3 max of |∇field| vs |∇ ln CHL| by coastal band", "Pearson r"),
                                    (ax[4], "align", "mean_abs_cos", "Mean |cos θ| between ∇field and ∇ln CHL (both > median)", "mean |cos θ|")):
        for j, b in enumerate(BANDS):
            vals = [B[key][f"{f} | band {b}"][stat] for f in FIELDS]
            vals = [np.nan if v is None else v for v in vals]
            a.bar(x + (j - 1.5) * w, vals, w, color=OKABE_ITO[j], label=f"excl. ≤{b} px from land" if b else "no band excluded")
        n0 = [B[key][f"{f} | band 0"]["n_px"] for f in FIELDS]
        a.set_xticks(x); a.set_xticklabels([f"{f}\n(n {n:,}, band 0)" for f, n in zip(FIELDS, n0)], rotation=25, ha="right", fontsize=7)
        a.set(title=title, xlabel="field", ylabel=yl); a.legend(fontsize=7); a.grid(axis="y", alpha=0.3); a.axhline(0, color="grey", lw=0.6)
        if key == "align":
            a.axhline(2 / np.pi, color="black", ls="--", lw=1, label="random directions (2/π)"); a.set_ylim(0.5, 1.0); a.legend(fontsize=7)
    a = ax[2]
    for j, f in enumerate(VALUE_FIELDS):
        a.plot(BANDS, [B["corr_value"][f"{f} | band {b}"]["pearson"] for b in BANDS], "o-", color=OKABE_ITO[j], label=f"|{f}|")
    a.axhline(0, color="grey", lw=0.6)
    a.set(title="Pearson |field value| vs |∇ ln CHL| by coastal band", xlabel="coastal band excluded (px from land)", ylabel="Pearson r")
    a.set_xticks(BANDS); a.legend(fontsize=7); a.grid(alpha=0.3)
    a = ax[5]
    for j, b in enumerate(BANDS):
        for sub_, ls in (("val", "-"), ("val+train", "--")):
            v = [B["sst_block"][f"{sub_} | band {b} | {s}x{s}"]["pearson"] for s in BLOCKS]
            a.plot(BLOCKS, v, marker="o", ls=ls, color=OKABE_ITO[j], label=f"{sub_}, excl. ≤{b} px" if b else f"{sub_}, no band")
    a.set(title="Pearson |∇ ln CHL| vs |∇SST| on s×s block means (4a method)", xlabel="block size s (px)", ylabel="Pearson r")
    a.set_xticks(BLOCKS); a.legend(fontsize=7, ncol=2); a.grid(alpha=0.3)
    fig.suptitle("Black Sea 4 km: candidate input fields vs ln CHL gradients on observed pixels")
    plt.tight_layout(); plt.savefig(fn, dpi=110); plt.close(fig)


def main():
    names = {int(k): v for k, v in json.load(open(ROOT / "analyses/blk/region_names.json")).items()}
    ctx, ocean, rd = setup()
    a = collect_A(ctx, ocean, rd); A = analyse_A(a, names)
    print("A data quality:", json.dumps(a["dq"]))
    b = collect_B(ctx, ocean, rd); B = analyse_B(b)
    print("B data quality:", json.dumps(b["dq"]))
    out = dict(run=str(RUN.relative_to(ROOT)), data_quality=dict(A=a["dq"], B=b["dq"]), A=A, B=B,
               notes=["A1: VAL days of preds.npz (182); pairs at lag L along x and y pooled; truth observed both ends, >= 1 hidden end",
                      "A1: x = dpred, y = dtruth (ln CHL); slope_origin = sum(xy)/sum(xx); slope with intercept = cov/var; rev = dp~dt",
                      "A1: distance / region of a pair as gradient_diagnosis (hidden end with the larger EDT distance to visible px)",
                      "A1: CI = 2.5/97.5 percentiles of 1000 day-block bootstrap resamples (val days)",
                      "A2: S = gaussian_filter(pred*ocean)/gaussian_filter(ocean), mode constant; MAE pooled over hidden px (= trainer's cnn_hidden)",
                      "A2: best g picked on val per sigma; test delta CI: day-block bootstrap of pooled-pixel MAE difference",
                      f"B: VAL days of preds.npz + every {TRAIN_STRIDE}th TRAIN day (ctx['tr'][::{TRAIN_STRIDE}]); observed CHL px whose 4 neighbours are observed",
                      "B: centred differences, dx = R cos(lat) dlon, dy = R dlat (R = 6371 km); scalar gradients per km; vorticity/strain 1/day (u, v in m/s)",
                      "B: a field gradient is NaN unless the pixel and its 4 neighbours are valid; ln CHL t-1 = store day i-1 (any split)",
                      "B2: medians of |grad ln CHL| and |grad field| recomputed within each evaluated subset (field x band)",
                      "B3: band b excludes ocean px with Euclidean distance to the nearest land px <= b (1 = adjacent to land)",
                      "B4: 3x3 max over finite neighbours, centre must be finite (same pixel set as B1)",
                      "B5: gradient_diagnosis 4a code (forward differences on block means, /s); coastal band applied to both CHL and SST masks",
                      "pred float16 in preds.npz"])
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    json.dump(out, open(RES / "gradient_calibration.json", "w"), indent=1, default=lambda o: None if o is None else (float(o) if np.isscalar(o) else str(o)))
    plot_A(A, a["tab"], FIG / "gradient_calibration_A.png")
    plot_B(B, b["dq"]["n_days"], FIG / "gradient_calibration_B.png")
    print(tables(A, B))
    print("\njson", RES / "gradient_calibration.json", "\nfigures", FIG / "gradient_calibration_A.png", FIG / "gradient_calibration_B.png")


if __name__ == "__main__":
    main()
