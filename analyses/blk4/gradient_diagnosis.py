"""Where does the 4 km model lose gradient amplitude? Diagnostics of the trainer's gradient ratio / correlation on
VAL days (preds.npz, 182 days) for the best Black Sea 4 km run and the lambda 0 reference.

Pair statistic (chl_cnn/train_pinn.py residual_stats): adjacent pixel pairs in x and y, truth observed at both ends,
pair rule as stated per diagnostic; ratio = sum|dpred| / sum|dtruth|, corr = sum(dp dt) / sqrt(sum dp^2 sum dt^2)
(uncentred). preds.npz 'pred' is float16 (the trainer used float32).

  0. reproduce the trainer's val ratio / corr (lag 1, truth at both ends, >= 1 hidden end)
  1. pair category: both visible / >= 1 hidden / exactly 1 hidden / both hidden
  2. lag L = 1, 2, 4, 8, 16 px; structure functions mean|d| vs lag for truth, pred, SST, current speed
  3. region (7, as error_breakdown) and distance of the hidden end(s) to the nearest visible pixel
  4. (a) |grad ln CHL| vs |grad SST| and vs current speed on observed pixels (1x1, 2x2, 4x4 blocks)
     (b) ratio on hidden pairs by decile of local |grad SST|
  5. persistence (truth t-1) and DOY climatology on the same pairs

Run from the repo root: python -m analyses.blk4.gradient_diagnosis   (CPU only)
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
from chl_cnn.viz import context_for
from chl_cnn.plotting import OKABE_ITO
from analyses.blk4.error_breakdown import regions_4km, doy_clim

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
RUNS = {"best (sl lam0.3 cont)": ROOT / "models/pinn/t3d/blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50",
        "lam0": ROOT / "models/pinn/t3d/blk4_t3d_c4_gdl100_lam0_b16_50ep"}
MODELS = list(RUNS)
LAGS = [1, 2, 4, 8, 16]
BLOCKS = [1, 2, 4]
DIST_BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, np.inf)]
CATS = ["both visible", ">=1 hidden (trainer)", "exactly 1 hidden", "both hidden"]
MIN_N = 5


# ============================== data ==============================
def pairs(a, L, ax):
    """(first end, second end) views of a 2D array for lag L along axis ax (1 = x, 0 = y)."""
    return (a[:, :-L], a[:, L:]) if ax == 1 else (a[:-L], a[L:])


def blockmean(a, valid, s):
    """s x s block mean of a, NaN where any pixel in the block is invalid."""
    if s == 1:
        return np.where(valid, a, np.nan)
    H, W = (a.shape[0] // s) * s, (a.shape[1] // s) * s
    v = np.where(valid, a, np.nan)[:H, :W].reshape(H // s, s, W // s, s)
    return v.mean((1, 3))                                         # NaN propagates: any invalid px -> NaN


def gradmag(a):
    """Forward-difference gradient magnitude per pixel; NaN unless the pixel, its x+1 and y+1 neighbours are finite."""
    g = np.full(a.shape, np.nan, "float32")
    g[:-1, :-1] = np.sqrt((a[:-1, 1:] - a[:-1, :-1]) ** 2 + (a[1:, :-1] - a[:-1, :-1]) ** 2)
    return g


class Acc:
    """Running sums for ratio / corr of a predictor's differences against truth differences."""
    def __init__(self):
        self.s = dict(ap=0.0, at=0.0, pt=0.0, pp=0.0, tt=0.0, n=0)

    def add(self, dp, dt):
        dp = dp.astype("float64"); dt = dt.astype("float64")
        self.s["ap"] += np.abs(dp).sum(); self.s["at"] += np.abs(dt).sum(); self.s["pt"] += (dp * dt).sum()
        self.s["pp"] += (dp * dp).sum(); self.s["tt"] += (dt * dt).sum(); self.s["n"] += dp.size

    def out(self):
        s = self.s
        if s["n"] == 0:
            return dict(ratio=None, corr=None, n_pairs=0)
        return dict(ratio=s["ap"] / s["at"], corr=s["pt"] / np.sqrt(s["pp"] * s["tt"]),
                    mean_abs_d_pred=s["ap"] / s["n"], mean_abs_d_truth=s["at"] / s["n"], n_pairs=int(s["n"]))


def collect():
    """One pass over the val days: running sums for 0-2, a flat lag-1 hidden-pair table for 3-5, and pixel samples for 4a."""
    ctx = context_for(RUNS[MODELS[0]])
    d, ocean = ctx["d"], ctx["ocean"]
    z = {k: np.load(p / "preds.npz") for k, p in RUNS.items()}
    z0 = z[MODELS[0]]
    for k in MODELS:
        assert (z[k]["idx"] == z0["idx"]).all() and (z[k]["hidden"] == z0["hidden"]).all(), k
    pos = {int(i): k for k, i in enumerate(z0["idx"])}
    reg, _ = regions_4km(ocean)
    clim, cbin = doy_clim(ctx)
    oc = np.asarray(d["ocean"]); rows, cols = np.flatnonzero(oc.any(1)), np.flatnonzero(oc.any(0))
    ys = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, oc.shape[0])); xs = slice(max(cols[0] - 1, 0), min(cols[-1] + 2, oc.shape[1]))
    assert (oc[ys, xs] == ocean).all()
    rd = lambda name, i: np.asarray(d[name][i, ys, xs]).astype("float32")

    cat = {(m, c): Acc() for m in MODELS for c in CATS}
    lag = {(m, L): Acc() for m in MODELS for L in LAGS}
    sf = {(f, L): [0.0, 0] for f in ("truth_all_obs", "sst", "speed") for L in LAGS}
    tab = {c: [] for c in ["dt", "nhid", "region", "dist", "sstg", "dpers", "dclim", "day"] + [f"dp_{m}" for m in MODELS]}
    phys = {(f, s): [] for f in ("sst", "speed") for s in BLOCKS}
    dq = dict(n_days=0, days_no_visible=0, days_no_hidden=0, sst_nan_ocean_px_day=0, speed_nan_ocean_px_day=0,
              pred_nan_ocean=dict.fromkeys(MODELS, 0), hidden_off_obs=0, persist_day_missing=0)

    for i in (int(v) for v in z0["val_idx"]):
        k = pos[i]; dq["n_days"] += 1
        y = ctx["y"](i); obs = np.isfinite(y) & ocean
        hid = z0["hidden"][k] & ocean
        dq["hidden_off_obs"] += int((z0["hidden"][k] & ~obs).sum())
        hid &= obs
        vis = obs & ~hid
        P = {m: z[m]["pred"][k].astype("float32") for m in MODELS}
        for m in MODELS:
            dq["pred_nan_ocean"][m] += int((~np.isfinite(P[m]) & ocean).sum())
        sst = rd("sst32", i); sok = ocean & np.isfinite(sst)
        spd = np.hypot(rd("u_curr", i), rd("v_curr", i)); vok = ocean & np.isfinite(spd)
        dq["sst_nan_ocean_px_day"] += int((ocean & ~sok).sum()); dq["speed_nan_ocean_px_day"] += int((ocean & ~vok).sum())
        if i - 1 >= 0:
            yp = ctx["y"](i - 1); okp = np.isfinite(yp) & ocean
        else:
            dq["persist_day_missing"] += 1; yp = np.full_like(y, np.nan); okp = np.zeros_like(obs)
        cl = clim[cbin[i]]
        if vis.any():
            dist = distance_transform_edt(~vis).astype("float32")
        else:
            dq["days_no_visible"] += 1; dist = np.full(y.shape, np.inf, "float32")
        if not hid.any():
            dq["days_no_hidden"] += 1
        sstg = np.full(y.shape, np.nan, "float32")                 # central-difference |grad SST|, NaN next to land/NaN
        gy_, gx_ = np.gradient(np.where(sok, sst, np.nan)); sstg[:] = np.hypot(gx_, gy_)

        for L in LAGS:
            for ax in (0, 1):
                o1, o2 = pairs(obs, L, ax); h1, h2 = pairs(hid, L, ax)
                t1, t2 = pairs(y, L, ax); dt = t2 - t1
                both = o1 & o2
                m_hid = both & (h1 | h2)
                for m in MODELS:
                    p1, p2 = pairs(P[m], L, ax); dp = p2 - p1
                    lag[(m, L)].add(dp[m_hid], dt[m_hid])
                    if L == 1:
                        cat[(m, CATS[0])].add(dp[both & ~h1 & ~h2], dt[both & ~h1 & ~h2])
                        cat[(m, CATS[1])].add(dp[m_hid], dt[m_hid])
                        cat[(m, CATS[2])].add(dp[both & (h1 ^ h2)], dt[both & (h1 ^ h2)])
                        cat[(m, CATS[3])].add(dp[both & h1 & h2], dt[both & h1 & h2])
                sf[("truth_all_obs", L)][0] += float(np.abs(dt[both]).astype("float64").sum()); sf[("truth_all_obs", L)][1] += int(both.sum())
                for f, a, ok in (("sst", sst, sok), ("speed", spd, vok)):
                    a1, a2 = pairs(a, L, ax); k1, k2 = pairs(ok, L, ax); mm = k1 & k2
                    sf[(f, L)][0] += float(np.abs(a2 - a1)[mm].astype("float64").sum()); sf[(f, L)][1] += int(mm.sum())
                if L == 1:                                         # flat table of the trainer's pairs
                    d1, d2 = pairs(dist, 1, ax); r1, r2 = pairs(reg, 1, ax); s1, s2 = pairs(sstg, 1, ax)
                    q1, q2 = pairs(yp, 1, ax); ok1, ok2 = pairs(okp, 1, ax); c1, c2 = pairs(cl, 1, ax)
                    dd1, dd2 = np.where(h1, d1, -1), np.where(h2, d2, -1)    # distance counted on hidden ends only
                    first = dd1 >= dd2
                    tab["dt"].append(dt[m_hid]); tab["nhid"].append((h1.astype(np.int8) + h2)[m_hid])
                    tab["dist"].append(np.maximum(dd1, dd2)[m_hid])
                    tab["region"].append(np.where(first, r1, r2)[m_hid])
                    tab["sstg"].append(((s1 + s2) / 2)[m_hid])
                    tab["dpers"].append(np.where(ok1 & ok2, q2 - q1, np.nan)[m_hid])
                    tab["dclim"].append((c2 - c1)[m_hid])
                    tab["day"].append(np.full(int(m_hid.sum()), i, np.int32))
                    for m in MODELS:
                        p1, p2 = pairs(P[m], 1, ax); tab[f"dp_{m}"].append((p2 - p1)[m_hid])
        for s in BLOCKS:                                           # 4a: gradient magnitudes on observed pixels
            gc = gradmag(blockmean(y, obs, s)) / s
            for f, a, ok, isgrad in (("sst", sst, sok, True), ("speed", spd, vok, False)):
                b = blockmean(a, ok, s)
                g = gradmag(b) / s if isgrad else b
                mm = np.isfinite(gc) & np.isfinite(g)
                phys[(f, s)].append(np.stack([gc[mm], g[mm]], 1).astype("float32"))
    tab = {c: np.concatenate(v) for c, v in tab.items()}
    phys = {k: np.concatenate(v) for k, v in phys.items()}
    sf = {k: (v[0] / v[1] if v[1] else None, v[1]) for k, v in sf.items()}
    lat = ctx["lat"]; dy_km = float(np.abs(np.diff(lat)).mean() * 111.32)
    dx_km = float(np.abs(np.diff(ctx["lon"])).mean() * 111.32 * np.cos(np.deg2rad(lat.mean())))
    dq.update(px_km=dict(dx=dx_km, dy=dy_km), ocean_px=int(ocean.sum()),
              table_nan=dict(sstg=int(np.isnan(tab["sstg"]).sum()), dpers=int(np.isnan(tab["dpers"]).sum()),
                             dclim=int(np.isnan(tab["dclim"]).sum()), dist_inf=int(np.isinf(tab["dist"]).sum()),
                             region_unlabelled=int((tab["region"] == 0).sum()),
                             **{f"dp_{m}": int(np.isnan(tab[f'dp_{m}']).sum()) for m in MODELS}))
    return dict(cat=cat, lag=lag, sf=sf, tab=tab, phys=phys, dq=dq)


# ============================== analysis ==============================
def rc(dp, dt):
    a = Acc(); a.add(dp, dt); return a.out()


def grouped(tab, gid, labels, preds):
    rows = []
    for g, lab in enumerate(labels):
        m = gid == g
        r = dict(group=lab, n_pairs=int(m.sum()), n_days=int(len(np.unique(tab["day"][m]))))
        for name, col in preds.items():
            mm = m & np.isfinite(tab[col])
            r[name] = rc(tab[col][mm], tab["dt"][mm]) if mm.any() else dict(ratio=None, corr=None, n_pairs=0)
        r["flag_n_lt5"] = r["n_pairs"] < MIN_N or r["n_days"] < MIN_N
        rows.append(r)
    return rows


def analyse(c, names):
    tab = c["tab"]; mp = {m: f"dp_{m}" for m in MODELS}
    res = {}
    res["reproduce"] = {m: c["cat"][(m, CATS[1])].out() for m in MODELS}
    res["category"] = {cc: {m: c["cat"][(m, cc)].out() for m in MODELS} for cc in CATS}
    res["lag"] = {L: {m: c["lag"][(m, L)].out() for m in MODELS} for L in LAGS}
    res["structure"] = {L: {f: dict(mean_abs_d=c["sf"][(f, L)][0], n_pairs=c["sf"][(f, L)][1]) for f in ("truth_all_obs", "sst", "speed")}
                        for L in LAGS}
    for L in LAGS:
        o = c["lag"][(MODELS[0], L)].out()
        res["structure"][L]["truth_hidden_pairs"] = dict(mean_abs_d=o["mean_abs_d_truth"], n_pairs=o["n_pairs"])
        for m in MODELS:
            o = c["lag"][(m, L)].out()
            res["structure"][L][f"pred_{m}_hidden_pairs"] = dict(mean_abs_d=o["mean_abs_d_pred"], n_pairs=o["n_pairs"])
    res["region"] = grouped(tab, tab["region"].astype(int) - 1, [names[r] for r in range(1, 8)], mp)
    dist = tab["dist"]; edges = np.array([0.5, 1.5, 3.5, 7.5, 15.5, np.inf])     # Euclidean px, rounded to bins
    gid = np.searchsorted(edges, dist, side="right") - 1
    gid[(dist < 0.5)] = -1
    res["dist_unbinned"] = int((gid < 0).sum())
    res["distance"] = grouped(tab, gid, ["1 (<1.5)", "2-3", "4-7", "8-15", "16+"], mp)
    sg = tab["sstg"]; fin = np.isfinite(sg)
    edges = np.quantile(sg[fin], np.linspace(0, 1, 11))
    dec = np.where(fin, np.clip(np.searchsorted(edges, sg, side="right") - 1, 0, 9), -1)
    res["sstg_decile_edges"] = [float(e) for e in edges]
    res["sstg_decile"] = grouped(tab, dec, [f"D{j + 1}" for j in range(10)], mp)
    phys = {}
    for (f, s), a in c["phys"].items():
        phys[f"{f} {s}x{s}"] = dict(pearson=float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
                                    spearman=float(spearmanr(a[:, 0], a[:, 1]).statistic), n_px=int(len(a)))
    res["phys_corr"] = phys
    base = {}
    allp = np.ones(len(tab["dt"]), bool); per = np.isfinite(tab["dpers"])
    for sub, m_ in (("all trainer pairs", allp), ("pairs with t-1 observed at both ends", per)):
        base[sub] = {**{m: rc(tab[mp[m]][m_], tab["dt"][m_]) for m in MODELS},
                     "DOY climatology": rc(tab["dclim"][m_], tab["dt"][m_])}
        if sub != "all trainer pairs":
            base[sub]["persistence (truth t-1)"] = rc(tab["dpers"][m_], tab["dt"][m_])
    res["baselines"] = base
    return res


# ============================== reporting ==============================
def fmt(o):
    return "-" if o.get("ratio") is None else f"{o['ratio']:.3f} | {o['corr']:.3f}"


def tables(res, dq):
    out = []
    out.append("\n### 0. Reproduction (lag 1, truth both ends, >=1 hidden end)\n| model | ratio | corr | n pairs |\n|---|---|---|---|")
    for m, o in res["reproduce"].items():
        out.append(f"| {m} | {fmt(o)} | {o['n_pairs']:,} |")
    hdr = lambda g: f"| {g} | " + " | ".join(f"ratio {m} | corr {m}" for m in MODELS) + " | n pairs |\n|" + "---|" * (2 + 2 * len(MODELS))
    out.append("\n### 1. Pair category (lag 1)\n" + hdr("category"))
    for cc, v in res["category"].items():
        out.append(f"| {cc} | " + " | ".join(fmt(v[m]) for m in MODELS) + f" | {v[MODELS[0]]['n_pairs']:,} |")
    km = (dq["px_km"]["dx"] + dq["px_km"]["dy"]) / 2
    out.append("\n### 2a. Lag (truth both ends, >=1 hidden end)\n" + hdr("lag px (km)"))
    for L, v in res["lag"].items():
        out.append(f"| {L} ({L * km:.0f}) | " + " | ".join(fmt(v[m]) for m in MODELS) + f" | {v[MODELS[0]]['n_pairs']:,} |")
    cols = ["truth_all_obs", "truth_hidden_pairs", f"pred_{MODELS[0]}_hidden_pairs", "sst", "speed"]
    out.append("\n### 2b. Structure function, mean |d| (raw units: ln CHL, store SST units, current speed units); n pairs in ()\n| lag px | "
               + " | ".join(cols) + " |\n|" + "---|" * (1 + len(cols)))
    for L, v in res["structure"].items():
        out.append(f"| {L} | " + " | ".join(f"{v[c]['mean_abs_d']:.4g} ({v[c]['n_pairs']:,})" for c in cols) + " |")
    for key, t in (("region", "3a. Region (region of the hidden end)"), ("distance", "3b. Distance of hidden end to nearest visible px (max over hidden ends)"),
                   ("sstg_decile", "4b. Decile of local |grad SST| (mean of the two ends)")):
        out.append(f"\n### {t}\n" + hdr("group").replace(" | n pairs |", " | n pairs | n days |").replace("|---|---|---|---|---|---|", "|---|---|---|---|---|---|---|"))
        for r in res[key]:
            fl = " (n<5)" if r["flag_n_lt5"] else ""
            out.append(f"| {r['group']}{fl} | " + " | ".join(fmt(r[m]) for m in MODELS) + f" | {r['n_pairs']:,} | {r['n_days']} |")
    out.append("\n### 4a. |grad ln CHL| vs |grad SST| / current speed, observed pixels pooled over val days\n| field, block | Pearson | Spearman | n px |\n|---|---|---|---|")
    for k, v in res["phys_corr"].items():
        out.append(f"| {k} | {v['pearson']:.3f} | {v['spearman']:.3f} | {v['n_px']:,} |")
    out.append("\n### 5. Baselines (lag 1)\n| pair subset | predictor | ratio | corr | n pairs |\n|---|---|---|---|---|")
    for sub, v in res["baselines"].items():
        for p, o in v.items():
            out.append(f"| {sub} | {p} | {fmt(o)} | {o['n_pairs']:,} |")
    return "\n".join(out)


def plot(res, dq, fn):
    km = (dq["px_km"]["dx"] + dq["px_km"]["dy"]) / 2
    fig, ax = plt.subplots(2, 4, figsize=(24, 11)); ax = ax.ravel()
    C = {MODELS[0]: OKABE_ITO[0], MODELS[1]: OKABE_ITO[1]}

    def bars(a, labels, getter, title, xlabel, ns):
        x = np.arange(len(labels)); w = 0.2
        for j, m in enumerate(MODELS):
            r = [np.nan if getter(i, m)["ratio"] is None else getter(i, m)["ratio"] for i in range(len(labels))]
            cr = [np.nan if getter(i, m)["corr"] is None else getter(i, m)["corr"] for i in range(len(labels))]
            a.bar(x + (j - 1.5) * w, r, w, color=C[m], label=f"ratio, {m}")
            a.bar(x + (j + 0.5) * w, cr, w, color=C[m], hatch="//", alpha=0.55, label=f"corr, {m}")
        a.set_xticks(x); a.set_xticklabels([f"{l}\n(n {n:,})" for l, n in zip(labels, ns)], rotation=30, ha="right", fontsize=7)
        a.set(title=title, xlabel=xlabel, ylabel="gradient ratio / gradient corr"); a.axhline(1, color="grey", lw=0.6)
        a.legend(fontsize=7); a.grid(axis="y", alpha=0.3)

    cv = res["category"]
    bars(ax[0], CATS, lambda i, m: cv[CATS[i]][m], "1. Gradient ratio and corr by pair visibility (lag 1)", "pair category",
         [cv[c][MODELS[0]]["n_pairs"] for c in CATS])
    a = ax[1]
    for m in MODELS:
        a.plot([L * km for L in LAGS], [res["lag"][L][m]["ratio"] for L in LAGS], "o-", color=C[m], label=f"ratio, {m}")
        a.plot([L * km for L in LAGS], [res["lag"][L][m]["corr"] for L in LAGS], "s--", color=C[m], label=f"corr, {m}")
    a.set_xscale("log"); a.set_xticks([L * km for L in LAGS]); a.set_xticklabels([f"{L * km:.0f}" for L in LAGS])
    a.set(title="2. Gradient ratio and corr vs pair lag (>=1 hidden end)", xlabel=f"lag (km; 1 px ~ {km:.1f} km)", ylabel="gradient ratio / gradient corr")
    a.legend(fontsize=7); a.grid(alpha=0.3)
    a = ax[2]
    series = [("truth_all_obs", "ln CHL truth, all observed pairs", OKABE_ITO[7], "-"),
              ("truth_hidden_pairs", "ln CHL truth, >=1-hidden pairs", OKABE_ITO[7], ":"),
              (f"pred_{MODELS[0]}_hidden_pairs", f"ln CHL pred ({MODELS[0]}), >=1-hidden pairs", OKABE_ITO[0], "-"),
              (f"pred_{MODELS[1]}_hidden_pairs", f"ln CHL pred ({MODELS[1]}), >=1-hidden pairs", OKABE_ITO[1], "--"),
              ("sst", "SST, ocean pairs", OKABE_ITO[3], "-"), ("speed", "current speed, ocean pairs", OKABE_ITO[2], "-")]
    for key, lab, col, ls in series:
        v = np.array([res["structure"][L][key]["mean_abs_d"] for L in LAGS])
        a.plot([L * km for L in LAGS], v / v[0], marker="o", color=col, ls=ls, label=lab)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xticks([L * km for L in LAGS]); a.set_xticklabels([f"{L * km:.0f}" for L in LAGS])
    a.set(title="2b. Structure function, mean |difference| / its lag-1 value", xlabel="lag (km)", ylabel="mean |Δ| at lag / mean |Δ| at lag 1")
    a.legend(fontsize=7); a.grid(alpha=0.3, which="both")
    for a, key, t, xl in ((ax[3], "region", "3a. Gradient ratio and corr by region (lag 1, >=1 hidden)", "region"),
                          (ax[4], "distance", "3b. Gradient ratio and corr by distance of hidden end to visible px", "distance (px)"),
                          (ax[6], "sstg_decile", "4b. Gradient ratio and corr by local |∇SST| decile (lag 1)", "|∇SST| decile")):
        rows = res[key]
        bars(a, [r["group"] + (" (n<5)" if r["flag_n_lt5"] else "") for r in rows], lambda i, m, rows=rows: rows[i][m], t, xl,
             [r["n_pairs"] for r in rows])
    a = ax[5]; pc = res["phys_corr"]; keys = list(pc); x = np.arange(len(keys))
    a.bar(x - 0.2, [pc[k]["pearson"] for k in keys], 0.4, color=OKABE_ITO[5], label="Pearson")
    a.bar(x + 0.2, [pc[k]["spearman"] for k in keys], 0.4, color=OKABE_ITO[4], label="Spearman")
    a.set_xticks(x); a.set_xticklabels([f"|∇lnCHL| vs {'|∇SST|' if k.startswith('sst') else 'speed'}\n{k.split()[1]} blocks (n {pc[k]['n_px']:,})" for k in keys],
                                       rotation=30, ha="right", fontsize=7)
    a.set(title="4a. Correlation of |∇ ln CHL| with |∇SST| and current speed (observed px)", ylabel="correlation"); a.legend(fontsize=7)
    a.axhline(0, color="grey", lw=0.6); a.grid(axis="y", alpha=0.3)
    b = res["baselines"]; sub = "pairs with t-1 observed at both ends"; preds = list(b[sub])
    bars_r = [(f"{s[:3]}…: {p}", b[s][p]) for s in b for p in b[s]]
    a = ax[7]; x = np.arange(len(bars_r))
    a.bar(x - 0.2, [o["ratio"] for _, o in bars_r], 0.4, color=OKABE_ITO[0], label="ratio")
    a.bar(x + 0.2, [o["corr"] for _, o in bars_r], 0.4, color=OKABE_ITO[1], label="corr")
    lab = [("all pairs" if s.startswith("all") else "t-1 obs pairs") + f": {p}\n(n {o['n_pairs']:,})" for s in b for p, o in b[s].items()]
    a.set_xticks(x); a.set_xticklabels(lab, rotation=35, ha="right", fontsize=7); a.axhline(1, color="grey", lw=0.6)
    a.set(title="5. Gradient ratio and corr, models vs baselines (lag 1)", ylabel="gradient ratio / gradient corr"); a.legend(fontsize=7)
    a.grid(axis="y", alpha=0.3)
    fig.suptitle("Black Sea 4 km, val days (182): predicted vs true ln CHL differences between pixel pairs "
                 "(ratio = Σ|Δpred|/Σ|Δtruth|, corr = uncentred)")
    plt.tight_layout(); plt.savefig(fn, dpi=110); plt.close(fig)


def main():
    names = {int(k): v for k, v in json.load(open(ROOT / "analyses/blk/region_names.json")).items()}
    c = collect()
    res = analyse(c, names)
    dq = c["dq"]
    out = dict(runs={k: str(v.relative_to(ROOT)) for k, v in RUNS.items()}, data_quality=dq, results=res,
               notes=["VAL days in preds.npz (182); pairs at lag L along x and y pooled",
                      "visible = observed & not hidden; hidden from preds.npz (all hidden px are observed)",
                      "ratio = sum|dpred|/sum|dtruth|; corr = sum(dp dt)/sqrt(sum dp^2 sum dt^2), uncentred",
                      "region / distance of a pair: the hidden end with the larger distance to the nearest visible px (EDT, Euclidean px)",
                      "local |grad SST|: central differences (np.gradient) on SST, mean over the two pair ends; NaN next to land/NaN SST",
                      "4a: forward-difference |grad| on s x s block means (blocks with any unobserved/NaN px dropped), divided by s; pooled over days",
                      "structure function SST / speed: all ocean pairs with finite values on val days",
                      "pred float16 in preds.npz; trainer used float32"])
    json.dump(out, open(RES / "gradient_diagnosis.json", "w"), indent=1, default=lambda o: None if o is None else float(o))
    fn = FIG / "gradient_diagnosis.png"
    plot(res, dq, fn)
    print("data quality:", json.dumps(dq, indent=1))
    print("dist unbinned:", res["dist_unbinned"], " sstg decile edges:", np.round(res["sstg_decile_edges"], 4).tolist())
    print(tables(res, dq))
    print("\njson", RES / "gradient_diagnosis.json", "\nfigure", fn)


if __name__ == "__main__":
    main()
