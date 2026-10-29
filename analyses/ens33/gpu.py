"""Bridge #33 steps 1-3, analysis on the GPU (torch) from the per-hidden-px caches of analyses/ens33/prep.py. No training.

Step 3 (Black Sea, blk4) runs first because step 1 uses its best ensemble.
  3. members' MAE; plain means (all 9, the 7 non-t3d runs, best subset = exhaustive search over all 511 subsets by val MAE, val = both
     halves; also chosen on one half and scored on the other); key-count-aware weights (as step 2) if a plain mean beats the best single.
  1. gradient calibration on lag-1 pairs with both px hidden: OLS slope (with intercept) of true difference on predicted difference,
     ratio sum|dp| / sum|dt| and uncentred corr (postrun's formulas), overall and by pair distance bin (min of the two px distances);
     spread-error: per-px member std (ddof 1), MAE of the ensemble mean by spread decile (deciles within split), Spearman(spread, |error|)
     overall and by distance bin (ordinal ranks).
  2. IO key-count-aware weights: per bin, non-negative weights summing to 1 fitted on val half A, applied to half B, and vice versa
     (analyses/io7/ensemble/report.fit on a smoothed |r|, objective and gradient on the GPU); bins (a) keys 0, 1-8, 9+; (b) distance
     1, 2-3, 4-7, 8-15, 16+ px; control: one global weight vector fitted the same way. Out-of-sample val = half A with the B fit plus
     half B with the A fit; test with each half's fit.
Writes analyses/ens33/results/gpu_raw.json. Usage (project root): python -W ignore -m analyses.ens33.gpu
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from analyses.ens33.prep import DATA, BLK_CORE
from analyses.io7.ensemble.report import EPS

OUT = Path(__file__).resolve().parent / "results"
DEV = torch.device("cuda")
DIST_BINS = [("1", 0.5, 1.5), ("2-3", 1.5, 3.5), ("4-7", 3.5, 7.5), ("8-15", 7.5, 15.5), ("16+", 15.5, np.inf)]
KEY_BINS = [("0", 0, 0), ("1-8", 1, 8), ("9+", 9, 10 ** 9)]
SPLITS = {"val": (0, 1), "test": (2,), "A": (0,), "B": (1,)}
MIN_N = 5


def T(a, dt=None):
    return torch.as_tensor(np.asarray(a), device=DEV, dtype=dt)


def load(store):
    z = np.load(DATA / f"{store}_hidden.npz")
    D = {k: z[k] for k in z.files}
    G = dict(X=T(D["X"], torch.float32), y=T(D["y"], torch.float32), lab=T(D["lab"].astype("int64")),
             dist=T(D["dist"].astype("float32")), nk=T(D["nk"].astype("int64")), cell=T(D["cell"].astype("int64")),
             i0=T(D["i0"]), i1=T(D["i1"]), members=list(D["members"]), tgt=D["tgt"])
    G["pdist"] = torch.minimum(G["dist"][G["i0"]], G["dist"][G["i1"]])
    G["plab"] = G["lab"][G["i0"]]
    G["other_t3d"] = dict(zip(D.get("other_t3d_names", []), map(bool, D.get("other_t3d_match", []))))
    return G


def smask(G, s, key="lab"):
    return torch.isin(G[key], T(SPLITS[s]))


def dist_bins(v):
    return {b: (v > lo) & (v <= hi) for b, lo, hi in DIST_BINS}


def key_bins(v):
    return {b: (v >= lo) & (v <= hi) for b, lo, hi in KEY_BINS}


def mae(p, y, m=None):
    if m is not None:
        p, y = p[m], y[m]
    n = int(p.numel())
    return dict(mae=float((p - y).abs().double().mean()) if n else None, n=n)


def rw(ae, cell, tgt, nc):
    """postrun.reweighted on hidden-px vectors (all hidden): val real-cloud cell target, cells with >= 5 px."""
    ok = cell >= 0
    cnt = torch.bincount(cell[ok], minlength=nc).double().cpu().numpy()
    s = torch.bincount(cell[ok], weights=ae[ok].double(), minlength=nc).cpu().numpy()
    cov = cnt >= MIN_N
    p = tgt / tgt.sum()
    m_c = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return dict(mae=float((p[cov] * m_c[cov]).sum() / p[cov].sum()), uncovered_real_mass=float(p[~cov].sum()))


def scores(G, p):
    """MAE val / test / halves, no key, >= 1 key, rw, by key bin and distance bin, for a hidden-px prediction vector."""
    y = G["y"]; nc = len(G["tgt"]); out = {}
    for s in ("val", "test", "A", "B"):
        m = smask(G, s)
        r = mae(p, y, m)
        r["no key"] = mae(p, y, m & (G["nk"] == 0))
        r[">=1 key"] = mae(p, y, m & (G["nk"] > 0))
        r["rw"] = rw((p - y).abs()[m], G["cell"][m], G["tgt"], nc)
        r["by_key"] = {b: mae(p, y, m & mk) for b, mk in key_bins(G["nk"]).items()}
        r["by_dist"] = {b: mae(p, y, m & mk) for b, mk in dist_bins(G["dist"]).items()}
        out[s] = r
    return out


# ------------------------------------------------------------------ fitting
def fit(X, y):
    """As analyses/io7/ensemble/report.fit (SLSQP, smoothed |r|, starts uniform + each vertex, best true MAE), objective on the GPU."""
    n, k = X.shape
    def f(w, eps):
        r = X @ T(w, torch.float32) - y
        s = torch.sqrt(r * r + eps * eps)
        return float(s.double().mean()), (X.T @ (r / s)).double().cpu().numpy() / n
    true = lambda w: float((X @ T(w, torch.float32) - y).abs().double().mean())
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},)
    best = None
    for w0 in [np.full(k, 1 / k)] + [np.eye(k)[i] * .9 + .1 / k for i in range(k)]:
        w = w0
        for eps in EPS:
            r = minimize(f, w, args=(eps,), jac=True, method="SLSQP", bounds=[(0, 1)] * k, constraints=cons,
                         options=dict(maxiter=300, ftol=1e-12))
            w = np.clip(r.x, 0, None); w /= w.sum()
        m = true(w)
        if best is None or m < best[1]:
            best = (w, m)
    w = best[0]
    w[w < 1e-4] = 0; w /= w.sum()
    return w


def binned_fit(G, bins):
    """bins: {name: hidden-px mask}. Weights per half and bin; cross-fit val prediction, test predictions with A / B weights."""
    X, y, lab = G["X"], G["y"], G["lab"]
    W = {h: {} for h in "AB"}
    for h, i in (("A", 0), ("B", 1)):
        for b, mk in bins.items():
            m = mk & (lab == i)
            W[h][b] = fit(X[m], y[m]) if int(m.sum()) >= MIN_N else np.ones(X.shape[1]) / X.shape[1]
    pA = torch.zeros_like(y); pB = torch.zeros_like(y)
    for b, mk in bins.items():
        pA[mk] = X[mk] @ T(W["A"][b], torch.float32)
        pB[mk] = X[mk] @ T(W["B"][b], torch.float32)
    cross = torch.where(lab == 0, pB, pA)              # half A scored with the B fit, half B with the A fit; test: A fit
    return W, cross, pA, pB


def weighted_block(G, bins, label):
    W, cross, pA, pB = binned_fit(G, bins)
    sc = {"oos": scores(G, cross), "fitA": scores(G, pA), "fitB": scores(G, pB)}
    return dict(bins=list(bins), weights={h: {b: dict(zip(G["members"], map(float, w))) for b, w in W[h].items()} for h in W},
                scores=sc), dict(cross=cross, fitA=pA, fitB=pB)


# ------------------------------------------------------------------ step 1
def grad_cal(G, p):
    y = G["y"]; i0, i1 = G["i0"], G["i1"]
    dp = (p[i1] - p[i0]).double(); dt = (y[i1] - y[i0]).double()
    def one(m):
        a, b = dp[m], dt[m]; n = int(a.numel())
        if n < MIN_N:
            return dict(n_pairs=n, slope=None)
        am, bm = a.mean(), b.mean()
        slope = float(((a - am) * (b - bm)).sum() / ((a - am) ** 2).sum())
        return dict(slope=slope, intercept=float(bm - slope * am), ratio=float(a.abs().sum() / b.abs().sum()),
                    corr=float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum())), n_pairs=n)
    out = {}
    for s in ("val", "test"):
        ms = smask(G, s, "plab")
        out[s] = dict(all=one(ms), by_dist={b: one(ms & mk) for b, mk in dist_bins(G["pdist"]).items()})
    return out


def ranks(v):
    r = torch.empty_like(v, dtype=torch.float64)
    r[torch.argsort(v, stable=True)] = torch.arange(v.numel(), device=DEV, dtype=torch.float64)
    return r


def spearman(a, b):
    if a.numel() < MIN_N:
        return None
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / torch.sqrt((ra * ra).sum() * (rb * rb).sum()))


def spread_error(G, cols):
    Xs = G["X"][:, cols]
    p = Xs.mean(1); sd = Xs.std(1, unbiased=True); ae = (p - G["y"]).abs()
    out = {}
    for s in ("val", "test"):
        m = smask(G, s)
        sp, e = sd[m], ae[m]
        n = sp.numel()
        dec = (ranks(sp) * 10 // n).long()
        out[s] = dict(spearman=spearman(sp, e), n=int(n),
                      deciles=[dict(mae=float(e[dec == d].double().mean()), mean_spread=float(sp[dec == d].double().mean()),
                                    spread_hi=float(sp[dec == d].max()), n=int((dec == d).sum())) for d in range(10)],
                      by_dist={b: dict(spearman=spearman(sp[mk[m]], e[mk[m]]), n=int(mk[m].sum()), mae=float(e[mk[m]].double().mean()),
                                       mean_spread=float(sp[mk[m]].double().mean())) for b, mk in dist_bins(G["dist"]).items()})
    return out


# ------------------------------------------------------------------ main
def step3(G):
    X, y = G["X"], G["y"]; mem = G["members"]; k = len(mem)
    vm, am, bm = smask(G, "val"), smask(G, "A"), smask(G, "B")
    single = {m: scores(G, X[:, j]) for j, m in enumerate(mem)}
    subs = {}
    for r in range(1, k + 1):
        for c in itertools.combinations(range(k), r):
            p = X[:, list(c)].mean(1)
            subs[c] = {s: float((p[mk] - y[mk]).abs().double().mean()) for s, mk in (("val", vm), ("A", am), ("B", bm))}
    best = {s: min(subs, key=lambda c: subs[c][s]) for s in ("val", "A", "B")}
    core = tuple(mem.index(m) for m in BLK_CORE)
    plain = {"mean_all9": tuple(range(k)), "mean_core7": core, "mean_bestsub_val": best["val"],
             "mean_bestsub_A": best["A"], "mean_bestsub_B": best["B"]}
    pm = {n: scores(G, X[:, list(c)].mean(1)) for n, c in plain.items()}
    best_single = min(mem, key=lambda m: single[m]["val"]["mae"])
    beats = {n: pm[n]["val"]["mae"] < single[best_single]["val"]["mae"] for n in ("mean_all9", "mean_core7", "mean_bestsub_val")}
    out = dict(members=mem, other_t3d_same_px=G["other_t3d"], single=single, best_single_val=best_single,
               plain={n: dict(members=[mem[i] for i in c], scores=pm[n]) for n, c in plain.items()}, plain_beats_best_single_val=beats,
               subset_search=dict(n_subsets=len(subs), top10_val=[dict(members=[mem[i] for i in c], **subs[c])
                                                                    for c in sorted(subs, key=lambda c: subs[c]["val"])[:10]]))
    if any(beats.values()):
        for lab, bins in (("keybins", key_bins(G["nk"])), ("distbins", dist_bins(G["dist"])), ("global", {"all": torch.ones_like(vm)})):
            out[f"w_{lab}"], _ = weighted_block(G, bins, lab)
    return out, list(best["val"])


def main():
    t0 = time.time()
    res = {}
    G = load("blk4")
    res["step3_blk4"], bcols = step3(G)
    bname = "blk4_mean_bestsub_val"
    pb = G["X"][:, bcols].mean(1)
    res["step1"] = {bname: dict(members=[G["members"][i] for i in bcols], grad=grad_cal(G, pb),
                                spread=spread_error(G, bcols) if len(bcols) > 1 else None)}
    del G; torch.cuda.empty_cache()
    print("blk4 done", round(time.time() - t0), "s", flush=True)

    G = load("io")
    mem = G["members"]; X = G["X"]
    seeds = [mem.index(m) for m in ("io6_tokdec_blend_spatial", "io6_tokdec_blend_spatial_s1", "io6_tokdec_blend_spatial_s2")]
    cols = {"ens11_mean": list(range(len(mem))), "io7_physmem": [mem.index("io7_physmem")], "io6_spatial_seed_mean": seeds}
    for n, c in cols.items():
        p = X[:, c].mean(1)
        res["step1"][n] = dict(members=[mem[i] for i in c], grad=grad_cal(G, p), spread=spread_error(G, c) if len(c) > 1 else None)
    print("io step1 done", round(time.time() - t0), "s", flush=True)
    s2 = dict(members=mem, ens11_mean=scores(G, X.mean(1)), io7_physmem=scores(G, X[:, mem.index("io7_physmem")]))
    for lab, bins in (("keybins", key_bins(G["nk"])), ("distbins", dist_bins(G["dist"])), ("global", {"all": torch.ones_like(G["y"], dtype=torch.bool)})):
        s2[f"w_{lab}"], _ = weighted_block(G, bins, lab)
        print("io", lab, round(time.time() - t0), "s", flush=True)
    res["step2_io"] = s2
    res["gpu_wall_s"] = time.time() - t0
    OUT.mkdir(exist_ok=True)
    (OUT / "gpu_raw.json").write_text(json.dumps(res, indent=1))
    print("done", round(res["gpu_wall_s"]), "s", flush=True)


if __name__ == "__main__":
    main()
