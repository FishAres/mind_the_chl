"""Ensembles of IO members, bridge #29 step 1 (eval only, CPU; from saved preds.npz).

  1. per-px mean ensembles: (a) the 6 Phase 0 members + io7_physmem (ens7_mean); (b) all 11 postrun-scored IO members (ens11_mean)
  2. fitted weights (non-negative, sum 1) minimising hidden-px MAE on one half of the val days (split by year: A = 2001, 2013;
     B = 2007, 2018), scored on the other half and on test; compared with the plain means on the same halves
Scoring of every ensemble: analyses/io7/postrun.score_pred (val/test MAE, key split, real-cloud reweighted, gradients lag 1/2/4).
Data generation (hidden-px matrix, full-array ensembles), analysis (fit), reporting (figure, JSON, FINDINGS entry) are separate.
Writes analyses/io7/ensemble/results/ensemble.json, figures/ens_models.png, one FINDINGS.md entry (replaced on rerun).
Usage (project root): python -W ignore -m analyses.io7.ensemble.report [--no-findings]
"""
import argparse
import datetime as dt
import fcntl
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from analyses.io7 import postrun as PR
from chl_cnn.plotting import OKABE_ITO

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "results", HERE / "figures"
CACHE = Path.home() / "mind_the_chl_gap" / "data" / "io7" / "ensemble_hidden_px.npz"
P0 = ["io6_tokdec_blend_spatial", "io4_tokdec_blend", "io3_ptok4", "io2_base_eval", "io2_seed1", "io2_seed2"]
SET_A = P0 + ["io7_physmem"]
SET_B = SET_A + ["io6_tokdec_blend_spatial_s1", "io6_tokdec_blend_spatial_s2", "io7_wmattn", "io7_analog"]
HALVES = {"A": (2001, 2013), "B": (2007, 2018)}
EPS = (1e-3, 1e-4)                                    # smoothing of |r| as sqrt(r^2 + eps^2), two stages


# ------------------------------------------------------------------ data generation
def val_half():
    C = PR.ctx(); c = C["c"]
    yr = np.asarray(c["times"])[c["va"]].astype("datetime64[Y]").astype(int) + 1970
    return {h: np.isin(yr, ys) for h, ys in HALVES.items()}, yr


def hidden_matrix():
    """X (n_hidden val+test px, n members) float32, y, split/half labels; cached."""
    if CACHE.exists():
        z = np.load(CACHE)
        if list(z["members"]) == SET_B:
            return {k: z[k] for k in z.files}
    C = PR.ctx(); c = C["c"]; nv = C["nv"]
    hid = c["hidden"]
    idx = np.r_[c["va"], c["te"]]
    day = np.nonzero(hid)[0]
    y = np.concatenate([c["y"][idx[i]][hid[i]] for i in range(len(idx))]).astype("float32")
    hv, _ = val_half()
    lab = np.full(len(idx), 2, "int8"); lab[:nv][hv["A"]] = 0; lab[:nv][hv["B"]] = 1      # 0 val A, 1 val B, 2 test
    X = np.empty((len(y), len(SET_B)), "float32")
    for j, m in enumerate(SET_B):
        p, _ = PR.load_pred(m)
        X[:, j] = p[hid]
        del p
        print("hidden px", m, flush=True)
    out = dict(X=X, y=y, lab=lab[day], members=np.array(SET_B))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def combos(weights):
    """Full-array (val+test days) weighted combinations, one pass over the members. weights: {name: vector over SET_B}."""
    acc = {}
    for j, m in enumerate(SET_B):
        if not any(w[j] for w in weights.values()):
            continue
        p, _ = PR.load_pred(m)
        for n, w in weights.items():
            if w[j]:
                if n not in acc:
                    acc[n] = np.zeros_like(p)
                acc[n] += np.float32(w[j]) * p
        del p
        print("combine", m, flush=True)
    return acc


# ------------------------------------------------------------------ analysis
def mae(X, y, w):
    return float(np.abs(X @ w.astype("float32") - y).astype("float64").mean())


def fit(X, y):
    """min_w mean|Xw - y| s.t. w >= 0, sum w = 1 (SLSQP on a smoothed |r|; starts: uniform and every vertex; best true MAE kept)."""
    n, k = X.shape
    Xd = X
    def f(w, eps):
        r = (Xd @ w.astype("float32") - y).astype("float64")
        s = np.sqrt(r * r + eps * eps)
        return s.mean(), (Xd.T @ (r / s).astype("float32")).astype("float64") / n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},)
    best = None
    for w0 in [np.full(k, 1 / k)] + [np.eye(k)[i] * .9 + .1 / k for i in range(k)]:
        w = w0
        for eps in EPS:
            r = minimize(f, w, args=(eps,), jac=True, method="SLSQP", bounds=[(0, 1)] * k, constraints=cons,
                         options=dict(maxiter=300, ftol=1e-12))
            w = np.clip(r.x, 0, None); w /= w.sum()
        m = mae(X, y, w)
        if best is None or m < best[1]:
            best = (w, m)
    w = best[0]
    w[w < 1e-4] = 0; w /= w.sum()
    return w


def analyse(H):
    X, y, lab = H["X"], H["y"], H["lab"]
    ia, ib = [SET_B.index(m) for m in SET_A], list(range(len(SET_B)))
    wmean = {"ens7_mean": np.isin(np.arange(len(SET_B)), ia) / len(ia), "ens11_mean": np.ones(len(SET_B)) / len(SET_B)}
    halves = {}
    fitted = {}
    for h, other in (("A", "B"), ("B", "A")):
        m = lab == "AB".index(h)
        w = fit(X[m], y[m])
        fitted[f"fit{h}_ens11"] = w
    mk = {h: lab == i for i, h in enumerate("AB")}
    mk["test"] = lab == 2
    for n, w in {**wmean, **fitted}.items():
        halves[n] = {h: dict(mae=mae(X[m], y[m], w), n=int(m.sum())) for h, m in mk.items()}
    for j, mem in enumerate(SET_B):
        e = np.eye(len(SET_B))[j]
        halves[mem] = {h: dict(mae=mae(X[m], y[m], e), n=int(m.sum())) for h, m in mk.items()}
    nonfinite = int((~np.isfinite(X)).sum())
    return wmean, fitted, halves, nonfinite


# ------------------------------------------------------------------ reporting
def ens_rows(res):
    return [n for n in ("ens6_mean", "ens7_mean", "ens11_mean", "fitA_ens11", "fitB_ens11") if n in res]


def figure(members, res, fitted, halves):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    names = SET_B + ens_rows(res)
    x = np.arange(len(names))
    for j, (s, k, mk, lab) in enumerate((("val", "mae", "o", "val (fake-cloud hidden px)"), ("test", "mae", "s", "test (fake-cloud hidden px)"))):
        ax[0].plot(x + (j - .5) * .25, [res[n][s]["mae"] for n in names], mk, color=OKABE_ITO[j], label=lab)
    ax[0].axvline(len(SET_B) - .5, color="0.6", lw=.8)
    ax[0].set(xticks=x, xticklabels=names, ylabel="MAE (ln CHL)", title="Hidden-px MAE: members and ensembles\n"
              f"(val n {res['ens11_mean']['val']['n_hidden']:,} px, test n {res['ens11_mean']['test']['n_hidden']:,} px)")
    ax[0].tick_params(axis="x", rotation=60, labelsize=7); plt.setp(ax[0].get_xticklabels(), ha="right", rotation_mode="anchor"); ax[0].legend(fontsize=8)
    k = len(SET_B); xw = np.arange(k)
    for i, (n, w) in enumerate(fitted.items()):
        ax[1].bar(xw + (i - .5) * .38, w, .38, color=OKABE_ITO[i + 2], label=f"{n} (fit on val half {n[3]})")
    ax[1].axhline(1 / k, color="0.5", ls=":", label=f"plain mean 1/{k}")
    ax[1].set(xticks=xw, xticklabels=SET_B, ylabel="weight", title="Fitted ensemble weights (≥ 0, sum 1) by member")
    ax[1].tick_params(axis="x", rotation=60, labelsize=7); plt.setp(ax[1].get_xticklabels(), ha="right", rotation_mode="anchor"); ax[1].legend(fontsize=8)
    rows = ["ens7_mean", "ens11_mean", "fitA_ens11", "fitB_ens11"]
    xs = np.arange(3)
    for i, n in enumerate(rows):
        ax[2].plot(xs + (i - 1.5) * .12, [halves[n][h]["mae"] for h in ("A", "B", "test")], "o", color=OKABE_ITO[i], label=n)
    na, nb = halves["ens11_mean"]["A"]["n"], halves["ens11_mean"]["B"]["n"]
    ax[2].set(xticks=xs, xticklabels=[f"val half A\n2001, 2013\nn {na:,} px", f"val half B\n2007, 2018\nn {nb:,} px", f"test\nn {halves['ens11_mean']['test']['n']:,} px"],
              ylabel="MAE (ln CHL)", title="Hidden-px MAE by val half and test:\nplain means vs fitted weights")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ens_models.png", dpi=130); plt.close(fig)


def entry(res, fitted, halves, hinfo, nonfinite):
    when = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    g = lambda r, s: ", ".join(f"{r[s]['grad'][L]['ratio']:.3f}/{r[s]['grad'][L]['corr']:.3f}" for L in ("1", "2", "4"))
    L = [f"## io7 ensembles, bridge #29 step 1 (IO; {when}; eval only; analyses/io7/ensemble/report.py)",
         "- Assumptions:",
         f"  - ens7_mean = per-px mean of {', '.join(SET_A)}. ens11_mean adds {', '.join(SET_B[len(SET_A):])}: every IO run with a "
         "postrun result in analyses/io7/results/ (all 60-epoch IO recipe runs). Other full-length IO runs with preds.npz (e.g. io6_tokdec_krig, "
         "io6_tokdec_krig_signed, io4_tokdec, io2_tattn8, io5_tokdec_blend_up0p3) are not included.",
         f"  - Val halves by year: A = {HALVES['A']} ({hinfo['A']['days']} days, {hinfo['A']['px']:,} hidden px), B = {HALVES['B']} "
         f"({hinfo['B']['days']} days, {hinfo['B']['px']:,} hidden px). Year split chosen over alternate days because adjacent days share "
         "clouds and fields.",
         "  - Fitted weights: over all 11 members, minimising hidden-px MAE on one half (SLSQP on sqrt(r² + ε²), ε 1e-3 then 1e-4; starts: "
         "uniform and each member-dominant vertex; lowest true MAE kept; weights < 1e-4 set to 0, renormalised). fitA = fit on half A.",
         "  - Scoring as analyses/io7/postrun.py (rw = real-cloud reweighted, val real-cloud target applied to test too). "
         f"Non-finite member preds on hidden px: {nonfinite}.",
         "",
         "| model set | val | test | val ≥ 1 key | val no key | test no key | val rw | test rw | grad ratio/corr lag 1 val |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n in ["io6_tokdec_blend_spatial"] + ens_rows(res):
        r = res[n]
        L.append(f"| {n} | {r['val']['mae']:.5f} | {r['test']['mae']:.5f} | {r['val']['by_key']['>=1 key']['mae']:.5f} | "
                 f"{r['val']['by_key']['no key']['mae']:.5f} | {r['test']['by_key']['no key']['mae']:.5f} | {r['val']['rw']['mae']:.5f} | "
                 f"{r['test']['rw']['mae']:.5f} | {r['val']['grad']['1']['ratio']:.3f}/{r['val']['grad']['1']['corr']:.3f} |")
    v = res["ens11_mean"]
    L += ["", f"- n: val {v['val']['n_hidden']:,} hidden px ({v['val']['n_days']} days; ≥ 1 key {v['val']['by_key']['>=1 key']['n']:,}, no key "
          f"{v['val']['by_key']['no key']['n']:,}); test {v['test']['n_hidden']:,} ({v['test']['n_days']} days). fitA/fitB val rows include "
          "their own fitting half (in-sample); see the half table for out-of-sample.",
          "- Gradient ratio/corr lags 1, 2, 4 px (val; test): " + "; ".join(f"{n} {g(res[n], 'val')}; {g(res[n], 'test')}" for n in ens_rows(res)) + ".",
          "- Fitted weights (zeros omitted): " + "; ".join(f"{n}: " + ", ".join(f"{m} {x:.3f}" for m, x in zip(SET_B, w) if x > 0)
                                                          for n, w in fitted.items()) + ".",
          "", "| combination | val half A | val half B | test |", "|---|---|---|---|"]
    for n in ["ens7_mean", "ens11_mean", "fitA_ens11", "fitB_ens11"]:
        L.append(f"| {n} | " + " | ".join(f"{halves[n][h]['mae']:.5f}" + (" (fit)" if n == f"fit{h}_ens11" else "") for h in ("A", "B", "test")) + " |")
    L += ["", f"- Half table n: A {halves['ens11_mean']['A']['n']:,} px, B {halves['ens11_mean']['B']['n']:,} px, test {halves['ens11_mean']['test']['n']:,} px. "
          f"Out-of-sample fitted − ens11_mean: half B (fitA) {halves['fitA_ens11']['B']['mae'] - halves['ens11_mean']['B']['mae']:+.5f}, "
          f"half A (fitB) {halves['fitB_ens11']['A']['mae'] - halves['ens11_mean']['A']['mae']:+.5f}; test fitA {halves['fitA_ens11']['test']['mae'] - halves['ens11_mean']['test']['mae']:+.5f}, "
          f"fitB {halves['fitB_ens11']['test']['mae'] - halves['ens11_mean']['test']['mae']:+.5f}.",
          "- Members' half MAE (A / B): " + ", ".join(f"{m} {halves[m]['A']['mae']:.4f}/{halves[m]['B']['mae']:.4f}" for m in SET_B) + ".",
          "- Results results/ensemble.json; figure figures/ens_models.png."]
    return "\n".join(L) + "\n"


def write_findings(text):
    with open(PR.FINDINGS.with_suffix(".md.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        s = PR.FINDINGS.read_text()
        s = re.sub(r"\n## io7 ensembles, bridge #29 step 1 \(.*?(?=\n## |\Z)", "", s, flags=re.S)
        PR.FINDINGS.write_text(s.rstrip("\n") + "\n\n" + text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-findings", action="store_true")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    H = hidden_matrix()
    wmean, fitted, halves, nonfinite = analyse(H)
    print(json.dumps({n: [round(x, 4) for x in w] for n, w in fitted.items()}), flush=True)
    hv, _ = val_half()
    hinfo = {h: dict(days=int(hv[h].sum()), px=halves["ens11_mean"][h]["n"]) for h in "AB"}
    res = {m: PR.score_run(m) for m in SET_B}
    P0J = json.loads((HERE.parent / "phase0" / "results" / "phase0.json").read_text())["items34"]["ens6_mean"]
    res["ens6_mean"] = P0J
    acc = combos({**wmean, **fitted})
    for n, P in list(acc.items()):
        w = {**wmean, **fitted}[n]
        res[n] = PR.score_pred(n, P, dict(members=SET_B, weights=[float(x) for x in w]))
        del acc[n], P
        print(n, f"val {res[n]['val']['mae']:.5f} test {res[n]['test']['mae']:.5f}", flush=True)
    keep = lambda r: {s: {k: r[s][k] for k in ("mae", "rmse", "bias", "n_hidden", "n_days", "by_key", "rw", "rw_own", "grad") if k in r[s]}
                      for s in ("val", "test")}
    j = dict(members_ens7=SET_A, members_ens11=SET_B, halves_def={h: list(v) for h, v in HALVES.items()}, half_info=hinfo,
             fitted_weights={n: dict(zip(SET_B, map(float, w))) for n, w in fitted.items()}, half_mae=halves,
             scores={n: keep(res[n]) for n in SET_B + ens_rows(res)}, nonfinite_hidden_preds=nonfinite)
    (OUT / "ensemble.json").write_text(json.dumps(j, indent=1))
    figure(SET_B, res, fitted, halves)
    text = entry(res, fitted, halves, hinfo, nonfinite)
    print(text)
    if not a.no_findings:
        write_findings(text)


if __name__ == "__main__":
    main()
