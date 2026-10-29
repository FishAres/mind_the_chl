"""Bridge #33 reporting (CPU): reads results/gpu_raw.json (analyses/ens33/gpu.py), adds postrun's standard lag-1 gradient ratio / corr
(pairs with truth at both ends and >= 1 hidden end, analyses/io_improve/score.grad_stats) for the Black Sea combinations (full arrays)
and the IO step-1 models, writes results/ens33.json, figures/ens33_*.png and three FINDINGS.md entries (replaced on rerun).
Weighted Black Sea combinations as full arrays: hidden px from the per-bin weights (val: cross-fit halves; test: A fit, and B fit
separately), every other px = the plain mean of all 9 members (only enters the >= 1-hidden-end gradient pairs).
Usage (project root): python -W ignore -m analyses.ens33.report [--no-findings]
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

from analyses.io7 import postrun as PR
from analyses.io_improve import score as SC
from analyses.ens33.gpu import DIST_BINS, KEY_BINS
from chl_cnn.plotting import OKABE_ITO

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "results", HERE / "figures"
IO7 = HERE.parent / "io7"
DB = [b for b, _, _ in DIST_BINS]
KB = [b for b, _, _ in KEY_BINS]


# ------------------------------------------------------------------ data (postrun-standard lag-1 gradients)
def g1(pred, c, sl, idx):
    return SC.grad_stats(pred[sl], c["y"][idx], c["obs"][idx], c["hidden"][sl], 1)


def grads_full(c, nv, pred):
    return {"val": g1(pred, c, slice(0, nv), c["va"]), "test": g1(pred, c, slice(nv, None), c["te"])}


def blk_full(R):
    """Postrun lag-1 grads (and a val/test MAE cross-check) for the Black Sea combinations."""
    PR.use_store("blk4")
    C = PR.ctx(); c = C["c"]; nv = C["nv"]; hid = c["hidden"]
    S = R["step3_blk4"]; mem = S["members"]
    P = np.stack([PR.load_pred(m)[0] for m in mem])
    out = {}
    for m in mem:
        out[m] = grads_full(c, nv, P[mem.index(m)])
    for n, d in S["plain"].items():
        p = P[[mem.index(m) for m in d["members"]]].mean(0)
        out[n] = grads_full(c, nv, p)
        out[n]["chk_val_mae"] = float(np.abs(p[:nv] - c["y"][c["va"]])[hid[:nv]].mean())
    base = P.mean(0)
    days_half = np.full(len(hid), 2, "int8")
    yr = np.asarray(c["times"])[c["va"]].astype("datetime64[Y]").astype(int) + 1970
    days_half[:nv][yr == 2017] = 0; days_half[:nv][yr == 2021] = 1
    dist = c["dist"].astype("float32"); nk = C["nk"]
    binm = {"keybins": {b: (nk >= lo) & (nk <= hi) for b, lo, hi in KEY_BINS},
            "distbins": {b: (dist > lo) & (dist <= hi) for b, lo, hi in DIST_BINS},
            "global": {"all": np.ones(hid.shape, bool)}}
    for lab, bins in binm.items():
        if f"w_{lab}" not in S:
            continue
        W = S[f"w_{lab}"]["weights"]
        def comp(choose):
            p = base.copy()
            for b, mk in bins.items():
                for d in range(len(hid)):
                    m = mk[d] & hid[d]
                    w = np.array([W[choose(days_half[d])][b][x] for x in mem], "float32")
                    p[d][m] = np.tensordot(w, P[:, d][:, m], 1)
            return p
        pc = comp(lambda h: "B" if h == 0 else "A")             # val cross-fit; test A fit
        pb = comp(lambda h: "B")
        out[f"w_{lab}"] = {"val": g1(pc, c, slice(0, nv), c["va"]), "test_fitA": g1(pc, c, slice(nv, None), c["te"]),
                           "test_fitB": g1(pb, c, slice(nv, None), c["te"]),
                           "chk_val_mae": float(np.abs(pc[:nv] - c["y"][c["va"]])[hid[:nv]].mean())}
    return out


def io_full(R):
    """Postrun lag-1 grads for the IO step-1 models (ens11_mean and io7_physmem from saved JSONs; seed mean from the 3 preds)."""
    PR.use_store("io")
    ens = json.loads((IO7 / "ensemble" / "results" / "ensemble.json").read_text())["scores"]["ens11_mean"]
    phy = json.loads((IO7 / "results" / "io7_physmem.json").read_text())
    out = {"ens11_mean": {s: ens[s]["grad"]["1"] for s in ("val", "test")},
           "io7_physmem": {s: phy[s]["grad"]["1"] for s in ("val", "test")}}
    C = PR.ctx(); c = C["c"]; nv = C["nv"]
    acc = None
    mem = R["step1"]["io6_spatial_seed_mean"]["members"]
    for m in mem:
        p, _ = PR.load_pred(m)
        acc = p if acc is None else acc + p
        del p
    acc /= len(mem)
    out["io6_spatial_seed_mean"] = grads_full(c, nv, acc)
    return out


# ------------------------------------------------------------------ figures
def fig_step1(R):
    S1 = R["step1"]; names = list(S1)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(DB) + 1)
    for j, s in enumerate(("val", "test")):
        for i, n in enumerate(names):
            g = S1[n]["grad"][s]
            ys = [g["all"]["slope"]] + [g["by_dist"][b]["slope"] for b in DB]
            ax[0, j].plot(x, ys, "o-", color=OKABE_ITO[i], label=n)
        ax[0, j].axhline(1, color="0.6", ls=":", lw=1)
        ax[0, j].set(xticks=x, xticklabels=["all"] + DB, xlabel="pair distance to nearest visible px (min of the 2 px; px)",
                     ylabel="OLS slope of true Δ on predicted Δ",
                     title=f"Lag-1 gradient calibration slope, both px hidden ({s})")
        ax[0, j].legend(fontsize=8)
        ens = [n for n in names if S1[n]["spread"]]
        for i, n in enumerate(ens):
            d = S1[n]["spread"][s]["deciles"]
            ax[1, 0].plot(np.arange(1, 11) + (j - .5) * .1, [q["mae"] for q in d], "o-" if s == "val" else "s--",
                          color=OKABE_ITO[names.index(n)], label=f"{n} ({s})", ms=4)
            sp = [S1[n]["spread"][s]["spearman"]] + [S1[n]["spread"][s]["by_dist"][b]["spearman"] for b in DB]
            ax[1, 1].plot(x + (j - .5) * .1, sp, "o-" if s == "val" else "s--", color=OKABE_ITO[names.index(n)], label=f"{n} ({s})", ms=4)
    ax[1, 0].set(xticks=range(1, 11), xlabel="member-spread decile (1 = lowest std; within split)", ylabel="MAE of ensemble mean (ln CHL)",
                 title="Hidden-px MAE by ensemble spread decile")
    ax[1, 1].set(xticks=x, xticklabels=["all"] + DB, xlabel="distance to nearest visible px (px)", ylabel="Spearman ρ(spread, |error|)",
                 title="Spread–error rank correlation by distance bin")
    for a in ax[1]:
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "ens33_step1_calibration.png", dpi=130); plt.close(fig)


def fig_step2(R):
    S = R["step2_io"]; mem = S["members"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 6), gridspec_kw=dict(width_ratios=[1.2, 1, 1.3]))
    rows = [("w_keybins", "by_key", KB), ("w_distbins", "by_dist", DB)]
    xt, xl = [], []
    off = 0
    for k, (w, grp, bins) in enumerate(rows):
        for i, (lab, col) in enumerate((("w_keybins", 0), ("w_distbins", 1), ("w_global", 2))):
            d = [S[lab]["scores"]["oos"]["val"][grp][b]["mae"] - S["ens11_mean"]["val"][grp][b]["mae"] for b in bins]
            ax[0].plot(np.arange(len(bins)) + off + (i - 1) * .2, d, "o", color=OKABE_ITO[col], label=lab if k == 0 else None)
        xt += list(np.arange(len(bins)) + off); xl += [f"keys {b}" if grp == "by_key" else f"dist {b}" for b in bins]
        off += len(bins) + 1
    ax[0].axhline(0, color="0.6", lw=.8)
    ax[0].set(xticks=xt, xticklabels=xl, ylabel="MAE − ens11_mean MAE (ln CHL)",
              title="IO val, out of sample (cross-fit halves):\nweighted ensembles minus ens11_mean, by key / distance bin")
    ax[0].tick_params(axis="x", rotation=45); ax[0].legend(fontsize=8)
    for a, lab, bins in ((ax[1], "w_keybins", KB), (ax[2], "w_distbins", DB)):
        W = S[lab]["weights"]
        M = np.array([[W[h][b][m] for h in "AB" for b in bins] for m in mem])
        im = a.imshow(M, cmap="cividis", vmin=0, aspect="auto")
        a.set(xticks=range(M.shape[1]), xticklabels=[f"{h}: {b}" for h in "AB" for b in bins], yticks=range(len(mem)), yticklabels=mem,
              xlabel="fit half: bin", title=f"IO fitted weights per {'key-count' if lab == 'w_keybins' else 'distance'} bin (≥ 0, sum 1)")
        a.tick_params(axis="x", rotation=45)
        for (i, j), v in np.ndenumerate(M):
            if v >= .005:
                a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w" if v < .5 * M.max() else "k")
        plt.colorbar(im, ax=a, fraction=.04, label="weight")
    fig.tight_layout(); fig.savefig(FIG / "ens33_step2_io_weights.png", dpi=130); plt.close(fig)


def fig_step3(R, rows):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows))
    for j, s in enumerate(("val", "test")):
        ax.plot(x + (j - .5) * .25, [r[s] for r in rows], "os"[j], color=OKABE_ITO[j], label=f"{s} (fake-cloud hidden px)")
    ax.set(xticks=x, xticklabels=[r["name"] for r in rows], ylabel="MAE (ln CHL)", title="Black Sea 4 km hidden-px MAE: members and ensembles")
    ax.tick_params(axis="x", rotation=60); plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ens33_step3_blk4.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------ FINDINGS
f5 = lambda v: "n/a" if v is None else f"{v:.5f}"
f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
fn = lambda n: f"{n:,}"


def entries(R, G):
    when = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    S1, S2, S3 = R["step1"], R["step2_io"], R["step3_blk4"]
    bname = next(n for n in S1 if n.startswith("blk4_"))
    E = {}
    # ---- step 1
    L = [f"## ens33 step 1: gradient and spread-error calibration of ensembles (IO + Black Sea 4 km; {when}; eval only; analyses/ens33/)",
         "- Assumptions: lag-1 pairs = adjacent px along y or x on the same day with BOTH px hidden (postrun's standard pairs need only one hidden "
         "end; its lag-1 ratio/corr are listed as 'postrun'). Slope = OLS (with intercept) of true Δ on predicted Δ (1 = calibrated amplitude). "
         "Pair distance bin = the smaller of the two px distances to the nearest day-t visible px. Spread = per-px std of members (ddof 1); error = "
         "|ensemble mean − truth|; deciles of spread within each split; Spearman with ordinal ranks. io7_physmem is one model (no spread). "
         f"io6_spatial_seed_mean = io6_tokdec_blend_spatial, _s1, _s2. Black Sea ensemble = {bname} ({', '.join(S1[bname]['members'])}), "
         "chosen in step 3 by val MAE.", "",
         "| model | split | slope all | 1 | 2-3 | 4-7 | 8-15 | 16+ | ratio/corr both-hidden | ratio/corr postrun | n pairs |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, d in S1.items():
        for s in ("val", "test"):
            g = d["grad"][s]
            L.append(f"| {n} | {s} | {f3(g['all']['slope'])} | " + " | ".join(f3(g["by_dist"][b]["slope"]) for b in DB)
                     + f" | {f3(g['all']['ratio'])}/{f3(g['all']['corr'])} | {f3(G[n][s]['ratio'])}/{f3(G[n][s]['corr'])} | {fn(g['all']['n_pairs'])} |")
    L += ["", "- n pairs by distance bin (val): " + "; ".join(f"{n} " + ", ".join(f"{b} {fn(S1[n]['grad']['val']['by_dist'][b]['n_pairs'])}" for b in DB)
                                                            for n in S1 if n in ("ens11_mean", bname)) + ".", "",
          "| ensemble | split | Spearman all | 1 | 2-3 | 4-7 | 8-15 | 16+ | MAE spread decile 1 | decile 5 | decile 10 | n px |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, d in S1.items():
        if not d["spread"]:
            continue
        for s in ("val", "test"):
            q = d["spread"][s]
            L.append(f"| {n} | {s} | {f3(q['spearman'])} | " + " | ".join(f3(q["by_dist"][b]["spearman"]) for b in DB)
                     + f" | {f5(q['deciles'][0]['mae'])} | {f5(q['deciles'][4]['mae'])} | {f5(q['deciles'][9]['mae'])} | {fn(q['n'])} |")
    L += ["", "- MAE by spread decile 1..10 (val): " + "; ".join(f"{n} " + ", ".join(f"{x['mae']:.4f}" for x in d["spread"]["val"]["deciles"])
                                                             for n, d in S1.items() if d["spread"]) + ".",
          "- Figure figures/ens33_step1_calibration.png; results results/ens33.json (step1)."]
    E["step 1"] = "\n".join(L) + "\n"
    # ---- step 2
    hA, hB = S2["ens11_mean"]["A"]["n"], S2["ens11_mean"]["B"]["n"]
    L = [f"## ens33 step 2: key-count-aware IO ensemble weights (IO; {when}; eval only; analyses/ens33/)",
         "- Assumptions: 11 ens11 members; per bin, weights ≥ 0 summing to 1 minimise hidden-px MAE on one val half (fit as analyses/io7/ensemble/"
         "report.fit; objective on the GPU). Halves A = 2001, 2013 (n " + fn(hA) + " px), B = 2007, 2018 (n " + fn(hB) + " px). Out of sample (oos) "
         "val = half A scored with the B fit + half B with the A fit. Test scored with the A fit and with the B fit. Key bins 0, 1-8, 9+ keys "
         "(kriging stencil, analyses/io7/keys.py); distance bins 1, 2-3, 4-7, 8-15, 16+ px. w_global = one weight vector, same procedure (#29 control). "
         "rw = postrun's real-cloud reweighted MAE (val target).", "",
         "| combination | val oos | Δ vs ens11 | test (A fit / B fit) | val no key | test no key (A/B) | val rw | test rw (A/B) |",
         "|---|---|---|---|---|---|---|---|"]
    e = S2["ens11_mean"]
    L.append(f"| ens11_mean | {f5(e['val']['mae'])} | 0 | {f5(e['test']['mae'])} | {f5(e['val']['no key']['mae'])} | {f5(e['test']['no key']['mae'])} | "
             f"{f5(e['val']['rw']['mae'])} | {f5(e['test']['rw']['mae'])} |")
    p = S2["io7_physmem"]
    L.append(f"| io7_physmem | {f5(p['val']['mae'])} | {p['val']['mae'] - e['val']['mae']:+.5f} | {f5(p['test']['mae'])} | {f5(p['val']['no key']['mae'])} | "
             f"{f5(p['test']['no key']['mae'])} | {f5(p['val']['rw']['mae'])} | {f5(p['test']['rw']['mae'])} |")
    for lab in ("w_global", "w_keybins", "w_distbins"):
        sc = S2[lab]["scores"]; o = sc["oos"]
        L.append(f"| {lab} | {f5(o['val']['mae'])} | {o['val']['mae'] - e['val']['mae']:+.5f} | {f5(sc['fitA']['test']['mae'])} / {f5(sc['fitB']['test']['mae'])} | "
                 f"{f5(o['val']['no key']['mae'])} | {f5(sc['fitA']['test']['no key']['mae'])} / {f5(sc['fitB']['test']['no key']['mae'])} | "
                 f"{f5(o['val']['rw']['mae'])} | {f5(sc['fitA']['test']['rw']['mae'])} / {f5(sc['fitB']['test']['rw']['mae'])} |")
    L += ["", "| bin | n val px | ens11_mean val | w_keybins oos − ens11 | w_distbins oos − ens11 | w_global oos − ens11 | physmem weight A / B (own binning) |",
          "|---|---|---|---|---|---|---|"]
    for grp, bins, lab in (("by_key", KB, "w_keybins"), ("by_dist", DB, "w_distbins")):
        for b in bins:
            m0 = e["val"][grp][b]["mae"]
            d = lambda L_: S2[L_]["scores"]["oos"]["val"][grp][b]["mae"] - m0
            pw = "/".join(f"{S2[lab]['weights'][h][b]['io7_physmem']:.3f}" for h in "AB")
            L.append(f"| {'keys' if grp == 'by_key' else 'dist'} {b} | {fn(e['val'][grp][b]['n'])} | {f5(m0)} | {d('w_keybins'):+.5f} | {d('w_distbins'):+.5f} | "
                     f"{d('w_global'):+.5f} | {pw} |")
    L += ["", "- Weights (fit A; zeros omitted): " + " ".join(f"{lab} {b}: " + ", ".join(f"{m} {w:.2f}" for m, w in S2[lab]["weights"]["A"][b].items() if w > 0) + ";"
                                                           for lab, bins in (("w_keybins", KB), ("w_distbins", DB)) for b in bins),
          "- Weights (fit B; zeros omitted): " + " ".join(f"{lab} {b}: " + ", ".join(f"{m} {w:.2f}" for m, w in S2[lab]["weights"]["B"][b].items() if w > 0) + ";"
                                                           for lab, bins in (("w_keybins", KB), ("w_distbins", DB)) for b in bins),
          f"- n: val {fn(e['val']['n'])} hidden px (no key {fn(e['val']['no key']['n'])}); test {fn(e['test']['n'])} (no key {fn(e['test']['no key']['n'])}).",
          "- Figure figures/ens33_step2_io_weights.png; results results/ens33.json (step2_io)."]
    E["step 2"] = "\n".join(L) + "\n"
    # ---- step 3
    oth = S3["other_t3d_same_px"]
    L = [f"## ens33 step 3: Black Sea 4 km ensembles from existing blk4 runs (blk4; {when}; eval only; analyses/ens33/)",
         f"- Assumptions: members = {', '.join(S3['members'])} (the 7 named runs + the two postrun blk4_t3d_* references; all share scored days and "
         f"hidden px). Other blk4_t3d_* runs with the same days / hidden px, not included: {', '.join(k for k, v in oth.items() if v) or 'none'}"
         f"{'; different: ' + ', '.join(k for k, v in oth.items() if not v) if any(not v for v in oth.values()) else ''}. "
         "Best subset = lowest val MAE over all 511 non-empty subsets (val in-sample for the choice; also chosen on one val half and scored on the "
         "other). Halves A = 2017, B = 2021 (one val year each). Key-aware weights as step 2 (out of sample val = cross-fit halves). "
         "grad lag-1 = postrun standard (full arrays; weighted rows: non-hidden px = mean of all 9).", "",
         "| model | val | test | val no key | test no key | val rw | test rw | grad lag-1 ratio/corr val | test |", "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    def row(name, sc, g, test_sc=None, gt=None):
        t = test_sc or sc
        L.append(f"| {name} | {f5(sc['val']['mae'])} | {f5(t['test']['mae'])} | {f5(sc['val']['no key']['mae'])} | {f5(t['test']['no key']['mae'])} | "
                 f"{f5(sc['val']['rw']['mae'])} | {f5(t['test']['rw']['mae'])} | {f3(g['val']['ratio'])}/{f3(g['val']['corr'])} | "
                 f"{f3((gt or g['test'])['ratio'])}/{f3((gt or g['test'])['corr'])} |")
        rows.append(dict(name=name, val=sc["val"]["mae"], test=t["test"]["mae"]))
    for m in sorted(S3["members"], key=lambda m: S3["single"][m]["val"]["mae"]):
        row(m, S3["single"][m], G["blk"][m])
    for n in ("mean_all9", "mean_core7", "mean_bestsub_val"):
        row(n, S3["plain"][n]["scores"], G["blk"][n])
    for lab in ("w_global", "w_keybins", "w_distbins"):
        if lab in S3:
            sc = S3[lab]["scores"]; g = G["blk"][lab]
            row(f"{lab} (oos val; test A fit)", sc["oos"], dict(val=g["val"], test=g["test_fitA"]), sc["fitA"])
            row(f"{lab} (test B fit)", sc["oos"], dict(val=g["val"], test=g["test_fitB"]), sc["fitB"])
    v = S3["single"][S3["members"][0]]
    L += ["", f"- Best single by val: {S3['best_single_val']}. Plain mean beats it on val: " + ", ".join(f"{k} {v_}" for k, v_ in S3["plain_beats_best_single_val"].items()) + ".",
          f"- Best subsets: val {', '.join(S3['plain']['mean_bestsub_val']['members'])}; chosen on half A: {', '.join(S3['plain']['mean_bestsub_A']['members'])} "
          f"(half B MAE {f5(S3['plain']['mean_bestsub_A']['scores']['B']['mae'])} vs mean_all9 {f5(S3['plain']['mean_all9']['scores']['B']['mae'])}, best single "
          f"{f5(S3['single'][S3['best_single_val']]['B']['mae'])}); chosen on half B: {', '.join(S3['plain']['mean_bestsub_B']['members'])} (half A MAE "
          f"{f5(S3['plain']['mean_bestsub_B']['scores']['A']['mae'])} vs mean_all9 {f5(S3['plain']['mean_all9']['scores']['A']['mae'])}, best single "
          f"{f5(S3['single'][S3['best_single_val']]['A']['mae'])}).",
          "- Top 5 subsets by val MAE: " + "; ".join(f"{'+'.join(t['members'])} {t['val']:.5f}" for t in S3["subset_search"]["top10_val"][:5]) + "."]
    for lab in ("w_keybins", "w_distbins"):
        if lab in S3:
            L.append(f"- {lab} weights (fit A | fit B; zeros omitted): " + " ".join(
                f"{b}: " + ", ".join(f"{m} {w:.2f}" for m, w in S3[lab]["weights"]["A"][b].items() if w > 0) + " | "
                + ", ".join(f"{m} {w:.2f}" for m, w in S3[lab]["weights"]["B"][b].items() if w > 0) + ";" for b in S3[lab]["bins"]))
    L += [f"- n: val {fn(v['val']['n'])} hidden px (182 days; half A {fn(v['A']['n'])}, B {fn(v['B']['n'])}; no key {fn(v['val']['no key']['n'])}); "
          f"test {fn(v['test']['n'])} (183 days; no key {fn(v['test']['no key']['n'])}).",
          "- Figure figures/ens33_step3_blk4.png; results results/ens33.json (step3_blk4)."]
    E["step 3"] = "\n".join(L) + "\n"
    return E, rows


def write_findings(E):
    with open(PR.FINDINGS.with_suffix(".md.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        s = PR.FINDINGS.read_text()
        for k, text in E.items():
            s = re.sub(rf"\n## ens33 {k}: .*?(?=\n## |\Z)", "", s, flags=re.S)
        PR.FINDINGS.write_text(s.rstrip("\n") + "\n\n" + "\n".join(E[k] for k in ("step 1", "step 2", "step 3")))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-findings", action="store_true")
    a = p.parse_args()
    FIG.mkdir(exist_ok=True)
    R = json.loads((OUT / "gpu_raw.json").read_text())
    G = {"blk": blk_full(R)}
    for k, x in G["blk"].items():
        if "chk_val_mae" in x:
            ref = R["step3_blk4"]["plain"][k]["scores"]["val"]["mae"] if k in R["step3_blk4"]["plain"] else R["step3_blk4"][k]["scores"]["oos"]["val"]["mae"]
            print("check val MAE", k, x["chk_val_mae"], ref, flush=True)
    bname = next(n for n in R["step1"] if n.startswith("blk4_"))
    G[bname] = G["blk"]["mean_bestsub_val"]
    G.update(io_full(R))
    R["postrun_grad_lag1"] = G
    (OUT / "ens33.json").write_text(json.dumps(R, indent=1))
    E, rows = entries(R, G)
    fig_step1(R); fig_step2(R); fig_step3(R, rows)
    for t in E.values():
        print(t)
    if not a.no_findings:
        write_findings(E)


if __name__ == "__main__":
    main()
