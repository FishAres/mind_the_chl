"""Feature motion of the anomaly A = ln C - B (static-base hypothesis, Test 3c). Re-runs the unchanged masked NCC of
analyses/blk4/spectral_motion.py (32 px patches, leads 1-3) and the block matcher of analyses/blk4/chl_motion.py (16 px
tiles, lags 1-2) on A instead of ln C, and compares with the cached ln C results on the same day pairs.  CPU only.

Method
- Bases from ~/mind_the_chl_gap/data/static_bases (chl_cnn/tracer_static.py; doy / harm fitted on TRAIN days; run = causal
  20-day mean, lp = 150 km low-pass of day t, both with the doy fallback), embedded in the full 197 x 275 grid (crop rows
  4-180, cols 0-272; NaN outside). For a pair (t, t+k): A_t = ln C(t) - B(t); A_t+k = ln C(t+k) - B(t+k) for doy / harm and
  ln C(t+k) - B(t) for run / lp (base held at day t, as in the forecast). Px with a NaN base are missing.
- NCC: same pairs as the spectral cache (null = 0), same candidates (>= 50 % observed both days), search, acceptance
  (peak r >= 0.6, not edge). No phase correlation / coherence. Block matching: chl_motion.rmap on A for T = 16, lags 1 and
  2 (all consecutive-day pairs), acceptance as chl_motion.build_table (zero shift or significant).
- Statistics as spectral_motion / chl_motion: accepted counts, zero share, OLS slope on the reanalysis-current
  displacement (patch / tile currents from the caches, east / north, 300 day-bootstrap CI), lead-k on lead-1 slope and
  median ratio. Pairs from all 2014-2024 days (bases are in-sample on train days); val-year-only rows added.

Run from the repo root: python -m analyses.tracer.static_anom_motion [--bases doy,harm,run,lp] [--no-bm]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, argparse
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from analyses.blk4 import spectral_motion as SM
from analyses.blk4 import chl_motion as CM
from chl_cnn import data as D

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
BASE_DIR = Path.home() / "mind_the_chl_gap/data/static_bases"
CROP = (4, 181, 0, 273)
HELD = {"doy": False, "harm": False, "run": True, "lp": True}
MIN_N = 5

_G = {}


def _init(base):
    _G.update(SM.load())
    pos, of = SM.patch_grid(_G["ocean"])
    _G["pos"], _G["pidx"] = pos[of >= SM.MIN_VALID], np.flatnonzero(of >= SM.MIN_VALID)
    _G["base"] = base
    _G["B"] = np.load(BASE_DIR / f"{base}.npy", mmap_mode="r")
    _G["doy"] = (_G["time"] - _G["time"].astype("datetime64[Y]")).astype(int)
    _G["shape"] = _G["ocean"].shape


def base_full(day):
    """Base for store day `day` on the full grid."""
    H, W = _G["shape"]
    out = np.full((H, W), np.nan)
    b = _G["base"]
    X = _G["B"][_G["doy"][day]] if b in ("doy", "harm") else _G["B"][day]
    out[CROP[0]:CROP[1], CROP[2]:CROP[3]] = X
    return out


def anom_pair(i, j):
    A = SM.ln_chl(_G["chl_l3"][i]) - base_full(i)
    Bj = SM.ln_chl(_G["chl_l3"][j]) - (base_full(i) if HELD[_G["base"]] else base_full(j))
    return A, Bj


def work_ncc(job):
    i, j, k = job
    pos, pidx = _G["pos"], _G["pidx"]
    A, B = anom_pair(i, j)
    PA, PB = SM.cut(A, pos), SM.cut(B, pos)
    fa, fb = np.isfinite(PA).mean((1, 2)), np.isfinite(PB).mean((1, 2))
    cand = (fa >= SM.MIN_VALID) & (fb >= SM.MIN_VALID)
    if not cand.any():
        return None
    c = np.flatnonzero(cand)
    Rm, N, Wf = SM.ncc_maps(PA[c], PB[c])
    p = SM.peaks(Rm, SM.SW[k])
    n = len(c)
    return dict(i=np.full(n, i), j=np.full(n, j), k=np.full(n, k), p=pidx[c], fy=p["fy"], fx=p["fx"], rpk=p["rpk"],
                edge=p["edge"], okpk=p["ok"])


def work_bm(job):
    i, k = job
    A, B = anom_pair(i, i + k)
    H, W = A.shape
    T = 16; gm = {T: CM.tile_geom(H, W, T)}
    fa = CM.tsum(np.isfinite(A).astype(float), T, gm[T]) / T ** 2
    fb = CM.tsum(np.isfinite(B).astype(float), T, gm[T]) / T ** 2
    cand = (fa >= CM.MIN_OBS) & (fb >= CM.MIN_OBS)
    recs = []
    if not cand.any():
        return recs
    R, N = CM.rmap(A, B, [T], gm)[T]
    S = 6 if k == 1 else 12
    for ty, tx in zip(*np.nonzero(cand)):
        p = CM.peak(R[ty, tx], N[ty, tx], S)
        if p is None:
            p = dict(sy=0, sx=0, fy=np.nan, fx=np.nan, rpk=np.nan, r0=np.nan, npk=0, n0=0, edge=False)
        recs.append((i, k, T, S, ty, tx, p["sy"], p["sx"], p["fy"], p["fx"], p["rpk"], p["r0"], p["npk"], p["n0"], p["edge"]))
    return recs


def run(base, nproc=6, bm=True):
    t0 = time.time()
    z = np.load(SM.CACHE)
    m = z["null"] == 0
    jobs = sorted(set(zip(z["i"][m].tolist(), z["j"][m].tolist(), z["k"][m].tolist())))
    with Pool(nproc, initializer=_init, initargs=(base,)) as pool:
        recs = [r for r in pool.imap_unordered(work_ncc, jobs, chunksize=16) if r is not None]
    ncc = {kk: np.concatenate([r[kk] for r in recs]) for kk in recs[0]}
    print(f"{base} ncc {len(jobs)} pairs {time.time() - t0:.0f}s", flush=True)
    out = dict(ncc=ncc)
    if bm:
        zb = np.load(CM.CACHE)
        days = sorted(set(zip(zb["day"][(zb["T"] == 16)].tolist(), zb["lag"][(zb["T"] == 16)].tolist())))
        with Pool(nproc, initializer=_init, initargs=(base,)) as pool:
            rr = []
            for r in pool.imap_unordered(work_bm, days, chunksize=8):
                rr.extend(r)
        out["bm"] = {f: np.array([r[q] for r in rr]) for q, f in enumerate(CM.FIELDS)}
        print(f"{base} bm {len(days)} day-lags {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(RES / f"static_anom_motion_{base}.npz", **{f"ncc_{k}": v for k, v in out["ncc"].items()},
                        **({f"bm_{k}": v for k, v in out["bm"].items()} if bm else {}))
    return out


# ============================== statistics ==============================
def ncc_stats(rec, z, va_days):
    """rec: dict of arrays (i, j, k, p, fy, fx, rpk, edge, okpk)."""
    out = {}
    acc = rec["okpk"] & ~rec["edge"] & (rec["rpk"] >= SM.R_MIN)
    pc = np.searchsorted(z["pidx"], rec["p"])
    key = {}
    for k in (1, 2, 3):
        mk = rec["k"] == k
        dc = np.full((2, len(mk)), np.nan)
        ik = np.flatnonzero(mk)
        dc[0, ik] = SM.lag_disp(z["vel_cu"], rec["i"][ik], pc[ik], k); dc[1, ik] = SM.lag_disp(z["vel_cv"], rec["i"][ik], pc[ik], k)
        do = np.stack([rec["fx"], rec["fy"]])
        A = mk & acc
        sel = A & np.isfinite(dc).all(0)
        zero = (np.rint(rec["fy"]) == 0) & (np.rint(rec["fx"]) == 0)
        e = dict(candidates=int(mk.sum()), accepted=int(A.sum()), zero_share=round(float(zero[A].mean()), 3),
                 abs_lt_0p5_share=round(float((np.hypot(*do[:, A]) < 0.5).mean()), 3),
                 median_abs_d=round(float(np.median(np.hypot(*do[:, A]))), 3),
                 slope_on_cur={c: SM.boot_slope(dc[q][sel], do[q][sel], rec["i"][sel]) for q, c in enumerate(["east", "north"])})
        selv = sel & np.isin(rec["i"], va_days)
        e["val_years"] = dict(n=int(selv.sum()), zero_share=round(float(zero[A & np.isin(rec["i"], va_days)].mean()), 3),
                              slope_on_cur={c: SM.boot_slope(dc[q][selv], do[q][selv], rec["i"][selv], nb=100) for q, c in enumerate(["east", "north"])})
        out[f"lead{k}"] = e
        key[k] = {(int(a), int(b)): n for n, (a, b) in enumerate(zip(rec["i"][A], rec["p"][A]))}
        key[k]["_idx"] = np.flatnonzero(A)
    for k in (2, 3):
        common = [(key[1]["_idx"][key[1][kk]], key[k]["_idx"][v]) for kk, v in key[k].items() if kk != "_idx" and kk in key[1]]
        if len(common) < MIN_N:
            out[f"lead{k}_on_lead1"] = dict(n=len(common), flag="n<5"); continue
        i1, ik = np.array(common).T
        d1 = np.stack([rec["fx"][i1], rec["fy"][i1]]); dk = np.stack([rec["fx"][ik], rec["fy"][ik]])
        out[f"lead{k}_on_lead1"] = dict(n=len(common), median_abs_lead1=round(float(np.median(np.hypot(*d1))), 3),
                                        median_abs_leadk=round(float(np.median(np.hypot(*dk))), 3),
                                        **{c: SM.boot_slope(d1[q], dk[q], rec["i"][i1], nb=100) for q, c in enumerate(["east", "north"])})
    return out


def bm_stats(arr, zb):
    out = {}
    zz = dict(zb)
    for f in CM.FIELDS:
        zz[f] = arr[f]
    for lag, S in ((1, 6), (2, 12)):
        d = CM.build_table(zz, lag, 16, S)
        A = d["accept"]; vc = A & np.isfinite(d["dc"]).all(0)
        out[f"lag{lag}_S{S}"] = dict(candidates=int(len(A)), accepted=int(A.sum()), ambiguous=int(d["ambig"].sum()),
                                     peak_below_0p6=int((d["rpk"] < CM.R_MIN).sum()),
                                     zero_share=round(float(d["zero"][A].mean()), 3),
                                     slope_on_cur={c: CM.boot_slope(d["dc"][q][vc], d["do"][q][vc], d["day"][vc]) for q, c in enumerate(["east", "north"])})
        out[f"_tab{lag}"] = d
    a, b = out.pop("_tab1"), out.pop("_tab2")
    k1 = {(dd, y, x): j for j, (dd, y, x) in enumerate(zip(a["day"], a["ty"], a["tx"])) if a["accept"][j]}
    pairs = [(k1[(dd, y, x)], j) for j, (dd, y, x) in enumerate(zip(b["day"], b["ty"], b["tx"])) if b["accept"][j] and (dd, y, x) in k1]
    if len(pairs) >= MIN_N:
        i1, i2 = np.array(pairs).T
        out["lag2_on_lag1"] = dict(n=len(pairs), **{c: CM.boot_slope(a["do"][q][i1], b["do"][q][i2], a["day"][i1], nb=100) for q, c in enumerate(["east", "north"])},
                                   median_abs_lag1=round(float(np.median(np.hypot(*a["do"][:, i1]))), 3), median_abs_lag2=round(float(np.median(np.hypot(*b["do"][:, i2]))), 3))
    return out


def raw_records():
    z = np.load(SM.CACHE)
    m = z["null"] == 0
    return {f: z[f][m] for f in ("i", "j", "k", "p", "fy", "fx", "rpk", "edge", "okpk")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bases", default="doy,harm,run,lp"); ap.add_argument("--no-bm", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    z = np.load(SM.CACHE); zb = np.load(CM.CACHE)
    g = SM.load()
    tr, va, te = D.split_by_years(g["time"], (2017, 2021), (2019, 2023))
    out_path = RES / "static_anom_motion.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res["assumptions"] = __doc__.split("Method")[1].split("Run from")[0].strip()
    if "raw" not in res:
        res["raw"] = dict(ncc=ncc_stats(raw_records(), z, va),
                          bm=bm_stats({f: zb[f] for f in CM.FIELDS}, zb))
        out_path.write_text(json.dumps(res, indent=1, default=str))
    for b in a.bases.split(","):
        f = RES / f"static_anom_motion_{b}.npz"
        if a.stats_only and f.exists():
            q = np.load(f)
            rec = {"ncc": {k[4:]: q[k] for k in q.files if k.startswith("ncc_")}}
            if any(k.startswith("bm_") for k in q.files):
                rec["bm"] = {k[3:]: q[k] for k in q.files if k.startswith("bm_")}
        else:
            rec = run(b, bm=not a.no_bm)
        res[b] = dict(ncc=ncc_stats(rec["ncc"], z, va))
        if "bm" in rec:
            res[b]["bm"] = bm_stats(rec["bm"], zb)
        out_path.write_text(json.dumps(res, indent=1, default=str))
        print(b, json.dumps(res[b])[:1500], flush=True)


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from chl_cnn.plotting import OKABE_ITO as C
    r = json.loads((RES / "static_anom_motion.json").read_text())
    fields = [f for f in ["raw", "doy", "harm", "run", "lp"] if f in r]
    lab = {"raw": "ln C (cached)", "doy": "A = ln C - doy clim", "harm": "A = ln C - harmonics", "run": "A = ln C - 20 d running mean", "lp": "A = ln C - 150 km low-pass"}
    sl = lambda e: np.mean([e["slope_on_cur"]["east"]["slope"], e["slope_on_cur"]["north"]["slope"]])
    L = ["# Feature motion of the anomaly A = ln C - B (masked NCC 32 px, block matching 16 px)", "",
         "| field | NCC accepted l1 / l2 / l3 | NCC zero share l1 / l2 / l3 | NCC slope on current l1 / l2 / l3 (e, n) | NCC median abs shift l1 -> l2 -> l3 (same pairs) | NCC OLS lead-2 / lead-3 on lead-1 (e, n) | BM accepted / ambiguous lag 1 | BM zero share lag 1 / 2 | BM slope on current lag 1 (e, n) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for f in fields:
        n = r[f]["ncc"]; b = r[f].get("bm", {})
        s_ = lambda e: f"{e['slope_on_cur']['east']['slope']:.2f}, {e['slope_on_cur']['north']['slope']:.2f}"
        L.append(f"| {lab[f]} | {n['lead1']['accepted']:,} / {n['lead2']['accepted']:,} / {n['lead3']['accepted']:,} | "
                 f"{n['lead1']['zero_share']:.3f} / {n['lead2']['zero_share']:.3f} / {n['lead3']['zero_share']:.3f} | "
                 f"{s_(n['lead1'])} / {s_(n['lead2'])} / {s_(n['lead3'])} | "
                 f"{n['lead2_on_lead1']['median_abs_lead1']:.2f} -> {n['lead2_on_lead1']['median_abs_leadk']:.2f} (n {n['lead2_on_lead1']['n']:,}); {n['lead3_on_lead1']['median_abs_lead1']:.2f} -> {n['lead3_on_lead1']['median_abs_leadk']:.2f} (n {n['lead3_on_lead1']['n']:,}) | "
                 f"{n['lead2_on_lead1']['east']['slope']:.2f}, {n['lead2_on_lead1']['north']['slope']:.2f} / {n['lead3_on_lead1']['east']['slope']:.2f}, {n['lead3_on_lead1']['north']['slope']:.2f} | "
                 + (f"{b['lag1_S6']['accepted']:,} / {b['lag1_S6']['ambiguous']:,} | {b['lag1_S6']['zero_share']:.3f} / {b['lag2_S12']['zero_share']:.3f} | "
                    f"{b['lag1_S6']['slope_on_cur']['east']['slope']:.2f}, {b['lag1_S6']['slope_on_cur']['north']['slope']:.2f} |" if b else "- | - | - |"))
    L += ["", "Val-year start days only, NCC slope on current l1 / l2 / l3 (mean e, n):", ""]
    for f in fields:
        n = r[f]["ncc"]
        L.append(f"- {lab[f]}: " + " / ".join(f"{np.mean([n[k]['val_years']['slope_on_cur']['east']['slope'], n[k]['val_years']['slope_on_cur']['north']['slope']]):.2f} (n {n[k]['val_years']['n']:,})" for k in ("lead1", "lead2", "lead3")))
    (RES / "static_anom_motion.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    x = np.arange(3)
    for u, f in enumerate(fields):
        n = r[f]["ncc"]; w = 0.8 / len(fields)
        ax[0].bar(x + (u - len(fields) / 2 + .5) * w, [n[k]["zero_share"] for k in ("lead1", "lead2", "lead3")], w, color=C[u], label=lab[f])
        ax[1].plot([1, 2, 3], [sl(n[k]) for k in ("lead1", "lead2", "lead3")], marker="o", color=C[u], label=lab[f])
        ax[2].plot([1, 2, 3], [n["lead2_on_lead1"]["median_abs_lead1"], n["lead2_on_lead1"]["median_abs_leadk"], n["lead3_on_lead1"]["median_abs_leadk"]], marker="o", color=C[u], label=lab[f])
        if "bm" in r[f]:
            ax[3].bar(u, r[f]["bm"]["lag1_S6"]["zero_share"], color=C[u], label=lab[f])
    ax[0].set(xticks=x, xticklabels=["lead 1", "lead 2", "lead 3"], ylabel="zero-shift share of accepted patch-pairs", title="Masked NCC: zero-shift share by lead")
    ax[1].set(xlabel="lead (days)", ylabel="OLS slope of shift on current displacement", title="Masked NCC: slope on reanalysis current (mean e/n)")
    ax[2].plot([1, 2, 3], [1.14, 2.28, 3.42], color="0.5", ls=":", label="linear in lead from raw lead 1")
    ax[2].set(xlabel="lead (days)", ylabel="median |shift| (px), same patch-pairs", title="Masked NCC: median |shift| by lead (lead-1 set of each pair)")
    ax[3].set(xticks=range(len(fields)), xticklabels=[f for f in fields], ylabel="zero-shift share of accepted tiles", title="Block matching lag 1: zero-shift share")
    for a_ in ax: a_.grid(alpha=.3)
    ax[1].legend(fontsize=7); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "static_anom_motion.png", dpi=110); plt.close(fig)


if __name__ == "__main__" and "--report" in sys.argv:
    report()
elif __name__ == "__main__":
    main()
