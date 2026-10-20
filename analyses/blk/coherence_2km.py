"""Scale-dependent lead-1 coherence of ln CHL at 2 km (Black Sea), compared with the 4 km result of
analyses/blk4/spectral_motion.py.

Method (also written to results/coherence_2km.json "assumptions")
- Store ~/mind_the_chl_gap/data/blk_store_2014_2024 (395 x 550, 2 km). ln CHL; CHL > 100 mg/m3 or <= 0 -> missing
  (spectral_motion.ln_chl). Nominal pixel 2.25 km (half the 4.5 km used at 4 km).
- Code imported from analyses.blk4.spectral_motion (cut, ncc_maps, peaks, coherence_pair, msc, med_curve, jobs_list,
  patch_grid, ln_chl); its module constants are rebound for 2 km: patch P = 64 px (~145 km), stride 32, origin
  (row 4, col 2) = the 4 km patch grid x 2 (same 176 positions, kept if >= 50 % ocean at 2 km); Hann 64 x 64; rings
  m = 1..32, wavelength 64 / m px (144 km .. 4.5 km); NCC minimum pixel overlap 1024 (25 % of the patch, as 256 / 1024
  at 4 km); lead-1 search window +-12 px (= +-6 px at 4 km, same ~27 km).
- Day pairs: start days every 5th store day (store index 0, 5, 10, ...), lead 1 only (t+1 the next calendar day in
  the store). Null: one random other day >= 30 days away per data pair (seed 0), all pairs (the 4 km run used a
  random 1/3 of pairs).
- Coherence on patches >= 90 % observed on both days; unaligned = same position on day t+1; aligned = day t+1 patch
  at the integer NCC peak (any non-edge peak) + sub-pixel phase ramp, >= 90 % observed there. Median MSC per ring
  over patch-pairs with all rings finite (spectral_motion.med_curve), aligned and "unaligned_same_as_aligned" on the
  same patch-pairs as aligned; "unaligned_all" on all clear patch-pairs.
- 4 km comparison: (a) the published 4 km lead-1 curves (results/spectral_motion.json, all days); (b) the 4 km cache
  (results/spectral_motion_patches.npz) restricted to the same start days; (c) matched: patch-pairs (start day,
  patch position) that are clear and aligned-valid at both 2 km and 4 km. Both patches span the same ~144 km, so
  4 km ring m <-> 2 km ring m (same wavelength 144 / m km, m = 1..16); 2 km rings 17-32 (8.5-4.5 km) have no 4 km
  counterpart.

Run from the repo root: python -m analyses.blk.coherence_2km   (CPU only)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from chl_cnn.plotting import OKABE_ITO as PAL
import analyses.blk4.spectral_motion as SM

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
STORE = Path.home() / "mind_the_chl_gap/data/blk_store_2014_2024"
EVERY = 5
CACHE = RES / "coherence_2km_patches.npz"

# ---- rebind spectral_motion constants for the 2 km grid ----
SM.P, SM.STRIDE, SM.R0, SM.C0 = 64, 32, 4, 2
SM.NR, SM.PX_KM, SM.MIN_OVL_PX = 32, 2.25, 1024
SM.LEADS, SM.SW = (1,), {1: 12}
SM.WIN = np.outer(np.hanning(SM.P + 2)[1:-1], np.hanning(SM.P + 2)[1:-1])
SM.FY, SM.FX = np.meshgrid(np.fft.fftfreq(SM.P), np.fft.fftfreq(SM.P), indexing="ij")
SM.RING = np.rint(np.hypot(SM.FY, SM.FX) * SM.P).astype(int)
SM.WL_KM = SM.P / np.arange(1, SM.NR + 1) * SM.PX_KM
S_SEARCH = 12

_G = {}


def _init():
    _G["chl"] = np.load(STORE / "chl_l3.npy", mmap_mode="r")
    oc = np.load(STORE / "ocean.npy").astype(bool)
    pos, of = SM.patch_grid(oc)
    _G["pos"], _G["pidx"] = pos[of >= SM.MIN_VALID], np.flatnonzero(of >= SM.MIN_VALID)


def work(job):
    i, j, k, null = job
    pos, pidx = _G["pos"], _G["pidx"]
    A, B = SM.ln_chl(_G["chl"][i]), SM.ln_chl(_G["chl"][j])
    PA, PB = SM.cut(A, pos), SM.cut(B, pos)
    fa, fb = np.isfinite(PA).mean((1, 2)), np.isfinite(PB).mean((1, 2))
    clear = (fa >= SM.CLEAR) & (fb >= SM.CLEAR)
    if not clear.any():
        return None
    q = np.flatnonzero(clear)
    Rm, _, _ = SM.ncc_maps(PA[q], PB[q], S=S_SEARCH)
    p = SM.peaks(Rm, SM.SW[k], S=S_SEARCH)
    okshift = p["ok"] & ~p["edge"]
    un, al = SM.coherence_pair(PA[q], B, pos[q], p["fy"], p["fx"], okshift)
    n = len(q)
    return dict(i=np.full(n, i), j=np.full(n, j), null=np.full(n, null), p=pidx[q], fy=p["fy"], fx=p["fx"], rpk=p["rpk"],
                u_xy=un[0].astype("complex64"), u_xx=un[1].astype("float32"), u_yy=un[2].astype("float32"),
                a_xy=al[0].astype("complex64"), a_xx=al[1].astype("float32"), a_yy=al[2].astype("float32"))


def compute(nproc=4):
    t0 = time.time()
    time_ = np.load(STORE / "time.npy")
    jobs, _ = SM.jobs_list(time_, EVERY)
    t = time_.astype("datetime64[D]").astype(int)
    rng = np.random.default_rng(0); nulls = []
    for (i, j, k, _) in jobs:
        while True:
            jj = int(rng.integers(0, len(t)))
            if abs(t[jj] - t[i]) >= 30:
                break
        nulls.append((i, jj, k, 1))
    print(f"{len(jobs)} pairs + {len(nulls)} null pairs", flush=True)
    recs = []
    with Pool(nproc, initializer=_init) as pool:
        for q, r in enumerate(pool.imap_unordered(work, jobs + nulls, chunksize=8)):
            if r is not None:
                recs.append(r)
            if q % 200 == 0:
                print(f"{q}/{len(jobs) + len(nulls)} {time.time() - t0:.0f}s", flush=True)
    arr = {kk: np.concatenate([r[kk] for r in recs]) for kk in recs[0]}
    oc = np.load(STORE / "ocean.npy").astype(bool)
    pos, of = SM.patch_grid(oc)
    np.savez_compressed(CACHE, **arr, meta=json.dumps(dict(n_pairs=len(jobs), n_null=len(nulls), seconds=round(time.time() - t0),
                        n_patches_kept=int((of >= SM.MIN_VALID).sum()), n_patches_grid=len(pos), start_days=sorted({j[0] for j in jobs}))))


def curves(z, sel):
    al_ok = sel & np.isfinite(z["a_xx"]).all(1)
    out = {}
    for kind, s, lab in [("u", sel, "unaligned_all"), ("u", al_ok, "unaligned_same_as_aligned"), ("a", al_ok, "aligned")]:
        mc, n = SM.med_curve(SM.ring_msc(z, s, kind))
        out[lab] = dict(n=n, median_msc=None if mc is None else [round(float(v), 4) for v in mc])
    return out


def main():
    if "--recompute" in sys.argv or not CACHE.exists():
        compute()
    z = dict(np.load(CACHE)); meta = json.loads(str(z.pop("meta")))
    wl2 = [round(float(w), 2) for w in SM.WL_KM]
    res = dict(wavelength_km=wl2, meta={k: v for k, v in meta.items() if k != "start_days"}, n_start_days=len(meta["start_days"]))
    res["lead1"] = curves(z, z["null"] == 0)
    res["lead1_null"] = curves(z, z["null"] == 1)
    # data quality
    chl = np.load(STORE / "chl_l3.npy", mmap_mode="r"); oc = np.load(STORE / "ocean.npy").astype(bool)
    days = np.array(meta["start_days"]); nn = nb = 0
    for d0 in range(0, len(days), 100):
        x = np.asarray(chl[days[d0:d0 + 100]])[:, oc]; nn += int(np.isnan(x).sum()); nb += int((x > 100).sum())
    res["data_quality"] = dict(start_days=len(days), ocean_px=int(oc.sum()), chl_nan_ocean_on_start_days=nn,
                               chl_gt100_set_missing=nb, ocean_px_days=int(len(days) * oc.sum()),
                               nonfinite_msc_rows_data=int((~np.isfinite(SM.ring_msc(z, z["null"] == 0, "u")).all(1)).sum()))
    # ---- 4 km comparison ----
    j4 = json.load(open(ROOT / "analyses/blk4/results/spectral_motion.json"))["coherence"]
    wl4 = j4["wavelength_km"]
    pub = {lab: j4["lead1"][lab] for lab in ["aligned", "unaligned_all", "unaligned_same_as_aligned"]}
    pub["null_aligned"] = j4["lead1"]["null_aligned"]
    z4 = dict(np.load(ROOT / "analyses/blk4/results/spectral_motion_patches.npz"))
    base4 = z4["clear"] & (z4["k"] == 1)
    sd = np.isin(z4["i"], days)
    sd4 = {}
    for nul in (0, 1):
        sel = base4 & (z4["null"] == nul) & sd
        al_ok = sel & np.isfinite(z4["a_xx"]).all(1)
        for kind, s, lab in [("u", sel, "unaligned_all"), ("a", al_ok, "aligned"), ("u", al_ok, "unaligned_same_as_aligned")]:
            mc, n = SM.med_curve(SM.msc(z4[f"{kind}_xy"][s], z4[f"{kind}_xx"][s], z4[f"{kind}_yy"][s]))
            sd4[("null_" if nul else "") + lab] = dict(n=n, median_msc=None if mc is None else [round(float(v), 4) for v in mc])
    # matched patch-pairs (data only)
    s4 = base4 & (z4["null"] == 0) & sd & np.isfinite(z4["a_xx"]).all(1)
    s2 = (z["null"] == 0) & np.isfinite(z["a_xx"]).all(1)
    key4 = z4["i"][s4].astype(np.int64) * 1000 + z4["p"][s4]; key2 = z["i"][s2].astype(np.int64) * 1000 + z["p"][s2]
    common, i4, i2 = np.intersect1d(key4, key2, return_indices=True)
    m4 = np.flatnonzero(s4)[i4]; m2 = np.flatnonzero(s2)[i2]
    matched = dict(n=int(len(common)))
    for kind in ("a", "u"):
        M4 = SM.msc(z4[f"{kind}_xy"][m4], z4[f"{kind}_xx"][m4], z4[f"{kind}_yy"][m4])
        M2 = SM.msc(z[f"{kind}_xy"][m2], z[f"{kind}_xx"][m2], z[f"{kind}_yy"][m2])
        matched[f"4km_{kind}"] = [round(float(v), 4) for v in np.median(M4, 0)]
        matched[f"2km_{kind}"] = [round(float(v), 4) for v in np.median(M2, 0)]
    # shift agreement on matched pairs (2 km px / 2 = 4 km px)
    dfy = z["fy"][m2] / 2 - z4["fy"][m4]; dfx = z["fx"][m2] / 2 - z4["fx"][m4]
    matched["shift_diff_4kmpx"] = dict(median_abs=round(float(np.median(np.hypot(dfy, dfx))), 3),
                                       frac_within_0p5=round(float((np.hypot(dfy, dfx) <= 0.5).mean()), 3))
    res["compare_4km"] = dict(wavelength_km_4km=wl4, published_all_days=pub, same_start_days=sd4, matched_patch_pairs=matched,
                              note="4 km ring m <-> 2 km ring m (m = 1..16, same wavelength)")
    doc = __doc__.split('Method (also written to results/coherence_2km.json "assumptions")')[1].split("Run from")[0].strip()
    res = dict(assumptions=doc.split("\n- "), **res)
    json.dump(res, open(RES / "coherence_2km.json", "w"), indent=1)
    plot(res)
    # concise print
    L = res["lead1"]; N = res["lead1_null"]
    print(json.dumps(dict(n=dict(aligned=L["aligned"]["n"], unaligned_all=L["unaligned_all"]["n"], null_aligned=N["aligned"]["n"]),
                          dq=res["data_quality"], meta=res["meta"], matched_n=matched["n"], shift=matched["shift_diff_4kmpx"])))
    for q, w in enumerate(wl2):
        m = q + 1
        x4 = ""
        if m <= 16:
            r4 = m - 1
            x4 = (f" | 4km pub a/u {pub['aligned']['median_msc'][r4]:.3f}/{pub['unaligned_all']['median_msc'][r4]:.3f}"
                  f" | 4km sd a {sd4['aligned']['median_msc'][r4]:.3f} | matched a 2km/4km {matched['2km_a'][q]:.3f}/{matched['4km_a'][r4]:.3f}")
        print(f"{w:6.1f} km a {L['aligned']['median_msc'][q]:.3f} u {L['unaligned_all']['median_msc'][q]:.3f} null {N['aligned']['median_msc'][q]:.3f}{x4}")


def plot(res):
    wl2 = np.array(res["wavelength_km"]); c4 = res["compare_4km"]; wl4 = np.array(c4["wavelength_km_4km"])
    L, N = res["lead1"], res["lead1_null"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    a = ax[0]
    a.semilogx(wl2, L["aligned"]["median_msc"], "o-", color=PAL[0], ms=4, label=f"2 km aligned (n {L['aligned']['n']:,})")
    a.semilogx(wl2, L["unaligned_all"]["median_msc"], "s-", color=PAL[1], ms=4, label=f"2 km unaligned (n {L['unaligned_all']['n']:,})")
    a.semilogx(wl2, N["aligned"]["median_msc"], "^-", color=PAL[7], ms=4, label=f"2 km null aligned (n {N['aligned']['n']:,})")
    p = c4["published_all_days"]
    a.semilogx(wl4, p["aligned"]["median_msc"], "o--", color=PAL[0], mfc="none", label=f"4 km aligned, all days (n {p['aligned']['n']:,})")
    a.semilogx(wl4, p["unaligned_all"]["median_msc"], "s--", color=PAL[1], mfc="none", label=f"4 km unaligned, all days (n {p['unaligned_all']['n']:,})")
    a.semilogx(wl4, p["null_aligned"]["median_msc"], "^--", color=PAL[7], mfc="none", label=f"4 km null aligned (n {p['null_aligned']['n']:,})")
    a.set_xlabel("wavelength (km)"); a.set_ylabel("median magnitude-squared coherence"); a.legend(fontsize=8)
    a.set_title(f"Lead-1 ln CHL coherence by wavelength, 2 km (every {EVERY}th start day) and 4 km")
    a.set_xticks([4.5, 9, 18, 36, 72, 144]); a.set_xticklabels([4.5, 9, 18, 36, 72, 144])
    m = c4["matched_patch_pairs"]; b = ax[1]
    b.semilogx(wl2, m["2km_a"], "o-", color=PAL[0], ms=4, label="2 km aligned")
    b.semilogx(wl4, m["4km_a"], "o--", color=PAL[0], mfc="none", label="4 km aligned")
    b.semilogx(wl2, m["2km_u"], "s-", color=PAL[1], ms=4, label="2 km unaligned")
    b.semilogx(wl4, m["4km_u"], "s--", color=PAL[1], mfc="none", label="4 km unaligned")
    b.set_xlabel("wavelength (km)"); b.set_ylabel("median magnitude-squared coherence"); b.legend(fontsize=8)
    b.set_title(f"Same patch-pairs clear at both resolutions (n {m['n']:,})")
    b.set_xticks([4.5, 9, 18, 36, 72, 144]); b.set_xticklabels([4.5, 9, 18, 36, 72, 144])
    fig.tight_layout(); fig.savefig(FIG / "coherence_2km.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
