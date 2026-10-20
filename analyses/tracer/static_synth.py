"""Test 1 of the static-base hypothesis: synthetic fields = static part (variance fraction f) + moving part, observed
through REAL cloud masks, measured with the unchanged estimators of analyses/blk4/spectral_motion.py (masked NCC on 32 px
patches) and analyses/blk4/chl_motion.py (16 px block matching). Do zero-shift spikes and non-scaling lead-2/3 shifts
appear as f grows, and which f reproduces the observed statistics?  CPU only.

Method
- Sequences: N_SEQ random (start day i, patch p) from all 2014-2024 store days with i..i+3 consecutive calendar days, one
  of the 91 spectral_motion patches (>= 50 % ocean) observed >= 50 % on day i, and finite patch-mean reanalysis current
  (spectral cache vel_cu / vel_cv, px/day) on days i..i+3. Seed 0.
- Masks: the real observed mask (ocean & finite ln CHL) of the 64 x 64 px window centred on the patch (patch = window
  [16:48, 16:48]; outside the grid = missing) on days i..i+3.
- Fields on a 128 x 128 periodic grid (window = [32:96, 32:96]), spectral amplitude |q|^-1.5 (power k^-3, as the
  spectral_motion synthetic check), unit variance:
    static B (same every day);  moving M_0 = GRF, M_{j+1}^(q) = rho(q) M_j^(q) exp(-2 pi i q.d_j) + sqrt(1 - rho^2) amp(q) xi_{j+1}(q)
    (AR(1) per Fourier mode with an exact sub-pixel shift; stationary spectrum), d_j = gamma x trapezoid of the real patch
    current over day j..j+1 (px; +row = north, +col = east);
    rho(q) = exp(-rate(lambda)) per day, rate = the per-wavelength aligned decay rates of FINDINGS / spectral_motion.json
    (coherence decay_fits aligned per_ring, interpolated in log wavelength; clamped to the 144 km value above 144 km and
    to the 9 km value below 9 km);
    observed field F_j^ = s(q) (sqrt(f) B^ + sqrt(1 - f) M_j^) + sqrt(1 - s^2) amp(q) eta_j(q), s^2 = min(exp(c(lambda)), 1)
    with c the per-wavelength intercept of the same fits (independent daily small-scale noise; reproduces the lead-0
    intercept of the measured coherence decay).
  The same random numbers are used for every (f, gamma) (common random numbers).
- Estimators (imported, unchanged): NCC = spectral_motion.ncc_maps + peaks on the window's central 32 x 32 patch of days
  0 and k, candidate >= 50 % observed on both days, search +-6/9/12 px for k = 1/2/3, accepted peak r >= 0.6 and not on the
  window edge. Block matching = the chl_motion definitions re-implemented for 16 x 16 tiles (the four tiles inside the
  patch; Pearson r over px observed in both, overlap >= 50 % of 256 px, shifts +-12), chl_motion.peak, acceptance as
  chl_motion.build_table (peak r >= 0.6, not edge, zero best shift or significant vs zero shift); candidate tile >= 80 %
  observed on both days; search +-6 at lead 1, +-12 at leads 2-3 (lead 3 is not in the chl_motion cache).
- Statistics per (f, gamma) and the same on the real data of the same sequences (spectral cache for NCC, chl_motion cache
  for block matching): NCC zero share (integer best shift (0, 0)) per lead, share |d| < 0.5 px, OLS slope of the measured
  shift on the current displacement over the lag (east / north; on the true moving-part displacement for synthetic),
  lead-k on lead-1 slope and median |d_k| / median |d_1| on sequences accepted at both leads; block-matching zero share at
  lead 1 and the lag-2 on lag-1 slope.
- Control: f = 0, gamma = 1, rho = 1, s = 1 (frozen field, pure shift): the measured shift must track the truth.

Run from the repo root: python -m analyses.tracer.static_synth
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from analyses.blk4 import spectral_motion as SM
from analyses.blk4 import chl_motion as CM

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
OUT = RES / "static_synth.json"
N_SEQ, NG, NWIN = 2000, 128, 64
F_GRID = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
G_GRID = [0.5, 0.75, 1.0]
LEADS = (1, 2, 3)
BM_S = {1: 6, 2: 12, 3: 12}
TILES = [(16, 16), (16, 32), (32, 16), (32, 32)]
MIN_N = 5


# ============================== inputs ==============================
def decay_tables():
    r = json.load(open(ROOT / "analyses/blk4/results/spectral_motion.json"))
    pr = r["coherence"]["decay_fits"]["aligned"]["per_ring"]
    wl = np.array([x["wavelength_km"] for x in pr]); rate = np.array([x["rate_per_day"] for x in pr]); c = np.array([x["intercept"] for x in pr])
    return wl, rate, c


def mode_tables():
    """amp, rho, s on the NG x NG periodic grid."""
    wl, rate, c = decay_tables()
    ky, kx = np.meshgrid(np.fft.fftfreq(NG), np.fft.fftfreq(NG), indexing="ij")
    q = np.hypot(ky, kx); q0 = q.copy(); q0[0, 0] = 1.0
    lam = SM.PX_KM / q0
    o = np.argsort(wl)
    ll = np.clip(np.log(lam), np.log(wl.min()), np.log(wl.max()))
    rt = np.interp(ll, np.log(wl[o]), rate[o]); cc = np.interp(ll, np.log(wl[o]), c[o])
    amp = q0 ** -1.5; amp[0, 0] = 0.0
    rho = np.exp(-rt); s2 = np.minimum(np.exp(cc), 1.0)
    return amp, rho, np.sqrt(s2), ky, kx


def sample_sequences(seed=0):
    z = np.load(SM.CACHE)
    g = SM.load()
    t = g["time"].astype("datetime64[D]").astype(int)
    pos, pidx = z["pos"], z["pidx"]
    cu, cv = z["vel_cu"], z["vel_cv"]
    ok_i = np.array([i + 3 < len(t) and t[i + 3] - t[i] == 3 for i in range(len(t))])
    rng = np.random.default_rng(seed)
    seqs = []
    obs_cache = {}
    def obs(i):
        if i not in obs_cache:
            obs_cache[i] = np.isfinite(SM.ln_chl(g["chl_l3"][i])) & g["ocean"]
        return obs_cache[i]
    tries = 0
    while len(seqs) < N_SEQ and tries < 200000:
        tries += 1
        i = int(rng.integers(0, len(t))); q = int(rng.integers(0, len(pos)))
        if not ok_i[i]:
            continue
        v = np.stack([cu[i:i + 4, q], cv[i:i + 4, q]])
        if not np.isfinite(v).all():
            continue
        r, c = pos[q]
        if obs(i)[r:r + 32, c:c + 32].mean() < 0.5:
            continue
        seqs.append((i, q))
    H, W = g["ocean"].shape
    masks = np.zeros((len(seqs), 4, NWIN, NWIN), bool)
    for n, (i, q) in enumerate(seqs):
        r, c = pos[q] - 16
        rr = np.arange(r, r + NWIN); cc = np.arange(c, c + NWIN)
        inr = (rr >= 0) & (rr < H); inc = (cc >= 0) & (cc < W)
        for j in range(4):
            o = obs(i + j)
            mm = np.zeros((NWIN, NWIN), bool)
            mm[np.ix_(inr, inc)] = o[np.ix_(rr[inr], cc[inc])]
            masks[n, j] = mm
    seqs = np.array(seqs)
    vel = np.stack([np.stack([cu[i:i + 4, q], cv[i:i + 4, q]], 1) for i, q in seqs])        # (N, 4, 2) east, north px/day
    return seqs, masks, vel, dict(tries=tries, n=len(seqs))


# ============================== synthetic fields ==============================
def make_fields(f, gamma, vel, seed=0, batch=200, frozen=False):
    """Returns F (N, 4, NWIN, NWIN) float64 (no mask) and the true moving-part displacement per day interval (N, 3, 2)
    (east, north) px."""
    amp, rho, s, ky, kx = mode_tables()
    if frozen:
        rho = np.ones_like(rho); s = np.ones_like(s)
    N = len(vel)
    out = np.zeros((N, 4, NWIN, NWIN))
    dtrue = gamma * 0.5 * (vel[:, :-1] + vel[:, 1:])                                       # (N, 3, 2)
    norm = np.sqrt((amp ** 2).sum()) / NG                                                  # std of real(ifft(amp xi^)) with xi^ = fft(white)
    for b0 in range(0, N, batch):
        rng = np.random.default_rng([seed, b0])
        nb = min(batch, N - b0)
        wn = lambda: np.fft.fft2(rng.standard_normal((nb, NG, NG)))
        Bh = amp * wn() / norm
        Mh = amp * wn() / norm
        innov = [amp * wn() / norm for _ in range(3)]
        noise = [amp * wn() / norm for _ in range(4)]
        for j in range(4):
            if j > 0:
                d = dtrue[b0:b0 + nb, j - 1]
                ph = np.exp(-2j * np.pi * (ky[None] * d[:, 1, None, None] + kx[None] * d[:, 0, None, None]))
                Mh = rho * Mh * ph + np.sqrt(1 - rho ** 2) * innov[j - 1]
            Fh = s * (np.sqrt(f) * Bh + np.sqrt(1 - f) * Mh) + np.sqrt(1 - s ** 2) * noise[j]
            out[b0:b0 + nb, j] = np.real(np.fft.ifft2(Fh))[:, 32:96, 32:96]
    return out, dtrue


# ============================== estimators ==============================
def run_ncc(F, masks):
    """F, masks (N, 4, 64, 64). Returns per lead dict of arrays (N,)."""
    res = {}
    A = np.where(masks[:, 0], F[:, 0], np.nan)[:, 16:48, 16:48]
    fa = np.isfinite(A).mean((1, 2))
    for k in LEADS:
        B = np.where(masks[:, k], F[:, k], np.nan)[:, 16:48, 16:48]
        fb = np.isfinite(B).mean((1, 2))
        cand = (fa >= SM.MIN_VALID) & (fb >= SM.MIN_VALID)
        fy = np.full(len(F), np.nan); fx = fy.copy(); sy = np.zeros(len(F), int); sx = sy.copy(); acc = np.zeros(len(F), bool)
        c = np.flatnonzero(cand)
        for b0 in range(0, len(c), 500):
            cc = c[b0:b0 + 500]
            R, _, _ = SM.ncc_maps(A[cc], B[cc])
            p = SM.peaks(R, SM.SW[k])
            fy[cc], fx[cc], sy[cc], sx[cc] = p["fy"], p["fx"], p["sy"], p["sx"]
            acc[cc] = p["ok"] & ~p["edge"] & (p["rpk"] >= SM.R_MIN)
        res[k] = dict(cand=cand, acc=acc, fy=fy, fx=fx, sy=sy, sx=sx)
    return res


def bm_rmap(A, Bw, S=12):
    """A (n, 16, 16) NaN = missing; Bw (n, 16 + 2S, 16 + 2S) the day-t+k window centred on the tile.
    Returns r, N (n, 2S+1, 2S+1) as chl_motion.rmap."""
    T = 16
    n = len(A)
    mA = np.isfinite(A); a0 = np.where(mA, A, 0.0)
    R = np.full((n, 2 * S + 1, 2 * S + 1), np.nan); Nn = np.zeros_like(R)
    for iy, sy in enumerate(range(-S, S + 1)):
        for ix, sx in enumerate(range(-S, S + 1)):
            Bs = Bw[:, S + sy:S + sy + T, S + sx:S + sx + T]
            m = mA & np.isfinite(Bs)
            a, b = np.where(m, a0, 0.0), np.where(m, Bs, 0.0)
            nn = m.sum((1, 2)).astype(float)
            sa, sb = a.sum((1, 2)), b.sum((1, 2)); saa, sbb, sab = (a * a).sum((1, 2)), (b * b).sum((1, 2)), (a * b).sum((1, 2))
            with np.errstate(invalid="ignore", divide="ignore"):
                den = np.sqrt((nn * saa - sa ** 2) * (nn * sbb - sb ** 2))
                r = (nn * sab - sa * sb) / den
            r[(nn < CM.MIN_OVL * T * T) | ~(den > 1e-12)] = np.nan
            R[:, iy, ix] = r; Nn[:, iy, ix] = nn
    return R, Nn


def run_bm(F, masks):
    """Per lead: arrays (N, 4 tiles) of cand, accept, zero, ambiguous, fy, fx."""
    res = {}
    N = len(F)
    for k in LEADS:
        S = BM_S[k]
        out = {nm: np.zeros((N, 4), bool) for nm in ("cand", "acc", "zero", "ambig")}
        out.update(fy=np.full((N, 4), np.nan), fx=np.full((N, 4), np.nan))
        for q, (r, c) in enumerate(TILES):
            A = np.where(masks[:, 0], F[:, 0], np.nan)[:, r:r + 16, c:c + 16]
            Bfull = np.where(masks[:, k], F[:, k], np.nan)
            Bt = Bfull[:, r:r + 16, c:c + 16]
            cand = (np.isfinite(A).mean((1, 2)) >= CM.MIN_OBS) & (np.isfinite(Bt).mean((1, 2)) >= CM.MIN_OBS)
            ci = np.flatnonzero(cand)
            if not len(ci):
                continue
            Bw = Bfull[ci, r - 12:r + 28, c - 12:c + 28]
            R, Nn = bm_rmap(A[ci], Bw, 12)
            for u, n in enumerate(ci):
                p = CM.peak(R[u], Nn[u], S)
                if p is None:
                    continue
                zero = p["sy"] == 0 and p["sx"] == 0
                good = (p["rpk"] >= CM.R_MIN) and not p["edge"]
                nE = 1.96 * np.sqrt(2 / max(p["npk"] / 4 - 3, 1))
                dz = np.arctanh(np.clip(p["rpk"], -.999, .999)) - np.arctanh(np.clip(p["r0"], -.999, .999)) if np.isfinite(p["r0"]) else np.nan
                sig = dz > nE
                out["acc"][n, q] = good and (zero or sig); out["ambig"][n, q] = good and not zero and not sig
                out["zero"][n, q] = zero; out["fy"][n, q] = p["fy"]; out["fx"][n, q] = p["fx"]
            out["cand"][:, q] = cand
        res[k] = out
    return res


# ============================== statistics ==============================
def ols(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_N or x[ok].std() == 0:
        return dict(n=int(ok.sum()), flag="n<5")
    b, a = np.polyfit(x[ok], y[ok], 1)
    return dict(slope=round(float(b), 3), r2=round(float(np.corrcoef(x[ok], y[ok])[0, 1] ** 2), 3), n=int(ok.sum()))


def stats(ncc, bm, dref, dtrue=None):
    """dref (N, 3, 2) per-interval reference displacement (current, px); dtrue optional true moving displacement."""
    s = {}
    for k in LEADS:
        r = ncc[k]; A = r["acc"]
        dr = dref[:, :k].sum(1)                                            # (N, 2) east, north over the lag
        dk = np.stack([r["fx"], r["fy"]], 1)
        e = dict(cand=int(r["cand"].sum()), acc=int(A.sum()),
                 zero_frac=round(float(((r["sy"] == 0) & (r["sx"] == 0))[A].mean()), 3) if A.sum() >= MIN_N else None,
                 abs_lt_0p5_frac=round(float((np.hypot(*dk[A].T) < 0.5).mean()), 3) if A.sum() >= MIN_N else None,
                 median_abs_d=round(float(np.median(np.hypot(*dk[A].T))), 3) if A.sum() >= MIN_N else None,
                 slope_on_cur_e=ols(dr[A, 0], dk[A, 0]), slope_on_cur_n=ols(dr[A, 1], dk[A, 1]))
        if dtrue is not None:
            dt = dtrue[:, :k].sum(1)
            e.update(slope_on_true_e=ols(dt[A, 0], dk[A, 0]), slope_on_true_n=ols(dt[A, 1], dk[A, 1]),
                     median_abs_true=round(float(np.median(np.hypot(*dt[A].T))), 3) if A.sum() >= MIN_N else None)
        s[f"ncc_lead{k}"] = e
    for k in (2, 3):
        both = ncc[1]["acc"] & ncc[k]["acc"]
        d1 = np.stack([ncc[1]["fx"], ncc[1]["fy"]], 1)[both]; dk = np.stack([ncc[k]["fx"], ncc[k]["fy"]], 1)[both]
        e = dict(n=int(both.sum()), slope_e=ols(d1[:, 0], dk[:, 0]), slope_n=ols(d1[:, 1], dk[:, 1]),
                 median_ratio=round(float(np.median(np.hypot(*dk.T)) / np.median(np.hypot(*d1.T))), 3) if both.sum() >= MIN_N else None)
        if dtrue is not None:
            t1 = dtrue[:, :1].sum(1)[both]; tk = dtrue[:, :k].sum(1)[both]
            e.update(true_slope_e=ols(t1[:, 0], tk[:, 0]), true_slope_n=ols(t1[:, 1], tk[:, 1]))
        s[f"ncc_lead{k}_on_lead1"] = e
    if bm is not None:
        for k in LEADS:
            if k not in bm:
                continue
            b = bm[k]; A = b["acc"]
            dr = np.repeat(dref[:, :k].sum(1)[:, None], 4, 1)                 # (N, 4, 2)
            s[f"bm_lead{k}"] = dict(cand=int(b["cand"].sum()), acc=int(A.sum()), ambig=int(b["ambig"].sum()),
                                    zero_frac=round(float(b["zero"][A].mean()), 3) if A.sum() >= MIN_N else None,
                                    slope_on_cur_e=ols(dr[..., 0][A], b["fx"][A]), slope_on_cur_n=ols(dr[..., 1][A], b["fy"][A]))
        if 2 in bm:
            both = bm[1]["acc"] & bm[2]["acc"]
            s["bm_lead2_on_lead1"] = dict(n=int(both.sum()), slope_e=ols(bm[1]["fx"][both], bm[2]["fx"][both]),
                                          slope_n=ols(bm[1]["fy"][both], bm[2]["fy"][both]))
    return s


# ============================== real data on the same sequences ==============================
def real_stats(seqs, vel):
    z = np.load(SM.CACHE)
    m = z["null"] == 0
    key = {(int(i), int(p), int(k)): n for n, (i, p, k) in enumerate(zip(z["i"][m], z["p"][m], z["k"][m]))}
    fy, fx, rpk, edge, okpk = (z[f][m] for f in ("fy", "fx", "rpk", "edge", "okpk"))
    pidx = z["pidx"]
    N = len(seqs)
    ncc = {}
    for k in LEADS:
        d = dict(cand=np.zeros(N, bool), acc=np.zeros(N, bool), fy=np.full(N, np.nan), fx=np.full(N, np.nan), sy=np.zeros(N, int), sx=np.zeros(N, int))
        for n, (i, q) in enumerate(seqs):
            j = key.get((int(i), int(pidx[q]), k))
            if j is None:
                continue
            d["cand"][n] = True
            d["acc"][n] = okpk[j] and not edge[j] and rpk[j] >= SM.R_MIN
            d["fy"][n], d["fx"][n] = fy[j], fx[j]
            d["sy"][n], d["sx"][n] = int(np.rint(fy[j])), int(np.rint(fx[j]))
        ncc[k] = d
    # block matching cache: tiles (a, b), (a, b+1), (a+1, b), (a+1, b+1) of patch at (2 + 16 a, 1 + 16 b)
    zb = np.load(CM.CACHE)
    bm = {}
    pos = z["pos"]
    for k, S in ((1, 6), (2, 12)):
        d = CM.build_table(dict(zb), k, 16, S)
        kk = {(int(dd), int(y), int(x)): n for n, (dd, y, x) in enumerate(zip(d["day"], d["ty"], d["tx"]))}
        out = {nm: np.zeros((N, 4), bool) for nm in ("cand", "acc", "zero", "ambig")}
        out.update(fy=np.full((N, 4), np.nan), fx=np.full((N, 4), np.nan))
        for n, (i, q) in enumerate(seqs):
            a, b = (pos[q][0] - 2) // 16, (pos[q][1] - 1) // 16
            for u, (ty, tx) in enumerate([(a, b), (a, b + 1), (a + 1, b), (a + 1, b + 1)]):
                j = kk.get((int(i), ty, tx))
                if j is None:
                    continue
                out["cand"][n, u] = True; out["acc"][n, u] = d["accept"][j]; out["zero"][n, u] = d["zero"][j]
                out["ambig"][n, u] = d["ambig"][j]; out["fy"][n, u] = d["fy"][j]; out["fx"][n, u] = d["fx"][j]
        bm[k] = out
    dref = 0.5 * (vel[:, :-1] + vel[:, 1:])
    return stats(ncc, bm, dref)


# ============================== driver ==============================
_S = {}


def _work(cfg):
    f, gamma, frozen = cfg
    F, dtrue = make_fields(f, gamma, _S["vel"], frozen=frozen)
    ncc = run_ncc(F, _S["masks"]); bm = run_bm(F, _S["masks"])
    dref = 0.5 * (_S["vel"][:, :-1] + _S["vel"][:, 1:])
    return cfg, stats(ncc, bm, dref, dtrue)


def _init(vel, masks):
    _S.update(vel=vel, masks=masks)


def main():
    t0 = time.time()
    seqs, masks, vel, sinfo = sample_sequences()
    print("sequences", sinfo, f"{time.time() - t0:.0f}s", flush=True)
    res = dict(assumptions=__doc__.split("Method")[1].split("Run from")[0].strip(), sample=sinfo,
               mask_obs_frac_day0_patch=round(float(masks[:, 0, 16:48, 16:48].mean()), 3),
               current_abs_px_day_median=round(float(np.median(np.hypot(*vel.reshape(-1, 2).T))), 3))
    res["real_same_sequences"] = real_stats(seqs, vel)
    print("real", json.dumps(res["real_same_sequences"])[:1500], flush=True)
    cfgs = [(0.0, 1.0, True)] + [(f, g, False) for g in G_GRID for f in F_GRID]
    res["synthetic"] = {}
    with Pool(6, initializer=_init, initargs=(vel, masks)) as pool:
        for cfg, st in pool.imap_unordered(_work, cfgs):
            nm = "control_frozen_f0_g1" if cfg[2] else f"f{cfg[0]}_g{cfg[1]}"
            res["synthetic"][nm] = st
            print(nm, f"{time.time() - t0:.0f}s", json.dumps({k: st[k] for k in ['ncc_lead1', 'ncc_lead2_on_lead1', 'bm_lead1']})[:600], flush=True)
    res["wall_s"] = round(time.time() - t0)
    OUT.write_text(json.dumps(res, indent=1))
    print("done", res["wall_s"], flush=True)


if __name__ == "__main__" and "--report" not in sys.argv:
    main()


# ============================== report ==============================
STATS = [("NCC zero share, lead 1", lambda s: s["ncc_lead1"]["zero_frac"]),
         ("NCC zero share, lead 2", lambda s: s["ncc_lead2"]["zero_frac"]),
         ("NCC zero share, lead 3", lambda s: s["ncc_lead3"]["zero_frac"]),
         ("Block-matching zero share, lead 1", lambda s: s["bm_lead1"]["zero_frac"]),
         ("NCC slope on current, lead 1 (mean e/n)", lambda s: np.mean([s["ncc_lead1"]["slope_on_cur_e"]["slope"], s["ncc_lead1"]["slope_on_cur_n"]["slope"]])),
         ("NCC slope on current, lead 3 (mean e/n)", lambda s: np.mean([s["ncc_lead3"]["slope_on_cur_e"]["slope"], s["ncc_lead3"]["slope_on_cur_n"]["slope"]])),
         ("NCC median abs-shift ratio lead 2 / lead 1", lambda s: s["ncc_lead2_on_lead1"]["median_ratio"]),
         ("NCC median abs-shift ratio lead 3 / lead 1", lambda s: s["ncc_lead3_on_lead1"]["median_ratio"]),
         ("NCC OLS slope lead-2 on lead-1 shift (mean e/n)", lambda s: np.mean([s["ncc_lead2_on_lead1"]["slope_e"]["slope"], s["ncc_lead2_on_lead1"]["slope_n"]["slope"]])),
         ("NCC OLS slope lead-3 on lead-1 shift (mean e/n)", lambda s: np.mean([s["ncc_lead3_on_lead1"]["slope_e"]["slope"], s["ncc_lead3_on_lead1"]["slope_n"]["slope"]]))]


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from chl_cnn.plotting import OKABE_ITO as C
    r = json.loads(OUT.read_text())
    real = r["real_same_sequences"]; syn = r["synthetic"]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8)); axes = axes.ravel()
    rows = []
    for q, (nm, fn) in enumerate(STATS):
        ax = axes[q]
        for u, gm in enumerate(G_GRID):
            ax.plot(F_GRID, [fn(syn[f"f{f}_g{gm}"]) for f in F_GRID], marker="o", color=C[u], label=f"synthetic, moving part at {gm} x current")
        ax.axhline(fn(real), color="k", lw=1.5, label="real ln CHL, same sequences")
        ax.axhline(fn(syn["control_frozen_f0_g1"]), color=C[3], ls=":", lw=1, label="control: frozen field, 1.0 x current")
        ax.set(title=nm, xlabel="static variance fraction f", ylabel=nm.split(",")[0].split(" (")[0])
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Synthetic static + moving fields through real cloud masks vs real ln CHL ({r['sample']['n']} patch sequences)")
    fig.tight_layout(); fig.savefig(FIG / "static_synth_stats.png", dpi=110); plt.close(fig)
    L = ["# Synthetic static + moving fields (Test 1)", "", f"{r['sample']['n']} sequences (start day, 32 px patch), real cloud masks; median |current| {r['current_abs_px_day_median']} px/day.", "",
         "| config | " + " | ".join(nm for nm, _ in STATS) + " | NCC accepted l1 / candidates | BM accepted l1 / candidates |", "|---" * (len(STATS) + 3) + "|"]
    def line(lab, s):
        v = [fn(s) for _, fn in STATS]
        return f"| {lab} | " + " | ".join(f"{x:.3f}" if x is not None else "-" for x in v) + f" | {s['ncc_lead1']['acc']} / {s['ncc_lead1']['cand']} | {s['bm_lead1']['acc']} / {s['bm_lead1']['cand']} |"
    L.append(line("real ln CHL", real))
    L.append(line("control frozen f0, 1.0x", syn["control_frozen_f0_g1"]))
    for gm in G_GRID:
        for f in F_GRID:
            L.append(line(f"f {f}, {gm}x", syn[f"f{f}_g{gm}"]))
    L += ["", "True lead-2 on lead-1 displacement slope (synthetic, same sequences): "
          + ", ".join(f"{k}: {np.mean([v['ncc_lead2_on_lead1']['true_slope_e']['slope'], v['ncc_lead2_on_lead1']['true_slope_n']['slope']]):.2f}" for k, v in list(syn.items())[:3])]
    (RES / "static_synth.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__" and "--report" in sys.argv:
    report()
