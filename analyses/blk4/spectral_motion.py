"""Spectral (Fourier-domain) estimates of CHL feature motion and of scale-dependent decorrelation, Black Sea 4 km.

Method (also written to the JSON "assumptions")
- Store ~/mind_the_chl_gap/data/blk4_store_2014_2024 (197 x 275, lat ascending -> +row = north, lon ascending -> +col =
  east). ln CHL; CHL > 100 mg/m3 or <= 0 -> missing. Pixel 4.50 km north-south, 4.19-4.80 km east-west (grid_metrics);
  wavelengths are quoted with a nominal 4.5 km pixel.
- Patches 32 x 32 px (~145 km), stride 16, origin (row 2, col 1) = the 16 px tile grid of chl_motion.py, so each patch
  is exactly 2 x 2 block-matching tiles; 11 x 16 = 176 patch positions, kept if >= 50 % ocean.
- Day pairs (t, t+k), k = 1, 2, 3 calendar days, all 2014-2024 store days; pairs are used only when t..t+k are
  consecutive calendar days in the store (the store skips 18 calendar days).
- Window: 2-D Hann, np.hanning(34)[1:-1] outer product (no zero edge weights).
- Masked NCC (Padfield 2012, weighted form): weights w = Hann x observed mask on each day; for every integer shift s the
  weighted Pearson r of day-t values at x and day-t+k values at x + s with pair weight w_t(x) w_t+k(x + s); all sums by
  zero-padded (64 x 64) FFT cross-correlations. Values mean-removed (weighted) per patch first. r = NaN where the
  pixel overlap < 256 (25 % of the patch) or the weighted overlap < 25 % of its full-patch zero-shift value.
  Candidate: >= 50 % of the 1024 px observed on each day. Search window +-S with S = 6 / 9 / 12 px for k = 1 / 2 / 3.
  Peak = argmax; sub-pixel 1-D parabolic fit per axis (offset clipped to +-0.5). Accepted: peak r >= 0.6 and peak not
  on the search-window edge. Shift s means a feature at x on day t is at x + s on day t+k.
- Phase correlation cross-check: patches >= 90 % observed on both days; gaps filled with the weighted patch mean,
  mean removed, Hann; normalised cross-power spectrum, inverse FFT (32 x 32, circular), peak in +-S, parabolic.
- Velocity references per patch and day (px/day): reanalysis current (uo, vo), geostrophic current from zos
  (chl_cnn.tracer.geostrophic, margin 2 px), and the learned tracer velocity u_eff of models/tracer/tr_rea_geo_wind
  (chl_cnn.tracer.VelocityModel imported read-only, weights from model.pt, inputs rebuilt as in train_tracer.load:
  a u_rea + b u_geo + wind term, all terms included). Hann-weighted patch mean over pixels with a finite field;
  valid if >= 50 % of the patch's 1024 px are finite. Displacement over the lag = trapezoid over days t..t+k
  (daily fields linear in time, as the tracer). Shuffled control: reanalysis current of the same patch on a random
  day with |t' - t| > 1 (seed 0).
- Comparison statistics as chl_motion.py: OLS with intercept per component (d_obs on d_ref), 95 % CI from 300
  bootstrap resamples of days; angles only where |d_obs| / k >= 0.5 and |d_ref| / k >= 0.5 px/day; mean resultant
  length R. Overlapping patches are not independent (day-clustered bootstrap only).
- Coherence: patches >= 90 % observed on both days (gaps filled, mean removed, Hann, 32 x 32 FFT). Ring-averaged
  magnitude-squared coherence MSC(m) = |sum_ring X Y*|^2 / (sum_ring |X|^2 sum_ring |Y|^2), rings m = round(|wavenumber
  index|) = 1..16, wavelength 32 / m px (145 km .. 9 km). Unaligned: day t+k patch at the same position. Aligned:
  day t+k patch extracted at the integer part of the masked-NCC peak (any non-edge peak, no r threshold; >= 90 %
  observed there) and the sub-pixel remainder removed by a Fourier phase ramp. Null: same patch position, random
  other day >= 30 days away, same procedure (incl. alignment by its own NCC peak), one null pair per data pair on a
  random 1/3 of the day pairs.
- Decay fits on median MSC per ring and lead: rho = sqrt(max((MSC - null) / (1 - null), 1e-4)). Per-ring:
  ln rho = c_m - r_m k (OLS over k = 1..3) -> e-folding lead 1 / r_m. Single diffusivity: ln rho = c_m - K q_m^2 k
  with q = 2 pi / wavelength (joint fit, free c_m); and the strict form ln rho = -K q^2 k (no intercept).
- Region of a patch = majority 4 km region label over its ocean px (error_breakdown.regions_4km labels, full grid).
  Season by month of day t.

Run from the repo root: python -m analyses.blk4.spectral_motion [--recompute] [--every N]   (CPU only)
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
from scipy.ndimage import gaussian_filter, shift as nd_shift
from chl_cnn.plotting import OKABE_ITO as C
from chl_cnn.physics import grid_metrics
from chl_cnn import store_blk4 as S4

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
CACHE = RES / "spectral_motion_patches.npz"
STORE = Path.home() / "mind_the_chl_gap/data/blk4_store_2014_2024"
TRACER = ROOT / "models/tracer/tr_rea_geo_wind"
NAMES = {1: "NW shelf", 2: "W open", 3: "E open", 4: "Anatolian", 5: "Crimea/Caucasus", 6: "Azov", 7: "Marmara"}
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
           9: "SON", 10: "SON", 11: "SON"}
P, STRIDE, R0, C0 = 32, 16, 2, 1
LEADS, SW = (1, 2, 3), {1: 6, 2: 9, 3: 12}
MIN_VALID, CLEAR, MIN_OVL_PX, MIN_OVL_W, R_MIN = 0.5, 0.9, 256, 0.25, 0.6
NR = 16
PX_KM = 4.5
MIN_N = 5
WIN = np.outer(np.hanning(P + 2)[1:-1], np.hanning(P + 2)[1:-1])
FY, FX = np.meshgrid(np.fft.fftfreq(P), np.fft.fftfreq(P), indexing="ij")          # cycles / px
RING = np.rint(np.hypot(FY, FX) * P).astype(int)
WL_KM = P / np.arange(1, NR + 1) * PX_KM


# ============================== data ==============================
def load():
    g = {k: np.load(STORE / f"{k}.npy", mmap_mode="r") for k in ["chl_l3", "u_curr", "v_curr", "sla", "u_wind", "v_wind"]}
    g["lat"], g["lon"] = np.load(STORE / "lat.npy").astype("float64"), np.load(STORE / "lon.npy").astype("float64")
    g["ocean"] = np.load(STORE / "ocean.npy").astype(bool)
    g["time"] = np.load(STORE / "time.npy").astype("datetime64[D]")
    return g


def ln_chl(c):
    c = np.asarray(c, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(c) & (c > 0) & (c <= 100), np.log(c), np.nan)


def patch_grid(ocean):
    H, W = ocean.shape
    pos = [(R0 + STRIDE * a, C0 + STRIDE * b) for a in range((H - R0 - P) // STRIDE + 1)
           for b in range((W - C0 - P) // STRIDE + 1)]
    pos = np.array(pos)
    ofrac = np.array([ocean[r:r + P, c:c + P].mean() for r, c in pos])
    return pos, ofrac


def cut(F, pos, dr=None, dc=None):
    """F (..., H, W) -> (..., n, P, P) patches at pos (+ integer offsets dr, dc per patch); NaN outside the grid."""
    H, W = F.shape[-2:]
    r = pos[:, 0] + (0 if dr is None else dr); c = pos[:, 1] + (0 if dc is None else dc)
    rr = r[:, None] + np.arange(P)[None]; cc = c[:, None] + np.arange(P)[None]
    inside = (rr >= 0) & (rr < H); insc = (cc >= 0) & (cc < W)
    out = F[..., np.clip(rr, 0, H - 1)[:, :, None], np.clip(cc, 0, W - 1)[:, None, :]]
    return np.where(inside[:, :, None] & insc[:, None, :], out, np.nan)


def regions_full(ocean):
    reg2 = np.load(ROOT / "analyses/blk/regions.npy")
    H2, W2 = reg2.shape[0] // 2, reg2.shape[1] // 2
    b = S4._blocks(reg2, H2, W2)
    cnt = np.stack([(b == k).sum(-1) for k in range(1, 8)], -1)
    reg = np.where(cnt.max(-1) > 0, cnt.argmax(-1) + 1, 0).astype(np.int8)
    assert reg.shape == ocean.shape
    return reg


# ============================== core estimators ==============================
def ncc_maps(A, B, S=12):
    """A, B (n, P, P) with NaN = missing. Weighted masked NCC (Padfield 2012 with Hann weights) for shifts in +-S.
    Returns r (n, 2S+1, 2S+1), pixel overlap N, weighted-overlap fraction Wf (index [S+sy, S+sx])."""
    mA, mB = np.isfinite(A), np.isfinite(B)
    w1, w2 = WIN * mA, WIN * mB
    a = np.where(mA, A, 0.0); b = np.where(mB, B, 0.0)
    a = np.where(mA, a - (w1 * a).sum((1, 2), keepdims=True) / w1.sum((1, 2), keepdims=True), 0.0)
    b = np.where(mB, b - (w2 * b).sum((1, 2), keepdims=True) / w2.sum((1, 2), keepdims=True), 0.0)
    Q = 2 * P
    F = lambda x: np.fft.rfft2(x, s=(Q, Q))
    xc = lambda X, Y: np.fft.irfft2(np.conj(X) * Y, s=(Q, Q))
    Fw1, Fw2, Fa, Fb = F(w1), F(w2), F(w1 * a), F(w2 * b)
    Wm = xc(Fw1, Fw2); S1 = xc(Fa, Fw2); S2 = xc(Fw1, Fb)
    S11 = xc(F(w1 * a * a), Fw2); S22 = xc(Fw1, F(w2 * b * b)); S12 = xc(Fa, Fb)
    N = xc(F(mA.astype(float)), F(mB.astype(float)))
    idx = np.arange(-S, S + 1) % Q
    sel = lambda X: X[:, idx][:, :, idx]
    Wm, S1, S2, S11, S22, S12, N = map(sel, (Wm, S1, S2, S11, S22, S12, N))
    with np.errstate(invalid="ignore", divide="ignore"):
        v1 = S11 - S1 ** 2 / Wm; v2 = S22 - S2 ** 2 / Wm
        r = (S12 - S1 * S2 / Wm) / np.sqrt(v1 * v2)
    Wf = Wm / (WIN ** 2).sum()
    bad = (N < MIN_OVL_PX - 0.5) | (Wf < MIN_OVL_W) | ~(v1 > 1e-9 * Wm) | ~(v2 > 1e-9 * Wm)
    r[bad] = np.nan
    return r, np.rint(N), Wf


def peaks(R, Suse, S=12):
    """R (n, 2S+1, 2S+1) -> fy, fx (sub-pixel), sy, sx (int), rpk, edge, ok(peak exists)."""
    n = R.shape[0]
    sub = R[:, S - Suse:S + Suse + 1, S - Suse:S + Suse + 1]
    flat = sub.reshape(n, -1)
    ok = np.isfinite(flat).any(1)
    am = np.nanargmax(np.where(np.isfinite(flat), flat, -np.inf), 1)
    iy, ix = np.unravel_index(am, sub.shape[1:])
    sy, sx = iy - Suse, ix - Suse
    edge = (np.abs(sy) == Suse) | (np.abs(sx) == Suse)
    rpk = flat[np.arange(n), am]
    off = []
    for axis in (0, 1):
        lo_i = (np.clip(iy - (axis == 0), 0, 2 * Suse), np.clip(ix - (axis == 1), 0, 2 * Suse))
        hi_i = (np.clip(iy + (axis == 0), 0, 2 * Suse), np.clip(ix + (axis == 1), 0, 2 * Suse))
        lo, hi = sub[np.arange(n), lo_i[0], lo_i[1]], sub[np.arange(n), hi_i[0], hi_i[1]]
        den = lo - 2 * rpk + hi
        with np.errstate(invalid="ignore", divide="ignore"):
            o = np.clip(0.5 * (lo - hi) / den, -0.5, 0.5)
        off.append(np.where(np.isfinite(lo) & np.isfinite(hi) & (den < 0) & ~edge, o, 0.0))
    return dict(fy=sy + off[0], fx=sx + off[1], sy=sy, sx=sx, rpk=np.where(ok, rpk, np.nan), edge=edge, ok=ok)


def prep_clear(X):
    """Fill gaps with the weighted mean, remove it, apply Hann."""
    m = np.isfinite(X)
    w = WIN * m
    mu = (w * np.where(m, X, 0)).sum((1, 2), keepdims=True) / w.sum((1, 2), keepdims=True)
    return (np.where(m, X, mu) - mu) * WIN


def phase_corr(A, B, Suse):
    X, Y = np.fft.fft2(prep_clear(A)), np.fft.fft2(prep_clear(B))
    Rx = np.conj(X) * Y
    c = np.real(np.fft.ifft2(Rx / (np.abs(Rx) + 1e-12)))
    idx = np.arange(-12, 13) % P
    sub = c[:, idx][:, :, idx]
    p = peaks(sub, Suse)
    return p


def ring_sums(X, Y):
    """X, Y (n, P, P) complex spectra -> Sxy (complex), Sxx, Syy per ring 1..NR."""
    out = []
    for Z in (X * np.conj(Y), np.abs(X) ** 2, np.abs(Y) ** 2):
        out.append(np.stack([Z[:, RING == m].sum(1) for m in range(1, NR + 1)], 1))
    return out


def coherence_pair(A, Bfull, pos, fy, fx, okshift):
    """A (n, P, P) day-t clear patches; Bfull (H, W) day t+k. Returns unaligned and aligned ring sums
    (aligned NaN where no shift or the displaced patch is < 90 % observed / outside)."""
    Bu = cut(Bfull, pos)
    X = np.fft.fft2(prep_clear(A))
    un = ring_sums(X, np.fft.fft2(prep_clear(Bu)))
    iy, ix = np.rint(np.nan_to_num(fy)).astype(int), np.rint(np.nan_to_num(fx)).astype(int)
    Ba = cut(Bfull, pos, iy, ix)
    good = okshift & (np.isfinite(Ba).mean((1, 2)) >= CLEAR)
    ry, rx = np.nan_to_num(fy) - iy, np.nan_to_num(fx) - ix
    Y = np.fft.fft2(prep_clear(np.where(np.isfinite(Ba).any((1, 2), keepdims=True), Ba, 0.0)))
    Y = Y * np.exp(2j * np.pi * (FY[None] * ry[:, None, None] + FX[None] * rx[:, None, None]))
    al = ring_sums(X, Y)
    al = [np.where(good[:, None], z, np.nan) for z in al]
    return un, al


def msc(sxy, sxx, syy):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(sxy) ** 2 / (sxx * syy)


# ============================== synthetic validation ==============================
def grf(n, beta, rng):
    ky, kx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
    k = np.hypot(ky, kx); k[0, 0] = 1
    amp = k ** (-beta / 2); amp[0, 0] = 0
    f = np.real(np.fft.ifft2(amp * np.fft.fft2(rng.standard_normal((n, n)))))
    return f / f.std()


def fshift(f, dy, dx):
    ky, kx = np.meshgrid(np.fft.fftfreq(f.shape[0]), np.fft.fftfreq(f.shape[1]), indexing="ij")
    return np.real(np.fft.ifft2(np.fft.fft2(f) * np.exp(-2j * np.pi * (ky * dy + kx * dx))))


def cloud_mask(frac, rng, smooth=3.0):
    if frac <= 0:
        return np.ones((P, P), bool)
    z = gaussian_filter(rng.standard_normal((P + 16, P + 16)), smooth)[8:-8, 8:-8]
    return z > np.quantile(z, frac)


def synthetic(g, n_trial=300, seed=0):
    """Known sub-pixel shifts, U(-5, 5) px per axis, recovered with random blob cloud masks (independent per day).
    Fields: (a) Gaussian random field, spectrum k^-3, noise 0.05 sd per day; (b) real clear ln CHL (64 x 64 region of
    the clearest summer day), shifted by cubic-spline interpolation, same noise."""
    rng = np.random.default_rng(seed)
    # clearest 64 x 64 window on the clearest day
    cov = [np.isfinite(ln_chl(g["chl_l3"][j])).mean() for j in range(0, len(g["time"]), 7)]
    best = None
    for j in np.argsort(cov)[::-1][:10] * 7:
        L = ln_chl(g["chl_l3"][j]); m = np.isfinite(L)
        for r in range(0, L.shape[0] - 64, 8):
            for c in range(0, L.shape[1] - 64, 8):
                f = m[r:r + 64, c:c + 64].mean()
                if f >= 0.995 and (best is None or np.nanstd(L[r:r + 64, c:c + 64]) > best[0]):
                    best = (np.nanstd(L[r:r + 64, c:c + 64]), j, r, c)
    real = None
    if best is not None:
        _, j, r, c = best
        L = ln_chl(g["chl_l3"][j])[r:r + 64, c:c + 64]
        real = np.where(np.isfinite(L), L, np.nanmean(L))
        real_info = dict(day=str(g["time"][j]), row=int(r), col=int(c), sd=round(float(np.std(real)), 3))
    out = {}
    fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for kind in ["grf", "real"]:
        if kind == "real" and real is None:
            continue
        rows = {}
        for fr in fracs:
            A, B, tru = [], [], []
            for _ in range(n_trial):
                dy, dx = rng.uniform(-5, 5, 2)
                if kind == "grf":
                    f0 = grf(128, 3.0, rng); f1 = fshift(f0, dy, dx)
                    a, b = f0[48:80, 48:80], f1[48:80, 48:80]
                else:
                    f1 = nd_shift(real, (dy, dx), order=3, mode="nearest")
                    a, b = real[16:48, 16:48], f1[16:48, 16:48]
                sd = a.std()
                a = a + 0.05 * sd * rng.standard_normal(a.shape); b = b + 0.05 * sd * rng.standard_normal(b.shape)
                a = np.where(cloud_mask(fr, rng), a, np.nan); b = np.where(cloud_mask(fr, rng), b, np.nan)
                A.append(a); B.append(b); tru.append((dy, dx))
            A, B, tru = np.array(A), np.array(B), np.array(tru)
            Rm, _, _ = ncc_maps(A, B)
            p = peaks(Rm, 6)
            acc = p["ok"] & ~p["edge"] & (p["rpk"] >= R_MIN)
            err = np.hypot(p["fy"] - tru[:, 0], p["fx"] - tru[:, 1])
            row = dict(n=n_trial, ncc_accepted=int(acc.sum()), ncc_med_abs_err_px=round(float(np.median(err[acc])), 3) if acc.any() else None,
                       ncc_p90_abs_err_px=round(float(np.percentile(err[acc], 90)), 3) if acc.any() else None,
                       ncc_frac_err_gt1px=round(float((err[acc] > 1).mean()), 3) if acc.any() else None,
                       ncc_bias_y=round(float(np.mean((p["fy"] - tru[:, 0])[acc])), 3) if acc.any() else None,
                       ncc_bias_x=round(float(np.mean((p["fx"] - tru[:, 1])[acc])), 3) if acc.any() else None,
                       ncc_med_rpk=round(float(np.nanmedian(p["rpk"])), 3))
            if fr <= 0.1:
                pc = phase_corr(A, B, 6)
                e2 = np.hypot(pc["fy"] - tru[:, 0], pc["fx"] - tru[:, 1])
                okp = pc["ok"] & ~pc["edge"]
                row.update(pc_n=int(okp.sum()), pc_med_abs_err_px=round(float(np.median(e2[okp])), 3),
                           pc_p90_abs_err_px=round(float(np.percentile(e2[okp], 90)), 3),
                           pc_frac_err_gt1px=round(float((e2[okp] > 1).mean()), 3))
            if fr == 0.0 and kind == "grf":
                # coherence sanity: aligned (true-shift) vs unaligned on clear GRF patches, via a full shifted field
                A2, B2, tr2 = [], [], []
                for _ in range(100):
                    dy, dx = rng.uniform(-5, 5, 2)
                    f0 = grf(128, 3.0, rng); f1 = fshift(f0, dy, dx)
                    A2.append(f0[48:80, 48:80]); B2.append(f1); tr2.append((dy, dx))
                A2, tr2 = np.array(A2), np.array(tr2)
                un, al = [], []
                for q in range(100):
                    u_, a_ = coherence_pair(A2[q:q + 1], B2[q], np.array([[48, 48]]), tr2[q:q + 1, 0], tr2[q:q + 1, 1], np.array([True]))
                    un.append(msc(*u_)[0]); al.append(msc(*a_)[0])
                out["coherence_check_grf_clear_no_noise"] = dict(
                    wavelength_km=[round(float(w), 1) for w in WL_KM],
                    median_msc_unaligned=[round(float(v), 3) for v in np.median(un, 0)],
                    median_msc_aligned_true_shift=[round(float(v), 3) for v in np.median(al, 0)])
            rows[f"cloud_{fr:.1f}"] = row
        out[kind] = rows
    if real is not None:
        out["real_field"] = real_info
    out["setup"] = "shift U(-5,5) px per axis, search +-6 px, n trials per cloud fraction; cloud fraction applied independently to each day (blob masks: Gaussian-smoothed noise, sigma 3 px, thresholded)"
    return out


# ============================== per-pair worker ==============================
_G = {}


def _init():
    _G.update(load())
    pos, of = patch_grid(_G["ocean"])
    _G["pos"], _G["pidx"] = pos[of >= MIN_VALID], np.flatnonzero(of >= MIN_VALID)


def work(job):
    i, j, k, null = job
    pos, pidx = _G["pos"], _G["pidx"]
    A, B = ln_chl(_G["chl_l3"][i]), ln_chl(_G["chl_l3"][j])
    PA, PB = cut(A, pos), cut(B, pos)
    fa, fb = np.isfinite(PA).mean((1, 2)), np.isfinite(PB).mean((1, 2))
    cand = (fa >= MIN_VALID) & (fb >= MIN_VALID)
    if not cand.any():
        return None
    c = np.flatnonzero(cand)
    Rm, N, Wf = ncc_maps(PA[c], PB[c])
    S = SW[k]
    p = peaks(Rm, S)
    n = len(c); ar = np.arange(n)
    iy, ix = np.clip(p["sy"] + 12, 0, 24), np.clip(p["sx"] + 12, 0, 24)
    rec = dict(i=np.full(n, i), j=np.full(n, j), k=np.full(n, k), null=np.full(n, null), p=pidx[c],
               fa=fa[c], fb=fb[c], fy=p["fy"], fx=p["fx"], rpk=p["rpk"], edge=p["edge"], okpk=p["ok"],
               r0=Rm[:, 12, 12], npk=N[ar, iy, ix], wpk=Wf[ar, iy, ix])
    clear = (fa[c] >= CLEAR) & (fb[c] >= CLEAR)
    nanf = np.full(n, np.nan)
    rec.update(pc_fy=nanf.copy(), pc_fx=nanf.copy(), pc_pk=nanf.copy(), pc_edge=np.zeros(n, bool), clear=clear)
    z = np.full((n, NR), np.nan, "float32"); zc = np.full((n, NR), np.nan, "complex64")
    rec.update(u_xy=zc.copy(), u_xx=z.copy(), u_yy=z.copy(), a_xy=zc.copy(), a_xx=z.copy(), a_yy=z.copy())
    if clear.any():
        q = np.flatnonzero(clear)
        if not null:
            pc = phase_corr(PA[c][q], PB[c][q], S)
            rec["pc_fy"][q], rec["pc_fx"][q], rec["pc_pk"][q], rec["pc_edge"][q] = pc["fy"], pc["fx"], pc["rpk"], pc["edge"]
        okshift = p["ok"][q] & ~p["edge"][q]
        un, al = coherence_pair(PA[c][q], B, pos[c][q], p["fy"][q], p["fx"][q], okshift)
        for nm, (sxy, sxx, syy) in (("u", un), ("a", al)):
            rec[f"{nm}_xy"][q], rec[f"{nm}_xx"][q], rec[f"{nm}_yy"][q] = sxy, sxx, syy
    return rec


def jobs_list(time_, every=1, seed=0):
    t = time_.astype("datetime64[D]").astype(int)
    pos_of = {v: q for q, v in enumerate(t)}
    jobs = []
    for i in range(0, len(t), every):
        for k in LEADS:
            if all((t[i] + d) in pos_of for d in range(1, k + 1)):
                jobs.append((i, pos_of[t[i] + k], k, 0))
    rng = np.random.default_rng(seed)
    nulls = []
    for (i, j, k, _) in jobs:
        if rng.random() < 1 / 3:
            while True:
                jj = int(rng.integers(0, len(t)))
                if abs(t[jj] - t[i]) >= 30:
                    break
            nulls.append((i, jj, k, 1))
    return jobs, nulls


# ============================== velocities ==============================
def ueff_and_refs(g, pos):
    """Per day and patch: Hann-weighted patch means (px/day, east / north) of reanalysis current, geostrophic current
    and tracer u_eff; plus a check of the rebuilt a factor against factor_maps.npz."""
    import torch
    torch.set_num_threads(4)
    from chl_cnn import tracer as TR
    cfg = json.loads((TRACER / "config.json").read_text())
    ocean_full = g["ocean"]; H, W = ocean_full.shape
    rows, cols = np.flatnonzero(ocean_full.any(1)), np.flatnonzero(ocean_full.any(0))
    ys = slice(max(rows[0] - 1, 0), min(rows[-1] + 2, H)); xs = slice(max(cols[0] - 1, 0), min(cols[-1] + 2, W))
    assert [ys.start, ys.stop, xs.start, xs.stop] == cfg["data"]["crop"], (ys, xs, cfg["data"]["crop"])
    ocean = ocean_full[ys, xs]; lat, lon = g["lat"][ys], g["lon"][xs]
    dx, dy, f, _ = grid_metrics(lat, lon)
    dxF, dyF, fF, _ = grid_metrics(g["lat"], g["lon"])
    dep = np.load(STORE / "depth.npy")[ys, xs].astype("float32")
    ldep = np.log10(np.maximum(np.nan_to_num(np.abs(dep), nan=5.0), 5.0))
    dist = -np.load(STORE / "topo.npy")[ys, xs].astype("float32")
    ldist = np.log10(1.0 + np.maximum(np.nan_to_num(dist, nan=0.0), 0.0))
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    st = [np.where(ocean, (x.astype("float32") - x.astype("float32")[ocean].mean()) / x.astype("float32")[ocean].std(), 0.0)
          for x in (ldep, ldist, la, lo)]
    STATIC = torch.as_tensor(np.stack(st).astype("float32"))
    model = TR.VelocityModel(cfg["terms"], n_static=4, hidden=cfg["hidden"], a0=cfg["a_init"], b0=cfg["b_init"],
                             b_range=(cfg["b_min"], 1.5), cell=cfg["corr_cell"], corr_hidden=cfg["corr_hidden"])
    model.load_state_dict(torch.load(TRACER / "model.pt", map_location="cpu")["model"]); model.eval()
    # check: a at the factor-map doys
    fm = np.load(TRACER / "factor_maps.npz")
    with torch.no_grad():
        ang = torch.as_tensor(2 * np.pi * fm["doy"] / 365.25, dtype=torch.float32)
        x = torch.cat([STATIC[None].expand(12, -1, -1, -1), torch.sin(ang).view(12, 1, 1, 1).expand(12, 1, *ocean.shape),
                       torch.cos(ang).view(12, 1, 1, 1).expand(12, 1, *ocean.shape)], 1)
        a_chk = model.a(x).numpy()
    chk = dict(max_abs_diff_a_vs_factor_maps=float(np.abs(a_chk - fm["a"])[:, ocean].max()))
    doy = (g["time"] - g["time"].astype("datetime64[Y]")).astype(int)
    nd = len(g["time"]); npch = len(pos)
    out = {nm: np.full((nd, npch), np.nan, "float32") for nm in ["cu", "cv", "gu", "gv", "eu", "ev"]}
    nfin = lambda X: np.isfinite(X)
    def pmean(F):                                         # (T, H, W) -> (T, npch)
        Pt = cut(F, pos)                                  # (T, npch, P, P)
        m = np.isfinite(Pt)
        s = (np.where(m, Pt, 0) * WIN).sum((-1, -2)); w = (m * WIN).sum((-1, -2))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(m.mean((-1, -2)) >= 0.5, s / w, np.nan)
    for b0 in range(0, nd, 50):
        s = slice(b0, min(b0 + 50, nd)); T_ = s.stop - s.start
        u = np.asarray(g["u_curr"][s], "float32"); v = np.asarray(g["v_curr"][s], "float32")
        z = np.asarray(g["sla"][s], "float32")
        okc = nfin(u) & nfin(v) & ocean_full[None]
        ugF, vgF, gok = TR.geostrophic(z, dxF, dyF, fF, ocean_full, 2)
        cu = np.where(okc, u * 86400 / dxF, np.nan); cv = np.where(okc, v * 86400 / dyF, np.nan)
        gu = np.where(gok, ugF * 86400 / dxF, np.nan); gv = np.where(gok, vgF * 86400 / dyF, np.nan)
        # u_eff on the crop (inputs as train_tracer.load: NaN -> 0, geostrophic 0 where invalid)
        uc, vc = np.where(okc, u, 0)[:, ys, xs], np.where(okc, v, 0)[:, ys, xs]
        tx = np.asarray(g["u_wind"][s, ys, xs], "float32"); ty = np.asarray(g["v_wind"][s, ys, xs], "float32")
        okw = nfin(tx) & nfin(ty) & ocean[None]
        tx, ty = np.where(okw, tx, 0), np.where(okw, ty, 0)
        an = 2 * np.pi * doy[s] / 365.25
        tt = lambda X: torch.as_tensor(np.ascontiguousarray(X, dtype="float32"))
        with torch.no_grad():
            ue, ve = model(STATIC, tt(np.sin(an)), tt(np.cos(an)), tt(uc), tt(vc), tt(ugF[:, ys, xs]), tt(vgF[:, ys, xs]),
                           tt(tx), tt(ty), tt(ocean))
        eu = np.full((T_, H, W), np.nan, "float32"); ev = np.full_like(eu, np.nan)
        okcr = okc[:, ys, xs]
        eu[:, ys, xs] = np.where(okcr, ue.numpy() * 86400 / dx, np.nan); ev[:, ys, xs] = np.where(okcr, ve.numpy() * 86400 / dy, np.nan)
        for nm, F in (("cu", cu), ("cv", cv), ("gu", gu), ("gv", gv), ("eu", eu), ("ev", ev)):
            out[nm][s] = pmean(F)
        print(f"  velocities {s.stop}/{nd}", flush=True)
    return out, chk


# ============================== compute ==============================
def compute(every=1, nproc=6):
    g = load()
    t0 = time.time()
    pos_all, of = patch_grid(g["ocean"])
    keep = of >= MIN_VALID
    syn = synthetic(g); print("synthetic done", round(time.time() - t0), "s", flush=True)
    jobs, nulls = jobs_list(g["time"], every)
    print(f"{len(jobs)} pairs + {len(nulls)} null pairs, {keep.sum()} patches", flush=True)
    recs = []
    t1 = time.time()
    with Pool(nproc, initializer=_init) as pool:
        for q, r in enumerate(pool.imap_unordered(work, jobs + nulls, chunksize=16)):
            if r is not None:
                recs.append(r)
            if q % 1000 == 0:
                print(f"{q}/{len(jobs) + len(nulls)} {time.time() - t1:.0f}s", flush=True)
    t_pairs = time.time() - t1
    arr = {kk: np.concatenate([r[kk] for r in recs]) for kk in recs[0]}
    print("velocities", flush=True)
    vel, chk = ueff_and_refs(g, pos_all[keep])
    np.savez_compressed(CACHE, **arr, **{f"vel_{k_}": v for k_, v in vel.items()}, pos=pos_all[keep], pidx=np.flatnonzero(keep),
                        meta=json.dumps(dict(synthetic=syn, ueff_check=chk, n_pairs=len(jobs), n_null_pairs=len(nulls),
                                             every=every, pair_seconds=round(t_pairs), nproc=nproc, total_seconds=round(time.time() - t0))))
    print(f"done {time.time() - t0:.0f}s", flush=True)


# ============================== analysis ==============================
def boot_slope(x, y, days, nb=300, seed=0):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, days = x[ok], y[ok], days[ok]
    if len(x) < MIN_N:
        return dict(n=int(len(x)), flag="n<5")
    b, a = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if x.std() > 0 else np.nan
    ud, inv = np.unique(days, return_inverse=True)
    order = np.argsort(inv, kind="stable"); idx_by = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        pick = np.concatenate([idx_by[d] for d in rng.integers(0, len(ud), len(ud))])
        if x[pick].std() > 0:
            bs.append(np.polyfit(x[pick], y[pick], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(slope=round(float(b), 3), ci95=[round(float(lo), 3), round(float(hi), 3)], intercept=round(float(a), 3),
                r2=round(r2, 3), n=int(len(x)), n_days=int(len(ud)))


def angle_stats(do, dr, k):
    m = (np.hypot(*do) / k >= 0.5) & (np.hypot(*dr) / k >= 0.5) & np.isfinite(dr).all(0) & np.isfinite(do).all(0)
    dth = np.angle(np.exp(1j * (np.arctan2(do[1][m], do[0][m]) - np.arctan2(dr[1][m], dr[0][m]))))
    if m.sum() < MIN_N:
        return dict(n=int(m.sum()), flag="n<5"), dth
    zz = np.exp(1j * dth).mean()
    return dict(R=round(float(abs(zz)), 3), mean_deg=round(float(np.degrees(np.angle(zz))), 1),
                frac_within_45=round(float((np.abs(dth) <= np.pi / 4).mean()), 3), n=int(m.sum())), dth


def compare(sel, do, dr, days, k, nb=300):
    sel = sel & np.isfinite(dr).all(0)
    out = {c: boot_slope(dr[q][sel], do[q][sel], days[sel], nb) for q, c in enumerate(["east", "north"])}
    out["angle"] = angle_stats(do[:, sel], dr[:, sel], k)[0]
    return out


def lag_disp(V, i, p, k):
    """Trapezoid displacement over days i..i+k (store indices, consecutive calendar days)."""
    st = np.stack([V[i + d, p] for d in range(k + 1)])
    w = np.ones(k + 1); w[0] = w[-1] = 0.5
    return (st * w[:, None]).sum(0)


def build(z, k):
    m = (z["k"] == k) & (z["null"] == 0)
    d = {f: z[f][m] for f in ["i", "j", "p", "fa", "fb", "fy", "fx", "rpk", "edge", "okpk", "r0", "npk", "wpk",
                               "pc_fy", "pc_fx", "pc_pk", "pc_edge", "clear"]}
    d["accept"] = d["okpk"] & ~d["edge"] & (d["rpk"] >= R_MIN)
    d["do"] = np.stack([d["fx"], d["fy"]])
    pcol = np.searchsorted(z["pidx"], d["p"])
    d["pc"] = pcol
    for nm, (a, b) in {"dc": ("cu", "cv"), "dg": ("gu", "gv"), "de": ("eu", "ev")}.items():
        d[nm] = np.stack([lag_disp(z[f"vel_{a}"], d["i"], pcol, k), lag_disp(z[f"vel_{b}"], d["i"], pcol, k)])
    rng = np.random.default_rng(0); nd = z["vel_cu"].shape[0]
    sh = np.full_like(d["dc"], np.nan); todo = np.arange(len(d["i"]))
    for _ in range(30):
        if not len(todo): break
        t2 = rng.integers(0, nd - k, len(todo))
        cand = np.stack([lag_disp(z[f"vel_{a}"], t2, pcol[todo], k) for a in ("cu", "cv")])
        ok = (np.abs(t2 - d["i"][todo]) > 1) & np.isfinite(cand).all(0)
        sh[:, todo[ok]] = cand[:, ok]; todo = todo[~ok]
    d["ds"] = sh
    return d


def ring_msc(z, sel, kind):
    return msc(z[f"{kind}_xy"][sel], z[f"{kind}_xx"][sel], z[f"{kind}_yy"][sel])


def band_msc(z, sel, kind, rings):
    r = np.array(rings) - 1
    sxy = z[f"{kind}_xy"][sel][:, r].sum(1); sxx = z[f"{kind}_xx"][sel][:, r].sum(1); syy = z[f"{kind}_yy"][sel][:, r].sum(1)
    return msc(sxy, sxx, syy)


def med_curve(M):
    ok = np.isfinite(M).all(1)
    if ok.sum() < MIN_N:
        return None, int(ok.sum())
    return np.median(M[ok], 0), int(ok.sum())


def decay_fits(med, null):
    """med[k] (NR,), null[k] (NR,) medians -> per-ring rate fits and single-K fits."""
    q = 2 * np.pi / WL_KM                                           # rad / km
    rho = {k: np.sqrt(np.maximum((med[k] - null[k]) / (1 - null[k]), 1e-4)) for k in LEADS}
    L = np.stack([np.log(rho[k]) for k in LEADS])                   # (3, NR)
    ks = np.array(LEADS, float)
    per = []
    for m in range(NR):
        b, c = np.polyfit(ks, L[:, m], 1)
        fit = c + b * ks
        per.append(dict(wavelength_km=round(float(WL_KM[m]), 1), rate_per_day=round(float(-b), 4), intercept=round(float(c), 3),
                        efold_days=round(float(-1 / b), 2) if b < 0 else None,
                        K_m2_s=round(float(-b / q[m] ** 2 * 1e6 / 86400), 1),
                        rho_by_lead=[round(float(rho[k][m]), 3) for k in LEADS],
                        rms_resid=round(float(np.sqrt(np.mean((L[:, m] - fit) ** 2))), 4)))
    # joint single K with free intercepts: L[k, m] = c_m - K q_m^2 k
    X = np.zeros((3 * NR, NR + 1)); yv = L.ravel()
    for a_, k in enumerate(LEADS):
        for m in range(NR):
            X[a_ * NR + m, m] = 1; X[a_ * NR + m, NR] = -q[m] ** 2 * k
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ coef
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    rm = np.sqrt((res.reshape(3, NR) ** 2).mean(0))
    # strict: L = -K q^2 k
    xs = np.concatenate([-q ** 2 * k for k in LEADS]); Ks = float((xs * yv).sum() / (xs * xs).sum())
    rs = yv - Ks * xs
    return dict(per_ring=per,
                single_K_free_intercept=dict(K_m2_s=round(float(coef[NR] * 1e6 / 86400), 1), R2=round(float(1 - (res ** 2).sum() / ss_tot), 3),
                                             rms_resid_by_wavelength=[round(float(v), 4) for v in rm]),
                single_K_strict=dict(K_m2_s=round(Ks * 1e6 / 86400, 1), R2=round(float(1 - (rs ** 2).sum() / ss_tot), 3),
                                     rms_resid_by_wavelength=[round(float(v), 4) for v in np.sqrt((rs.reshape(3, NR) ** 2).mean(0))])), rho


def main():
    every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 1
    if "--recompute" in sys.argv or not CACHE.exists():
        compute(every)
    z = dict(np.load(CACHE))
    meta = json.loads(str(z.pop("meta")))
    g = load(); H, W = g["ocean"].shape
    reg = regions_full(g["ocean"])
    pos = z["pos"]
    preg = np.array([np.bincount(reg[r:r + P, c:c + P][reg[r:r + P, c:c + P] > 0], minlength=8)[1:].argmax() + 1
                     if (reg[r:r + P, c:c + P] > 0).any() else 0 for r, c in pos])
    months = g["time"].astype("datetime64[M]").astype(int) % 12 + 1
    doc = __doc__.split("Method (also written to the JSON \"assumptions\")")[1].split("Run from")[0].strip()
    res = dict(assumptions=doc.split("\n- "), synthetic=meta["synthetic"], ueff_check=meta["ueff_check"],
               timing=dict(pair_seconds=meta["pair_seconds"], total_seconds=meta["total_seconds"], nproc=meta["nproc"],
                           n_pairs=meta["n_pairs"], n_null_pairs=meta["n_null_pairs"], every_nth_day=meta["every"]))
    chl = g["chl_l3"]
    oc = g["ocean"]
    dq = dict(days=len(g["time"]), patches=len(pos), ocean_px=int(oc.sum()))
    cnt_nan = cnt_big = 0
    for b0 in range(0, len(g["time"]), 500):
        x = np.asarray(chl[b0:b0 + 500])[:, oc]
        cnt_nan += int(np.isnan(x).sum()); cnt_big += int((x > 100).sum())
    dq.update(chl_nan_ocean=cnt_nan, chl_gt100_set_missing=cnt_big)
    for nm in ["cu", "gu", "eu"]:
        dq[f"vel_{nm}_nan_patchdays"] = int(np.isnan(z[f"vel_{nm}"]).sum())
    dq["vel_patchdays"] = int(z["vel_cu"].size)
    res["data_quality"] = dq
    tabs = {}
    res["motion"] = {}
    for k in LEADS:
        d = build(z, k); tabs[k] = d
        d["reg"] = preg[d["pc"]]; d["season"] = np.array([SEASONS[m] for m in months[d["i"]]])
        A = d["accept"]; days = d["i"]
        c = dict(candidates=int(len(A)), no_peak=int((~d["okpk"]).sum()), edge=int((d["okpk"] & d["edge"]).sum()),
                 peak_below_0p6=int((d["okpk"] & ~d["edge"] & (d["rpk"] < R_MIN)).sum()),
                 accepted=int(A.sum()), accepted_days=int(len(np.unique(days[A]))),
                 accepted_clear=int((A & d["clear"]).sum()),
                 accepted_cloudy_50_90=int((A & ~d["clear"]).sum()),
                 frac_abs_d_lt_0p5px=round(float((np.hypot(*d["do"][:, A]) < 0.5).mean()), 3))
        qf = lambda x: [round(float(v), 2) for v in np.nanpercentile(x, [10, 25, 50, 75, 90])]
        c["abs_d_per_day_pctl_10_25_50_75_90"] = {nm: qf(np.hypot(*d[key][:, A & np.isfinite(d[key]).all(0)]) / k)
                                                 for nm, key in [("obs", "do"), ("cur", "dc"), ("geo", "dg"), ("ueff", "de")]}
        c["overall"] = {nm: compare(A, d["do"], d[key], days, k) for nm, key in
                        [("cur", "dc"), ("geo", "dg"), ("ueff", "de"), ("shuffled_cur", "ds")]}
        c["overall_r_ge_0p8"] = {nm: compare(A & (d["rpk"] >= 0.8), d["do"], d[key], days, k) for nm, key in
                                 [("cur", "dc"), ("ueff", "de")]}
        c["overall_clear_only"] = {nm: compare(A & d["clear"], d["do"], d[key], days, k) for nm, key in [("cur", "dc"), ("ueff", "de")]}
        c["overall_cloudy_50_90"] = {nm: compare(A & ~d["clear"], d["do"], d[key], days, k) for nm, key in [("cur", "dc"), ("ueff", "de")]}
        nb = 300 if k == 1 else 100
        c["by_region"] = {NAMES[r]: {nm: compare(A & (d["reg"] == r), d["do"], d[key], days, k, nb) for nm, key in
                                     [("cur", "dc"), ("geo", "dg"), ("ueff", "de")]} for r in range(1, 8)}
        c["by_season"] = {s: {nm: compare(A & (d["season"] == s), d["do"], d[key], days, k, nb) for nm, key in
                              [("cur", "dc"), ("geo", "dg"), ("ueff", "de")]} for s in ["DJF", "MAM", "JJA", "SON"]}
        sp = np.hypot(*d["dc"]); vc = A & np.isfinite(sp)
        e = np.nanpercentile(sp[vc], [100 / 3, 200 / 3])
        c["cur_tercile_edges_px_over_lag"] = [round(float(x), 2) for x in e]
        c["by_cur_tercile"] = {f"T{q + 1}": {nm: compare(vc & (np.digitize(sp, e) == q), d["do"], d[key], days, k, nb)
                                             for nm, key in [("cur", "dc"), ("ueff", "de")]} for q in range(3)}
        # phase-correlation cross-check on clear patches
        cl = d["clear"] & np.isfinite(d["pc_fy"]) & ~d["pc_edge"]
        both = cl & A
        diff = np.hypot(d["pc_fx"] - d["fx"], d["pc_fy"] - d["fy"])
        c["phase_corr_check"] = dict(n_clear_pc_valid=int(cl.sum()), n_also_ncc_accepted=int(both.sum()),
                                     median_abs_diff_px=round(float(np.median(diff[both])), 3) if both.any() else None,
                                     frac_diff_le_1px=round(float((diff[both] <= 1).mean()), 3) if both.any() else None,
                                     slope_pc_on_ncc={cc: boot_slope(d[f"f{a}"][both], d[f"pc_f{a}"][both], days[both], 100)
                                                      for cc, a in [("east", "x"), ("north", "y")]},
                                     pc_vs_cur={cc: boot_slope(d["dc"][q][cl], d[f"pc_f{a}"][cl], days[cl], 100)
                                                for q, (cc, a) in enumerate([("east", "x"), ("north", "y")])},
                                     pc_vs_cur_on_ncc_accepted={cc: boot_slope(d["dc"][q][both], d[f"pc_f{a}"][both], days[both], 100)
                                                                for q, (cc, a) in enumerate([("east", "x"), ("north", "y")])},
                                     ncc_vs_cur_same_patches={cc: boot_slope(d["dc"][q][both], d["do"][q][both], days[both], 100)
                                                              for q, cc in enumerate(["east", "north"])})
        for thr in (0.99,):
            b2 = both & (d["fa"] >= thr) & (d["fb"] >= thr)
            c["phase_corr_check"][f"subset_ge_{thr}_observed"] = dict(
                n=int(b2.sum()), median_abs_diff_px=round(float(np.median(diff[b2])), 3) if b2.sum() >= MIN_N else None,
                frac_diff_le_1px=round(float((diff[b2] <= 1).mean()), 3) if b2.sum() >= MIN_N else None,
                median_abs_pc=round(float(np.median(np.hypot(d["pc_fx"][b2], d["pc_fy"][b2]))), 2) if b2.sum() >= MIN_N else None,
                median_abs_ncc=round(float(np.median(np.hypot(d["fx"][b2], d["fy"][b2]))), 2) if b2.sum() >= MIN_N else None)
        c["phase_corr_check"]["median_abs_pc"] = round(float(np.median(np.hypot(d["pc_fx"][both], d["pc_fy"][both]))), 2)
        c["phase_corr_check"]["median_abs_ncc"] = round(float(np.median(np.hypot(d["fx"][both], d["fy"][both]))), 2)
        res["motion"][f"lead{k}"] = c
    # lead consistency: same patch and start day accepted at lead 1 and lead k
    key = lambda d_: d_["i"] * 1000 + d_["p"]
    k1 = {v: q for q, v in enumerate(key(tabs[1])) if tabs[1]["accept"][q]}
    res["lead_consistency"] = {}
    for k in (2, 3):
        d = tabs[k]
        a, b = np.array([(k1[v], q) for q, v in enumerate(key(d)) if d["accept"][q] and v in k1]).T
        res["lead_consistency"][f"lead{k}_on_lead1"] = dict(
            n=int(len(a)), **{c: boot_slope(tabs[1]["do"][q][a], d["do"][q][b], d["i"][b], 100) for q, c in enumerate(["east", "north"])},
            median_abs_lead1=round(float(np.median(np.hypot(*tabs[1]["do"][:, a]))), 2), median_abs_leadk=round(float(np.median(np.hypot(*d["do"][:, b]))), 2),
            leadk_on_cur_same_set={c: boot_slope(d["dc"][q][b], d["do"][q][b], d["i"][b], 100) for q, c in enumerate(["east", "north"])})
    res["vs_block_matching"] = vs_blockmatch(tabs, g)
    res["coherence"], curves = coherence_analysis(z, preg, months)
    with open(RES / "spectral_motion.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    plots(tabs, res, z, g, preg, curves)
    print(json.dumps({k: res["motion"][k]["overall"] for k in res["motion"]}, indent=0)[:3000])


def vs_blockmatch(tabs, g):
    """Same day / lag: patch displacement (this script) vs mean of the accepted block-matching tiles inside it."""
    from analyses.blk4 import chl_motion as CM
    zc = dict(np.load(CM.CACHE)); zc.pop("self_test")
    t = g["time"].astype(int)
    out = {}
    for lag, S in [(1, 6), (2, 12)]:
        bt = CM.build_table(zc, lag, 16, S)
        A = bt["accept"] & (t[np.minimum(bt["day"] + lag, len(t) - 1)] - t[bt["day"]] == lag)
        key = {}
        for q in np.flatnonzero(A):
            key.setdefault((bt["day"][q], bt["ty"][q], bt["tx"][q]), q)
        d = tabs[lag]
        ty0 = (np.load(CACHE)["pos"][:, 0] - R0) // STRIDE; tx0 = (np.load(CACHE)["pos"][:, 1] - C0) // STRIDE
        ok = d["accept"]
        pm, tm, dd, ntile = [], [], [], []
        for q in np.flatnonzero(ok):
            pc = d["pc"][q]; dl = []
            for a in (0, 1):
                for b in (0, 1):
                    h = key.get((d["i"][q], ty0[pc] + a, tx0[pc] + b))
                    if h is not None:
                        dl.append(bt["do"][:, h])
            if dl:
                pm.append(d["do"][:, q]); tm.append(np.mean(dl, 0)); dd.append(d["i"][q]); ntile.append(len(dl))
        pm, tm, dd = np.array(pm).T, np.array(tm).T, np.array(dd)
        diff = np.hypot(*(pm - tm))
        out[f"lag{lag}"] = dict(n_patches_with_accepted_tiles=int(len(dd)), median_tiles_per_patch=float(np.median(ntile)),
                                median_abs_diff_px=round(float(np.median(diff)), 3), frac_diff_le_1px=round(float((diff <= 1).mean()), 3),
                                slope_patch_on_tilemean={c: boot_slope(tm[q], pm[q], dd, 100) for q, c in enumerate(["east", "north"])},
                                median_abs_patch=round(float(np.median(np.hypot(*pm))), 2), median_abs_tilemean=round(float(np.median(np.hypot(*tm))), 2),
                                n_spectral_accepted=int(ok.sum()), n_block_tiles_accepted_calendar=int(A.sum()))
        tabs[f"bm{lag}"] = (pm, tm)
    return out


def coherence_analysis(z, preg, months):
    out = {"wavelength_km": [round(float(w), 1) for w in WL_KM], "method": "ring-averaged MSC on >=90 % observed patches; see assumptions"}
    curves = {}
    reg = preg[np.searchsorted(z["pidx"], z["p"])]
    seas = np.array([SEASONS[m] for m in months[z["i"]]])
    base = z["clear"]
    for k in LEADS:
        for null in (0, 1):
            s = base & (z["k"] == k) & (z["null"] == null)
            al_ok = s & np.isfinite(z["a_xx"]).all(1)
            for kind, sel, lab in [("u", s, "unaligned_all"), ("u", al_ok, "unaligned_same_as_aligned"), ("a", al_ok, "aligned")]:
                mc, n = med_curve(ring_msc(z, sel, kind))
                curves[(k, null, lab)] = (mc, n)
        # aligned, NCC accepted (r >= 0.6) only
        s = base & (z["k"] == k) & (z["null"] == 0) & np.isfinite(z["a_xx"]).all(1) & (z["rpk"] >= R_MIN)
        curves[(k, 0, "aligned_ncc_accepted")] = med_curve(ring_msc(z, s, "a"))
    for key, (mc, n) in curves.items():
        out.setdefault(f"lead{key[0]}", {})[("null_" if key[1] else "") + key[2]] = dict(
            n=n, median_msc=[round(float(v), 3) for v in mc] if mc is not None else None)
    # by region / season (aligned, data, with the lead's null) for each lead
    for k in LEADS:
        s0 = base & (z["k"] == k) & (z["null"] == 0) & np.isfinite(z["a_xx"]).all(1)
        out[f"lead{k}"]["by_region_aligned"] = {}
        out[f"lead{k}"]["by_season_aligned"] = {}
        for r in range(1, 8):
            mc, n = med_curve(ring_msc(z, s0 & (reg == r), "a")); curves[(k, "reg", r)] = (mc, n)
            out[f"lead{k}"]["by_region_aligned"][NAMES[r]] = dict(n=n, median_msc=[round(float(v), 3) for v in mc] if mc is not None else "n<5")
        for se in ["DJF", "MAM", "JJA", "SON"]:
            mc, n = med_curve(ring_msc(z, s0 & (seas == se), "a")); curves[(k, "season", se)] = (mc, n)
            out[f"lead{k}"]["by_season_aligned"][se] = dict(n=n, median_msc=[round(float(v), 3) for v in mc] if mc is not None else "n<5")
    # decay fits
    fits = {}
    for lab in ["aligned", "unaligned_same_as_aligned"]:
        med = {k: curves[(k, 0, lab)][0] for k in LEADS}; nul = {k: curves[(k, 1, lab)][0] for k in LEADS}
        fits[lab], rho = decay_fits(med, nul)
        curves[("rho", lab)] = rho
    for nm, keyf in [("season", ["DJF", "MAM", "JJA", "SON"]), ("reg", [1, 2, 3])]:
        for v in keyf:
            med = {k: curves[(k, nm, v)][0] for k in LEADS}
            if any(m is None for m in med.values()):
                continue
            nul = {k: curves[(k, 1, "aligned")][0] for k in LEADS}
            fits[f"aligned_{nm}_{NAMES[v] if nm == 'reg' else v}"], _ = decay_fits(med, nul)
    out["decay_fits"] = fits
    # per-patch small-scale band (rings 8..16 = 18-9 km) at lead 1 for maps
    s1 = base & (z["k"] == 1) & (z["null"] == 0)
    rings = list(range(8, 17))
    bu = band_msc(z, s1, "u", rings); ba = band_msc(z, s1, "a", rings)
    pp = z["p"][s1]
    npch = len(z["pidx"])
    mp = {}
    for nm, v in (("unaligned", bu), ("aligned", ba)):
        arr = np.full(npch, np.nan); nn = np.zeros(npch, int)
        for q, pid in enumerate(z["pidx"]):
            x = v[(pp == pid) & np.isfinite(v)]
            nn[q] = len(x)
            if len(x) >= 20:
                arr[q] = np.median(x)
        mp[nm] = (arr, nn)
    sn = base & (z["k"] == 1) & (z["null"] == 1)
    nb_ = band_msc(z, sn, "a", rings); nu_ = band_msc(z, sn, "u", rings)
    out["band_9_18km_lead1"] = dict(median_aligned=round(float(np.nanmedian(ba)), 3), median_unaligned=round(float(np.nanmedian(bu)), 3),
                                    n_aligned=int(np.isfinite(ba).sum()), n_unaligned=int(np.isfinite(bu).sum()),
                                    null_median_aligned=round(float(np.nanmedian(nb_)), 3), null_median_unaligned=round(float(np.nanmedian(nu_)), 3),
                                    n_null=int(np.isfinite(nb_).sum()), map_min_n_per_patch=20,
                                    patches_mapped=int(np.isfinite(mp["aligned"][0]).sum()))
    curves["map"] = mp
    curves["null_band"] = (float(np.nanmedian(nu_)), float(np.nanmedian(nb_)))
    return out, curves


# ============================== plots ==============================
def patch_map(ax, vals, z, g, title, cmap, vmin, vmax, label):
    H, W = g["ocean"].shape
    img = np.full((H, W), np.nan)
    for v, (r, c) in zip(vals, z["pos"]):
        if np.isfinite(v):
            img[r + 8:r + 24, c + 8:c + 24] = v                        # central 16 x 16 px of each patch
    img[~g["ocean"]] = np.nan
    ext = [g["lon"][0], g["lon"][-1], g["lat"][0], g["lat"][-1]]
    im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, aspect=1.35)
    ax.contour(g["lon"], g["lat"], g["ocean"].astype(float), [0.5], colors="k", linewidths=0.5)
    ax.set(title=title, xlabel="longitude (°E)", ylabel="latitude (°N)")
    plt.colorbar(im, ax=ax, fraction=0.03, label=label)


def plots(tabs, res, z, g, preg, curves):
    syn = res["synthetic"]
    # 1 synthetic
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for q, kind in enumerate(["grf", "real"]):
        if kind not in syn: continue
        rows = syn[kind]; fr = [float(k_.split("_")[1]) for k_ in rows]
        ax[0].plot(fr, [rows[k_]["ncc_med_abs_err_px"] for k_ in rows], "o-", color=C[q], label=f"masked NCC, median ({kind})")
        ax[0].plot(fr, [rows[k_]["ncc_p90_abs_err_px"] for k_ in rows], "s--", color=C[q], label=f"masked NCC, 90th pct ({kind})")
        pcf = [f for f, k_ in zip(fr, rows) if "pc_med_abs_err_px" in rows[k_]]
        ax[0].plot(pcf, [rows[k_]["pc_med_abs_err_px"] for k_ in rows if "pc_med_abs_err_px" in rows[k_]], "^", ms=9, color=C[q + 2],
                   label=f"phase corr, median ({kind})")
        ax[1].plot(fr, [rows[k_]["ncc_accepted"] / rows[k_]["n"] for k_ in rows], "o-", color=C[q], label=f"{kind}")
    ax[0].set(xlabel="cloud fraction per day (independent masks)", ylabel="|recovered − true shift|, px",
              title="Synthetic shift recovery error vs cloud fraction (n=300 per point)"); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="cloud fraction per day", ylabel="fraction accepted (r ≥ 0.6, not on edge)",
              title="Synthetic: masked-NCC acceptance vs cloud fraction"); ax[1].legend(fontsize=8)
    cc = syn.get("coherence_check_grf_clear_no_noise")
    if cc:
        ax[2].semilogx(cc["wavelength_km"], cc["median_msc_unaligned"], "o-", color=C[1], label="unaligned")
        ax[2].semilogx(cc["wavelength_km"], cc["median_msc_aligned_true_shift"], "o-", color=C[0], label="aligned by true shift")
        ax[2].set(xlabel="wavelength, km", ylabel="median ring MSC", title="Synthetic GRF, shift U(−5,5) px: coherence (n=100)")
        ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "spectral_synthetic.png", dpi=130); plt.close(fig)

    # 2 components vs references (lead 1)
    d = tabs[1]; A = d["accept"]
    fig, ax = plt.subplots(3, 2, figsize=(10, 13))
    for r_, (lab, key, kk) in enumerate([("reanalysis current", "dc", "cur"), ("geostrophic (SSH)", "dg", "geo"), ("tracer u_eff", "de", "ueff")]):
        sel = A & np.isfinite(d[key]).all(0)
        for q, comp in enumerate(["east", "north"]):
            a_ = ax[r_, q]
            a_.hist2d(d[key][q][sel], d["do"][q][sel], bins=[np.linspace(-6, 6, 49)] * 2, cmap="viridis", cmin=1)
            s = res["motion"]["lead1"]["overall"][kk][comp]
            xx = np.array([-6, 6]); a_.plot(xx, s["intercept"] + s["slope"] * xx, color=C[1], lw=2, label=f"OLS slope {s['slope']}")
            a_.plot(xx, xx, "w--", lw=1, label="1:1")
            a_.set(xlabel=f"{comp}, {lab}, px/day", ylabel=f"{comp}, observed patch shift (masked NCC), px/day",
                   title=f"{comp}: patch shift vs {lab}, lead 1 (n={s['n']})"); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "spectral_components.png", dpi=130); plt.close(fig)

    # 3 slopes by group (lead 1) + by lead
    c = res["motion"]["lead1"]
    groups = [("all", c["overall"])] + list(c["by_season"].items()) + [(k_, v) for k_, v in c["by_region"].items()] + \
             [(f"|d_cur| {k_}", v) for k_, v in c["by_cur_tercile"].items()]
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for a_, key, lab in [(ax[0], "cur", "reanalysis current"), (ax[1], "geo", "geostrophic (SSH)"), (ax[2], "ueff", "tracer u_eff")]:
        for q, (comp, off) in enumerate([("east", -0.15), ("north", 0.15)]):
            for jx, (gname, v) in enumerate(groups):
                s = v.get(key, {}).get(comp, {})
                if "slope" not in s: continue
                a_.errorbar(jx + off, s["slope"], yerr=[[s["slope"] - s["ci95"][0]], [s["ci95"][1] - s["slope"]]], fmt="o",
                            color=C[q], label=comp if jx == 0 else None)
                a_.annotate(str(s["n"]), (jx + off, s["ci95"][1]), fontsize=6, ha="center", va="bottom")
        a_.axhline(0, color="k", lw=.6); a_.axhline(1, color="grey", lw=.6, ls="--")
        a_.set(ylabel="OLS slope, patch shift on reference", title=f"Slope of masked-NCC patch shift on {lab}, lead 1, by group (numbers = n patches)")
        a_.legend(fontsize=8)
    ax[2].set_xticks(range(len(groups))); ax[2].set_xticklabels([g_[0] for g_ in groups], rotation=35, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "spectral_slopes_by_group.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for q, comp in enumerate(["east", "north"]):
        for j_, (key, lab) in enumerate([("cur", "reanalysis"), ("geo", "geostrophic"), ("ueff", "tracer u_eff"), ("shuffled_cur", "shuffled-day reanalysis")]):
            ys_ = [res["motion"][f"lead{k}"]["overall"][key][comp] for k in LEADS]
            ax[q].errorbar(np.array(LEADS) + 0.08 * (j_ - 1.5), [s["slope"] for s in ys_],
                           yerr=[[s["slope"] - s["ci95"][0] for s in ys_], [s["ci95"][1] - s["slope"] for s in ys_]], fmt="o-", color=C[j_], label=lab)
        ax[q].set(xlabel="lead k, days", ylabel="OLS slope, patch shift on reference", title=f"{comp}: slope of patch shift on reference velocity by lead",
                  xticks=LEADS); ax[q].axhline(0, color="k", lw=.5); ax[q].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "spectral_slopes_by_lead.png", dpi=130); plt.close(fig)

    # 4 angles
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for a_, (lab, key) in zip(ax, [("reanalysis current", "dc"), ("geostrophic (SSH)", "dg"), ("tracer u_eff", "de"), ("reanalysis, random other day", "ds")]):
        sel = A & np.isfinite(d[key]).all(0)
        st, dth = angle_stats(d["do"][:, sel], d[key][:, sel], 1)
        a_.hist(np.degrees(dth), np.linspace(-180, 180, 37), color=C[0], density=True)
        a_.set(xlabel="angle(patch shift) − angle(reference), degrees", title=f"vs {lab}\nR={st.get('R')}, n={st['n']}")
    ax[0].set_ylabel("density")
    fig.suptitle("Direction difference, masked-NCC patch shift vs reference velocity (|d| ≥ 0.5 px/day both; lead 1)")
    fig.tight_layout(); fig.savefig(FIG / "spectral_angles.png", dpi=130); plt.close(fig)

    # 5 vs block matching and phase correlation
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    pm, tm = tabs["bm1"]
    for q, comp in enumerate(["east", "north"]):
        ax[0, q].hist2d(tm[q], pm[q], bins=[np.linspace(-6, 6, 49)] * 2, cmap="viridis", cmin=1)
        ax[0, q].plot([-6, 6], [-6, 6], "w--", lw=1)
        ax[0, q].set(xlabel=f"{comp}, mean block-matching shift of accepted 16 px tiles, px", ylabel=f"{comp}, 32 px patch shift (masked NCC), px",
                     title=f"{comp}: patch vs block matching, lag 1 (n={pm.shape[1]})")
        both = d["clear"] & A & np.isfinite(d["pc_fy"]) & ~d["pc_edge"]
        ax[1, q].hist2d(d[f"f{'xy'[q]}"][both], d[f"pc_f{'xy'[q]}"][both], bins=[np.linspace(-6, 6, 49)] * 2, cmap="viridis", cmin=1)
        ax[1, q].plot([-6, 6], [-6, 6], "w--", lw=1)
        ax[1, q].set(xlabel=f"{comp}, masked NCC shift, px", ylabel=f"{comp}, phase correlation shift, px",
                     title=f"{comp}: phase corr vs masked NCC, ≥90 % clear, lead 1 (n={both.sum()})")
    fig.tight_layout(); fig.savefig(FIG / "spectral_vs_blockmatch_phasecorr.png", dpi=130); plt.close(fig)

    # 6 coherence curves
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for q, k in enumerate(LEADS):
        a_ = ax[q]
        for lab, col, ls in [("aligned", C[0], "-"), ("unaligned_same_as_aligned", C[1], "-"), ("unaligned_all", C[1], ":"),
                             ("aligned_ncc_accepted", C[2], "-")]:
            mc, n = curves[(k, 0, lab)]
            if mc is not None: a_.semilogx(WL_KM, mc, ls, color=col, marker="o", ms=3, label=f"{lab.replace('_', ' ')} (n={n})")
        for lab, col in [("aligned", C[0]), ("unaligned_same_as_aligned", C[1])]:
            mc, n = curves[(k, 1, lab)]
            if mc is not None: a_.semilogx(WL_KM, mc, "--", color=col, label=f"null, random day, {lab.split('_')[0]} (n={n})")
        a_.set(xlabel="wavelength, km", title=f"Median ring MSC, day t vs t+{k}, ≥90 % clear 32 px patches")
        a_.legend(fontsize=7); a_.invert_xaxis()
    ax[0].set_ylabel("median magnitude-squared coherence")
    fig.tight_layout(); fig.savefig(FIG / "spectral_coherence_curves.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), sharey=True)
    for q, k in enumerate(LEADS):
        for r_ in range(1, 8):
            mc, n = curves[(k, "reg", r_)]
            if mc is not None: ax[0, q].semilogx(WL_KM, mc, "o-", ms=3, color=C[(r_ - 1) % 8], label=f"{NAMES[r_]} (n={n})")
        for s_, se in enumerate(["DJF", "MAM", "JJA", "SON"]):
            mc, n = curves[(k, "season", se)]
            if mc is not None: ax[1, q].semilogx(WL_KM, mc, "o-", ms=3, color=C[s_], label=f"{se} (n={n})")
        for a_ in ax[:, q]:
            mc, n = curves[(k, 1, "aligned")]
            a_.semilogx(WL_KM, mc, "k--", label="null (random day)"); a_.invert_xaxis(); a_.legend(fontsize=6); a_.set_xlabel("wavelength, km")
        ax[0, q].set_title(f"Aligned median MSC by region, lead {k}"); ax[1, q].set_title(f"Aligned median MSC by season, lead {k}")
    ax[0, 0].set_ylabel("median MSC"); ax[1, 0].set_ylabel("median MSC")
    fig.tight_layout(); fig.savefig(FIG / "spectral_coherence_by_group.png", dpi=130); plt.close(fig)

    # 7 decay fits
    fits = res["coherence"]["decay_fits"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for j_, lab in enumerate(["aligned", "unaligned_same_as_aligned"]):
        rho = curves[("rho", lab)]
        for m in [0, 2, 5, 9, 15]:
            ax[0].plot(LEADS, [np.log(rho[k][m]) for k in LEADS], "o-" if j_ == 0 else "s:", color=C[[0, 2, 5, 9, 15].index(m)],
                       label=f"{WL_KM[m]:.0f} km, {lab.split('_')[0]}")
        per = fits[lab]["per_ring"]
        ef = [p_["efold_days"] if p_["efold_days"] is not None else np.nan for p_ in per]
        ax[1].loglog(WL_KM, ef, "o-" if j_ == 0 else "s:", color=C[j_], label=lab.split("_")[0])
        Km = [p_["K_m2_s"] for p_ in per]
        ax[2].semilogx(WL_KM, Km, "o-" if j_ == 0 else "s:", color=C[j_],
                       label=f"{lab.split('_')[0]}: per-wavelength K; joint K={fits[lab]['single_K_free_intercept']['K_m2_s']} m²/s (R²={fits[lab]['single_K_free_intercept']['R2']})")
    ax[0].set(xlabel="lead, days", ylabel="ln ρ (null-corrected, from median MSC)", title="Decay of null-corrected coherence with lead", xticks=LEADS)
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].set(xlabel="wavelength, km", ylabel="e-folding lead of ρ, days", title="e-folding lead time by wavelength (OLS over leads 1-3)"); ax[1].legend(fontsize=8)
    ax[2].set(xlabel="wavelength, km", ylabel="implied diffusivity K = rate / q², m²/s", title="Per-wavelength diffusivity vs single-K fit"); ax[2].legend(fontsize=7)
    for a_ in ax[1:]: a_.invert_xaxis()
    fig.tight_layout(); fig.savefig(FIG / "spectral_decay_fits.png", dpi=130); plt.close(fig)

    # 8 maps
    mp = curves["map"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    nu, na = curves["null_band"]
    patch_map(ax[0, 0], mp["aligned"][0], z, g, f"Median 9-18 km band MSC, lead 1, aligned (null median {na:.2f}; ≥20 pairs)", "viridis", 0, 0.2, "MSC")
    patch_map(ax[0, 1], mp["unaligned"][0], z, g, f"Median 9-18 km band MSC, lead 1, unaligned (null median {nu:.2f}; ≥20 pairs)", "viridis", 0, 0.2, "MSC")
    # agreement with currents per patch position
    npch = len(z["pidx"]); Rv = np.full(npch, np.nan); Sv = np.full(npch, np.nan)
    for q in range(npch):
        s = A & (d["pc"] == q) & np.isfinite(d["dc"]).all(0)
        if s.sum() < 20: continue
        do, dc = d["do"][:, s], d["dc"][:, s]
        Sv[q] = (do * dc).sum() / (dc * dc).sum()
        st, _ = angle_stats(do, dc, 1)
        if "R" in st and st["n"] >= 20: Rv[q] = st["R"]
    patch_map(ax[1, 0], Sv, z, g, "Projection slope Σ d_obs·d_cur / Σ|d_cur|², lead 1 (≥20 accepted pairs)", "cividis", 0, 1, "slope")
    patch_map(ax[1, 1], Rv, z, g, "Mean resultant length of direction difference vs current, lead 1 (≥20 pairs)", "cividis", 0, 1, "R")
    fig.tight_layout(); fig.savefig(FIG / "spectral_maps.png", dpi=130); plt.close(fig)
    res_maps = dict(proj_slope_patches=int(np.isfinite(Sv).sum()), R_patches=int(np.isfinite(Rv).sum()))
    print("maps", res_maps)


if __name__ == "__main__":
    main()
