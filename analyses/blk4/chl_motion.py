"""Do observed CHL features move the way the reanalysis currents say? Maximum cross-correlation (block matching)
of ln CHL between day t and day t+k on the Black Sea 4 km store, compared with tile-mean reanalysis currents
(d_cur) and geostrophic currents from the model SSH (d_geo).

Method (all stated in the JSON "assumptions" as well)
- Store ~/mind_the_chl_gap/data/blk4_store_2014_2024 (4 km, lat ascending -> +row = north, lon ascending -> +col =
  east). ln CHL; CHL > 100 mg/m3 or <= 0 -> missing. u_curr/v_curr = CMEMS uo/vo (m/s); "sla" = CMEMS zos (m).
- Tiles: non-overlapping T x T px (T = 16, 8), tile grid centred in the 197 x 275 grid (remainder rows/cols at the
  edges dropped). Candidate tile: >= 80 % of its T*T pixels observed on day t AND on day t+k (land counts as missing).
- For every integer shift (sy, sx) in [-12, 12]^2 the day-t tile is compared with day t+k at (row+sy, col+sx):
  Pearson r of ln CHL over pixels observed in both (= mean removed per shift); r = NaN if the overlap < 50 % of T*T.
  Search windows S = 6 (main) and S = 12 are taken from the same r map.
- Peak = argmax r inside the window. Sub-pixel: 1-D parabolic fit along each axis through the peak and its two
  neighbours (offset clipped to +-0.5; integer value kept if a neighbour is NaN).
- Acceptance: peak r >= 0.6; peak not on the window edge; and either the best shift is (0, 0) or
  atanh(r_peak) - atanh(r_0) > 1.96 * sqrt(2 / (n_eff - 3)) with n_eff = n_overlap_at_peak / 4 (pixels assumed
  correlated over ~2 x 2 px). Non-zero best shift failing that test = "ambiguous" (rejected; sensitivity rows count
  them as zero motion, or drop the significance test altogether).
- d_cur: per-pixel u * 86400 / dx, v * 86400 / dy (grid_metrics, px/day), averaged over the tile (valid if >= 50 % of
  tile pixels finite) and over days t..t+k, times k -> px over the lag. d_geo: same from geostrophic u_g = -g/f
  d(zos)/dy, v_g = g/f d(zos)/dx, centred differences, pixels within 2 px of land (or of a non-finite zos) masked.
- Shuffled control: d_cur of the same tile on a random day drawn uniformly from all days with |t' - t| > 1 and a valid
  tile current (seed 0).
- OLS with intercept per component (d_obs on d_ref); 95 % CI from 300 bootstrap resamples of days (tiles clustered by
  day). Angles: d = (east, north) vector; angle difference wrapped to (-180, 180]; only tiles with |d_obs| >= 0.5 px
  and |d_ref| >= 0.5 px. Mean resultant length R = |mean(exp(i dtheta))|.
- Region of a tile = majority 4 km region label over its ocean pixels (2 km analyses/blk/regions.npy, 2x2 mode as
  error_breakdown.regions_4km, no crop). Season by the month of day t.

Run from the repo root: python -m analyses.blk4.chl_motion [--recompute]   (CPU, ~8 processes)
"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from chl_cnn.plotting import OKABE_ITO
from chl_cnn.physics import grid_metrics
from chl_cnn import store_blk4 as S4

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
CACHE = RES / "chl_motion_tiles.npz"
STORE = Path.home() / "mind_the_chl_gap/data/blk4_store_2014_2024"
NAMES = {1: "NW shelf", 2: "Western open sea", 3: "Eastern open sea", 4: "Anatolian shelf",
         5: "Crimea/Kerch/Caucasus shelf", 6: "Azov Sea + Kerch strait", 7: "Marmara/Bosporus corner"}
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
           9: "SON", 10: "SON", 11: "SON"}
SMAX, TILES, MIN_OBS, MIN_OVL, R_MIN = 12, (16, 8), 0.8, 0.5, 0.6
G = 9.81
MIN_N = 5


# ============================== data ==============================
def load():
    g = {k: np.load(STORE / f"{k}.npy", mmap_mode="r") for k in ["chl_l3", "u_curr", "v_curr", "sla"]}
    g["lat"], g["lon"] = np.load(STORE / "lat.npy"), np.load(STORE / "lon.npy")
    g["ocean"] = np.load(STORE / "ocean.npy").astype(bool)
    g["time"] = np.load(STORE / "time.npy")
    return g


def ln_chl(c):
    c = np.asarray(c, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(c) & (c > 0) & (c <= 100), np.log(c), np.nan)


def regions(ocean):
    reg2 = np.load(ROOT / "analyses/blk/regions.npy")
    H2, W2 = reg2.shape[0] // 2, reg2.shape[1] // 2
    b = S4._blocks(reg2, H2, W2)
    cnt = np.stack([(b == k).sum(-1) for k in range(1, 8)], -1)
    reg = np.where(cnt.max(-1) > 0, cnt.argmax(-1) + 1, 0).astype(np.int8)
    assert reg.shape == ocean.shape
    return reg


def tile_geom(H, W, T):
    nty, ntx = H // T, W // T
    return (H - nty * T) // 2, (W - ntx * T) // 2, nty, ntx


def tsum(X, T, geom):
    r0, c0, nty, ntx = geom
    return X[r0:r0 + nty * T, c0:c0 + ntx * T].reshape(nty, T, ntx, T).sum((1, 3))


def tmean_valid(X, T, geom, min_frac=0.5):
    m = np.isfinite(X)
    s, n = tsum(np.where(m, X, 0.0), T, geom), tsum(m.astype(float), T, geom)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_frac * T * T, s / n, np.nan)


# ============================== matching ==============================
_G = {}


def _init():
    _G.update(load())


def rmap(A, B, Tlist, geoms, S=SMAX):
    """r[T]: (nty, ntx, 2S+1, 2S+1) Pearson r of day-t tile vs day-t+k shifted; n[T] overlap counts."""
    H, W = A.shape
    mA = np.isfinite(A)
    Bp = np.full((H + 2 * S, W + 2 * S), np.nan); Bp[S:S + H, S:S + W] = B
    out = {T: (np.full(geoms[T][2:] + (2 * S + 1, 2 * S + 1), np.nan), np.zeros(geoms[T][2:] + (2 * S + 1, 2 * S + 1)))
           for T in Tlist}
    for iy, sy in enumerate(range(-S, S + 1)):
        for ix, sx in enumerate(range(-S, S + 1)):
            Bs = Bp[S + sy:S + sy + H, S + sx:S + sx + W]
            m = mA & np.isfinite(Bs)
            a, b = np.where(m, A, 0.0), np.where(m, Bs, 0.0)
            for T in Tlist:
                g = geoms[T]
                n = tsum(m.astype(float), T, g)
                sa, sb = tsum(a, T, g), tsum(b, T, g)
                saa, sbb, sab = tsum(a * a, T, g), tsum(b * b, T, g), tsum(a * b, T, g)
                with np.errstate(invalid="ignore", divide="ignore"):
                    den = np.sqrt((n * saa - sa ** 2) * (n * sbb - sb ** 2))
                    r = (n * sab - sa * sb) / den
                r[(n < MIN_OVL * T * T) | ~(den > 1e-12)] = np.nan
                out[T][0][..., iy, ix] = r; out[T][1][..., iy, ix] = n
    return out


def peak(R, N, Suse, S=SMAX):
    """Peak of one tile's r map inside +-Suse. Returns dict or None if all NaN."""
    sub = R[S - Suse:S + Suse + 1, S - Suse:S + Suse + 1]
    if not np.isfinite(sub).any():
        return None
    iy, ix = np.unravel_index(np.nanargmax(sub), sub.shape)
    sy, sx = iy - Suse, ix - Suse
    edge = abs(sy) == Suse or abs(sx) == Suse
    fy = fx = 0.0
    if not edge:
        for axis in (0, 1):
            lo = sub[iy - 1, ix] if axis == 0 else sub[iy, ix - 1]
            hi = sub[iy + 1, ix] if axis == 0 else sub[iy, ix + 1]
            c = sub[iy, ix]; den = lo - 2 * c + hi
            if np.isfinite(lo) and np.isfinite(hi) and den < 0:
                off = float(np.clip(0.5 * (lo - hi) / den, -0.5, 0.5))
                if axis == 0: fy = off
                else: fx = off
    return dict(sy=sy, sx=sx, fy=sy + fy, fx=sx + fx, rpk=sub[iy, ix], r0=R[S, S],
                npk=N[S - Suse + iy, S - Suse + ix], n0=N[S, S], edge=edge)


def match_day(args):
    i, k = args
    lat, lon = _G["lat"], _G["lon"]
    A, B = ln_chl(_G["chl_l3"][i]), ln_chl(_G["chl_l3"][i + k])
    H, W = A.shape
    geoms = {T: tile_geom(H, W, T) for T in TILES}
    cand = {}
    for T in TILES:
        fa = tsum(np.isfinite(A).astype(float), T, geoms[T]) / T ** 2
        fb = tsum(np.isfinite(B).astype(float), T, geoms[T]) / T ** 2
        cand[T] = (fa >= MIN_OBS) & (fb >= MIN_OBS)
    Tl = [T for T in TILES if cand[T].any()]
    recs = []
    if not Tl:
        return recs
    rm = rmap(A, B, Tl, geoms)
    for T in Tl:
        R, N = rm[T]
        for ty, tx in zip(*np.nonzero(cand[T])):
            for Suse in (6, 12):
                p = peak(R[ty, tx], N[ty, tx], Suse)
                if p is None:
                    p = dict(sy=0, sx=0, fy=np.nan, fx=np.nan, rpk=np.nan, r0=np.nan, npk=0, n0=0, edge=False)
                recs.append((i, k, T, Suse, ty, tx, p["sy"], p["sx"], p["fy"], p["fx"], p["rpk"], p["r0"],
                             p["npk"], p["n0"], p["edge"]))
    return recs


FIELDS = ["day", "lag", "T", "S", "ty", "tx", "sy", "sx", "fy", "fx", "rpk", "r0", "npk", "n0", "edge"]


def tile_currents(g):
    """Per day and tile size: tile-mean current and geostrophic current in px/day (east, north)."""
    lat, lon, ocean = g["lat"], g["lon"], g["ocean"]
    dx, dy, f, _ = grid_metrics(lat, lon)
    H, W = ocean.shape
    coast = binary_dilation(~ocean, structure=np.ones((5, 5), bool))       # within 2 px of land
    nd = len(g["time"])
    out = {T: {k: np.full((nd,) + tile_geom(H, W, T)[2:], np.nan, "float32") for k in ["cu", "cv", "gu", "gv"]}
           for T in TILES}
    for i0 in range(0, nd, 100):
        sl = slice(i0, min(i0 + 100, nd))
        u = np.asarray(g["u_curr"][sl], "float64"); v = np.asarray(g["v_curr"][sl], "float64")
        z = np.asarray(g["sla"][sl], "float64")
        z = np.where(ocean[None], z, np.nan)
        dzx = np.full_like(z, np.nan); dzy = np.full_like(z, np.nan)
        dzx[:, :, 1:-1] = (z[:, :, 2:] - z[:, :, :-2]) / (2 * dx[None, :, 1:-1])
        dzy[:, 1:-1, :] = (z[:, 2:, :] - z[:, :-2, :]) / (2 * dy[None, 1:-1, :])
        ug, vg = -G / f[None] * dzy, G / f[None] * dzx
        # mask: within 2 px of land or of a non-finite zos
        bad = np.stack([binary_dilation(~np.isfinite(zz), structure=np.ones((5, 5), bool)) for zz in z]) | coast[None]
        ug[bad] = np.nan; vg[bad] = np.nan
        pu, pv = u * 86400 / dx[None], v * 86400 / dy[None]
        gu, gv = ug * 86400 / dx[None], vg * 86400 / dy[None]
        for T in TILES:
            gm = tile_geom(H, W, T)
            for j, d in enumerate(range(sl.start, sl.stop)):
                out[T]["cu"][d] = tmean_valid(pu[j], T, gm); out[T]["cv"][d] = tmean_valid(pv[j], T, gm)
                out[T]["gu"][d] = tmean_valid(gu[j], T, gm); out[T]["gv"][d] = tmean_valid(gv[j], T, gm)
    return out


def self_test(g):
    """Known shift: day t+1 := day t moved by (+2 rows north, -3 cols west); the matcher must return sy=2, sx=-3."""
    i = int(np.argmax([np.isfinite(ln_chl(g["chl_l3"][j])).mean() for j in range(150, 250)])) + 150
    A = ln_chl(g["chl_l3"][i])
    B = np.full_like(A, np.nan); B[2:, :-3] = A[:-2, 3:]          # B[r, c] = A[r-2, c+3]: feature at (r,c) -> (r+2, c-3)
    gm = {16: tile_geom(*A.shape, 16)}
    R, N = rmap(A, B, [16], gm)[16]
    fa = tsum(np.isfinite(A).astype(float), 16, gm[16]) / 256
    sh = [peak(R[a, b], N[a, b], 6) for a, b in zip(*np.nonzero(fa >= MIN_OBS))]
    sh = [(p["sy"], p["sx"]) for p in sh if p is not None]
    vals, cnt = np.unique(np.array(sh), axis=0, return_counts=True)
    return dict(day=str(g["time"][i])[:10], imposed_rows_north=2, imposed_cols_east=-3, n_tiles=len(sh),
                recovered={f"{a},{b}": int(c) for (a, b), c in zip(vals, cnt)})


def compute():
    g = load()
    t0 = time.time()
    st = self_test(g); print("self test", st, flush=True)
    nd = len(g["time"])
    jobs = [(i, 1) for i in range(nd - 1)] + [(i, 2) for i in range(nd - 2)]
    recs = []
    with Pool(8, initializer=_init) as pool:
        for j, r in enumerate(pool.imap(match_day, jobs, chunksize=8)):
            recs.extend(r)
            if j % 500 == 0:
                print(f"{j}/{len(jobs)} {time.time() - t0:.0f}s {len(recs)} recs", flush=True)
    arr = {f: np.array([r[q] for r in recs]) for q, f in enumerate(FIELDS)}
    print("currents", flush=True)
    cur = tile_currents(g)
    np.savez_compressed(CACHE, **arr, **{f"{k}_{T}": cur[T][k] for T in TILES for k in cur[T]},
                        self_test=json.dumps(st))
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
    idx_by = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        pick = np.concatenate([idx_by[d] for d in rng.integers(0, len(ud), len(ud))])
        if x[pick].std() > 0:
            bs.append(np.polyfit(x[pick], y[pick], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(slope=round(float(b), 3), ci95=[round(float(lo), 3), round(float(hi), 3)],
                intercept=round(float(a), 3), r2=round(r2, 3), n=int(len(x)), n_days=int(len(ud)))


def angle_stats(do, dr):
    m = (np.hypot(*do) >= 0.5) & (np.hypot(*dr) >= 0.5) & np.isfinite(dr).all(0)
    dth = np.angle(np.exp(1j * (np.arctan2(do[1][m], do[0][m]) - np.arctan2(dr[1][m], dr[0][m]))))
    if m.sum() < MIN_N:
        return dict(n=int(m.sum()), flag="n<5"), dth
    z = np.exp(1j * dth).mean()
    return dict(R=round(float(abs(z)), 3), mean_deg=round(float(np.degrees(np.angle(z))), 1),
                frac_within_45=round(float((np.abs(dth) <= np.pi / 4).mean()), 3), n=int(m.sum())), dth


def compare(sel, do, dr, days):
    out = {c: boot_slope(dr[q][sel], do[q][sel], days[sel]) for q, c in enumerate(["east", "north"])}
    out["angle"] = angle_stats(do[:, sel], dr[:, sel])[0]
    return out


def build_table(z, lag, T, S, rule="main"):
    """Tiles of one configuration with d_obs, d_cur, d_geo (px over the lag)."""
    m = (z["lag"] == lag) & (z["T"] == T) & (z["S"] == S)
    d = {f: z[f][m] for f in FIELDS}
    nE = 1.96 * np.sqrt(2 / np.maximum(d["npk"] / 4 - 3, 1))
    with np.errstate(invalid="ignore"):
        dz = np.arctanh(np.clip(d["rpk"], -.999, .999)) - np.arctanh(np.clip(d["r0"], -.999, .999))
    zero = (d["sy"] == 0) & (d["sx"] == 0)
    good = (d["rpk"] >= R_MIN) & ~d["edge"]
    sig = dz > nE
    d["accept"] = good & (zero | sig)
    d["ambig"] = good & ~zero & ~sig
    d["zero"] = zero
    if rule == "ambig_as_zero":
        d["fy"] = np.where(d["ambig"], 0.0, d["fy"]); d["fx"] = np.where(d["ambig"], 0.0, d["fx"])
        d["accept"] = d["accept"] | d["ambig"]
    if rule == "no_sig":
        d["accept"] = good
    d["do"] = np.stack([d["fx"], d["fy"]])                        # (east, north) px over the lag
    cu, cv, gu, gv = (z[f"{k}_{T}"] for k in ["cu", "cv", "gu", "gv"])
    def lagmean(F):
        stack = np.stack([F[d["day"] + j, d["ty"], d["tx"]] for j in range(lag + 1)])
        return stack.mean(0) * lag                                # NaN if any day invalid
    d["dc"] = np.stack([lagmean(cu), lagmean(cv)]); d["dg"] = np.stack([lagmean(gu), lagmean(gv)])
    # shuffled: same tile, random other day with valid current
    rng = np.random.default_rng(0); nd = cu.shape[0]
    sh = np.full_like(d["dc"], np.nan)
    todo = np.arange(len(d["day"]))
    for _ in range(20):
        if not len(todo): break
        t2 = rng.integers(0, nd - lag, len(todo))
        far = np.abs(t2 - d["day"][todo]) > 1
        cand = np.stack([np.stack([F[t2 + j, d["ty"][todo], d["tx"][todo]] for j in range(lag + 1)]).mean(0) * lag
                         for F in (cu, cv)])
        ok = far & np.isfinite(cand).all(0)
        sh[:, todo[ok]] = cand[:, ok]; todo = todo[~ok]
    d["ds"] = sh
    return d


def main():
    if "--recompute" in sys.argv or not CACHE.exists():
        compute()
    z = dict(np.load(CACHE))
    st = json.loads(str(z.pop("self_test")))
    g = load(); H, W = g["ocean"].shape
    reg = regions(g["ocean"])
    months = g["time"].astype("datetime64[M]").astype(int) % 12 + 1
    res = dict(assumptions=__doc__.split("Method (all stated in the JSON \"assumptions\" as well)")[1].split("Run from")[0].strip().split("\n- "),
               self_test=st, configs={})
    nan_chl = int(np.isnan(np.asarray(g["chl_l3"][:, g["ocean"]])).sum())
    big = int((np.asarray(g["chl_l3"][:, g["ocean"]]) > 100).sum())
    res["data_quality"] = dict(days=len(g["time"]), ocean_px=int(g["ocean"].sum()), chl_nan_ocean=nan_chl,
                               chl_gt100_set_missing=big)
    tabs = {}
    for lag, T, S in [(1, 16, 6), (1, 16, 12), (1, 8, 6), (2, 16, 6), (2, 16, 12)]:
        d = build_table(z, lag, T, S)
        tabs[(lag, T, S)] = d
        gm = tile_geom(H, W, T)
        # tile region (majority over ocean px) and season
        rr = reg[gm[0]:gm[0] + gm[2] * T, gm[1]:gm[1] + gm[3] * T].reshape(gm[2], T, gm[3], T).swapaxes(1, 2).reshape(gm[2], gm[3], -1)
        treg = np.array([[np.bincount(rr[a, b][rr[a, b] > 0], minlength=8)[1:].argmax() + 1 if (rr[a, b] > 0).any() else 0
                          for b in range(gm[3])] for a in range(gm[2])])
        d["reg"] = treg[d["ty"], d["tx"]]; d["season"] = np.array([SEASONS[m] for m in months[d["day"]]])
        A = d["accept"]; vc = A & np.isfinite(d["dc"]).all(0); vg = A & np.isfinite(d["dg"]).all(0)
        c = dict(candidates=int(len(A)), no_valid_r=int(np.isnan(d["rpk"]).sum()),
                 peak_below_0p6=int((d["rpk"] < R_MIN).sum()), edge_hits=int((d["edge"] & (d["rpk"] >= R_MIN)).sum()),
                 ambiguous=int(d["ambig"].sum()), accepted=int(A.sum()), accepted_days=int(len(np.unique(d["day"][A]))),
                 accepted_with_cur=int(vc.sum()), accepted_with_geo=int(vg.sum()),
                 zero_shift_frac=round(float(d["zero"][A].mean()), 3),
                 zero_shift_frac_all_peak_ge_0p6=round(float(d["zero"][(d["rpk"] >= R_MIN)].mean()), 3),
                 frac_cur_lt_0p5px=round(float((np.hypot(*d["dc"][:, vc]) < 0.5).mean()), 3))
        q = lambda x: [round(float(v), 2) for v in np.nanpercentile(x, [10, 25, 50, 75, 90])]
        c["abs_d_pctl_10_25_50_75_90"] = dict(obs=q(np.hypot(*d["do"][:, A])), cur=q(np.hypot(*d["dc"][:, vc])),
                                              geo=q(np.hypot(*d["dg"][:, vg])), obs_on_cur_tiles=q(np.hypot(*d["do"][:, vc])))
        days = d["day"]
        c["overall"] = {"cur": compare(vc, d["do"], d["dc"], days), "geo": compare(vg, d["do"], d["dg"], days),
                        "shuffled_cur": compare(A & np.isfinite(d["ds"]).all(0), d["do"], d["ds"], days)}
        if (lag, T, S) == (1, 16, 6):
            d2 = build_table(z, lag, T, S, rule="ambig_as_zero")
            v2 = d2["accept"] & np.isfinite(d2["dc"]).all(0)
            c["sensitivity_ambiguous_as_zero"] = {"cur": compare(v2, d2["do"], d2["dc"], days),
                                                  "n_accepted": int(d2["accept"].sum())}
            nz = ~d["zero"]
            c["split_by_best_shift"] = {"nonzero_shift": {"cur": compare(vc & nz, d["do"], d["dc"], days),
                                                          "geo": compare(vg & nz, d["do"], d["dg"], days)},
                                        "zero_shift_abs_d_cur_pctl_10_50_90": [round(float(v), 2) for v in np.nanpercentile(np.hypot(*d["dc"][:, vc & ~nz]), [10, 50, 90])],
                                        "nonzero_shift_abs_d_cur_pctl_10_50_90": [round(float(v), 2) for v in np.nanpercentile(np.hypot(*d["dc"][:, vc & nz]), [10, 50, 90])]}
            d3 = build_table(z, lag, T, S, rule="no_sig")
            v3 = d3["accept"] & np.isfinite(d3["dc"]).all(0); v3g = d3["accept"] & np.isfinite(d3["dg"]).all(0)
            c["sensitivity_no_significance_test"] = {"cur": compare(v3, d3["do"], d3["dc"], days),
                                                     "geo": compare(v3g, d3["do"], d3["dg"], days),
                                                     "n_accepted": int(d3["accept"].sum()),
                                                     "zero_shift_frac": round(float(d3["zero"][d3["accept"]].mean()), 3),
                                                     "abs_d_obs_pctl_10_25_50_75_90": [round(float(v), 2) for v in np.nanpercentile(np.hypot(*d3["do"][:, v3]), [10, 25, 50, 75, 90])]}
            c["by_region"] = {NAMES[k]: {"cur": compare(vc & (d["reg"] == k), d["do"], d["dc"], days),
                                         "geo": compare(vg & (d["reg"] == k), d["do"], d["dg"], days)}
                              for k in range(1, 8)}
            c["by_season"] = {s: {"cur": compare(vc & (d["season"] == s), d["do"], d["dc"], days),
                                  "geo": compare(vg & (d["season"] == s), d["do"], d["dg"], days)}
                              for s in ["DJF", "MAM", "JJA", "SON"]}
            sp = np.hypot(*d["dc"]); e = np.nanpercentile(sp[vc], [100 / 3, 200 / 3])
            gp = np.hypot(*d["dg"]); eg = np.nanpercentile(gp[vg], [100 / 3, 200 / 3])
            c["cur_tercile_edges_px"] = [round(float(x), 2) for x in e]
            c["geo_tercile_edges_px"] = [round(float(x), 2) for x in eg]
            c["by_cur_tercile"] = {f"T{j + 1}": {"cur": compare(vc & (np.digitize(sp, e) == j), d["do"], d["dc"], days)}
                                   for j in range(3)}
            c["by_geo_tercile"] = {f"T{j + 1}": {"geo": compare(vg & (np.digitize(gp, eg) == j), d["do"], d["dg"], days)}
                                   for j in range(3)}
        res["configs"][f"lag{lag}_T{T}_S{S}"] = c
    # lag 2 vs lag 1 on the same tile/day (both accepted, same T=16; lag-1 S6, lag-2 S12)
    a, b = tabs[(1, 16, 6)], tabs[(2, 16, 12)]
    k1 = {(dd, y, x): j for j, (dd, y, x) in enumerate(zip(a["day"], a["ty"], a["tx"])) if a["accept"][j]}
    pairs = [(k1[(dd, y, x)], j) for j, (dd, y, x) in enumerate(zip(b["day"], b["ty"], b["tx"])) if b["accept"][j] and (dd, y, x) in k1]
    i1, i2 = np.array(pairs).T
    res["lag2_vs_lag1"] = dict(n_pairs=len(pairs),
                               **{c: boot_slope(a["do"][q][i1], b["do"][q][i2], a["day"][i1]) for q, c in enumerate(["east", "north"])},
                               median_abs_lag1=round(float(np.median(np.hypot(*a["do"][:, i1]))), 2),
                               median_abs_lag2=round(float(np.median(np.hypot(*b["do"][:, i2]))), 2))
    with open(RES / "chl_motion.json", "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    plots(tabs, res, g, (i1, i2))
    print(json.dumps(res['configs']['lag1_T16_S6']['split_by_best_shift'], default=str))


# ============================== plots ==============================
def plots(tabs, res, g, pairs):
    d = tabs[(1, 16, 6)]; A = d["accept"]
    vc = A & np.isfinite(d["dc"]).all(0); vg = A & np.isfinite(d["dg"]).all(0)
    vs = A & np.isfinite(d["ds"]).all(0)
    C = OKABE_ITO
    # 1 magnitudes
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 8, 41)
    for lab, x, col in [(f"observed CHL shift (n={vc.sum()})", np.hypot(*d["do"][:, vc]), C[0]),
                        (f"reanalysis current (n={vc.sum()})", np.hypot(*d["dc"][:, vc]), C[1]),
                        (f"geostrophic from SSH (n={vg.sum()})", np.hypot(*d["dg"][:, vg]), C[2])]:
        ax[0].hist(np.clip(x, 0, 8), bins, histtype="step", lw=1.8, color=col, label=lab, density=True)
    ax[0].set(xlabel="displacement magnitude, px/day (1 px ≈ 4.5 km; clipped at 8)", ylabel="density",
              title="Tile displacement magnitude, day t to t+1 (16 px tiles)")
    ax[0].legend(fontsize=8)
    ax[1].hist2d(np.hypot(*d["dc"][:, vc]), np.hypot(*d["do"][:, vc]), bins=[np.linspace(0, 6, 31)] * 2, cmap="viridis", cmin=1)
    ax[1].plot([0, 6], [0, 6], "w--", lw=1)
    ax[1].set(xlabel="|d_cur| reanalysis current, px/day", ylabel="|d_obs| observed CHL shift, px/day",
              title=f"Observed vs current displacement magnitude (n={vc.sum()})")
    fig.tight_layout(); fig.savefig(FIG / "chl_motion_magnitudes.png", dpi=130); plt.close(fig)
    # 2 angle histograms
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a_, (lab, dr, sel) in zip(ax, [("reanalysis current", d["dc"], vc), ("geostrophic (SSH)", d["dg"], vg),
                                        ("current on a random other day", d["ds"], vs)]):
        st, dth = angle_stats(d["do"][:, sel], dr[:, sel])
        a_.hist(np.degrees(dth), np.linspace(-180, 180, 37), color=C[0], density=True)
        a_.set(xlabel="angle(d_obs) − angle(d_ref), degrees", title=f"vs {lab}\nR={st.get('R')}, n={st['n']}")
    ax[0].set_ylabel("density")
    fig.suptitle("Direction difference, observed CHL shift vs reference (|d| ≥ 0.5 px/day both; 16 px tiles, lag 1)")
    fig.tight_layout(); fig.savefig(FIG / "chl_motion_angles.png", dpi=130); plt.close(fig)
    # 3 component scatter
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    for r_, (lab, dr, sel, key) in enumerate([("reanalysis current", d["dc"], vc, "cur"), ("geostrophic (SSH)", d["dg"], vg, "geo")]):
        for q, comp in enumerate(["east", "north"]):
            a_ = ax[r_, q]; x, y = dr[q][sel], d["do"][q][sel]
            a_.hist2d(x, y, bins=[np.linspace(-6, 6, 49)] * 2, cmap="viridis", cmin=1)
            s = res["configs"]["lag1_T16_S6"]["overall"][key][comp]
            xx = np.array([-6, 6]); a_.plot(xx, s["intercept"] + s["slope"] * xx, color=C[1], lw=2, label=f"OLS slope {s['slope']}")
            a_.plot(xx, xx, "w--", lw=1, label="1:1")
            a_.set(xlabel=f"{comp} component, {lab}, px/day", ylabel=f"{comp} component, observed CHL shift, px/day",
                   title=f"{comp}: observed shift vs {lab} (n={s['n']})"); a_.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "chl_motion_components.png", dpi=130); plt.close(fig)
    # 4 slopes by group
    c = res["configs"]["lag1_T16_S6"]
    groups = [("all", c["overall"])] + [(k, v) for k, v in c["by_season"].items()] + \
             [(k.split(" ")[0].split("/")[0], v) for k, v in c["by_region"].items()]
    groups += [(f"|d_cur| {k}", v) for k, v in c["by_cur_tercile"].items()]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    xs = np.arange(len(groups))
    for a_, key, lab in [(ax[0], "cur", "reanalysis current"), (ax[1], "geo", "geostrophic (SSH)")]:
        for q, (comp, off) in enumerate([("east", -0.15), ("north", 0.15)]):
            for j, (gname, v) in enumerate(groups):
                s = v.get(key, {}).get(comp, {})
                if "slope" not in s: continue
                a_.errorbar(j + off, s["slope"], yerr=[[s["slope"] - s["ci95"][0]], [s["ci95"][1] - s["slope"]]],
                            fmt="o", color=C[q], label=comp if j == 0 else None)
                a_.annotate(str(s["n"]), (j + off, s["ci95"][1]), fontsize=6, ha="center", va="bottom")
        a_.axhline(0, color="k", lw=.6); a_.axhline(1, color="grey", lw=.6, ls="--")
        a_.set(ylabel="OLS slope d_obs on d_ref", title=f"Effective current factor (slope of observed shift on {lab}) by group, 16 px tiles, lag 1; numbers = n tiles")
        a_.legend(fontsize=8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([g_[0] for g_ in groups], rotation=35, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "chl_motion_slopes_by_group.png", dpi=130); plt.close(fig)
    # 5 lag 2 vs lag 1
    a, b = tabs[(1, 16, 6)], tabs[(2, 16, 12)]; i1, i2 = pairs
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for q, comp in enumerate(["east", "north"]):
        ax[q].hist2d(a["do"][q][i1], b["do"][q][i2], bins=[np.linspace(-6, 6, 25), np.linspace(-12, 12, 49)], cmap="viridis", cmin=1)
        ax[q].plot([-6, 6], [-12, 12], "w--", lw=1, label="lag2 = 2 × lag1")
        ax[q].set(xlabel=f"{comp} shift t→t+1, px", ylabel=f"{comp} shift t→t+2, px",
                  title=f"{comp}: lag-2 vs lag-1 observed shift (n={len(i1)})"); ax[q].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "chl_motion_lag2_vs_lag1.png", dpi=130); plt.close(fig)
    # 6 example tiles
    examples(d, g)


def examples(d, g):
    H, W = g["ocean"].shape; T = 16; gm = tile_geom(H, W, T)
    A = d["accept"] & np.isfinite(d["dc"]).all(0)
    _, dth = None, np.angle(np.exp(1j * (np.arctan2(d["do"][1], d["do"][0]) - np.arctan2(d["dc"][1], d["dc"][0]))))
    big = (np.hypot(*d["do"]) >= 1.5) & (np.hypot(*d["dc"]) >= 1.5)
    rng = np.random.default_rng(1)
    picks = []
    for lab, m in [("agrees with current", A & big & (np.abs(dth) < np.pi / 6)),
                   ("disagrees with current", A & big & (np.abs(dth) > 2 * np.pi / 3)),
                   ("zero shift, current ≥ 1.5 px/day", A & d["zero"] & (np.hypot(*d["dc"]) >= 1.5)),
                   ("random accepted tile", A)]:
        idx = np.flatnonzero(m)
        if len(idx): picks.append((lab, rng.choice(idx)))
    fig, ax = plt.subplots(len(picks), 2, figsize=(8.5, 4 * len(picks)))
    ax = np.atleast_2d(ax)
    M = 8
    for r_, (lab, j) in enumerate(picks):
        day, ty, tx = d["day"][j], d["ty"][j], d["tx"][j]
        y0, x0 = gm[0] + ty * T, gm[1] + tx * T
        ys, xs = slice(max(y0 - M, 0), min(y0 + T + M, H)), slice(max(x0 - M, 0), min(x0 + T + M, W))
        f0, f1 = ln_chl(g["chl_l3"][day])[ys, xs], ln_chl(g["chl_l3"][day + 1])[ys, xs]
        vmin, vmax = np.nanpercentile(np.r_[f0.ravel(), f1.ravel()], [2, 98])
        cy, cx = y0 - ys.start + T / 2 - .5, x0 - xs.start + T / 2 - .5
        for c_, (f, title) in enumerate([(f0, "day t"), (f1, "day t+1")]):
            a_ = ax[r_, c_]
            im = a_.imshow(f, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            a_.add_patch(plt.Rectangle((x0 - xs.start - .5, y0 - ys.start - .5), T, T, fill=False, ec="w", lw=1.5))
            if c_ == 1:
                a_.add_patch(plt.Rectangle((x0 - xs.start - .5 + d["sx"][j], y0 - ys.start - .5 + d["sy"][j]), T, T,
                                           fill=False, ec=OKABE_ITO[3], lw=1.5, ls="--"))
                a_.quiver(cx, cy, d["do"][0][j], d["do"][1][j], color=OKABE_ITO[3], angles="xy", scale_units="xy", scale=1/2, width=.012, label="observed shift ×2")
                a_.quiver(cx, cy, d["dc"][0][j], d["dc"][1][j], color=OKABE_ITO[1], angles="xy", scale_units="xy", scale=1/2, width=.012, label="current ×2")
                a_.legend(fontsize=7, loc="lower right")
            a_.set(title=f"{lab}: {title} {str(g['time'][day + c_])[:10]}\nr_peak={d['rpk'][j]:.2f}, r0={d['r0'][j]:.2f}, "
                         f"d_obs=({d['do'][0][j]:.1f},{d['do'][1][j]:.1f}) d_cur=({d['dc'][0][j]:.1f},{d['dc'][1][j]:.1f})",
                   xlabel="column (east →)", ylabel="row (north ↑)")
            a_.title.set_fontsize(7)
            plt.colorbar(im, ax=a_, fraction=.046, label="ln CHL")
    fig.suptitle("Example 16 px tiles: ln CHL on day t and t+1; white = tile, dashed = best-match position\n(arrows: observed shift and tile-mean reanalysis current, both ×2, px/day)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "chl_motion_examples.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
