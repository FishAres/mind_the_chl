"""Diagnostics for the learned Lagrangian tracer (chl_cnn/tracer.py, train_tracer.py; both imported read-only, nothing
there is changed). Used by analyses/tracer/diag_*.py.

  setup()          train_tracer.load with the decomp climatology (CLIMD, for the geo_mean component), val / train starts.
  make_vel()       base velocity fields at factor 1 (px/day): reanalysis (optionally shifted by +-n days) or the step-1b
                   geo_mean component (train-years doy climatology of u_geo, x current validity).
  depart_sample_mode()  departure-point sampling of ln C with grid_sample nearest / bilinear / bicubic. Validity always
                   from the bilinear observed fraction (>= min_obs, as tracer.Integrator.depart_sample); strict=True also
                   requires the whole 4 x 4 bicubic support (rows/cols floor-1..floor+2) observed.
  scan()           loss landscape of the scalar model u_eff = a x base field over a grid of a (no training): per lead,
                   pixel L1 and band-passed L1 (tracer.BandPass), Pearson r of prediction vs target (pixels and band),
                   on a fixed pixel set (target observed and prediction valid for every a and every sampling mode) and on
                   each a's own valid set; optional extra pixel subsets intersected with the fixed set.
  Synth            synthetic ln CHL: base + Gaussian random anomaly with a ring spectrum calibrated to a target, advected
                   with truth_alpha x u_rea by backward trajectories (n_sub 8, bicubic sampling), observed through the real
                   cloud masks; optional iid noise, day-specific correlated noise, static (pinned) part, Lagrangian decay.
  train_scalar()   scalar a = 1.5 sigmoid(theta) (= the bias of tracer.Factor with lo 0, hi 1.5), Adam, loss as
                   train_tracer (pixel L1, mean per lead, leads 1-5 equal weights).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from . import tracer as TR
from . import train_tracer as TT

ALPHAS = sorted(set(np.round(np.arange(0.0, 1.2001, 0.05), 3).tolist() + [0.17]))
SETS_BASE = ("fixed", "own")
# accumulator fields: n, sum p, sum t, sum p^2, sum t^2, sum p t, sum |p - t|
NF = 7


# ============================== data ==============================
def setup(dev, K=5):
    a = TT.parse(["--run", "diag", "--vel-param", "decomp", "--terms=", "--leads", str(K)])
    d = TT.load(a, dev)
    return a, d


def calendar_ok(d, s, K, pre=1, post=1):
    """Start indices s whose store days s-pre .. s+K+post are consecutive calendar days."""
    dn = np.asarray(d["times"]).astype("datetime64[D]").astype(int)
    n = len(dn)
    keep = []
    for t in s:
        lo, hi = t - pre, t + K + post
        if lo < 0 or hi >= n:
            continue
        if dn[hi] - dn[lo] == hi - lo:
            keep.append(t)
    return np.array(keep, dtype=np.int64)


def make_vel(d, kind="rea", shift=0, K=5):
    H, W = d["ocean"].shape

    def f(days):
        B = days.shape[0]
        fl = (days + shift).reshape(-1)
        ok = d["CUROK"][fl].float()
        if kind == "rea":
            ue, ve = d["U"][fl].float(), d["V"][fl].float()
        elif kind == "geo_mean":
            c = d["CLIMD"][d["DOYI"][fl]]
            ue, ve = c[:, 0] * ok, c[:, 1] * ok
        else:
            raise ValueError(kind)
        U = torch.stack([ue * d["CPDX"], ve * d["CPDY"]], 1).view(B, K + 1, 2, H, W)
        return U, ok.view(B, K + 1, H, W)
    return f


def real_data(d):
    return lambda bt, days: (d["Y"][days], d["OBS"][days])


# ============================== sampling ==============================
def erode_support4(obs):
    """obs (B, H, W) bool -> True where rows i-1..i+2 and cols j-1..j+2 are all observed (outside grid = unobserved)."""
    x = F.pad((~obs).float()[:, None], (1, 2, 1, 2), value=1.0)
    return F.max_pool2d(x, 4, stride=1)[:, 0] < 0.5


def sample_at_floor(mask, pos):
    B, H, W = mask.shape
    ix, iy = pos[..., 0].floor().long(), pos[..., 1].floor().long()
    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    idx = (iy.clamp(0, H - 1) * W + ix.clamp(0, W - 1)).view(B, -1)
    return torch.gather(mask.reshape(B, -1), 1, idx).view(B, H, W) & inside


def depart_sample_mode(it, y, obs, pos, mode="bilinear", strict=False):
    if mode == "bilinear" and not strict:
        return it.depart_sample(y, obs, pos)
    den = it.sample(obs.float()[:, None], pos)[:, 0]
    valid = den >= it.min_obs
    yz = torch.where(obs, y, torch.zeros_like(y))[:, None]
    if mode == "bilinear":
        val = it.sample(yz, pos)[:, 0] / den.clamp(min=1e-3)
    else:
        val = F.grid_sample(yz, pos * it.scale - 1.0, mode=mode, padding_mode="zeros", align_corners=True)[:, 0]
    if strict:
        valid = valid & sample_at_floor(erode_support4(obs), pos)
    return val, valid


# ============================== landscape scan ==============================
def _acc(p, t, m):
    m = m.double()
    p = torch.where(m > 0, p.double(), 0.0); t = torch.where(m > 0, t.double(), 0.0)
    return torch.stack([m.sum(), p.sum(), t.sum(), (p * p).sum(), (t * t).sum(), (p * t).sum(), (p - t).abs().sum()])


def _acc_rows(p, t, m):
    """per batch row: sum |p - t| and n -> (B, 2)."""
    m = m.double()
    e = torch.where(m > 0, (p.double() - t.double()).abs(), 0.0)
    return torch.stack([e.sum((1, 2)), m.sum((1, 2))], 1)


@torch.no_grad()
def scan(d, it, starts, alphas, vel_fn, data_fn, modes=("bilinear",), strict=False, band=True, subsets_fn=None, K=5,
         batch=16, bp=None, log=print):
    """Returns dict(stats={mode: {set: (n_a, K, 2[pix, band], NF)}}, perday={mode: (n_starts, n_a, K, 2[pix, band], 2)}
    on the fixed set, nonfinite=count of non-finite predictions inside own masks, n_fixed_px per lead)."""
    dev = d["Y"].device
    H, W = d["ocean"].shape
    na = len(alphas)
    sets = list(SETS_BASE) + (["fixed_moving", "fixed_zero"] if subsets_fn is not None else [])
    stats = {m: {s: torch.zeros(na, K, 2, NF, dtype=torch.float64, device=dev) for s in sets} for m in modes}
    perday = {m: torch.zeros(len(starts), na, K, 2, 2, dtype=torch.float64, device=dev) for m in modes}
    nonfinite = 0
    nfix = torch.zeros(K, dtype=torch.float64, device=dev)
    for b0 in range(0, len(starts), batch):
        bt = starts[b0:b0 + batch]; B = len(bt)
        days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev)
        U, ok = vel_fn(days)
        Y, OBS = data_fn(bt, days)
        y0, o0 = Y[:, 0], OBS[:, 0]
        sub = subsets_fn(bt) if subsets_fn is not None else None
        for k in range(1, K + 1):
            truth, tobs = Y[:, k], OBS[:, k]
            preds, owns = {}, {}
            fixed = tobs.clone()
            for ai, al in enumerate(alphas):
                pos, val = it.departure(U * al, ok, k)
                for m in modes:
                    pr, pv = depart_sample_mode(it, y0, o0, pos, m, strict)
                    own = val & pv & tobs
                    preds[m, ai], owns[m, ai] = pr, own
                    fixed &= own
                    nonfinite += int((~torch.isfinite(pr) & own).sum())
            nfix[k - 1] += fixed.double().sum()
            if band:
                tbf, vbf = bp(truth, fixed)
            for m in modes:
                for ai in range(na):
                    pr, own = preds[m, ai], owns[m, ai]
                    st = stats[m]
                    st["fixed"][ai, k - 1, 0] += _acc(pr, truth, fixed)
                    st["own"][ai, k - 1, 0] += _acc(pr, truth, own)
                    perday[m][b0:b0 + B, ai, k - 1, 0] += _acc_rows(pr, truth, fixed)
                    if sub is not None:
                        st["fixed_moving"][ai, k - 1, 0] += _acc(pr, truth, fixed & sub[0])
                        st["fixed_zero"][ai, k - 1, 0] += _acc(pr, truth, fixed & sub[1])
                    if band:
                        pbf, _ = bp(pr, fixed)
                        st["fixed"][ai, k - 1, 1] += _acc(pbf, tbf, vbf)
                        perday[m][b0:b0 + B, ai, k - 1, 1] += _acc_rows(pbf, tbf, vbf)
                        if sub is not None:
                            st["fixed_moving"][ai, k - 1, 1] += _acc(pbf, tbf, vbf & sub[0])
                            st["fixed_zero"][ai, k - 1, 1] += _acc(pbf, tbf, vbf & sub[1])
                        pbo, vbo = bp(pr, own); tbo, _ = bp(truth, own)
                        st["own"][ai, k - 1, 1] += _acc(pbo, tbo, vbo)
        if log and (b0 // batch) % 5 == 0:
            log(f"  scan batch {b0 // batch + 1}/{(len(starts) + batch - 1) // batch}")
    return dict(stats={m: {s: v.cpu().numpy() for s, v in stats[m].items()} for m in modes},
                perday={m: v.cpu().numpy() for m, v in perday.items()}, nonfinite=nonfinite, n_fixed_px=nfix.cpu().numpy(),
                alphas=np.asarray(alphas), sets=sets, modes=list(modes))


def curve_metrics(acc):
    """acc (..., NF) -> mae, pearson r, n."""
    n, sp, st, spp, stt, spt, sab = np.moveaxis(acc, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = sab / n
        cov = spt / n - sp / n * st / n
        r = cov / np.sqrt((spp / n - (sp / n) ** 2) * (stt / n - (st / n) ** 2))
    return mae, r, n


def vertex(x, y, find="min"):
    """Grid arg-extremum and the parabolic vertex through it and its two neighbours (NaN at the grid edge)."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    i = int(np.nanargmin(y) if find == "min" else np.nanargmax(y))
    if 0 < i < len(x) - 1:
        c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
        v = -c[1] / (2 * c[0]) if c[0] != 0 else np.nan
    else:
        v = np.nan
    return float(x[i]), float(v)


# ============================== patch subsets ==============================
def patch_masks(cache_path, start_days, crop, H, W, P=32, r_min=0.6):
    """Lead-1 accepted masked-NCC patch pairs (spectral_motion cache) on start day t: moving = union of patches with
    |shift| >= 1 px, zero = union of patches with integer peak (0, 0) (|fx|, |fy| <= 0.5). Pixels in both -> neither.
    Returns dict day -> (moving (H, W) bool, zero (H, W) bool) and counts."""
    z = np.load(cache_path)
    sel = (z["k"] == 1) & (z["null"] == 0) & z["okpk"] & ~z["edge"] & (z["rpk"] >= r_min)
    i, p, fx, fy = z["i"][sel], z["p"][sel], z["fx"][sel], z["fy"][sel]
    pos = z["pos"]; pidx = z["pidx"]
    col = np.searchsorted(pidx, p)
    want = set(int(t) for t in start_days)
    out = {}
    cnt = dict(n_moving_patches=0, n_zero_patches=0, n_other_patches=0, px_both=0)
    for q in np.flatnonzero(np.isin(i, list(want))):
        t = int(i[q])
        if t not in out:
            out[t] = [np.zeros((H, W), bool), np.zeros((H, W), bool)]
        r, c = pos[col[q]]
        r0, c0 = r - crop[0], c - crop[2]
        sl = (slice(max(r0, 0), max(min(r0 + P, H), 0)), slice(max(c0, 0), max(min(c0 + P, W), 0)))
        mag = math.hypot(fx[q], fy[q])
        if mag >= 1.0:
            out[t][0][sl] = True; cnt["n_moving_patches"] += 1
        elif abs(fx[q]) <= 0.5 and abs(fy[q]) <= 0.5:
            out[t][1][sl] = True; cnt["n_zero_patches"] += 1
        else:
            cnt["n_other_patches"] += 1
    for t, (mv, zr) in out.items():
        both = mv & zr
        cnt["px_both"] += int(both.sum())
        mv &= ~both; zr &= ~both
    cnt["n_days_with_patches"] = len(out)
    return out, cnt


# ============================== synthetic fields ==============================
def ring_index(P=32):
    fy, fx = np.meshgrid(np.fft.fftfreq(P), np.fft.fftfreq(P), indexing="ij")
    return np.rint(np.hypot(fy, fx) * P).astype(int)


def hann2(P=32):
    return np.outer(np.hanning(P + 2)[1:-1], np.hanning(P + 2)[1:-1])


def ring_power_tiles(f, P=32, NR=16):
    """f (N, Hp, Wp) torch, fully defined -> mean ring power (NR,) over all non-overlapping P x P tiles, with the
    spectral_motion prep (Hann-weighted mean removed, Hann window, FFT, ring sums of |X|^2, rings 1..NR)."""
    N, Hp, Wp = f.shape
    t = f[:, :Hp // P * P, :Wp // P * P].reshape(N, Hp // P, P, Wp // P, P).permute(0, 1, 3, 2, 4).reshape(-1, P, P)
    win = torch.as_tensor(hann2(P), dtype=t.dtype, device=t.device)
    mu = (t * win).sum((1, 2), keepdim=True) / win.sum()
    X = torch.fft.fft2((t - mu) * win)
    pw = X.abs() ** 2
    ring = torch.as_tensor(ring_index(P), device=t.device)
    return torch.stack([pw[:, ring == m].sum(1).mean() for m in range(1, NR + 1)]).double().cpu().numpy()


class Synth:
    """Synthetic ln CHL observations for the scan (data_fn). spec: dict with ring targets (16 rings, wavelength
    32 / m px) 'signal', 'noise' (correlated noise), 'efold_days'; case: dict(base, static_frac, iid_sigma, corr_noise,
    decay). Each start day gets its own realisation (seeded by start index; the signal realisation is identical across
    cases)."""

    def __init__(self, d, dev, spec, case, K=5, seed=0, truth_alpha=0.5, n_sub_truth=8, Hp=256, Wp=384, P=32):
        self.d, self.dev, self.case, self.K, self.seed, self.P = d, dev, case, K, seed, P
        self.H, self.W = d["ocean"].shape
        self.Hp, self.Wp = Hp, Wp
        qy, qx = np.meshgrid(np.fft.fftfreq(Hp), np.fft.fftfreq(Wp), indexing="ij")
        self.q = np.hypot(qy, qx)
        self.it = TR.Integrator(self.H, self.W, dev, n_sub=n_sub_truth, vmax=10.0)
        self.vel = make_vel(d, "rea", 0, K)
        self.truth_alpha = truth_alpha
        self.calib = {}
        self.amp_sig = self.calibrate(np.asarray(spec["signal"])); self.calib["signal"] = self.last_check
        self.amp_noise = None
        if case.get("corr_noise"):
            self.amp_noise = self.calibrate(np.asarray(spec["noise"])); self.calib["noise"] = self.last_check
        tau = self.tau_of_q(np.asarray(spec["efold_days"]))
        self.r = torch.as_tensor(np.exp(-1.0 / tau), device=dev, dtype=torch.float32)

    def density(self, ring_vals):
        """Spectral density on the padded grid from ring values at q_m = m / P (log-log linear; flat below q_1;
        extrapolated with the last three rings' slope above q_16); 0 at q = 0."""
        P = self.P
        qm = np.arange(1, len(ring_vals) + 1) / P
        lv = np.log(np.maximum(ring_vals, 1e-30))
        lq = np.log(np.clip(self.q, qm[0], None))
        D = np.interp(lq, np.log(qm), lv)
        hi = self.q > qm[-1]
        s = np.polyfit(np.log(qm[-3:]), lv[-3:], 1)[0]
        D[hi] = lv[-1] + s * (np.log(self.q[hi]) - np.log(qm[-1]))
        D = np.exp(D); D[0, 0] = 0.0
        return D

    def tau_of_q(self, efold):
        P = self.P
        qm = np.arange(1, len(efold) + 1) / P
        tq = np.interp(np.log(np.clip(self.q, qm[0], qm[-1])), np.log(qm), np.asarray(efold, float))
        return tq

    def _grf(self, amp, seeds):
        out = []
        for s in seeds:
            g = torch.Generator(device=self.dev); g.manual_seed(int(s))
            wn = torch.randn(self.Hp, self.Wp, generator=g, device=self.dev)
            out.append(torch.fft.fft2(wn) * amp)
        return torch.stack(out)                                 # spectra (B, Hp, Wp)

    def _real(self, Xs):
        return torch.fft.ifft2(Xs).real[:, :self.H, :self.W].float()

    def calibrate(self, target, n=48, iters=3):
        """Ring values -> amplitude sqrt(density) whose measured tile ring power matches target."""
        vals = np.asarray(target, float).copy()
        for _ in range(iters):
            amp = torch.as_tensor(np.sqrt(self.density(vals)), device=self.dev, dtype=torch.complex64)
            f = torch.fft.ifft2(self._grf(amp, range(10_000_000, 10_000_000 + n))).real
            meas = ring_power_tiles(f, self.P)
            vals = vals * target / meas
        amp = torch.as_tensor(np.sqrt(self.density(vals)), device=self.dev, dtype=torch.complex64)
        f = torch.fft.ifft2(self._grf(amp, range(20_000_000, 20_000_000 + n))).real
        self.last_check = dict(target=np.asarray(target).tolist(), measured=ring_power_tiles(f, self.P).tolist())
        return amp

    def seeds(self, bt, stream):
        return [self.seed * 1_000_003 + int(t) * 101 + stream for t in bt]

    @torch.no_grad()
    def __call__(self, bt, days):
        c, K, d = self.case, self.K, self.d
        B = len(bt)
        sig = self._grf(self.amp_sig, self.seeds(bt, 0))
        f_st = float(c.get("static_frac", 0.0))
        lag = []                                               # Lagrangian-label fields D_0..D_K (B, H, W)
        Dk = sig
        lag.append(self._real(Dk))
        for j in range(1, K + 1):
            if c.get("decay"):
                inn = self._grf(self.amp_sig, self.seeds(bt, 10 + j))
                Dk = self.r * Dk + torch.sqrt(1 - self.r ** 2) * inn
            lag.append(self._real(Dk))
        static = self._real(self._grf(self.amp_sig, self.seeds(bt, 1))) if f_st > 0 else None
        U, ok = self.vel(days)
        U = U * self.truth_alpha
        Y = torch.empty(B, K + 1, self.H, self.W, device=self.dev)
        OBS = d["OBS"][days].clone()
        Y[:, 0] = lag[0]
        for k in range(1, K + 1):
            pos, val = self.it.departure(U, ok, k)
            Y[:, k] = F.grid_sample(lag[k][:, None], pos * self.it.scale - 1.0, mode="bicubic", padding_mode="zeros",
                                    align_corners=True)[:, 0]
            OBS[:, k] &= val
        if f_st > 0:
            Y = math.sqrt(1 - f_st) * Y + math.sqrt(f_st) * static[:, None]
        if c.get("base", True):
            Y = Y + d["CLIM"][d["BIN"][days]]
        for k in range(K + 1):
            if c.get("iid_sigma", 0.0) > 0:
                g = torch.Generator(device=self.dev)
                for b, s in enumerate(self.seeds(bt, 30 + k)):
                    g.manual_seed(s)
                    Y[b, k] += c["iid_sigma"] * torch.randn(self.H, self.W, generator=g, device=self.dev)
            if self.amp_noise is not None:
                Y[:, k] += self._real(self._grf(self.amp_noise, self.seeds(bt, 50 + k)))
        return Y, OBS


# ============================== scalar training ==============================
def train_scalar(d, it, base_vel, tr_s, va_s, a0, lr, epochs, batch=16, seed=0, K=5, eval_every=51, log=print, data_fn=None):
    """Scalar a = 1.5 sigmoid(theta), init a0. data_fn(bt, days) -> (Y, OBS) (B, K+1, H, W); default the store. Returns dict(steps, a (per step, after the update), loss (per step),
    eval_steps, val_mae (per eval, (K,) own set, x per lead), val_n)."""
    dev = d["Y"].device
    data_fn = data_fn or real_data(d)
    rng = np.random.default_rng(seed)
    p0 = a0 / 1.5
    theta = torch.tensor(math.log(p0 / (1 - p0)), device=dev, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    out = dict(a=[], loss=[], eval_steps=[], val_mae=[], val_n=[], val_a=[])

    @torch.no_grad()
    def val():
        al = float(1.5 * torch.sigmoid(theta))
        acc = np.zeros((K, 2))
        for b0 in range(0, len(va_s), 16):
            bt = va_s[b0:b0 + 16]
            days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev)
            U, ok = base_vel(days)
            Y, OBS = data_fn(bt, days)
            for k in range(1, K + 1):
                pos, v = it.departure(U * al, ok, k)
                pr, pv = it.depart_sample(Y[:, 0], OBS[:, 0], pos)
                m = v & pv & OBS[:, k]
                acc[k - 1] += [float((pr - Y[:, k]).abs()[m].double().sum()), float(m.sum())]
        return acc[:, 0] / acc[:, 1], acc[:, 1], al

    step = 0
    for ep in range(epochs):
        perm = rng.permutation(len(tr_s)); nb = len(perm) // batch
        for bi in range(nb):
            if step % eval_every == 0:
                mae, n, al = val()
                out["eval_steps"].append(step); out["val_mae"].append(mae.tolist()); out["val_n"].append(n.tolist()); out["val_a"].append(al)
            bt = tr_s[perm[bi * batch:(bi + 1) * batch]]
            days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev)
            U, ok = base_vel(days)
            Y, OBS = data_fn(bt, days)
            al = 1.5 * torch.sigmoid(theta)
            ls = []
            for k in range(1, K + 1):
                pos, v = it.departure(U * al, ok, k)
                pr, pv = it.depart_sample(Y[:, 0], OBS[:, 0], pos)
                m = v & pv & OBS[:, k]
                ls.append((pr - Y[:, k]).abs()[m].mean())
            loss = torch.stack(ls).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            out["a"].append(float(1.5 * torch.sigmoid(theta))); out["loss"].append(float(loss))
        if log:
            log(f"  a0 {a0} lr {lr} epoch {ep + 1} a {out['a'][-1]:.4f} loss {np.mean(out['loss'][-nb:]):.5f}")
    mae, n, al = val()
    out["eval_steps"].append(step); out["val_mae"].append(mae.tolist()); out["val_n"].append(n.tolist()); out["val_a"].append(al)
    out["steps_per_epoch"] = nb
    return out
