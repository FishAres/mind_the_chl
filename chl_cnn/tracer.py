"""Learned Lagrangian tracer model of surface ln CHL. Step 1: velocity terms only; step 2: plus a pointwise source term
(SourceModel, train_tracer.py --source). No latent channels. Used by train_tracer.py; tests in test_tracer.py.

    u_eff(x, t) = a(x, s) u_rea + b(x, s) u_geo + c(s) R(theta(s)) tau + du_coarse(x, t)        (m/s, then px/day)

  geostrophic  u_g = -g/f d(eta)/dy, v_g = g/f d(eta)/dx from the model SSH eta (CMEMS zos), centred differences in
               metres; pixels within `margin` px of land or of a non-finite eta -> 0, with a validity mask.
  VelocityModel a, b: 1x1-conv MLPs of (static geography, doy sin/cos) -> lo + (hi - lo) sigmoid(.), last layer zero so
               a = a0, b = b0 everywhere at the start. Wind: complex coefficient w(s) = p(s) + i q(s), p, q first-order
               harmonics of the day of year, applied to tau = tau_x + i tau_y (u + i v = w tau), i.e. c(s) = |w|,
               theta(s) = arg w (counter-clockwise rotation), units m/s per Pa; starts at 0. du_coarse: 3x3-conv net on
               physics inputs average-pooled to cells of `cell` px, bilinearly upsampled; last layer zero.
               No CHL input anywhere.
  Integrator   backward trajectories from every pixel at time t+k to time t (time in days, daily fields at integer
               times, linear interpolation in time), RK2 (midpoint), n_sub sub-steps per day, bilinear sampling
               (F.grid_sample, align_corners=True). Velocity components clipped to +-vmax px/day. The validity field
               (ocean & valid currents) is sampled at every velocity evaluation; a trajectory with any sample < min_valid
               (land, missing currents, outside the grid) is invalid.
  depart_sample ln C_obs(t) sampled at the departure point as sum(w C obs) / sum(w obs); valid only if the sampled
               observed fraction is >= min_obs (0.99, as ude.Transport).
  SourceModel  (step 2) S = g_max tanh(MLP(inputs)), 1x1 convs only (pointwise: no neighbourhood or gradient inputs),
               last layer zero so S = 0 at the start. Integrated along the backward trajectory, forward in time from the
               departure point: c_0 = ln C_obs(departure), c_{i+1} = c_i + h S(features at the sub-step midpoint position
               and time, c_i), h = 1 / n_sub day (the midpoints are those of the backward RK2 step; c enters at the
               start of the sub-step, explicit). departure(..., path=list) records the midpoints; it does not change
               the trajectory computation.
  Step 1b (train_tracer.py --vel-param decomp / --loss-band; appended at the end of this file, unused otherwise)
  DecompVelocity  u_eff = sum_j alpha_j(region, season) u_j, u_j in DECOMP_COMPS = geostrophic and ageostrophic
               (u_rea - u_geo) velocities, each split into a train-years day-of-year climatology (decomp_clim) and the
               daily anomaly (decomp_components); alpha free scalars per (component, region, season), init 0.25.
  BandPass     mask-aware (normalised-convolution) Difference-of-Gaussians band-pass used for the band-passed loss.
"""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation

G = 9.81


def geostrophic(z, dx, dy, f, ocean, margin=2):
    """z (T, H, W) SSH in m (NaN where missing); dx, dy (H, W) signed metres along increasing index; f (H, W) 1/s.
    Returns ug, vg (m/s, float32; 0 where invalid) and the validity mask (bool): ocean, all four centred-difference
    neighbours finite, and further than `margin` px from land / non-finite eta."""
    z = np.where(ocean[None], np.asarray(z, "float64"), np.nan)
    dzx = np.full_like(z, np.nan); dzy = np.full_like(z, np.nan)
    dzx[:, :, 1:-1] = (z[:, :, 2:] - z[:, :, :-2]) / (2 * dx[None, :, 1:-1])
    dzy[:, 1:-1, :] = (z[:, 2:, :] - z[:, :-2, :]) / (2 * dy[None, 1:-1, :])
    ug, vg = -G / f[None] * dzy, G / f[None] * dzx
    st = np.ones((2 * margin + 1, 2 * margin + 1), bool)
    bad = np.stack([binary_dilation(~np.isfinite(zz), structure=st) for zz in z]) | ~np.isfinite(ug) | ~np.isfinite(vg)
    ok = ~bad & ocean[None]
    return np.where(ok, ug, 0.0).astype("float32"), np.where(ok, vg, 0.0).astype("float32"), ok


class Integrator:
    """Backward RK2 trajectories on a fixed (H, W) grid."""

    def __init__(self, H, W, device, n_sub=4, vmax=10.0, min_valid=0.99, min_obs=0.99):
        gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        self.base = torch.stack([gx, gy], -1).float()                        # (H, W, 2) cells (x = col, y = row)
        self.scale = torch.tensor([2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1)], device=device)
        self.n_sub, self.vmax, self.min_valid, self.min_obs = n_sub, vmax, min_valid, min_obs

    def sample(self, f, pos):
        """f (B, C, H, W) at pos (B, H, W, 2) in cells -> (B, C, H, W); zero outside the grid."""
        return F.grid_sample(f, pos * self.scale - 1.0, mode="bilinear", padding_mode="zeros", align_corners=True)

    def departure(self, U, ok, k, path=None):
        """U (B, D, 2, H, W) velocity in px/day (x = col, y = row) of days t..t+D-1 (D >= k + 1); ok (B, D, H, W) float
        validity. Trajectories start at every pixel at time k and run back to time 0.
        Returns pos (B, H, W, 2) departure points in cells and valid (B, H, W) bool. If path is a list, appends
        (mid position (B, H, W, 2), day interval j, weight of day j + 1 at the midpoint time) for every sub-step, in
        backward order."""
        B = U.shape[0]
        U = U.clamp(-self.vmax, self.vmax)
        pos = self.base[None].expand(B, -1, -1, -1)
        valid = torch.ones(pos.shape[:3], dtype=torch.bool, device=pos.device)
        h = 1.0 / self.n_sub
        for j in range(k - 1, -1, -1):                       # day interval [j, j + 1]
            fld = torch.cat([U[:, j], U[:, j + 1], ok[:, j, None], ok[:, j + 1, None]], 1)       # (B, 6, H, W)

            def vel(p, w):                                   # w: weight of day j + 1 (time within the interval)
                s = self.sample(fld, p)
                v = (1 - w) * s[:, 0:2] + w * s[:, 2:4]
                o = (1 - w) * s[:, 4] + w * s[:, 5]
                return v.permute(0, 2, 3, 1), o
            for i in range(self.n_sub):
                w0 = 1.0 - i * h                             # time j + w0 at the start of the sub-step (going back)
                v1, o1 = vel(pos, w0)
                mid = pos - 0.5 * h * v1
                v2, o2 = vel(mid, w0 - 0.5 * h)
                if path is not None:
                    path.append((mid, j, w0 - 0.5 * h))
                valid = valid & (o1 >= self.min_valid) & (o2 >= self.min_valid)
                pos = pos - h * v2
        return pos, valid

    def depart_sample(self, y, obs, pos):
        """y (B, H, W) ln CHL (any value where unobserved), obs (B, H, W) bool at day t; pos from departure().
        Returns (prediction, valid) with valid = sampled observed fraction >= min_obs."""
        of = obs.float()
        s = self.sample(torch.stack([torch.where(obs, y, torch.zeros_like(y)), of], 1), pos)
        den = s[:, 1]
        return s[:, 0] / den.clamp(min=1e-3), den >= self.min_obs


def _mlp(in_ch, hidden, out0):
    net = nn.Sequential(nn.Conv2d(in_ch, hidden, 1), nn.GELU(), nn.Conv2d(hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
    nn.init.zeros_(net[-1].weight); nn.init.constant_(net[-1].bias, out0)
    return net


class Factor(nn.Module):
    """Bounded pointwise factor lo + (hi - lo) sigmoid(MLP(x)), equal to init everywhere at the start."""

    def __init__(self, in_ch, hidden, init, lo, hi):
        super().__init__()
        self.lo, self.hi = lo, hi
        p = (init - lo) / (hi - lo)
        self.net = _mlp(in_ch, hidden, math.log(p / (1 - p)))

    def forward(self, x):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.net(x)[:, 0])


class Coarse(nn.Module):
    """du_coarse: inputs (B, C, H, W) average-pooled to cells of `cell` px (zero-filled; mean over valid px), 3x3 convs,
    2 outputs (m/s, east / north), bilinear upsampling. Returns (du (B, 2, H, W), coarse field (B, 2, h, w))."""

    def __init__(self, in_ch, hidden=16, cell=8):
        super().__init__()
        self.cell = cell
        self.net = nn.Sequential(nn.Conv2d(in_ch + 1, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(hidden, 2, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, m):
        H, W = x.shape[-2:]
        k = self.cell
        ph, pw = (-H) % k, (-W) % k
        xs = F.pad(x * m[:, None], (0, pw, 0, ph)); ms = F.pad(m[:, None], (0, pw, 0, ph))
        num = F.avg_pool2d(xs, k); den = F.avg_pool2d(ms, k)
        xc = torch.cat([num / den.clamp(min=1e-3), den], 1)
        c = self.net(xc)
        up = F.interpolate(c, size=(H + ph, W + pw), mode="bilinear", align_corners=False)[..., :H, :W]
        return up, c


class VelocityModel(nn.Module):
    """u_eff in m/s from the day's physics fields and static geography; terms subset of {rea, geo, wind, corr}."""

    def __init__(self, terms, n_static, hidden=16, a0=0.25, b0=0.0, a_range=(0.0, 1.5), b_range=(-0.5, 1.5), cell=8, corr_hidden=16):
        super().__init__()
        self.terms = set(terms)
        fin = n_static + 2
        if "rea" in self.terms:
            self.a = Factor(fin, hidden, a0, *a_range)
        if "geo" in self.terms:
            self.b = Factor(fin, hidden, b0, *b_range)
        if "wind" in self.terms:
            self.wp = nn.Parameter(torch.zeros(3)); self.wq = nn.Parameter(torch.zeros(3))     # [const, sin, cos]
        if "corr" in self.terms:
            self.corr = Coarse(8, corr_hidden, cell)

    def wind_coef(self, ds, dc):
        """ds, dc: doy sin / cos tensors (any shape) -> (p, q) = Re, Im of w(s), m/s per Pa."""
        p = self.wp[0] + self.wp[1] * ds + self.wp[2] * dc
        q = self.wq[0] + self.wq[1] * ds + self.wq[2] * dc
        return p, q

    def forward(self, static, ds, dc, u, v, ug, vg, tx, ty, ocean, collect=None):
        """static (S, H, W); ds, dc (N,) doy sin / cos; u, v, ug, vg, tx, ty (N, H, W) (0 where invalid); ocean (H, W) float.
        Returns ue, ve (N, H, W) m/s and, in collect, the factor fields and the coarse-correction penalty pieces."""
        N, H, W = u.shape
        x = torch.cat([static[None].expand(N, -1, -1, -1), ds.view(N, 1, 1, 1).expand(N, 1, H, W), dc.view(N, 1, 1, 1).expand(N, 1, H, W)], 1)
        ue = torch.zeros_like(u); ve = torch.zeros_like(v)
        if "rea" in self.terms:
            a = self.a(x); ue = ue + a * u; ve = ve + a * v
            if collect is not None: collect["a"] = a
        if "geo" in self.terms:
            b = self.b(x); ue = ue + b * ug; ve = ve + b * vg
            if collect is not None: collect["b"] = b
        if "wind" in self.terms:
            p, q = self.wind_coef(ds, dc)
            p, q = p.view(N, 1, 1), q.view(N, 1, 1)
            ue = ue + p * tx - q * ty; ve = ve + q * tx + p * ty
        if "corr" in self.terms:
            xin = torch.stack([u / 0.2, v / 0.2, ug / 0.2, vg / 0.2, tx / 0.1, ty / 0.1,
                               ds.view(N, 1, 1).expand(N, H, W), dc.view(N, 1, 1).expand(N, H, W)], 1)
            du, c = self.corr(xin, ocean[None].expand(N, -1, -1))
            ue = ue + du[:, 0]; ve = ve + du[:, 1]
            if collect is not None:
                collect["corr_mag"] = (c ** 2).mean()
                collect["corr_grad"] = (torch.diff(c, dim=-1) ** 2).mean() + (torch.diff(c, dim=-2) ** 2).mean()
                collect["du"] = du
        return ue * ocean, ve * ocean


class SourceModel(nn.Module):
    """Pointwise bounded source S (ln CHL per day) = g_max tanh(MLP(x)); x (N, C, H, W); 1x1 convs; S = 0 at the start."""

    def __init__(self, in_ch, hidden=32, g_max=0.5):
        super().__init__()
        self.g_max = g_max
        self.net = _mlp(in_ch, hidden, 0.0)

    def forward(self, x):
        return self.g_max * torch.tanh(self.net(x)[:, 0])


def grad_acc(pred, truth, mask, lags=(1, 4, 16)):
    """Pairs (x, x + lag) along both axes with mask at both ends: sums for ratio sum|dp| / sum|dt| and the uncentred
    correlation sum(dp dt) / sqrt(sum dp^2 sum dt^2) (as train_pinn.py). Returns {lag: [gp, gt, gpt, gpp, gtt, n]}."""
    out = {}
    for L in lags:
        acc = torch.zeros(6, dtype=torch.float64, device=pred.device)
        for ax in (-1, -2):
            n = pred.shape[ax]
            if n <= L:
                continue
            dp = torch.narrow(pred, ax, L, n - L) - torch.narrow(pred, ax, 0, n - L)
            dt = torch.narrow(truth, ax, L, n - L) - torch.narrow(truth, ax, 0, n - L)
            m = (torch.narrow(mask, ax, L, n - L) & torch.narrow(mask, ax, 0, n - L)).double()
            dp, dt = dp.double(), dt.double()
            acc += torch.stack([(dp.abs() * m).sum(), (dt.abs() * m).sum(), (dp * dt * m).sum(), (dp * dp * m).sum(), (dt * dt * m).sum(), m.sum()])
        out[L] = acc
    return out


# ============================== step 1b: decomposed velocity, band-pass loss ==============================
# Both are used only with train_tracer.py --vel-param decomp / --loss-band; nothing above depends on them.

DECOMP_COMPS = ["geo_mean", "geo_anom", "ageo_mean", "ageo_anom"]


def decomp_clim(U, V, UG, VG, OK, doy, train_idx, sigma_days=15.0, chunk=200):
    """Day-of-year climatology of the geostrophic and ageostrophic velocities over the train days only.
    U, V, UG, VG (T, H, W) arrays (m/s; 0 where invalid; UG, VG 0 where the geostrophic field is invalid); OK (T, H, W)
    bool current validity; doy (T,) int 0..365; train_idx: store indices of the train days.
    Fields: geo = (UG, VG) * OK; ageo = (U - UG, V - VG) * OK (so geo + ageo = u_rea wherever OK; where the geostrophic
    field is invalid, geo = 0 and ageo = u_rea). Per doy bin (366 bins) the per-pixel sum and count over train days with
    OK, then circular Gaussian smoothing along doy (sigma_days) of both, mean = smoothed sum / smoothed count (0 where the
    smoothed count < 1e-6). Returns clim (366, 4, H, W) float32 [geo_u, geo_v, ageo_u, ageo_v] and count (366, H, W)."""
    from scipy.ndimage import gaussian_filter1d
    T, H, W = U.shape
    s = np.zeros((366, 4, H, W)); n = np.zeros((366, H, W))
    tr = np.asarray(train_idx)
    for b0 in range(0, len(tr), chunk):
        ii = tr[b0:b0 + chunk]
        ok = np.asarray(OK[ii]).astype("float64")
        u, v, ug, vg = (np.asarray(X[ii]).astype("float64") for X in (U, V, UG, VG))
        f = np.stack([ug, vg, u - ug, v - vg], 1) * ok[:, None]
        np.add.at(s, doy[ii], f); np.add.at(n, doy[ii], ok)
    ss = gaussian_filter1d(s, sigma_days, axis=0, mode="wrap"); ns = gaussian_filter1d(n, sigma_days, axis=0, mode="wrap")
    clim = np.where(ns[:, None] > 1e-6, ss / np.maximum(ns[:, None], 1e-6), 0.0).astype("float32")
    return clim, n.astype("int32")


def decomp_components(u, v, ug, vg, ok, clim_day):
    """u, v, ug, vg (N, H, W) m/s (0 where invalid); ok (N, H, W) float current validity; clim_day (N, 4, H, W) from
    decomp_clim at each day's doy. Returns comps (N, 4, 2, H, W) m/s in DECOMP_COMPS order, all 0 where ok = 0;
    sum over components = (u, v) * ok exactly up to float rounding."""
    geo = torch.stack([ug, vg], 1) * ok[:, None]
    ageo = torch.stack([u - ug, v - vg], 1) * ok[:, None]
    gm = clim_day[:, 0:2] * ok[:, None]; am = clim_day[:, 2:4] * ok[:, None]
    return torch.stack([gm, geo - gm, am, ageo - am], 1)


class DecompVelocity(nn.Module):
    """u_eff = sum_j alpha_j(region, time) u_j (+ optional wind term as VelocityModel).
    alpha_time 'season': one free scalar per (component, region 0..n_reg-1, season DJF/MAM/JJA/SON), init alpha0;
    'harm': alpha = c0 + c1 sin(doy) + c2 cos(doy) per (component, region), c0 init alpha0, c1 = c2 = 0.
    Unbounded (no sigmoid). Region and season piecewise constant: u_eff is discontinuous across region boundaries and
    month boundaries between seasons."""

    def __init__(self, n_comp=4, n_reg=8, alpha_time="season", alpha0=0.25, wind=False):
        super().__init__()
        self.alpha_time = alpha_time
        self.terms = {"wind"} if wind else set()
        if alpha_time == "season":
            self.alpha = nn.Parameter(torch.full((n_comp, n_reg, 4), float(alpha0)))
        else:
            c = torch.zeros(n_comp, n_reg, 3); c[..., 0] = alpha0
            self.alpha = nn.Parameter(c)
        if wind:
            self.wp = nn.Parameter(torch.zeros(3)); self.wq = nn.Parameter(torch.zeros(3))

    wind_coef = VelocityModel.wind_coef

    def alpha_field(self, reg, season, ds, dc):
        """reg (H, W) int64; season (N,) int64; ds, dc (N,) -> alpha (N, n_comp, H, W)."""
        if self.alpha_time == "season":
            a = self.alpha[:, :, season]                        # (C, R, N)
        else:
            a = self.alpha[..., 0:1] + self.alpha[..., 1:2] * ds[None, None] + self.alpha[..., 2:3] * dc[None, None]
        return a.permute(2, 0, 1)[:, :, reg]                    # (N, C, H, W)

    def forward(self, comps, reg, season, ds, dc, tx, ty, ocean, collect=None):
        """comps (N, C, 2, H, W) m/s; tx, ty (N, H, W) Pa; ocean (H, W) float. Returns ue, ve (N, H, W) m/s."""
        al = self.alpha_field(reg, season, ds, dc)
        uv = (al[:, :, None] * comps).sum(1)
        ue, ve = uv[:, 0], uv[:, 1]
        if "wind" in self.terms:
            N = ue.shape[0]
            p, q = self.wind_coef(ds, dc)
            p, q = p.view(N, 1, 1), q.view(N, 1, 1)
            ue = ue + p * tx - q * ty; ve = ve + q * tx + p * ty
        if collect is not None:
            collect["alpha"] = al
        return ue * ocean, ve * ocean


SIGMA_PER_WAVELENGTH = math.sqrt(math.log(2) / 2) / math.pi          # 0.1874: Gaussian with 0.5 response at wavelength


def gauss_kernel(sigma, trunc=4.0, device=None):
    r = max(int(math.ceil(trunc * sigma)), 1)
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


class BandPass:
    """Mask-aware Difference-of-Gaussians band-pass on a grid of nominal px_km pixels.

    Low-pass by normalised convolution (Knutsson & Westin 1993): L_s(x) = G_s * (m x) / G_s * m, m = 0/1 mask (land and
    clouds 0), G_s separable Gaussian (sum 1, truncated at 4 sigma, zero padding outside the grid). Band-pass
    B = L_s1 - L_s2 with s = 0.1874 lambda / px_km (Gaussian amplitude response exp(-2 pi^2 s^2 / lambda^2) = 0.5 at
    lambda): s1 from lo_km, s2 from hi_km. On a fully observed field the response to a sinusoid of wavelength lambda is
    R(lambda) = exp(-2 pi^2 s1^2 / lambda^2) - exp(-2 pi^2 s2^2 / lambda^2) (response()); it is not flat inside
    [lo, hi] and does not reach 0 outside (a DoG is a broad band-pass). Masked pixels never enter the sums, so cloud and
    land edges add no step; near them the low-passes are local weighted means over the observed side only.
    Valid output: m = 1 and both normalised weights G_s * m >= min_frac."""

    def __init__(self, lo_km, hi_km, px_km=4.5, min_frac=0.5, device=None):
        self.lo_km, self.hi_km, self.px_km, self.min_frac = lo_km, hi_km, px_km, min_frac
        self.s1 = SIGMA_PER_WAVELENGTH * lo_km / px_km
        self.s2 = SIGMA_PER_WAVELENGTH * hi_km / px_km
        self.k1 = gauss_kernel(self.s1, device=device); self.k2 = gauss_kernel(self.s2, device=device)

    def response(self, wavelength_km):
        lam = np.asarray(wavelength_km, dtype="float64") / self.px_km
        return np.exp(-2 * np.pi ** 2 * self.s1 ** 2 / lam ** 2) - np.exp(-2 * np.pi ** 2 * self.s2 ** 2 / lam ** 2)

    @staticmethod
    def _blur(x, kr):
        k, r = kr
        k = k.to(x.device, x.dtype)
        C = x.shape[1]
        x = F.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), padding=(0, r), groups=C)
        return F.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), padding=(r, 0), groups=C)

    def __call__(self, x, m):
        """x (N, H, W) (any value where m = 0), m (N, H, W) bool -> (band-passed (N, H, W), valid (N, H, W) bool).
        Linear in x (differentiable)."""
        mf = m.to(x.dtype)
        st = torch.stack([torch.where(m, x, torch.zeros_like(x)), mf], 1)
        b1 = self._blur(st, self.k1); b2 = self._blur(st, self.k2)
        l1 = b1[:, 0] / b1[:, 1].clamp(min=1e-6); l2 = b2[:, 0] / b2[:, 1].clamp(min=1e-6)
        valid = m & (b1[:, 1] >= self.min_frac) & (b2[:, 1] >= self.min_frac)
        return torch.where(valid, l1 - l2, torch.zeros_like(x)), valid
