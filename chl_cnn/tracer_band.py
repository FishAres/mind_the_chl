"""Band-split tracer: large-scale base + band-passed anomalies, each band advected and damped separately.
Nothing in tracer.py / train_tracer.py depends on this file (it only imports from them).

    ln C^(x, t+k) = B(x, t+k) + sum_b D_b(k) A_b(X^b_k(x), t)
    A(., t) = ln C_obs(., t) - B(., t) split into bands by mask-aware (normalised-convolution) low-passes of day t:
        L_lam = G_lam * (m y) / G_lam * m   (Gaussian, sigma = 0.1874 lam / 4.5 km px -> 0.5 amplitude response at lam;
                                              truncated at 4 sigma, zero padding; m = observed & B finite)
        A_lt20   = y - L_20                 (< 20 km)
        A_20_50  = L_20 - L_50              (20-50 km)
        A_50_150 = L_50 - B                 (50-150 km; B = 'lp' base of tracer_static: 150 km low-pass of day t with the
                                              doy fallback where its weight < 0.05)
      so B + A_lt20 + A_20_50 + A_50_150 = y exactly on every px with m = 1 (telescoping; check_reconstruction).
      On px with m = 1 the 20 / 50 km normalised weights are > 0 (centre weight alone 0.23 / 0.037), so no fallback.
    X^b_k(x) = departure point at day t of the backward RK2 trajectory (tracer.Integrator, unchanged) from x at day t+k with
      the band's velocity; A_b sampled there as sum(w m A_b) / sum(w m), valid if sum(w m) >= 0.99 (as depart_sample).
    D_b(k) = exp(-k / tau_b) (constant), or exp(-sum_sub-steps h / tau_b(x_mid, t_mid)) along the band's trajectory
      (physics-dependent tau, BandModel tau_mode 'phys').
    B(x, t+k) = B(x, t) (test 1) or B(x, t) + sum_sub-steps h S_B(...) (BandModel: pointwise source on B only, Eulerian at
      the arrival px, explicit Euler with n_sub sub-steps per day; S_B = g_max tanh(MLP), zero at init).
    Equal velocities and D_b = 1 for all bands reduce exactly (up to float rounding) to tracer_static.predict with the
    'lp' base (sampling is linear and the three bands share the mask m).

TAU_MEASURED: per band 1 / (mean null-corrected coherence decay rate over the spectral_motion rings whose wavelength lies
in the band); rings <= 12 km excluded (rho at lead 1 0.07-0.15, near the null). analyses/blk4/results/spectral_motion.json
coherence.decay_fits.aligned.per_ring:
    < 20 km   rings 18, 16, 14.4, 13.1 km: rates 0.603, 0.619, 0.587, 0.591 /day -> mean 0.600 -> tau 1.67 d
    20-50 km  rings 48, 36, 28.8, 24, 20.6 km: 0.207, 0.319, 0.445, 0.536, 0.590 -> mean 0.419 -> tau 2.38 d
    50-150 km rings 144, 72 km: 0.046, 0.099 -> mean 0.073 -> tau 13.8 d
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .tracer import SIGMA_PER_WAVELENGTH, gauss_kernel, BandPass, SourceModel, _mlp

BAND_NAMES = ["lt20", "20_50", "50_150"]
BAND_LABEL = {"lt20": "< 20 km", "20_50": "20-50 km", "50_150": "50-150 km"}
EDGES_KM = (20.0, 50.0)            # inner edges; the outer one (150 km) is the lp base
PX_KM = 4.5
RING_EXCLUDE_KM = 12.0


def tau_measured(spec_json=None):
    """(tau per band (3,), derivation dict) from the spectral_motion per-ring decay fits."""
    if spec_json is None:
        spec_json = Path(__file__).resolve().parents[1] / "analyses/blk4/results/spectral_motion.json"
    rings = json.load(open(spec_json))["coherence"]["decay_fits"]["aligned"]["per_ring"]
    lims = [(RING_EXCLUDE_KM, 20.0), (20.0, 50.0), (50.0, 150.0)]
    taus, info = [], {}
    for nm, (lo, hi) in zip(BAND_NAMES, lims):
        sel = [r for r in rings if lo < r["wavelength_km"] <= hi]
        rate = float(np.mean([r["rate_per_day"] for r in sel]))
        taus.append(1.0 / rate)
        info[nm] = dict(rings_km=[r["wavelength_km"] for r in sel], rates=[r["rate_per_day"] for r in sel], mean_rate=rate, tau_days=1.0 / rate)
    return np.array(taus), info


def lowpass(y, m, lam_km, px_km=PX_KM):
    """Normalised-convolution Gaussian low-pass. y (N, H, W), m (N, H, W) bool -> (L, den)."""
    kr = gauss_kernel(SIGMA_PER_WAVELENGTH * lam_km / px_km, device=y.device)
    st = torch.stack([torch.where(m, y, torch.zeros_like(y)), m.to(y.dtype)], 1)
    b = BandPass._blur(st, kr)
    return b[:, 0] / b[:, 1].clamp(min=1e-6), b[:, 1]


def decompose(y, obs, B):
    """y (N, H, W) ln C (any value where unobserved); obs bool; B (N, H, W) base at day t (NaN allowed).
    Returns A (N, 3, H, W) in BAND_NAMES order (0 where ~m) and m = obs & finite(B)."""
    m = obs & torch.isfinite(B)
    L20, _ = lowpass(y, m, EDGES_KM[0]); L50, _ = lowpass(y, m, EDGES_KM[1])
    Bz = torch.nan_to_num(B)
    A = torch.stack([y - L20, L20 - L50, L50 - Bz], 1)
    return torch.where(m[:, None], A, torch.zeros_like(A)), m


def check_reconstruction(y, obs, B):
    """max |B + sum A - y| over px with m = 1, and the number of such px."""
    A, m = decompose(y, obs, B)
    r = (torch.nan_to_num(B) + A.sum(1) - y).abs()[m]
    return float(r.max()) if r.numel() else 0.0, int(m.sum())


def band_sample(it, A, m, pos):
    """A (N, C, H, W), m (N, H, W) bool, pos departure points -> (sampled (N, C, H, W), valid (N, H, W)) as
    Integrator.depart_sample applied to every channel with the shared mask."""
    of = m.to(A.dtype)
    s = it.sample(torch.cat([A * of[:, None], of[:, None]], 1), pos)
    den = s[:, -1]
    return s[:, :-1] / den.clamp(min=1e-3)[:, None], den >= it.min_obs


def strain_speed(U, ok):
    """U (N, 2, H, W) px/day (x = col east, y = row north), ok (N, H, W) bool -> strain (1/day), speed (px/day); centred
    differences per px, 0 where the px or a neighbour is invalid."""
    u, v = U[:, 0], U[:, 1]
    okf = ok.float()

    def dx(f):
        g = torch.zeros_like(f); g[:, :, 1:-1] = 0.5 * (f[:, :, 2:] - f[:, :, :-2]); return g

    def dy(f):
        g = torch.zeros_like(f); g[:, 1:-1, :] = 0.5 * (f[:, 2:, :] - f[:, :-2, :]); return g
    nb = okf.clone()
    nb[:, :, 1:-1] *= okf[:, :, 2:] * okf[:, :, :-2]; nb[:, 1:-1, :] *= okf[:, 2:, :] * okf[:, :-2, :]
    nb[:, :, 0] = 0; nb[:, :, -1] = 0; nb[:, 0, :] = 0; nb[:, -1, :] = 0
    st = torch.sqrt((dx(u) - dy(v)) ** 2 + (dx(v) + dy(u)) ** 2 + 1e-12) * nb
    return st, torch.hypot(u, v) * okf


class BandModel(nn.Module):
    """Learned band-split tracer (test 2).
    a (3,) velocity factor per band on u_rea (free scalars, init a0); log_tau (3,) (init log tau0).
    tau_mode 'const': D_b(k) = exp(-k / tau_b). 'phys': tau_b(x, t) = tau_b * softplus(z_b) / ln 2, z = MLP(tau inputs)
    (1x1 convs, hidden tau_hidden, 3 outputs, last layer zero -> multiplier 1 at init); the rate 1 / tau_b(x, t) is
    evaluated on the daily grid and sampled (bilinear, linear in time) at the band trajectory's sub-step midpoints.
    src: SourceModel on B (inputs src_in channels, S_B = 0 at init) or None."""

    def __init__(self, tau0, a0=0.25, tau_mode="const", tau_in=5, tau_hidden=16, src_in=13, src_hidden=32, g_max=0.5, source=True):
        super().__init__()
        self.tau_mode = tau_mode
        self.a = nn.Parameter(torch.full((3,), float(a0)))
        self.log_tau = nn.Parameter(torch.log(torch.as_tensor(np.asarray(tau0, "float32"))))
        if tau_mode == "phys":
            net = nn.Sequential(nn.Conv2d(tau_in, tau_hidden, 1), nn.GELU(), nn.Conv2d(tau_hidden, tau_hidden, 1), nn.GELU(), nn.Conv2d(tau_hidden, 3, 1))
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
            self.taunet = net
        self.src = SourceModel(src_in, src_hidden, g_max) if source else None

    def tau(self):
        return torch.exp(self.log_tau)

    def tau_mult(self, x):
        """x (N, tau_in, H, W) -> multiplier (N, 3, H, W), 1 at init."""
        return F.softplus(self.taunet(x)) / math.log(2.0)
