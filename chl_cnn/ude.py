"""Pieces of the daily UDE for surface ln CHL (train_ude.py):

    C_{k+1} = T(C_k; u_k, v_k) + dt [ K lap(C_k) + g_theta(C_k, X_k) ],   dt = 1 day

  Transport   semi-Lagrangian advection: the departure point of every pixel is found by n_sub backward Euler sub-steps along
              the day's currents (cells/day along axis -1 = u, axis -2 = v), the field is then sampled once there (bilinear,
              F.grid_sample). The ocean mask and the current-validity mask are sampled the same way; where the sampled
              validity is below min_valid at any point of the trajectory (land, missing currents, outside the grid) the
              pixel keeps its value. Integer displacements are reproduced exactly.
  laplacian   5-point Laplacian in cells^-2 with zero flux into land / across the grid edge.
  GNet        1x1-conv MLP (two hidden layers) or 3x3 conv net (three layers) on the pixel covariate stack;
              g = g_max tanh(.) per day; the last layer starts at zero so g starts at 0.
  Relax       g = -lambda_r (C - clim) with one learned positive scalar, integrated exactly over the step
              (the Euler update then multiplies the anomaly by exp(-lambda_r dt)).
  AdvScale    learned advection factor alpha multiplying the currents: 'scalar' alpha = 1.5 sigmoid(a) in (0, 1.5), or
              'field' alpha(x) = 1.5 sigmoid(MLP(features)) (1x1 conv) with the last layer zero so alpha starts at alpha0
              everywhere; reg(alpha) = mean |alpha - mean alpha| over ocean pixels (L1 deviation from the scalar mean).
  rollout     the L-step recursion with switches for the transport and the source term.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class Transport:
    """Semi-Lagrangian advection on a fixed (H, W) grid with an ocean mask (bool tensor on the device)."""

    def __init__(self, ocean, n_sub=4, lag_max=20.0, min_valid=0.99):
        self.ocean = ocean
        self.H, self.W = ocean.shape
        dev = ocean.device
        gy, gx = torch.meshgrid(torch.arange(self.H, device=dev), torch.arange(self.W, device=dev), indexing="ij")
        self.base = torch.stack([gx, gy], -1).float()                                  # (H, W, 2) pixel coordinates in cells (x, y)
        self.scale = torch.tensor([2.0 / max(self.W - 1, 1), 2.0 / max(self.H - 1, 1)], device=dev)
        self.oceanf = ocean.float()[None, None]
        self.n_sub, self.lag_max, self.min_valid = n_sub, lag_max, min_valid

    def sample(self, f, pos):
        """f (B, C, H, W) sampled at pos (B, H, W, 2) in cells -> (B, C, H, W); zero outside the grid."""
        return F.grid_sample(f, pos * self.scale - 1.0, mode="bilinear", padding_mode="zeros", align_corners=True)

    def __call__(self, C, u, v, ok, dt=1.0, alpha=None):
        """C (B, H, W); u, v (B, H, W) cells/day; ok (B, H, W) bool validity of the currents; alpha None (= 1), a scalar
        tensor or a (B, H, W) field multiplying the currents. Returns (advected field, fraction of ocean pixels NOT advected)."""
        B = C.shape[0]
        okf = (ok & self.ocean).float()
        uv = torch.stack([u * okf, v * okf], 1)
        if alpha is not None:
            uv = uv * (alpha if alpha.dim() == 0 else alpha[:, None])
        pos = self.base[None].expand(B, -1, -1, -1)
        valid = okf.clone()
        h = dt / self.n_sub
        for s in range(self.n_sub):
            if s == 0:
                vel, vok = uv, okf
            else:
                vel = self.sample(uv, pos); vok = self.sample(okf[:, None], pos)[:, 0]
            valid = valid * (vok >= self.min_valid).float()
            pos = pos - h * vel.permute(0, 2, 3, 1)
        disp = (pos - self.base[None]).clamp(-self.lag_max, self.lag_max)
        pos = self.base[None] + disp
        num = self.sample((C * self.ocean)[:, None], pos)[:, 0]
        den = self.sample(self.oceanf.expand(B, -1, -1, -1), pos)[:, 0]
        valid = valid * (den >= self.min_valid).float()
        Cd = num / den.clamp(min=1e-3)
        out = torch.where(valid > 0.5, Cd, C) * self.ocean
        frac = 1.0 - valid[:, self.ocean].mean()
        return out, frac


def laplacian(C, ocean):
    """5-point Laplacian (cells^-2) of C (B, H, W) on ocean pixels; land / edge neighbours contribute no flux."""
    H, W = ocean.shape
    oc = ocean.float()
    Cp = F.pad(C * oc, (1, 1, 1, 1)); op = F.pad(oc, (1, 1, 1, 1))
    lap = torch.zeros_like(C)
    for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        Cn = Cp[:, 1 + di:1 + di + H, 1 + dj:1 + dj + W]; on = op[1 + di:1 + di + H, 1 + dj:1 + dj + W]
        lap = lap + on * (Cn - C)
    return lap * oc


class GNet(nn.Module):
    """Learned local tendency g_theta (ln CHL per day) from the pixel covariate stack; |g| <= g_max."""

    def __init__(self, kind, in_ch, hidden=32, g_max=0.5):
        super().__init__()
        self.kind, self.g_max = kind, g_max
        if kind == "mlp":
            self.net = nn.Sequential(nn.Conv2d(in_ch, hidden, 1), nn.GELU(), nn.Conv2d(hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        elif kind == "conv":
            self.net = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(hidden, 1, 3, padding=1))
        else:
            raise ValueError(kind)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.g_max * torch.tanh(self.net(x)[:, 0])


class Relax(nn.Module):
    """g = -lambda_r (C - clim), lambda_r = exp(log_lam) > 0, integrated exactly over one step of length dt."""

    def __init__(self, lam0=0.1):
        super().__init__()
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam0)))

    def lam(self):
        return self.log_lam.exp()

    def forward(self, C, clim, dt=1.0):
        return (C - clim) * (torch.exp(-self.lam() * dt) - 1.0) / dt


class AdvScale(nn.Module):
    """Learned advection factor alpha in (0, 1.5): kind 'scalar' (one parameter) or 'field' (1x1-conv MLP on the pixel stack)."""

    def __init__(self, kind, in_ch=0, hidden=16, alpha0=0.25, alpha_max=1.5):
        super().__init__()
        self.kind, self.alpha_max = kind, alpha_max
        b0 = math.log(alpha0 / (alpha_max - alpha0))                                 # sigmoid(b0) alpha_max = alpha0
        if kind == "scalar":
            self.a = nn.Parameter(torch.tensor(b0))
        elif kind == "field":
            self.net = nn.Sequential(nn.Conv2d(in_ch, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
            nn.init.zeros_(self.net[-1].weight); nn.init.constant_(self.net[-1].bias, b0)
        else:
            raise ValueError(kind)

    def forward(self, x=None):
        """scalar: 0-dim tensor; field: (B, H, W) from the stack x (B, F, H, W)."""
        if self.kind == "scalar":
            return self.alpha_max * torch.sigmoid(self.a)
        return self.alpha_max * torch.sigmoid(self.net(x)[:, 0])

    @staticmethod
    def reg(alpha, ocean):
        """Mean over ocean pixels of |alpha - mean_ocean(alpha)| (per sample, then averaged)."""
        if alpha.dim() == 0:
            return alpha * 0.0
        v = alpha[:, ocean]
        return (v - v.mean(1, keepdim=True)).abs().mean()


def rollout(C0, L, ocean, transport=None, currents=None, K=0.0, gfun=None, use_T=True, use_g=True, dt=1.0, collect=None, afun=None):
    """L daily steps from C0 (B, H, W). currents(k) -> (u, v, ok) in cells/day for step k; gfun(k, C) -> g (B, H, W);
    afun(k, C) -> advection factor alpha (scalar tensor or (B, H, W)), None = 1.
    transport None or use_T False: T = identity. Returns (preds (B, L, H, W), mean fraction not advected).
    collect: dict that receives 'g' (list of g per step) and 'alpha' (list per step, when afun is given)."""
    C, preds, fr = C0, [], []
    for k in range(L):
        tend = torch.zeros_like(C)
        if K > 0:
            tend = tend + K * laplacian(C, ocean)
        if use_g and gfun is not None:
            g = gfun(k, C)
            tend = tend + g
            if collect is not None:
                collect.setdefault("g", []).append(g)
        if use_T and transport is not None:
            u, v, ok = currents(k)
            alpha = afun(k, C) if afun is not None else None
            if collect is not None and alpha is not None:
                collect.setdefault("alpha", []).append(alpha)
            Cadv, f = transport(C, u, v, ok, dt, alpha=alpha); fr.append(f)
        else:
            Cadv = C
        C = (Cadv + dt * tend) * ocean
        preds.append(C)
    frac = torch.stack(fr).mean() if fr else torch.zeros((), device=C0.device)
    return torch.stack(preds, 1), frac
