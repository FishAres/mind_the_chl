"""Deep equilibrium gap filler: the fill is the fixed point of a learned update, not the output of a fixed-depth net.

State  z (B, 3, H, W): z-scored ln CHL for days t-1, t, t+1 (the same 3-day head as train_pinn.py).
Cell   f(z, x) = P( z + eta * Net([z, x]) )     Net = small dilated CNN with GroupNorm (no BatchNorm: its running
       statistics would change from one solver iteration to the next); x = the 26 injected input channels of
       train_pinn.py (features, visible CHL, context days, climatology, coords); P overwrites the visible pixels
       of each of the 3 days with their observed value (the inpainting data step of ELDER / Gilton et al. 2021),
       so observed pixels are a hard constraint and the equilibrium is a fill consistent with them.
       The last conv is zero-initialised: at init f(z) = P(z), so z0 (the climatology) is already a fixed point
       and training starts from a contractive, well-posed map.
Init   z0 = day-of-year climatology of the 3 days (the --residual baseline of the explicit CNN).

Solvers (forward)
  unroll    K applications of f, gradients by backprop through all K (weight-tied unrolled net; the cheap baseline
            that FNO-DEQ found to match a true DEQ on Darcy flow)
  fixed     plain fixed-point iteration to tolerance (no grad), then phantom-gradient steps
  anderson  Anderson acceleration (Walker & Ni 2011; the implicit-layers tutorial implementation) to tolerance,
            then phantom-gradient steps
Gradients for fixed/anderson: phantom gradient (Geng et al. 2021): after the no-grad solve, S damped steps
  z <- tau f(z) + (1 - tau) z with autograd on; S = 1, tau = 1 is Jacobian-free backprop (Fung et al. 2022).

Diagnostics: every solve records the iteration count and the final relative residual |f(z) - z| / |z| in
LAST_STATS; iteration_curve() scores the fill after k = 0, 1, 2, ... iterations (the test-time compute curve).
"""
import math

import torch
import torch.utils.checkpoint
from torch import nn

LAST_STATS = {}
import os


class ResBlockGN(nn.Module):
    def __init__(self, w, dilation, groups=8):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(groups, w), nn.GELU(), nn.Conv2d(w, w, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, w), nn.GELU(), nn.Conv2d(w, w, 3, padding=dilation, dilation=dilation))

    def forward(self, x):
        return x + self.body(x)


class DEQCell(nn.Module):
    """z_new = z + eta * Net([z, x]); eta a learned positive scalar (log-parameterised)."""

    def __init__(self, in_x, n_state=3, width=64, dilations=(1, 2, 4, 8), eta=1.0, learn_eta=True, spectral=False):
        super().__init__()
        self.inp = nn.Conv2d(in_x + n_state, width, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlockGN(width, d) for d in dilations])
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.GELU(), nn.Conv2d(width, n_state, 1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)
        if spectral:   # spectral normalisation of every conv (Ryu et al. 2019 / Gilton et al. 2021): bounds the Lipschitz constant of Net
            for mod in [self.inp] + [m for b in self.blocks for m in b.body if isinstance(m, nn.Conv2d)]:   # not the zero-init output conv (sigma = 0)
                nn.utils.parametrizations.spectral_norm(mod)
        le = torch.tensor(math.log(eta))
        if learn_eta:
            self.log_eta = nn.Parameter(le)
        else:
            self.register_buffer("log_eta", le)

    def eta(self):
        return self.log_eta.exp()

    def forward(self, z, x):
        return z + self.eta() * self.out(self.blocks(self.inp(torch.cat([z, x], 1))))


def _rel_res(fz, z):
    """Per-batch relative residual |f(z) - z| / |z| (float32), shape (B,)."""
    d = (fz - z).float().flatten(1).norm(dim=1)
    n = z.float().flatten(1).norm(dim=1).clamp(min=1e-6)
    return d / n


@torch.no_grad()
def fixed_point(f, z0, max_iter, tol):
    """Plain iteration z <- f(z) until max over the batch of the relative residual < tol. Returns z, iters, res (B,)."""
    z = z0
    res = None
    for k in range(1, max_iter + 1):
        fz = f(z)
        res = _rel_res(fz, z)
        z = fz
        if res.max() < tol:
            break
    return z, k, res


@torch.no_grad()
def anderson(f, z0, m=5, lam=1e-2, max_iter=50, tol=1e-3, beta=1.0):
    """Anderson acceleration (implicit-layers tutorial, ch. 4). History and least squares in float32 per batch element."""
    B = z0.shape[0]
    shape = z0.shape
    X = torch.zeros(B, m, z0[0].numel(), dtype=torch.float32, device=z0.device)
    Fh = torch.zeros_like(X)
    X[:, 0] = z0.flatten(1).float(); Fh[:, 0] = f(z0).flatten(1).float()
    X[:, 1] = Fh[:, 0]; Fh[:, 1] = f(Fh[:, 0].view(shape)).flatten(1).float()
    Hm = torch.zeros(B, m + 1, m + 1, dtype=torch.float32, device=z0.device)
    Hm[:, 0, 1:] = Hm[:, 1:, 0] = 1
    y = torch.zeros(B, m + 1, 1, dtype=torch.float32, device=z0.device); y[:, 0] = 1
    res = (Fh[:, 1] - X[:, 1]).norm(dim=1) / X[:, 1].norm(dim=1).clamp(min=1e-6)
    gbest = (Fh[:, 1] - X[:, 1]).norm(dim=1)       # smallest absolute residual |f(X) - X| seen so far, per batch element
    k = 1
    for k in range(2, max_iter):
        n = min(k, m)
        G = Fh[:, :n] - X[:, :n]
        GGt = torch.bmm(G, G.transpose(1, 2))
        # Tikhonov term relative to the scale of G G^T (its mean diagonal), not absolute: when the residuals of the history are
        # nearly collinear (early training: f(z) - z is almost independent of z) an absolute 1e-4 is negligible against |g|^2 ~ 1e2
        # and the least squares returns coefficients of O(10-100) that extrapolate the iterate geometrically (observed: |z| ~ 1e4
        # after 17 iterations, then fp16 overflow -> NaN).
        reg = lam * (torch.diagonal(GGt, dim1=1, dim2=2).mean(1) + 1e-12)
        Hm[:, 1:n + 1, 1:n + 1] = GGt + reg[:, None, None] * torch.eye(n, dtype=torch.float32, device=z0.device)[None]
        # The Tikhonov term pulls alpha towards the plain step e_{k-1} (rhs lam * e_{k-1}), not towards 0: with collinear residuals
        # the solve then returns the plain fixed-point step instead of an arbitrary extrapolation.
        y[:, 1:n + 1, 0] = 0.0; y[:, 1 + (k - 1) % m, 0] = reg
        try:
            alpha = torch.linalg.solve(Hm[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]
        except RuntimeError:          # singular (a batch element diverged to inf/NaN, or G == 0 exactly): fall back to a plain step
            alpha = torch.zeros(B, n, dtype=torch.float32, device=z0.device); alpha[:, (k - 1) % m if k - 1 < m else (k - 1) % m] = 1.0
        bad = ~torch.isfinite(alpha).all(1)
        if bad.any():
            alpha[bad] = 0.0; alpha[bad, (k - 1) % m] = 1.0
        X[:, k % m] = beta * (alpha[:, None] @ Fh[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        Fh[:, k % m] = f(X[:, k % m].view(shape)).flatten(1).float()
        gk = (Fh[:, k % m] - X[:, k % m]).norm(dim=1)
        # Safeguard (Zhang, O'Donoghue & Boyd 2020): an accelerated step whose residual is non-finite or more than 2x the best seen
        # is rejected for that batch element and replaced by the plain step f(X_{k-1}) (one extra evaluation of f for the batch).
        # Non-finite residual, residual > 2x the best seen, or max|state| > 1.25x the plain step's (GroupNorm makes the cell nearly
        # scale invariant, so a runaway |z| on the hidden pixels does not show in the residual, and the state norm is dominated by
        # the visible pixels that P holds fixed; observed: max|z| x1.3-1.8 per iteration, 5 -> 280 in 12 iterations): reject.
        bad = ~torch.isfinite(gk) | (gk > 2.0 * gbest) | (X[:, k % m].abs().amax(1) > 1.25 * Fh[:, (k - 1) % m].abs().amax(1))
        if bad.any():
            X[bad, k % m] = Fh[bad, (k - 1) % m]
            Fh[:, k % m] = torch.where(bad[:, None], f(X[:, k % m].view(shape)).flatten(1).float(), Fh[:, k % m])
            gk = (Fh[:, k % m] - X[:, k % m]).norm(dim=1)
        gbest = torch.minimum(gbest, torch.nan_to_num(gk, nan=float("inf")))
        res = gk / X[:, k % m].norm(dim=1).clamp(min=1e-6)
        if res.max() < tol:
            break
    return X[:, k % m].view(shape), k, res


class DEQ(nn.Module):
    """Wraps a cell and a projection; forward(x, z0, proj) returns the equilibrium (or the K-step unroll)."""

    def __init__(self, cell, solver="anderson", steps=24, tol=1e-3, grad="phantom", phantom_steps=1, phantom_tau=1.0,
                 anderson_m=5, anderson_beta=1.0, checkpoint=False):
        super().__init__()
        self.cell = cell
        self.checkpoint = checkpoint
        self.solver, self.steps, self.tol = solver, steps, tol
        self.grad, self.phantom_steps, self.phantom_tau = grad, phantom_steps, phantom_tau
        self.anderson_m, self.anderson_beta = anderson_m, anderson_beta

    def step(self, z, x, proj):
        return proj(self.cell(z, x))

    def forward(self, x, z0, proj, steps=None, solver=None):
        steps = self.steps if steps is None else steps
        solver = self.solver if solver is None else solver
        f = lambda z: self.step(z, x, proj).float()
        if solver == "unroll":
            z = z0
            res = None
            for _ in range(steps):
                fz = torch.utils.checkpoint.checkpoint(f, z, use_reentrant=False) if (self.checkpoint and torch.is_grad_enabled()) else f(z)
                res = _rel_res(fz.detach(), z.detach())
                z = fz
            LAST_STATS.update(iters=steps, res=float(res.mean()) if res is not None else 0.0, res_max=float(res.max()) if res is not None else 0.0)
            return z
        if solver == "anderson":
            z, k, res = anderson(f, z0, m=self.anderson_m, max_iter=steps, tol=self.tol, beta=self.anderson_beta)
        else:
            z, k, res = fixed_point(f, z0, max_iter=steps, tol=self.tol)
        LAST_STATS.update(iters=k, res=float(res.mean()), res_max=float(res.max()))
        # autocast caches fp16 weight copies made inside the no-grad solve; reused in the phantom step they are detached from
        # the parameters (no gradient reaches the convolutions). Clear the cache before the step that needs gradients.
        torch.clear_autocast_cache()
        if self.training or torch.is_grad_enabled():
            if self.grad == "phantom":
                for _ in range(self.phantom_steps):
                    z = self.phantom_tau * f(z) + (1 - self.phantom_tau) * z
            else:   # 'none': the solve is the output (eval)
                pass
        return z


@torch.no_grad()
def iteration_curve(deq, x, z0, proj, ks, solver="fixed"):
    """Fill after k iterations for each k in ks (sorted, k=0 -> z0). Returns {k: z_k}. Plain iteration reuses the state;
    Anderson restarts for each k."""
    out = {}
    f = lambda z: torch.nan_to_num(deq.step(z, x, proj).float(), nan=1e3, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
    if solver == "fixed":
        z = z0; k_done = 0
        for k in ks:
            for _ in range(k - k_done):
                z = f(z)
            k_done = k
            out[k] = z
    else:
        for k in ks:
            out[k] = z0 if k == 0 else anderson(f, z0, m=deq.anderson_m, max_iter=max(k, 2), tol=0.0, beta=deq.anderson_beta)[0]
    return out
