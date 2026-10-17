"""Spatial "neural ODE" gap filler: steady state of a pseudo-time anisotropic diffusion equation.

    dC/ds = div( D(x) grad C ) + beta (C_prior - C),   C = C_obs on observed pixels (Dirichlet)

The steady state minimises the energy
    E(C) = 1/2 sum_corners a_c  g^T D g   +  1/2 sum_pixels a_p beta (C - C_prior)^2
over the free (unobserved ocean) pixels, where g is the gradient of C at cell corners and a_c, a_p are
cos(lat) area weights. The operator A = G^T (a D) G + a beta is therefore symmetric positive definite
by construction and the steady state is one batched conjugate-gradient solve per day. Gradients with
respect to D and the prior come from the implicit function theorem (one more CG solve with the same
operator), so no pseudo-time trajectory is stored or integrated backwards.

Grid conventions: fields (B, H, W); row = lat index, col = lon index; corners (B, H-1, W-1) sit between
pixels (i,j), (i,j+1), (i+1,j), (i+1,j+1). Gradients are per cell in the lat direction; the lon
gradient is scaled by 1 / cos(lat) so D is in (lat-cell)^2 per pseudo-time unit and D = I is isotropic.
Land pixels never enter a gradient: a corner's x gradient averages the row pairs that are both ocean,
its y gradient the column pairs that are both ocean (missing pairs are dropped).

D(x) at a corner is a 2x2 SPD tensor given as (Dxx, Dxy, Dyy) = R(phi) diag(l1, l2) R(phi)^T:
  IsoD        D = d I                                        (1 parameter)
  SSTTensorD  l_par = d0 along the SST isotherm, l_perp = d0 / (1 + (|grad SST| / g0)^2) across it
  MLPD        1x1-conv MLP from corner covariates to (phi, l1, l2)
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

SHIFTS = ((0, 0), (0, 1), (1, 0), (1, 1))     # (di, dj) of the four pixels around a corner
LAST_STATS = {}                                # CG diagnostics of the last forward/backward solve


class Grid:
    """Static stencil weights for a (H, W) grid with an ocean mask and latitude vector (deg)."""

    def __init__(self, ocean, lat, device):
        ocean = torch.as_tensor(ocean, dtype=torch.bool, device=device)
        lat = torch.as_tensor(lat, dtype=torch.float32, device=device)
        self.H, self.W = ocean.shape
        self.ocean = ocean
        o = ocean.float()
        c00, c01, c10, c11 = (o[i:i + self.H - 1, j:j + self.W - 1] for i, j in SHIFTS)
        r0, r1 = c00 * c01, c10 * c11            # row pairs (x gradient)
        q0, q1 = c00 * c10, c01 * c11            # column pairs (y gradient)
        nx, ny = (r0 + r1).clamp(min=1), (q0 + q1).clamp(min=1)
        coslat = torch.cos(torch.deg2rad(lat))
        cosc = 0.5 * (coslat[:-1] + coslat[1:]).view(-1, 1)
        self.wx = torch.stack([-r0 / nx, r0 / nx, -r1 / nx, r1 / nx]) / cosc      # (4, H-1, W-1), order SHIFTS
        self.wy = torch.stack([-q0 / ny, -q1 / ny, q0 / ny, q1 / ny])
        self.area_c = cosc.expand(self.H - 1, self.W - 1).contiguous()           # corner area weight
        self.area_p = coslat.view(-1, 1).expand(self.H, self.W).contiguous()     # pixel area weight
        self.corner_ocean = (r0 + r1 + q0 + q1) > 0                              # corner touches an ocean pair

    def grad(self, c):
        """(gx, gy) at corners, each (B, H-1, W-1)."""
        H, W = self.H, self.W
        gx = gy = 0.0
        for k, (i, j) in enumerate(SHIFTS):
            v = c[:, i:i + H - 1, j:j + W - 1]
            gx = gx + self.wx[k] * v
            gy = gy + self.wy[k] * v
        return gx, gy

    def grad_t(self, fx, fy):
        """Transpose of grad: corner fluxes (B, H-1, W-1) -> pixels (B, H, W)."""
        H, W = self.H, self.W
        out = fx.new_zeros((fx.shape[0], H, W))
        for k, (i, j) in enumerate(SHIFTS):
            out[:, i:i + H - 1, j:j + W - 1] += self.wx[k] * fx + self.wy[k] * fy
        return out

    def to_corners(self, p, valid=None):
        """Mean of the four surrounding pixel values at each corner (NaN-free input; `valid` (B,H,W) bool
        restricts the mean to valid pixels). Returns (B, H-1, W-1) and the count of valid pixels."""
        H, W = self.H, self.W
        s = 0.0; n = 0.0
        for i, j in SHIFTS:
            v = p[:, i:i + H - 1, j:j + W - 1]
            if valid is None:
                s = s + v; n = n + 1.0
            else:
                m = valid[:, i:i + H - 1, j:j + W - 1].to(v.dtype)
                s = s + v * m; n = n + m
        n = torch.as_tensor(n, device=p.device, dtype=p.dtype) if not torch.is_tensor(n) else n
        return s / n.clamp(min=1), n


def apply_operator(grid, Dc, beta, c):
    """A c = G^T (a_c D) G c + a_p beta c  for the whole field (B, H, W). Dc: (B, 3, H-1, W-1) = (Dxx, Dxy, Dyy)."""
    gx, gy = grid.grad(c)
    a = grid.area_c
    fx = a * (Dc[:, 0] * gx + Dc[:, 1] * gy)
    fy = a * (Dc[:, 1] * gx + Dc[:, 2] * gy)
    return grid.grad_t(fx, fy) + grid.area_p * beta * c


def operator_diag(grid, Dc, beta):
    """Diagonal of A, (B, H, W), for the Jacobi preconditioner."""
    H, W = grid.H, grid.W
    a = grid.area_c
    out = Dc.new_zeros((Dc.shape[0], H, W))
    for k, (i, j) in enumerate(SHIFTS):
        wx, wy = grid.wx[k], grid.wy[k]
        out[:, i:i + H - 1, j:j + W - 1] += a * (Dc[:, 0] * wx * wx + 2 * Dc[:, 1] * wx * wy + Dc[:, 2] * wy * wy)
    return out + grid.area_p * beta


def _bdot(u, v):
    return (u * v).flatten(1).sum(1)


def cg_solve(grid, Dc, beta, rhs, free, x0, tol=1e-6, maxiter=500):
    """Solve P A P x = P rhs for the free pixels (x = x0 on the others), preconditioned CG, batched.
    rhs, x0, free: (B, H, W). Returns (x, relative residual per sample, iterations)."""
    free = free.to(rhs.dtype)
    x = x0.clone()
    r = (rhs - apply_operator(grid, Dc, beta, x)) * free
    Minv = free / operator_diag(grid, Dc, beta).clamp(min=1e-12)
    z = Minv * r
    p = z.clone()
    rz = _bdot(r, z)
    bnorm = _bdot(rhs * free, rhs * free).sqrt().clamp(min=1e-30)
    it = 0
    for it in range(1, maxiter + 1):
        Ap = apply_operator(grid, Dc, beta, p) * free
        alpha = rz / _bdot(p, Ap).clamp(min=1e-30)
        x = x + alpha.view(-1, 1, 1) * p
        r = r - alpha.view(-1, 1, 1) * Ap
        rel = _bdot(r, r).sqrt() / bnorm
        if bool((rel < tol).all()):
            break
        z = Minv * r
        rz_new = _bdot(r, z)
        p = z + (rz_new / rz.clamp(min=1e-30)).view(-1, 1, 1) * p
        rz = rz_new
    return x, rel, it


def residual_free(grid, Dc, beta, prior, fixed, free, x):
    """F(x, theta) = P (b - A x) with x = fixed on the Dirichlet pixels; b = a_p beta prior."""
    xf = torch.where(free, x, fixed)
    b = grid.area_p * beta * prior
    return (b - apply_operator(grid, Dc, beta, xf)) * free.to(x.dtype)


class SteadyState(torch.autograd.Function):
    """x* = argmin E(C) with Dirichlet data; backward via the implicit function theorem."""

    @staticmethod
    def forward(ctx, Dc, beta, prior, fixed, free, grid, tol, maxiter):
        with torch.no_grad():
            b = grid.area_p * beta * prior
            x0 = torch.where(free, prior, fixed)
            x, rel, it = cg_solve(grid, Dc, beta, b, free, x0, tol, maxiter)
            x = torch.where(free, x, fixed)
        ctx.save_for_backward(Dc, beta, prior, fixed, free, x)
        ctx.grid, ctx.tol, ctx.maxiter = grid, tol, maxiter
        LAST_STATS.update(fwd_rel=rel.max().item(), fwd_it=it)
        return x

    @staticmethod
    def backward(ctx, gbar):
        Dc, beta, prior, fixed, free, x = ctx.saved_tensors
        grid = ctx.grid
        with torch.no_grad():
            zero = torch.zeros_like(x)
            lam, rel, it = cg_solve(grid, Dc, beta, gbar * free.to(gbar.dtype), free, zero, ctx.tol, ctx.maxiter)
        LAST_STATS.update(bwd_rel=rel.max().item(), bwd_it=it)
        with torch.enable_grad():
            Dc_, beta_, prior_, fixed_ = (t.detach().requires_grad_(True) for t in (Dc, beta, prior, fixed))
            r = residual_free(grid, Dc_, beta_, prior_, fixed_, free, x.detach())
            gD, gb, gp, gf = torch.autograd.grad((lam * r).sum(), [Dc_, beta_, prior_, fixed_], allow_unused=True)
        gf = (gf if gf is not None else 0.0) + gbar * (~free).to(gbar.dtype)
        return gD, gb, gp, gf, None, None, None, None


def steady_state(grid, Dc, beta, prior, fixed, free, tol=1e-6, maxiter=500):
    """Differentiable steady state. beta: tensor broadcastable to (B, H, W). Returns (B, H, W)."""
    beta = torch.as_tensor(beta, dtype=prior.dtype, device=prior.device).expand_as(prior).contiguous()
    return SteadyState.apply(Dc, beta, prior, fixed, free, grid, tol, maxiter)


# ---------------------------------------------------------------- D models

def tensor_from_axes(cos_phi, sin_phi, l1, l2):
    """(Dxx, Dxy, Dyy) of R(phi) diag(l1, l2) R(phi)^T with the first axis along (cos_phi, sin_phi) in (x, y)."""
    dxx = l1 * cos_phi ** 2 + l2 * sin_phi ** 2
    dyy = l1 * sin_phi ** 2 + l2 * cos_phi ** 2
    dxy = (l1 - l2) * cos_phi * sin_phi
    return torch.stack([dxx, dxy, dyy], 1)


def tensor_eig(Dc):
    """Eigenvalues (larger, smaller) and the angle of the larger eigenvector; Dc (B, 3, H-1, W-1)."""
    dxx, dxy, dyy = Dc[:, 0], Dc[:, 1], Dc[:, 2]
    tr, det = dxx + dyy, dxx * dyy - dxy ** 2
    disc = (tr ** 2 / 4 - det).clamp(min=0).sqrt()
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    phi = 0.5 * torch.atan2(2 * dxy, dxx - dyy)
    return l1, l2, phi


def normalise_trace(Dc, corner_ocean, target=2.0):
    """Divide D by its per-sample mean trace over ocean corners / target, so mean trace(D) = target (= 2: D = I on
    average). Removes the common scale of D, beta and the neighbour weights w_k. The mean is NOT detached: the gradient
    of the loss with respect to the overall scale of the D-net output is then exactly zero. Dc (B, 3, H-1, W-1)."""
    m = corner_ocean.to(Dc.dtype)
    tr = ((Dc[:, 0] + Dc[:, 2]) * m).flatten(1).sum(1) / m.sum().clamp(min=1)
    return Dc * (target / tr.clamp(min=1e-8)).view(-1, 1, 1, 1)


class IsoD(nn.Module):
    def __init__(self, d0=1.0):
        super().__init__()
        self.log_d = nn.Parameter(torch.tensor(math.log(d0)))

    def forward(self, feats):
        d = self.log_d.exp()
        gx = feats["sst_gx"]
        z = torch.zeros_like(gx)
        return torch.stack([d + z, z, d + z], 1)


class SSTTensorD(nn.Module):
    """Diffusion d0 along SST isotherms, d0 / (1 + (|grad SST| / g0)^2) across them; isotropic where SST is missing."""

    def __init__(self, d0=1.0, g0=0.5):
        super().__init__()
        self.log_d0 = nn.Parameter(torch.tensor(math.log(d0)))
        self.log_g0 = nn.Parameter(torch.tensor(math.log(g0)))

    def forward(self, feats):
        gx, gy, ok = feats["sst_gx"], feats["sst_gy"], feats["sst_ok"]
        d0, g0 = self.log_d0.exp(), self.log_g0.exp()
        mag = (gx ** 2 + gy ** 2).sqrt()
        # isotherm tangent = grad SST rotated by 90 degrees
        cos_phi, sin_phi = -gy / mag.clamp(min=1e-8), gx / mag.clamp(min=1e-8)
        l_par = d0 * torch.ones_like(mag)
        l_perp = d0 / (1 + (mag / g0) ** 2)
        l_perp = torch.where(ok, l_perp, l_par)
        return tensor_from_axes(cos_phi, sin_phi, l_par, l_perp)


class MLPD(nn.Module):
    """1x1-conv MLP: corner covariates -> (phi via (cos, sin), l1, l2 in (0, d_max))."""

    def __init__(self, feature_names, hidden=32, d_max=4.0):
        super().__init__()
        self.names = list(feature_names)
        self.d_max = d_max
        self.net = nn.Sequential(nn.Conv2d(len(self.names), hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 4, 1))
        # init: small weights, orientation bias (1, 0) so the direction is defined, eigenvalues d_max*sigmoid(b) = 1 (isotropic)
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([1.0, 0.0, math.log(1 / (d_max - 1)), math.log(1 / (d_max - 1))]))

    def forward(self, feats):
        x = torch.stack([feats[n] for n in self.names], 1)
        o = self.net(x)
        v = o[:, :2]
        nrm = v.norm(dim=1).clamp(min=1e-6)
        cos_phi, sin_phi = v[:, 0] / nrm, v[:, 1] / nrm
        l1 = self.d_max * torch.sigmoid(o[:, 2])
        l2 = self.d_max * torch.sigmoid(o[:, 3])
        return tensor_from_axes(cos_phi, sin_phi, l1, l2)


class SSTResidualMLPD(MLPD):
    """MLP correction on top of the SST tensor: axis = SST isotherm tangent rotated by a learned angle, eigenvalues
    l = d_max sigmoid(logit(l_sst / d_max) + m) with (m1, m2) learned. At init the net outputs zero rotation and
    m = 0, so the model starts exactly at SSTTensorD(d0, g0) (both scalars also learned) and only departs from it
    where the covariates say so."""

    def __init__(self, feature_names, hidden=32, d_max=4.0, d0=1.0, g0=0.5):
        super().__init__(feature_names, hidden=hidden, d_max=d_max)
        self.base = SSTTensorD(d0, g0)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))   # (cos, sin) of zero rotation, zero log-multipliers

    def forward(self, feats):
        gx, gy, ok = feats["sst_gx"], feats["sst_gy"], feats["sst_ok"]
        d0, g0 = self.base.log_d0.exp(), self.base.log_g0.exp()
        mag = (gx ** 2 + gy ** 2).sqrt()
        c0, s0 = -gy / mag.clamp(min=1e-8), gx / mag.clamp(min=1e-8)
        l_par = d0 * torch.ones_like(mag)
        l_perp = torch.where(ok, d0 / (1 + (mag / g0) ** 2), l_par)
        x = torch.stack([feats[n] for n in self.names], 1)
        o = self.net(x)
        v = o[:, :2]; nrm = v.norm(dim=1).clamp(min=1e-6)
        cd, sd = v[:, 0] / nrm, v[:, 1] / nrm                    # rotation by delta phi
        cos_phi, sin_phi = c0 * cd - s0 * sd, s0 * cd + c0 * sd
        eps = 1e-4
        logit = lambda l: torch.logit((l / self.d_max).clamp(eps, 1 - eps))
        l1 = self.d_max * torch.sigmoid(logit(l_par) + o[:, 2])
        l2 = self.d_max * torch.sigmoid(logit(l_perp) + o[:, 3])
        return tensor_from_axes(cos_phi, sin_phi, l1, l2)


MLP_FEATURES = ["sst_gx", "sst_gy", "sst_ok", "u_cur", "v_cur", "cur_ok", "u_wind", "v_wind", "log_depth"]
EXTRA_FEATURES = ["sst_mag", "clim_gx", "clim_gy", "depth_gx", "depth_gy", "lat", "clim"]


def masked_grad(grid, field, valid):
    """Corner gradient of `field` using only corners whose four pixels are all valid; elsewhere 0. Returns gx, gy, ok."""
    v = valid.to(field.dtype)
    gx, gy = grid.grad(field * v)
    frac, _ = grid.to_corners(v)
    ok = frac > 0.999
    return torch.where(ok, gx, torch.zeros_like(gx)), torch.where(ok, gy, torch.zeros_like(gy)), ok


def corner_features(grid, sst, sst_ok, u_cur, v_cur, cur_ok, u_wind, v_wind, log_depth, clim=None, lat=None, extra=None):
    """Covariates at corners. sst: z-scored, NaN -> 0 with sst_ok (B,H,W) its validity. Currents: v is positive
    along increasing row index (the caller flips the sign when latitude descends). clim: (B,H,W) z-scored climatology,
    lat: (H,W) normalised latitude; extra: dict of further (B,H,W) pixel fields averaged to corners over ocean pixels.
    Returns dict of (B, H-1, W-1)."""
    gx, gy, sst_ok_c = masked_grad(grid, sst, sst_ok)
    uc, nc = grid.to_corners(u_cur, cur_ok); vc, _ = grid.to_corners(v_cur, cur_ok)
    uw, _ = grid.to_corners(u_wind); vw, _ = grid.to_corners(v_wind)
    oc = grid.ocean.expand_as(sst)
    ld, _ = grid.to_corners(log_depth.expand_as(sst), oc)
    f = dict(sst_gx=gx, sst_gy=gy, sst_ok=sst_ok_c, u_cur=uc, v_cur=vc, cur_ok=(nc > 0).to(sst.dtype),
             u_wind=uw, v_wind=vw, log_depth=ld, sst_mag=(gx ** 2 + gy ** 2).sqrt())
    dgx, dgy = grid.grad(log_depth.expand_as(sst) * oc.to(sst.dtype))
    f["depth_gx"], f["depth_gy"] = dgx, dgy
    if clim is not None:
        cgx, cgy = grid.grad(clim * oc.to(sst.dtype))
        f["clim_gx"], f["clim_gy"] = cgx, cgy
        f["clim"], _ = grid.to_corners(clim, oc)
    if lat is not None:
        f["lat"], _ = grid.to_corners(lat.expand_as(sst))
    for k, v in (extra or {}).items():
        f[k], _ = grid.to_corners(v, oc)
    return f


class ConvD(nn.Module):
    """Convolutional D-net: pixel-level covariate stack -> 3x3 conv layers -> averaged to corners -> 1x1 head to
    (orientation, two eigenvalues in (0, d_max)). Gives the tensor a neighbourhood view (front detection needs more
    than a point gradient)."""

    def __init__(self, in_ch, width=32, depth=3, d_max=4.0, dilation=1, channels=None):
        super().__init__()
        self.d_max = d_max
        self.channels = None if channels is None else list(channels)   # optional subset of input channel indices
        if self.channels is not None:
            in_ch = len(self.channels)
        layers, c = [], in_ch
        for i in range(depth):
            dil = dilation ** i
            layers += [nn.Conv2d(c, width, 3, padding=dil, dilation=dil), nn.GELU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, 4, 1)
        nn.init.normal_(self.head.weight, std=1e-2)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1.0, 0.0, math.log(1 / (d_max - 1)), math.log(1 / (d_max - 1))]))

    def forward(self, x):
        """x: (B, C, H, W) pixel stack. Returns (B, 3, H-1, W-1)."""
        if self.channels is not None:
            x = x[:, self.channels]
        h = self.body(x)
        h = F.avg_pool2d(h, 2, stride=1)                   # pixel features -> corners
        o = self.head(h)
        v = o[:, :2]; nrm = v.norm(dim=1).clamp(min=1e-6)
        l1 = self.d_max * torch.sigmoid(o[:, 2]); l2 = self.d_max * torch.sigmoid(o[:, 3])
        return tensor_from_axes(v[:, 0] / nrm, v[:, 1] / nrm, l1, l2)


def build_dmodel(kind, **kw):
    if kind == "iso":
        return IsoD(kw.get("d0", 1.0))
    if kind == "sst":
        return SSTTensorD(kw.get("d0", 1.0), kw.get("g0", 0.5))
    if kind == "mlp":
        drop = set(kw.get("drop", ()))                                   # feature names to remove (ablations via --dkw)
        feats = [f for f in kw.get("features", MLP_FEATURES) if f not in drop]
        if kw.get("sst_residual", False):
            return SSTResidualMLPD(feats, hidden=kw.get("hidden", 32), d_max=kw.get("d_max", 4.0), d0=kw.get("d0", 1.0), g0=kw.get("g0", 0.5))
        return MLPD(feats, hidden=kw.get("hidden", 32), d_max=kw.get("d_max", 4.0))
    if kind == "conv":
        return ConvD(kw["in_ch"], width=kw.get("hidden", 32), depth=kw.get("depth", 3), d_max=kw.get("d_max", 4.0), dilation=kw.get("dilation", 1),
                     channels=kw.get("channels", None))                  # channels: subset of pixel-stack indices (see train_spatial.pixel_stack)
    if kind in DMODEL_REGISTRY:
        return DMODEL_REGISTRY[kind](**kw)
    raise ValueError(kind)


DMODEL_REGISTRY = {}   # name -> callable(**kw) -> nn.Module producing (B, 3, H-1, W-1); register new D models here

class ConvDS(ConvD):
    """ConvD plus a learned additive source field g(x) at pixels (the g_theta of the pseudo-time equation
    dC/ds = div(D grad C) + g + data terms). forward returns (Dc, g); g (B, H, W) starts at zero."""

    def __init__(self, in_ch, width=32, depth=3, d_max=4.0, dilation=1, channels=None, g_scale=1.0):
        super().__init__(in_ch, width=width, depth=depth, d_max=d_max, dilation=dilation, **({"channels": channels} if channels is not None else {}))
        self.g_scale = g_scale
        self.src = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.src.weight); nn.init.zeros_(self.src.bias)

    def forward(self, x):
        if getattr(self, "channels", None) is not None:
            x = x[:, self.channels]
        h = self.body(x)
        g = self.g_scale * torch.tanh(0.02 * self.src(h)[:, 0])   # bounded prior shift |g| <= g_scale (ln CHL); the 0.02 damps the head's effective learning rate
        hc = F.avg_pool2d(h, 2, stride=1)
        o = self.head(hc)
        v = o[:, :2]; nrm = v.norm(dim=1).clamp(min=1e-6)
        l1 = self.d_max * torch.sigmoid(o[:, 2]); l2 = self.d_max * torch.sigmoid(o[:, 3])
        return tensor_from_axes(v[:, 0] / nrm, v[:, 1] / nrm, l1, l2), g


DMODEL_REGISTRY["convsrc"] = lambda **kw: ConvDS(kw["in_ch"], width=kw.get("hidden", 32), depth=kw.get("depth", 3), d_max=kw.get("d_max", 4.0),
                                                 dilation=kw.get("dilation", 1), channels=kw.get("channels"), g_scale=kw.get("g_scale", 1.0))
