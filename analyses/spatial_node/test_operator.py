"""Unit checks for chl_cnn.spatial_node: transpose, symmetry, consistency with the continuous operator,
CG vs dense solve, implicit gradient vs autograd / finite differences, and a synthetic tilted-front gap.
Run: python analyses/spatial_node/test_operator.py   (figures -> analyses/spatial_node/figures/)"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chl_cnn import spatial_node as SN
from chl_cnn.plotting import OKABE_ITO

torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
FIG = Path(__file__).resolve().parent / "figures"
report = []


def ok(name, cond, detail=""):
    report.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def random_spd(B, H, W):
    l1 = torch.rand(B, H - 1, W - 1, device=dev) * 2 + 0.1
    l2 = torch.rand(B, H - 1, W - 1, device=dev) * 2 + 0.1
    phi = torch.rand(B, H - 1, W - 1, device=dev) * math.pi
    return SN.tensor_from_axes(torch.cos(phi), torch.sin(phi), l1, l2)


# ---- 1. grad_t is the transpose of grad (grid with land and varying latitude)
H, W, B = 23, 31, 3
ocean = torch.rand(H, W, device=dev) > 0.25
lat = torch.linspace(30, -5, H)
grid = SN.Grid(ocean, lat, dev)
c = torch.randn(B, H, W, device=dev); fx = torch.randn(B, H - 1, W - 1, device=dev); fy = torch.randn_like(fx)
gx, gy = grid.grad(c)
lhs = (gx * fx + gy * fy).sum(); rhs = (c * grid.grad_t(fx, fy)).sum()
ok("grad_t == grad^T", abs(lhs - rhs) < 1e-10 * abs(lhs), f"|<Gc,f> - <c,G^T f>| = {abs(lhs - rhs).item():.2e}")

# ---- 2. operator symmetric and positive semi-definite
Dc = random_spd(B, H, W); beta = torch.full((B, H, W), 0.1, device=dev)
x, y = torch.randn(B, H, W, device=dev), torch.randn(B, H, W, device=dev)
Ax, Ay = SN.apply_operator(grid, Dc, beta, x), SN.apply_operator(grid, Dc, beta, y)
sym = abs((x * Ay).sum() - (y * Ax).sum()).item()
ok("A symmetric", sym < 1e-10, f"|x^T A y - y^T A x| = {sym:.2e}")
ok("A positive definite on random x", bool(((x * Ax).flatten(1).sum(1) > 0).all()), f"min x^T A x = {(x * Ax).flatten(1).sum(1).min().item():.3f}")
dg = SN.operator_diag(grid, Dc, beta)
e = torch.zeros(1, H, W, device=dev); e[0, 7, 9] = 1
ok("operator_diag matches A e_i", abs(dg[0, 7, 9] - SN.apply_operator(grid, Dc[:1], beta[:1], e)[0, 7, 9]) < 1e-10)

# ---- 3. continuous-operator consistency on an all-ocean grid at the equator (cos lat = 1)
H2, W2 = 16, 20
g2 = SN.Grid(torch.ones(H2, W2, dtype=torch.bool), torch.zeros(H2), dev)
ii, jj = torch.meshgrid(torch.arange(H2, device=dev, dtype=torch.float64), torch.arange(W2, device=dev, dtype=torch.float64), indexing="ij")
Diso = torch.stack([torch.ones(1, H2 - 1, W2 - 1, device=dev), torch.zeros(1, H2 - 1, W2 - 1, device=dev), torch.ones(1, H2 - 1, W2 - 1, device=dev)], 1)
zero_b = torch.zeros(1, H2, W2, device=dev)
lin = (2 * jj - 3 * ii).unsqueeze(0)
r_lin = SN.apply_operator(g2, Diso, zero_b, lin)[0, 1:-1, 1:-1].abs().max().item()
ok("A(linear field) = 0 in the interior", r_lin < 1e-10, f"max |A c| = {r_lin:.2e}")
quad = (jj ** 2).unsqueeze(0)
r_quad = SN.apply_operator(g2, Diso, zero_b, quad)[0, 1:-1, 1:-1]
ok("A(x^2) = -2 Dxx in the interior (A = -div D grad)", (r_quad + 2).abs().max().item() < 1e-10, f"max |A c + 2| = {(r_quad + 2).abs().max().item():.2e}")
# anisotropic: D with l1 along x only: A(y^2) must be 0, A(x^2) = -2 l1
Dx_only = SN.tensor_from_axes(torch.ones(1, H2 - 1, W2 - 1, device=dev), torch.zeros(1, H2 - 1, W2 - 1, device=dev), torch.full((1, H2 - 1, W2 - 1), 1.5, device=dev), torch.zeros(1, H2 - 1, W2 - 1, device=dev))
ry = SN.apply_operator(g2, Dx_only, zero_b, (ii ** 2).unsqueeze(0))[0, 1:-1, 1:-1].abs().max().item()
rx = (SN.apply_operator(g2, Dx_only, zero_b, quad)[0, 1:-1, 1:-1] + 3).abs().max().item()
ok("anisotropic D: no diffusion across its null axis", ry < 1e-10 and rx < 1e-10, f"A(y^2)={ry:.1e}, A(x^2)+2*1.5={rx:.1e}")

# ---- 4. CG vs dense solve (Dirichlet on observed pixels)
prior = torch.randn(B, H, W, device=dev) * 0.3
obs = (torch.rand(B, H, W, device=dev) > 0.6) & grid.ocean
fixed = torch.where(obs, torch.randn(B, H, W, device=dev), torch.zeros_like(prior))
free = grid.ocean & ~obs
xs = SN.steady_state(grid, Dc, beta, prior, fixed, free, tol=1e-12, maxiter=2000)
# dense: columns of P A P
idx = torch.nonzero(free[0].flatten()).squeeze(1)
E = torch.zeros(len(idx), H * W, device=dev); E[torch.arange(len(idx)), idx] = 1
AE = SN.apply_operator(grid, Dc[:1].expand(len(idx), -1, -1, -1), beta[:1].expand(len(idx), -1, -1), E.view(-1, H, W)).view(len(idx), -1)[:, idx]
b = (grid.area_p * beta[0] * prior[0]).flatten()[idx] - SN.apply_operator(grid, Dc[:1], beta[:1], fixed[:1])[0].flatten()[idx]
xd = torch.linalg.solve(AE.T, b)
err = (xs[0].flatten()[idx] - xd).abs().max().item()
ok("CG solution == dense solve", err < 1e-8, f"max |x_cg - x_dense| = {err:.2e}, n_free = {len(idx)}, dense A symmetric err {(AE - AE.T).abs().max().item():.1e}")
ok("fixed pixels kept", bool((xs[obs] == fixed[obs]).all()))

# ---- 5. implicit gradient vs autograd through unrolled CG and vs finite differences
def loss_of(Dc_, beta_, prior_, fixed_, implicit=True):
    if implicit:
        x = SN.steady_state(grid, Dc_, beta_, prior_, fixed_, free, tol=1e-13, maxiter=3000)
    else:
        bb = grid.area_p * beta_ * prior_
        x0 = torch.where(free, prior_, fixed_)
        x, _, _ = SN.cg_solve(grid, Dc_, beta_, bb, free, x0, tol=1e-13, maxiter=3000)
        x = torch.where(free, x, fixed_)
    wgt = torch.randn_like(x)                                    # arbitrary smooth loss
    torch.manual_seed(1); wgt = torch.randn_like(x)
    return (x * wgt).sum() + (x ** 2 * grid.ocean).sum() * 0.1

leaves = [t.clone().requires_grad_(True) for t in (Dc, beta, prior, fixed)]
gi = torch.autograd.grad(loss_of(*leaves, implicit=True), leaves)
leaves2 = [t.clone().requires_grad_(True) for t in (Dc, beta, prior, fixed)]
gu = torch.autograd.grad(loss_of(*leaves2, implicit=False), leaves2)
for name, a_, b_ in zip(["dL/dD", "dL/dbeta", "dL/dprior", "dL/dfixed"], gi, gu):
    rel = (a_ - b_).norm() / b_.norm().clamp(min=1e-30)
    ok(f"implicit grad == unrolled autograd ({name})", rel < 1e-6, f"rel err {rel.item():.2e}")
# finite difference on one D entry and one prior entry
for name, k, (bi, ci, i, j) in [("D", 0, (1, 1, 5, 6)), ("prior", 2, (1, None, 8, 10))]:
    eps = 1e-5
    base = [t.clone() for t in (Dc, beta, prior, fixed)]
    plus = [t.clone() for t in base]; minus = [t.clone() for t in base]
    if ci is None:
        plus[k][bi, i, j] += eps; minus[k][bi, i, j] -= eps
        g_an = gi[k][bi, i, j].item()
    else:
        plus[k][bi, ci, i, j] += eps; minus[k][bi, ci, i, j] -= eps
        g_an = gi[k][bi, ci, i, j].item()
    g_fd = ((loss_of(*plus) - loss_of(*minus)) / (2 * eps)).item()
    ok(f"implicit grad == finite difference ({name})", abs(g_an - g_fd) < 1e-5 * max(1, abs(g_fd)), f"analytic {g_an:.6f} fd {g_fd:.6f}")

# ---- 6. synthetic tilted front with along-front modulation, band of clouds across it
N = 64
gN = SN.Grid(torch.ones(N, N, dtype=torch.bool), torch.zeros(N), dev)
yy, xx = torch.meshgrid(torch.arange(N, device=dev, dtype=torch.float64), torch.arange(N, device=dev, dtype=torch.float64), indexing="ij")
ang = math.radians(30)                                    # front direction
n_hat = (math.sin(ang), -math.cos(ang))                  # normal (x, y)
t_hat = (math.cos(ang), math.sin(ang))
s_n = (xx - N / 2) * n_hat[0] + (yy - N / 2) * n_hat[1]   # across-front coordinate
s_t = (xx - N / 2) * t_hat[0] + (yy - N / 2) * t_hat[1]   # along-front coordinate
truth = torch.tanh(s_n / 3.0) * (1 + 0.5 * torch.sin(2 * math.pi * s_t / 24)) + 0.3 * torch.sin(2 * math.pi * s_n / 20)
sst = torch.tanh(s_n / 4.0) * 3                           # SST shares the front geometry (z-scored scale)
truth, sst = truth.unsqueeze(0), sst.unsqueeze(0)
cloud = ((xx > 20) & (xx < 44)).unsqueeze(0)              # vertical band crossing the front
obsN = ~cloud
freeN = cloud.clone()
feats = SN.corner_features(gN, sst, torch.ones_like(sst, dtype=torch.bool), torch.zeros_like(sst), torch.zeros_like(sst), torch.ones_like(sst, dtype=torch.bool),
                           torch.zeros_like(sst), torch.zeros_like(sst), torch.zeros(N, N, device=dev))
priorN = torch.zeros_like(truth)
rows = {}
for name, model in [("isotropic D", SN.IsoD(1.0)), ("SST-aligned D (g0=0.3)", SN.SSTTensorD(1.0, 0.3))]:
    model = model.to(dev)
    with torch.no_grad():
        Dn = model(feats)
        fill = SN.steady_state(gN, Dn, torch.tensor(1e-3, device=dev), priorN, truth * obsN, freeN, tol=1e-10, maxiter=5000)
    mae = (fill - truth).abs()[cloud].mean().item()
    l1, l2, _ = SN.tensor_eig(Dn)
    rows[name] = dict(fill=fill[0].cpu().numpy(), mae=mae, aniso=(l1 / l2.clamp(min=1e-9)).median().item())
    print(f"synthetic front: {name:26s} MAE on hidden band {mae:.4f}   median l1/l2 {rows[name]['aniso']:.1f}")
ok("synthetic front: SST-aligned D beats isotropic D", rows["SST-aligned D (g0=0.3)"]["mae"] < rows["isotropic D"]["mae"],
   f"iso {rows['isotropic D']['mae']:.4f} vs aligned {rows['SST-aligned D (g0=0.3)']['mae']:.4f}")

fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
vmin, vmax = -1.6, 1.6
inp = truth[0].cpu().numpy().copy(); inp[cloud[0].cpu().numpy()] = np.nan
panels = [("Truth (tilted front, along-front modulation)", truth[0].cpu().numpy()), ("Observed input (band hidden)", inp)]
panels += [(f"Fill, {k}\nMAE on hidden band {v['mae']:.3f}", v["fill"]) for k, v in rows.items()]
for a_, (t, img) in zip(ax, panels):
    im = a_.imshow(img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax); a_.set_title(t, fontsize=10)
    a_.set_xlabel("x (pixel)"); a_.set_ylabel("y (pixel)")
    a_.contour(sst[0].cpu().numpy(), levels=[-2, 0, 2], colors="white", linewidths=0.6, alpha=0.7)
fig.colorbar(im, ax=ax, shrink=0.8, label="synthetic field value")
fig.suptitle("Synthetic test: steady-state fill of a band gap with isotropic vs SST-aligned diffusion (white: SST isotherms)")
fig.savefig(FIG / "synthetic_front.png", dpi=130, bbox_inches="tight")

n_fail = sum(not r[1] for r in report)
print(f"\n{len(report) - n_fail}/{len(report)} checks passed")
sys.exit(1 if n_fail else 0)
