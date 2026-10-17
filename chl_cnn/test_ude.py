"""Unit checks of chl_cnn/ude.py on synthetic grids (CPU). Run: python -m chl_cnn.test_ude"""
import math
import torch
from . import ude


def check_transport():
    H = W = 64
    ocean = torch.ones(H, W, dtype=torch.bool)
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    bump = torch.exp(-((xx - 24) ** 2 + (yy - 30) ** 2) / (2 * 4.0 ** 2))[None]
    tr = ude.Transport(ocean, n_sub=4, lag_max=20.0)
    ok = torch.ones(1, H, W, dtype=torch.bool)
    # uniform current of 1 cell/day along x: the bump should move to x = 25
    out, frac = tr(bump, torch.ones(1, H, W), torch.zeros(1, H, W), ok, dt=1.0)
    ref = torch.exp(-((xx - 25) ** 2 + (yy - 30) ** 2) / (2 * 4.0 ** 2))[None]
    inner = torch.zeros(H, W, dtype=torch.bool); inner[2:-2, 2:-2] = True
    e_x = (out - ref)[0][inner].abs().max().item()
    # same along y (1 cell/day): bump to y = 31
    out_y, _ = tr(bump, torch.zeros(1, H, W), torch.ones(1, H, W), ok, dt=1.0)
    ref_y = torch.exp(-((xx - 24) ** 2 + (yy - 31) ** 2) / (2 * 4.0 ** 2))[None]
    e_y = (out_y - ref_y)[0][inner].abs().max().item()
    # zero currents: identity
    out0, frac0 = tr(bump, torch.zeros(1, H, W), torch.zeros(1, H, W), ok, dt=1.0)
    e_0 = (out0 - bump).abs().max().item()
    # 3 cells/day over one day: x = 27
    out3, _ = tr(bump, 3 * torch.ones(1, H, W), torch.zeros(1, H, W), ok, dt=1.0)
    ref3 = torch.exp(-((xx - 27) ** 2 + (yy - 30) ** 2) / (2 * 4.0 ** 2))[None]
    e_3 = (out3 - ref3)[0][inner].abs().max().item()
    print(f"transport: 1 cell/day x err {e_x:.2e}  y err {e_y:.2e}  3 cells/day err {e_3:.2e}  zero-current err {e_0:.2e}  "
          f"not-advected frac {frac.item():.4f} (edge row/col of departure points outside the grid)")
    assert e_x < 1e-3 and e_y < 1e-3 and e_3 < 1e-3 and e_0 < 1e-6
    # mass check: the sum of the interior is conserved for a translation
    print(f"transport: sum in {bump.sum().item():.4f} out {out[0][inner].sum().item():.4f}")


def check_relax():
    H = W = 16
    ocean = torch.ones(H, W, dtype=torch.bool)
    clim = torch.full((1, H, W), 0.5)
    C0 = clim + torch.linspace(-1, 1, H * W).view(1, H, W)
    relax = ude.Relax(lam0=1.0)
    gfun = lambda k, C: relax(C, clim, 1.0)
    preds, _ = ude.rollout(C0, 2, ocean, transport=None, K=0.0, gfun=gfun, use_T=False, use_g=True)
    errs = []
    for k in (1, 2):
        ref = clim + (C0 - clim) * math.exp(-k)
        errs.append((preds[:, k - 1] - ref).abs().max().item())
    print(f"relax lambda=1/day: max |C_k - (clim + (C_0 - clim) e^-k)| at k=1 {errs[0]:.2e}, k=2 {errs[1]:.2e}")
    assert max(errs) < 1e-6


def check_laplacian():
    H = W = 8
    ocean = torch.ones(H, W, dtype=torch.bool); ocean[3, 3] = False
    C = torch.arange(H * W).float().view(1, H, W) ** 2   # quadratic in index -> lap of j^2 along a row
    lap = ude.laplacian(C, ocean)
    # interior pixel far from land: C = (8 i + j)^2 ; lap = 2 (along j) + 2 * 64 (along i) = 130
    print(f"laplacian interior (6,6): {lap[0, 6, 6].item():.1f} (expect 130); land pixel (3,3): {lap[0, 3, 3].item():.1f} (expect 0)")
    assert abs(lap[0, 6, 6].item() - 130) < 1e-3 and lap[0, 3, 3].item() == 0.0


if __name__ == "__main__":
    check_transport(); check_relax(); check_laplacian(); print("ok")
