"""CPU unit tests of the blk9 two-level token model (chl_cnn/tokdec.py --tokdec-spatial-pool S). python -m analyses.blk9.test_blk9
Exits non-zero on a failure.

 1. Default path unchanged: TokDecNet(blend, spatial 2) == the pre-blk9 tokdec.py (tokdec.py.pre_blk9.bak) with the same weights,
    after perturbing the zero-init spatial weights (same state-dict keys, identical output).
 2. 177 x 273 (blk4 crop), P 4, S 3: patch grid 45 x 69 -> coarse 15 x 23; coarse ocean cells == the P 12 net's ocean patches on the same mask;
    forward / backward, output (B, 3, H, W), finite grads, every parameter gets a grad.
 3. Pooling / padding: on 22 x 26 (patch grid 6 x 7, padded to 6 x 9 for S 3 and 8 x 8 for S 4) the coarse tokens == a brute-force loop mean
    over the ocean patches of each S x S block; all-land blocks are not tokens; patches map to the right cell.
 4. Block == a float64 brute-force reference (tol 1e-6: the block pools in float32) (explicit softmax with the relative bias, nearest copy back, MLP) with random weights.
 5. Nearest upsampling: the attention update is identical for all patches of a coarse cell.
 6. Zero init: S 3 net == TokDecNet(blend, no spatial blocks) with the same weights.
 7. day_ok masking still works with S 3 (a masked, not-visible day's features / climatology change nothing). Land patches: see 3 (never tokens, never pooled).
 8. CLI: --tokdec-spatial-pool default 1; S >= 2 without --tokdec-spatial, or with --physmem, is a parse error.
"""
import importlib.util
import sys
from pathlib import Path

import torch

from chl_cnn import tokdec as TD
from chl_cnn.tokdec import TokDecNet

FAIL = []
KW = dict(dim=32, ch=8, heads=4, layers=1, width=16, out_channels=3, dropout=0.0, dilations=None)


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    if not ok:
        FAIL.append(name)


def inputs(B, F, T, H, W, seed=0, p_land=0.2):
    g = torch.Generator().manual_seed(seed)
    ocean = torch.rand(H, W, generator=g) > p_land
    ocean[: H // 3, : W // 3] = False                                            # a land block (all-land coarse cells)
    x3d = torch.randn(B, F + 2, T, H, W, generator=g); vis = torch.rand(B, T, H, W, generator=g) > 0.4
    x3d[:, -1] = vis.float(); x3d[:, -2] *= vis
    return ocean, x3d, torch.randn(B, 6, H, W, generator=g), torch.ones(B, T, dtype=torch.bool), torch.randn(B, T, H, W, generator=g)


def perturb(net, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for blk in net.sp_blocks:
            for prm in (blk.proj.weight, blk.proj.bias, blk.ff[2].weight, blk.ff[2].bias):
                prm.copy_(0.05 * torch.randn(prm.shape, generator=g))
            blk.rel.copy_(torch.randn(blk.rel.shape, generator=g))


def test_default_unchanged():
    p = Path(TD.__file__).with_name("tokdec.py.pre_blk9.bak")
    spec = importlib.util.spec_from_loader("tokdec_old", loader=None); old = importlib.util.module_from_spec(spec)
    old.__dict__["__package__"] = "chl_cnn"; exec(compile(p.read_text(), str(p), "exec"), old.__dict__)
    F, T = 8, 17; ocean, x3d, xs, ok, clim = inputs(2, F, T, 22, 26)
    torch.manual_seed(0); new = TokDecNet(F + 2, 6, T, 8, ocean, blend=True, spatial=2, **KW).eval()
    torch.manual_seed(0); ref = old.TokDecNet(F + 2, 6, T, 8, ocean, blend=True, spatial=2, **KW).eval()
    perturb(new); ref.load_state_dict(new.state_dict())
    with torch.no_grad():
        y0, y1 = ref(x3d, xs, ok, clim), new(x3d, xs, ok, clim)
    check("1 default path (spatial 2, pool 1) == pre-blk9 tokdec.py: same keys, identical output",
          list(ref.state_dict()) == list(new.state_dict()) and torch.equal(y0, y1) and type(new.sp_blocks[0]) is TD._SpatialBlock)


def test_blk4_shape():
    H, W, F, T, B = 177, 273, 8, 17, 1
    ocean, x3d, xs, ok, clim = inputs(B, F, T, H, W)
    torch.manual_seed(0); net = TokDecNet(F + 2, 6, T, 8, ocean, blend=True, spatial=2, spatial_pool=3, P=4, **KW)
    p12 = TokDecNet(F + 2, 6, T, 8, ocean, blend=True, spatial=2, P=12, **KW)
    perturb(net)
    y = net(x3d, xs, ok, clim); y.square().mean().backward()
    nog = [n for n, p in net.named_parameters() if p.grad is None]
    print(f"  P 4 grid {net.nPy} x {net.nPx} ({net.Np} ocean patches) -> coarse {net.nCy} x {net.nCx} ({net.Nc} cells); P 12 net {p12.Np} patches; "
          f"rel table {net.sp_blocks[0].rel.shape[0]}; ocean patches / cell {net.sp_count.min():.0f}..{net.sp_count.max():.0f}", flush=True)
    check("2 177 x 273, P 4, S 3: 45 x 69 -> 15 x 23, coarse cells == P 12 ocean patches, output shape, finite grads, all params get grads",
          (net.nPy, net.nPx, net.nCy, net.nCx) == (45, 69, 15, 23) and torch.equal(net.sp_cell, p12.patch_idx) and tuple(y.shape) == (B, 3, H, W)
          and not nog and all(torch.isfinite(p.grad).all() for p in net.parameters()))


def brute_pool(net, x, S):
    """loop reference: coarse tokens over the padded nCy x nCx grid (None for all-land cells)."""
    py, px = (net.patch_idx // net.nPx).tolist(), (net.patch_idx % net.nPx).tolist()
    cells = {}
    for i, (a, b) in enumerate(zip(py, px)):
        cells.setdefault((a // S) * net.nCx + b // S, []).append(i)
    keys = sorted(cells)
    return keys, torch.stack([x[:, cells[k]].double().mean(1) for k in keys], 1), cells


def test_pool_padding():
    F, T = 8, 5
    for S in (3, 4):
        ocean, *_ = inputs(1, F, T, 22, 26, seed=S)
        ocean[:16, :16] = False                                                  # >= one all-land coarse cell for S 3 and S 4
        net = TokDecNet(F + 2, 6, T, 2, ocean, blend=True, spatial=1, spatial_pool=S, **KW)
        x = torch.randn(2, net.Np, 32, dtype=torch.float64).float()
        keys, ref, cells = brute_pool(net, x, S)
        xc = x.new_zeros((2, net.Nc, 32)).index_add_(1, net.sp_parent, x) / net.sp_count.view(1, -1, 1)
        par_ok = all(net.sp_parent[i].item() == keys.index(k) for k, ii in cells.items() for i in ii)
        land = (net.nCy * net.nCx) - net.Nc
        check(f"3 pooling S {S}: patch grid {net.nPy} x {net.nPx} padded to {net.nCy * S} x {net.nCx * S}, {net.Nc} coarse cells "
              f"({land} all-land cells dropped): == brute-force mean over ocean patches, parent map",
              keys == net.sp_cell.tolist() and par_ok and torch.allclose(xc.double(), ref, atol=1e-6) and land > 0
              and (net.nCy, net.nCx) == (-(-net.nPy // S), -(-net.nPx // S)))


def test_block_reference():
    F, T = 8, 5
    ocean, *_ = inputs(1, F, T, 22, 26, seed=3)
    net = TokDecNet(F + 2, 6, T, 2, ocean, blend=True, spatial=1, spatial_pool=3, **KW).double()
    perturb(net)
    blk = net.sp_blocks[0]
    x = torch.randn(2, net.Np, 32, dtype=torch.float64)
    y = blk(x, net.rel_idx, net.sp_parent, net.sp_count)
    keys, xc, cells = brute_pool(net, x, 3)
    h, d = 4, 8
    qkv = blk.qkv(blk.n1(xc)).view(2, net.Nc, 3, h, d)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    cy, cx = [c // net.nCx for c in keys], [c % net.nCx for c in keys]
    bias = torch.empty(h, net.Nc, net.Nc, dtype=torch.float64)
    for i in range(net.Nc):
        for j in range(net.Nc):
            bias[:, i, j] = blk.rel[(cy[i] - cy[j] + net.nCy - 1) * (2 * net.nCx - 1) + (cx[i] - cx[j] + net.nCx - 1)]
    att = torch.softmax(torch.einsum("bihd,bjhd->bhij", q, k) / d ** 0.5 + bias, -1)
    o = blk.proj(torch.einsum("bhij,bjhd->bihd", att, v).reshape(2, net.Nc, 32))
    up = torch.stack([o[:, keys.index(c)] for i in range(net.Np) for c in [next(k_ for k_, ii in cells.items() if i in ii)]], 1)
    ref = x + up; ref = ref + blk.ff(blk.n2(ref))
    check(f"4 block == float64 brute-force reference (max |diff| {(y - ref).abs().max():.1e})", torch.allclose(y, ref, rtol=0, atol=1e-6))   # the block pools in float32
    with torch.no_grad():
        blk.ff[2].weight.zero_(); blk.ff[2].bias.zero_()
        dlt = blk(x, net.rel_idx, net.sp_parent, net.sp_count) - x
    same = all((dlt[:, ii] - dlt[:, ii[:1]]).abs().max() < 1e-12 for ii in cells.values())
    check("5 nearest upsampling: attention update identical within each coarse cell", same and dlt.abs().max() > 1e-3)


def test_zero_init_and_masks():
    F, T = 8, 9
    ocean, x3d, xs, ok, clim = inputs(2, F, T, 22, 26, seed=5)
    torch.manual_seed(4); base = TokDecNet(F + 2, 6, T, 4, ocean, blend=True, **KW).eval()
    torch.manual_seed(4); net = TokDecNet(F + 2, 6, T, 4, ocean, blend=True, spatial=2, spatial_pool=3, **KW).eval()
    sb = base.state_dict(); sn = net.state_dict()
    with torch.no_grad():
        eq = all(torch.equal(sb[k], sn[k]) for k in sb) and torch.equal(base(x3d, xs, ok, clim), net(x3d, xs, ok, clim))
    check("6 zero init: S 3 net == TokDecNet(blend, no spatial) with the same weights", eq)
    perturb(net)
    with torch.no_grad():
        y0 = net(x3d, xs, ok, clim)
        xm = x3d.clone(); xm[:, -2:, 1] = 0                                     # a missing day: not visible (as the data pipeline gives it)
        ok2 = ok.clone(); ok2[:, 1] = False; x2 = xm.clone(); x2[:, :-2, 1] += 5.0; cl2 = clim.clone(); cl2[:, 1] += 3.0
        ya, yb = net(xm, xs, ok2, clim), net(x2, xs, ok2, cl2)
    check("7 day_ok mask: a masked day's inputs change nothing (and masking it changes the output)",
          torch.equal(ya, yb) and not torch.equal(ya, y0))


def test_cli():
    from chl_cnn.train_pinn import parse
    base = ["--run", "x", "--store", "blk4", "--model", "temporal3d", "--t3d-stem", "tokdec", "--clim", "--tokdec-blend", "--context", "8", "--feat-context", "8"]
    a = parse(base + ["--tokdec-spatial", "2"]); errs = []
    for extra in (["--tokdec-spatial-pool", "3"], ["--tokdec-spatial", "2", "--tokdec-spatial-pool", "3", "--physmem"]):
        try:
            parse(base + extra); errs.append(False)
        except SystemExit:
            errs.append(True)
    b = parse(base + ["--tokdec-spatial", "2", "--tokdec-spatial-pool", "3"])
    check("8 CLI: default 1; S 3 with --tokdec-spatial 2 parses; without spatial / with --physmem errors",
          a.tokdec_spatial_pool == 1 and b.tokdec_spatial_pool == 3 and all(errs))


def main():
    test_default_unchanged(); test_blk4_shape(); test_pool_padding(); test_block_reference(); test_zero_init_and_masks(); test_cli()
    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
