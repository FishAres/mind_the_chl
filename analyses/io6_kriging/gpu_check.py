"""GPU check of the kriging stem (chl_cnn/krig.py) at the IO size, on random inputs (no store needed). Run under the GPU lock.

  agree   Triton kernels (chl_cnn/krig_triton.py, the training path) vs the eager PyTorch streamed reference (verified against a dense
          softmax on CPU, test_krig.py): forward outputs and every parameter gradient, batch 2; max |diff| of the outputs and max relative
          |diff| of the gradients. tl.dot precision "ieee" (fp32; criteria outputs < 1e-4, gradients < 1e-3 relative to each tensor's max)
          and "tf32" (the training setting; criteria outputs < 1e-3, gradients < 1e-2)
  inputs  random fields on a 149 x 181 grid with a smooth synthetic ocean mask; visibility = spatially clustered (Gaussian-smoothed noise
          thresholded at ~15 % of ocean px per day, as cloudy IO days), so that the kernels' skipping of offsets with no visible key in a
          16-px block is exercised as on real data
  timing  forward + backward of the stem alone at batch 8 (Triton path): median of 5 steps after 2 warm-up steps, peak memory allocated,
          no-grad forward; both variants (K1, K1-signed); --torch-timing adds the compiled PyTorch path for --chunks

Usage (project root): python -m analyses.io6_kriging.gpu_check [--chunks 32 16] [--dim 32] [--out results/gpu_check.json]
Exit code 1 if the agreement check fails.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from chl_cnn import krig as KR
from chl_cnn import krig_triton as KT

RES = Path(__file__).resolve().parent / "results"
H, W, T, TI = 149, 181, 17, 8


def inputs(B, dev, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = np.mgrid[:H, :W]
    ocean = torch.from_numpy(((np.sin(yy / 17.0) + np.cos(xx / 23.0)) > -0.6) & ~((yy < 40) & (xx < 70)))
    x3d = torch.randn(B, 11, T, H, W, generator=g)
    from scipy.ndimage import gaussian_filter
    noise = torch.from_numpy(gaussian_filter(np.random.default_rng(seed).standard_normal((B, T, H, W)), (0, 0, 4, 4))).float()
    thr = torch.quantile(noise.flatten()[::97], 0.85)
    vis = (noise > thr) & ocean
    x3d[:, -1] = vis.float(); x3d[:, -2] *= vis
    xs = torch.randn(B, 7, H, W, generator=g)
    ok = torch.ones(B, T, dtype=torch.bool)
    cl = 0.3 * torch.randn(B, T, H, W, generator=g)
    lat = torch.linspace(30, -12, H).view(-1, 1).expand(H, W)
    vel = dict(mu_u=0.048, sd_u=0.311, mu_v=0.002, sd_v=0.221, dx=27830.0 * torch.cos(torch.deg2rad(lat)), dy=torch.full((H, W), -27830.0))
    return ocean, [t.to(dev) for t in (x3d, xs, ok, cl)], vel


def build(ocean, vel, dev, impl, chunk=32, dim=32, signed=False, seed=1, compile=False):
    torch.manual_seed(seed)
    s = KR.KrigStem(9, 7, T, TI, ocean, 0, 1, 7, 8, vel, dim=dim, chunk=chunk, signed=signed, compile=compile, impl=impl).to(dev)
    with torch.no_grad():
        s.relbias.normal_(0, 0.5)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks", type=int, nargs="+", default=[32])
    p.add_argument("--torch-timing", action="store_true")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default=str(RES / "gpu_check.json"))
    a = p.parse_args()
    dev = torch.device("cuda")
    res = dict(grid=[H, W], agree={}, timing={})
    ocean, x, vel = inputs(2, dev)
    res["n_ocean"] = int(ocean.sum())
    fail = False
    for signed in (False, True):
        tag = "K1_signed" if signed else "K1"
        outs, grads = {}, {}
        for impl in ("torch", "ieee", "tf32"):                      # eager PyTorch reference, Triton fp32, Triton tf32
            KT.CFG["PREC"] = impl if impl != "torch" else KT.CFG["PREC"]
            s = build(ocean, vel, dev, "torch" if impl == "torch" else "triton", dim=a.dim, signed=signed)
            s.train()
            out, _ = s(*x)
            gout = torch.randn(out.shape, generator=torch.Generator().manual_seed(5)).to(dev)
            (out * gout).sum().backward()
            outs[impl] = out.detach()
            grads[impl] = {n: q.grad.detach().clone() for n, q in s.named_parameters() if q.grad is not None}
        for impl, tol_o, tol_g in (("ieee", 1e-4, 1e-3), ("tf32", 1e-3, 1e-2)):
            d_out = (outs[impl] - outs["torch"]).abs().max().item()
            d_g = {n: ((grads[impl][n] - grads["torch"][n]).abs().max() / grads["torch"][n].abs().max().clamp(min=1e-12)).item() for n in grads["torch"]}
            ok_ = d_out < tol_o and max(d_g.values()) < tol_g and set(grads[impl]) == set(grads["torch"])
            res["agree"][f"{tag}_{impl}"] = dict(max_abs_out=d_out, out_scale=outs["torch"].abs().max().item(), max_rel_grad=max(d_g.values()),
                                                 worst=max(d_g, key=d_g.get), passed=bool(ok_))
            print(tag, impl, "agree", json.dumps(res["agree"][f"{tag}_{impl}"]), flush=True)
            fail |= not ok_
    KT.CFG["PREC"] = "tf32"
    del x
    torch.cuda.empty_cache()
    ocean, x, vel = inputs(a.batch, dev)
    cfgs = [(sg, "triton", 0) for sg in (False, True)] + ([(sg, "torch", c) for sg in (False, True) for c in a.chunks] if a.torch_timing else [])
    for signed, impl, chunk in cfgs:
            tag = f"{'K1_signed' if signed else 'K1'}_{impl}{chunk or ''}_dim{a.dim}"
            s = build(ocean, vel, dev, impl, chunk=chunk or 32, dim=a.dim, signed=signed, compile=impl == "torch")
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ts = []
            t_first = None
            for it in range(7):
                torch.cuda.synchronize(); t0 = time.time()
                out, _ = s(*x)
                out.square().mean().backward()
                torch.cuda.synchronize(); dt = time.time() - t0
                if it == 0:
                    t_first = dt
                if it >= 2:
                    ts.append(dt)
                s.zero_grad(set_to_none=True)
            peak = (torch.cuda.max_memory_allocated() - base) / 1e9
            with torch.no_grad():
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(3):
                    s(*x)
                torch.cuda.synchronize(); t_eval = (time.time() - t0) / 3
            res["timing"][tag] = dict(batch=a.batch, s_per_step_fwd_bwd=float(np.median(ts)), s_first_step_incl_compile=t_first,
                                      s_per_forward_no_grad=t_eval, peak_gb_above_inputs=peak,
                                      est_s_per_epoch_stem_only=float(np.median(ts)) * int(np.ceil(5540 / a.batch)))
            print(tag, json.dumps(res["timing"][tag]), flush=True)
            del s
            torch.cuda.empty_cache()
    RES.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("GPU CHECK", "FAILED" if fail else "PASSED", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
