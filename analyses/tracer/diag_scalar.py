"""Test 3a: training dynamics of the scalar tracer a x u_rea (tracer_diag.train_scalar: a = 1.5 sigmoid(theta), Adam,
batch 16, pixel L1 mean per lead, leads 1-5 equal weights, n_sub 4, bilinear; all train starts, shuffled per epoch,
seed 0). One lr per call, inits a0 in --inits. Val MAE (own set, the test-1 val starts) every 51 steps and at the end.
GPU; gpu.lock. Output results/diag_scalar_lr<lr>.json.
"""
import sys, json, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

RES = Path(__file__).resolve().parent / "results"
K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--inits", default="0.1,0.5,1.0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    from chl_cnn import tracer as TR, tracer_diag as TD, train_tracer as TT
    dev = torch.device("cuda")
    t0 = time.time()
    a, d = TD.setup(dev, K)
    H, W = d["ocean"].shape
    va = TD.calendar_ok(d, TT.starts(d["times"], d["va"], K)[::2], K, 1, 1)
    tr = TT.starts(d["times"], d["tr"], K)
    if args.debug:
        tr, va, args.epochs = tr[:16 * 10], va[:32], 1
    it = TR.Integrator(H, W, dev, n_sub=4, vmax=10.0)
    vel = TD.make_vel(d, "rea", 0, K)
    print(f"load {time.time() - t0:.0f}s  train starts {len(tr)}  val starts {len(va)}", flush=True)
    out = dict(lr=args.lr, epochs=args.epochs, n_train=int(len(tr)), n_val=int(len(va)), runs={})
    for a0 in [float(x) for x in args.inits.split(",")]:
        out["runs"][str(a0)] = TD.train_scalar(d, it, vel, tr, va, a0, args.lr, args.epochs)
        print(f"a0 {a0} final a {out['runs'][str(a0)]['a'][-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    (RES / f"diag_scalar_lr{args.lr:g}{'_debug' if args.debug else ''}.json").write_text(json.dumps(out))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
