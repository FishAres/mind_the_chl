"""Test 1: loss landscape of the scalar tracer u_eff = a x base field (no training), val days. GPU; run under gpu.lock.

Variants (--variant):
  base        reanalysis current, n_sub 4, bilinear departure sampling; + pixel subsets of accepted lead-1 spectral-motion
              patches (moving |shift| >= 1 px / zero shift) intersected with the fixed set.
  shift_m1 / shift_p1   currents of day j-1 / j+1 used for day j.
  nsub1 / nsub16        RK2 sub-steps per day.
  sampling    departure sampling nearest / bilinear / bicubic; one shared validity rule (bilinear observed fraction
              >= 0.99 and the full 4 x 4 bicubic support observed), fixed set = valid for every a and every mode.
  geo_mean    step-1b geo_mean component (train-years doy climatology of u_geo, x current validity) instead of u_rea.
  train_base  as base on every 3rd train start (pixel metrics only; for the comparison with trained a).
  check       --eval-only consistency: unfiltered trainer val starts, a in {0.25, 1.0}, own-set pixel MAE vs
              models/tracer/tr_rea_geo_wind metrics.json (adv_x0.25 / adv_x1.0, best).
Val starts: trainer val starts (every 2nd, leads 1-5) restricted to those with store days t-1..t+6 consecutive (so the
+-1 day shifts exist); the same starts for every val variant. a grid tracer_diag.ALPHAS (0..1.2 step 0.05, + 0.17).
Band: tracer.BandPass(24, 150 km), min_frac 0.5, filter mask = the scored set (fixed or own).
Output results/diag_landscape_<variant>.npz.
"""
import sys, json, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RES = HERE / "results"; RES.mkdir(exist_ok=True)
CACHE = ROOT / "analyses/blk4/results/spectral_motion_patches.npz"
K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--debug", action="store_true")
    v = ap.parse_args().variant
    debug = ap.parse_args().debug
    from chl_cnn import tracer as TR, tracer_diag as TD, train_tracer as TT
    dev = torch.device("cuda")
    t0 = time.time()
    a, d = TD.setup(dev, K)
    H, W = d["ocean"].shape
    va_all = TT.starts(d["times"], d["va"], K)[::2]
    va = TD.calendar_ok(d, va_all, K, 1, 1)
    print(f"load {time.time() - t0:.0f}s  val starts {len(va_all)} -> {len(va)} with t-1..t+6 consecutive", flush=True)
    bp = TR.BandPass(24.0, 150.0, min_frac=0.5, device=dev)
    alphas = TD.ALPHAS
    kw = dict(bp=bp, K=K)
    starts = va
    n_sub, kind, shift, modes, strict, band, subsets = 4, "rea", 0, ("bilinear",), False, True, None
    info = {}
    if v == "base":
        pm, cnt = TD.patch_masks(CACHE, va, d["info"]["crop"], H, W)
        info["patch_counts"] = cnt
        zero = np.zeros((H, W), bool)

        def subsets(bt):
            mv = np.stack([pm.get(int(t), (zero, zero))[0] for t in bt]); zr = np.stack([pm.get(int(t), (zero, zero))[1] for t in bt])
            return torch.as_tensor(mv, device=dev), torch.as_tensor(zr, device=dev)
    elif v in ("shift_m1", "shift_p1"):
        shift = -1 if v == "shift_m1" else 1
    elif v in ("nsub1", "nsub16"):
        n_sub = int(v[4:])
    elif v == "sampling":
        modes, strict = ("nearest", "bilinear", "bicubic"), True
    elif v == "geo_mean":
        kind = "geo_mean"
    elif v == "train_base":
        starts = TT.starts(d["times"], d["tr"], K)[::3]; band = False
    elif v == "check":
        starts, alphas, band = va_all, [0.25, 1.0], False
    else:
        raise ValueError(v)
    if debug:
        starts = starts[:32]
    it = TR.Integrator(H, W, dev, n_sub=n_sub, vmax=10.0)
    res = TD.scan(d, it, starts, alphas, TD.make_vel(d, kind, shift, K), TD.real_data(d), modes=modes, strict=strict, band=band,
                  subsets_fn=subsets, **kw)
    info.update(variant=v, n_starts=int(len(starts)), n_sub=n_sub, kind=kind, shift=shift, modes=list(modes), strict=strict, band=band,
                nonfinite_in_own=res["nonfinite"], n_fixed_px=res["n_fixed_px"].tolist(), seconds=time.time() - t0,
                start_days=[str(x)[:10] for x in d["times"][starts]])
    if v == "check":
        mt = json.load(open(ROOT / "models/tracer/tr_rea_geo_wind/metrics.json"))["best"]["own"]
        mae, r, n = TD.curve_metrics(res["stats"]["bilinear"]["own"][:, :, 0])
        info["check"] = {lab: dict(diag_mae=mae[i].tolist(), diag_n=n[i].astype(int).tolist(),
                                   trainer_mae=[mt[lab][str(k)]["mae"] for k in range(1, K + 1)], trainer_n=[mt[lab][str(k)]["n"] for k in range(1, K + 1)])
                         for i, lab in enumerate(["adv_x0.25", "adv_x1.0"])}
        print(json.dumps(info["check"], indent=0))
    out = {f"stats__{m}__{s}": res["stats"][m][s] for m in modes for s in res["sets"]}
    out.update({f"perday__{m}": res["perday"][m] for m in modes})
    np.savez_compressed(RES / f"diag_landscape_{v}{'_debug' if debug else ''}.npz", alphas=np.asarray(alphas), starts=starts,
                        info=json.dumps(info), **out)
    for m in modes:
        mae, r, n = TD.curve_metrics(res["stats"][m]["fixed"][:, :, 0])
        print(m, "fixed pixel MAE argmin a per lead", [TD.vertex(alphas, mae[:, k]) for k in range(K)], flush=True)
    print(f"done {time.time() - t0:.0f}s nonfinite {res['nonfinite']}", flush=True)


if __name__ == "__main__":
    main()
