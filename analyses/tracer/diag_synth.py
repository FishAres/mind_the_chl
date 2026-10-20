"""Test 2: recovery of a known advection factor from synthetic ln CHL under realistic conditions. GPU; gpu.lock.

Truth (tracer_diag.Synth): per val start day t (the test-1 val starts), an anomaly field = Gaussian random field whose
ring spectrum (spectral_motion patch procedure) is calibrated to S(m) from results/diag_prep.json, carried in Lagrangian
labels and advected with 0.5 x the real reanalysis current of days t..t+k (backward RK2, n_sub 8, vmax 10 px/day,
bicubic sampling of the label field at the departure point; trajectory validity as tracer.Integrator). Observed through
the real cloud masks of days t..t+5 (and only where the truth trajectory is valid). Fitted with the test-1 scan
(u_eff = a x u_rea, n_sub 4, bilinear) on the same a grid.
Cases (--cases, comma list):
  anom_only   anomaly only (no base)
  none        base (train-years doy climatology of ln CHL, static in place) + anomaly
  iid         none + iid Gaussian pixel noise per day, sigma = sqrt(semivariance at lag 1 px) (upper bound: the nugget
              extrapolated from lags 1-3 px is negative)
  corr_noise  none + day-specific Gaussian noise field with ring spectrum N(m) = (1 - rho0(m)) P(m) (lead-0 coherence
              intercepts)
  static30 / static50   anomaly = sqrt(1-f) x advected + sqrt(f) x static field (same spectrum), f = 0.3 / 0.5
  decay       Lagrangian AR(1) per wavenumber: D_j = r D_{j-1} + sqrt(1 - r^2) innovation, r = exp(-1 / tau(q)),
              tau = per-wavelength e-folding of the aligned coherence (>= 14.4 km as fitted; shorter = the 14.4 km value)
  all         corr_noise + static30 + decay
--train: also fit a by training (tracer_diag.train_scalar, a init 0.25, --lr (3e-3), --epochs (1) of train starts) for each
case; output diag_synth_train_<case>[_lr<lr>_ep<epochs>].json.
Output results/diag_synth_<case>.npz (+ .json for training).
"""
import sys, json, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
K = 5
CASES = dict(anom_only=dict(base=False), none=dict(), iid=dict(iid_sigma=None), corr_noise=dict(corr_noise=True),
             static30=dict(static_frac=0.3), static50=dict(static_frac=0.5), decay=dict(decay=True),
             all=dict(corr_noise=True, static_frac=0.3, decay=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    from chl_cnn import tracer as TR, tracer_diag as TD, train_tracer as TT
    dev = torch.device("cuda")
    t0 = time.time()
    a, d = TD.setup(dev, K)
    H, W = d["ocean"].shape
    prep = json.load(open(RES / "diag_prep.json"))
    sigma_iid = float(np.sqrt(prep["semivariance_lag1to4"][0]))
    spec = dict(signal=prep["signal_ring_power"], noise=prep["noise_ring_power"], efold_days=prep["efold_days_used"])
    va = TD.calendar_ok(d, TT.starts(d["times"], d["va"], K)[::2], K, 1, 1)
    tr = TT.starts(d["times"], d["tr"], K)
    if args.debug:
        va = va[:32]
    it = TR.Integrator(H, W, dev, n_sub=4, vmax=10.0)
    bp = TR.BandPass(24.0, 150.0, min_frac=0.5, device=dev)
    vel = TD.make_vel(d, "rea", 0, K)
    print(f"load {time.time() - t0:.0f}s  val starts {len(va)}  sigma_iid {sigma_iid:.4f}", flush=True)
    for name in args.cases.split(","):
        case = dict(CASES[name])
        if "iid_sigma" in case:
            case["iid_sigma"] = sigma_iid
        syn = TD.Synth(d, dev, spec, case)
        info = dict(case=name, case_cfg=case, truth_alpha=0.5, calib=syn.calib)
        # realised field statistics on the first 32 starts: anomaly variance at observed px, lead 0
        Yc, Oc = syn(va[:32], torch.as_tensor(va[:32, None] + np.arange(K + 1)[None], device=dev))
        clim0 = d["CLIM"][d["BIN"][torch.as_tensor(va[:32], device=dev)]]
        an = (Yc[:, 0] - (clim0 if case.get("base", True) else 0))[Oc[:, 0]]
        info["anomaly_var_obs_px_lead0"] = float(an.var()); info["obs_frac_lead1_vs_real"] = [float(Oc[:, 1].sum()), float(d["OBS"][torch.as_tensor(va[:32] + 1, device=dev)].sum())]
        info["anomaly_var_real_val"] = prep["anomaly_var_observed_val"]
        if not args.no_scan:
            res = TD.scan(d, it, va, TD.ALPHAS, vel, syn, bp=bp, K=K, log=None)
            mae, r, n = TD.curve_metrics(res["stats"]["bilinear"]["fixed"][:, :, 0])
            info.update(nonfinite=res["nonfinite"], n_fixed_px=res["n_fixed_px"].tolist(), seconds=time.time() - t0,
                        argmin_fixed_pixel=[TD.vertex(TD.ALPHAS, mae[:, k]) for k in range(K)])
            np.savez_compressed(RES / f"diag_synth_{name}{'_debug' if args.debug else ''}.npz", alphas=np.asarray(TD.ALPHAS), starts=va,
                                info=json.dumps(info), **{f"stats__bilinear__{s}": res["stats"]["bilinear"][s] for s in res["sets"]},
                                perday__bilinear=res["perday"]["bilinear"])
            print(name, "fixed pixel argmin per lead", info["argmin_fixed_pixel"], f"({time.time() - t0:.0f}s)", flush=True)
        if args.train:
            trs = tr[:16 * 20] if args.debug else tr
            out = TD.train_scalar(d, it, vel, trs, va, 0.25, args.lr, args.epochs, data_fn=syn, eval_every=51)
            out.update(info=info, lr=args.lr, epochs=args.epochs)
            sfx = "" if (args.lr == 3e-3 and args.epochs == 1) else f"_lr{args.lr:g}_ep{args.epochs}"
            (RES / f"diag_synth_train_{name}{sfx}{'_debug' if args.debug else ''}.json").write_text(json.dumps(out))
            print(name, "trained a", out["a"][-1], f"({time.time() - t0:.0f}s)", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
