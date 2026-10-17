"""One table and one figure for all PINN-family runs in models/: lambda sweep, priors, 60-epoch, physics-only
inputs, U-Net. Val scores only (test held out).
  python analyses/pinn/pinn_runs_summary.py   -> analyses/pinn/figures/pinn_runs_summary.png + table
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chl_cnn.plotting import OKABE_ITO
from chl_cnn.runs import find_run

FIG = Path(__file__).resolve().parent / "figures"


def _exists(r):
    try: return (find_run(r) / "metrics.json").exists()
    except FileNotFoundError: return False
GROUPS = {
    "dilated CNN, 12 ep, lambda sweep": ["pinn_lam0", "pinn_lam0p1", "pinn_lam0p3", "pinn_lam1p0"],
    "dilated CNN, 12 ep, priors (lambda 0.3)": ["pinn_v1_clim", "pinn_v1_ekman", "pinn_v1_courant", "pinn_v1_clim_ekman", "pinn_v1_all"],
    "dilated CNN, 60 ep": ["pinn_long_lam0", "pinn_long_lam0p3", "pinn_long_lam1p0"],
    "physics-only inputs (no CHL image at all), 12 ep": ["nochl_lam0", "nochl_lam0p3", "nochl_cloud_lam0", "nochl_cloud_lam0p3"],
    "no CHL at t, CHL context t±1 + physics t-1..t+1, 12 ep": ["nochl_today_lam0", "nochl_today_lam0p3"],
    "U-Net as in mind-the-chl-gap (their inputs, MSE), 30 ep": ["unet_mtcg_lam0", "unet_mtcg_lam0p3"],
    "U-Net PINN, full inputs, ~1.9 M params, 30 ep": ["unet_d2w64_lam0", "unet_d2w64_lam0p3", "unet_d3w32_lam0", "unet_d3w32_lam0p3"],
}
REF = {"perf_dilated_vis01": "reference: data-only dilated CNN, 30 ep (train2.py)"}


def row(r):
    D = find_run(r); cfg = json.load(open(D / "config.json")); m = json.load(open(D / "metrics.json"))["scores"]["val"]; h = json.load(open(D / "history.json"))
    ph = m.get("physics", {})
    return dict(run=r, lam=cfg.get("lam", 0.0), model=cfg["model"], epochs=len(h["val"]), best_epoch=cfg["best_epoch"], params=cfg["n_params"],
                mae=m["cnn_hidden"]["mae"], rmse=m["cnn_hidden"]["rmse"], bias=m["cnn_hidden"]["bias"], r2=m["cnn_hidden"]["r2"],
                r_cloud=ph.get("abs_rbase_cloud", ph.get("abs_r_cloud", np.nan)), n=m["cnn_hidden"]["n"],
                inputs=("no CHL" + (" + cloud mask" if cfg.get("cloud_input") else "")) if cfg.get("no_chl_input") else
                       (f"CHL t±{cfg.get('context', 0)} only, feat t±{cfg.get('feat_context', 0)}" if cfg.get("no_chl_today") else
                        f"{len(cfg['features'])} feat + CHL t±{cfg.get('context', 0)}" + (f" {cfg['model']} d{cfg.get('unet_depth', 4)}w{cfg['width']}" if cfg["model"] == "unet" else "")),
                priors=",".join(k for k, v in [("clim", cfg.get("src_clim")), ("ekman", cfg.get("src_ekman")), ("courant", cfg.get("phys_weight") == "courant")] if v) or "-",
                hist=h["val"])


print(f"{'group / run':48s} {'lam':>4} {'inputs':>22} {'priors':>12} {'ep':>3} {'val MAE':>8} {'RMSE':>7} {'bias':>7} {'R2':>6} {'|r| cloud':>10} {'params':>8} {'n px':>8}")
fig, ax = plt.subplots(1, 2, figsize=(18, 5.6))
k = 0
for g, runs in GROUPS.items():
    print(f"-- {g}")
    for r in runs:
        if not _exists(r):
            print(f"   {r:45s} (missing)"); continue
        d = row(r)
        print(f"   {r:45s} {d['lam']:4g} {d['inputs']:>22} {d['priors']:>12} {d['epochs']:3d} {d['mae']:8.4f} {d['rmse']:7.4f} {d['bias']:7.4f} {d['r2']:6.3f} {d['r_cloud']:10.4f} {d['params']:8d} {d['n']:8d}")
        if g.startswith("dilated CNN, 60") or g.startswith("physics-only") or g.startswith("no CHL") or g.startswith("U-Net"):
            ax[0].plot(range(1, len(d["hist"]) + 1), d["hist"], color=OKABE_ITO[k % 8], ls="-" if k < 8 else "--", lw=1.2, label=f"{r} ({d['epochs']} ep)")
            k += 1
for r, lab in REF.items():
    if _exists(r):
        D = find_run(r); m = json.load(open(D / "metrics.json"))["scores"]["val"]["cnn_hidden"]; h = json.load(open(D / "history.json"))
        print(f"-- {lab}\n   {r:45s} {'-':>4} {'9 feat + CHL t±2':>22} {'-':>12} {len(h['val']):3d} {m['mae']:8.4f} {m['rmse']:7.4f} {m['bias']:7.4f} {m['r2']:6.3f} {'-':>10}")
        ax[0].plot(range(1, len(h["val"]) + 1), h["val"], color="k", ls=":", lw=1.2, label=f"{r} (data-only reference)")
ax[0].set(xlabel="epoch", ylabel="val MAE on hidden pixels, ln CHL", title="Validation hidden-pixel MAE by epoch: 60-epoch, physics-only-input and U-Net runs", ylim=(0.13, 0.26))
ax[0].grid(alpha=.3); ax[0].legend(fontsize=6, ncol=2)
# final MAE by group
labels, vals, cols = [], [], []
for gi, (g, runs) in enumerate(GROUPS.items()):
    for r in runs:
        if _exists(r):
            d = row(r); labels.append(r.replace("pinn_", "").replace("_lam", " λ").replace("p", ".")); vals.append(d["mae"]); cols.append(OKABE_ITO[gi % 8])
xi = np.arange(len(vals))
ax[1].bar(xi, vals, color=cols)
ax[1].axhline(0.358, color="k", ls="--", lw=1, label="climatology 0.358")
ax[1].set(xticks=xi, xticklabels=labels, ylabel="val MAE on hidden pixels, ln CHL", title="Final val hidden-pixel MAE, all PINN-family runs (colour = group)", ylim=(0.12, 0.24))
ax[1].tick_params(axis="x", rotation=70, labelsize=6); ax[1].grid(alpha=.3, axis="y"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "pinn_runs_summary.png", dpi=110)
