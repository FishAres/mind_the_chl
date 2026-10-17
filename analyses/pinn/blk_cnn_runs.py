"""Black Sea CNN runs (train_pinn.py --store blk, cnn_blk_*): summary table and per-epoch curves, next to the spatial-nODE
blk_* runs (models/spatial) and the climatology references.

  python analyses/pinn/blk_cnn_runs.py            -> prints a markdown table, writes analyses/pinn/figures/blk_cnn_curves.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chl_cnn.plotting import OKABE_ITO  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
PINN = ROOT / "models" / "pinn" / "cnn"
SPATIAL = ROOT / "models" / "spatial"


def load(run_dir):
    out = {}
    for k in ("config", "history", "metrics"):
        p = run_dir / f"{k}.json"
        out[k] = json.loads(p.read_text()) if p.exists() else None
    return out


def cnn_rows(pattern="cnn_blk_*"):
    rows = []
    for rd in sorted(PINN.glob(pattern)):
        r = load(rd)
        if r["history"] is None or r["history"].get("val") in (None, []):
            continue
        c, h, m = r["config"], r["history"], r["metrics"]
        v = (m or {}).get("scores", {}).get("val", {})
        ph = v.get("physics", {})
        ch = v.get("cnn_hidden", {})
        row = dict(run=rd.name, phys=c.get("phys", "upwind") if c else "?", lam=c.get("lam") if c else None,
                   features=len(c["features"]) if c else None, batch=c.get("batch") if c else None, crop=c.get("crop") if c else None,
                   epochs=len(h["val"]), val_final=h["val"][-1], val_best=min(h["val"]), best_ep=int(np.argmin(h["val"])) + 1,
                   rmse=ch.get("rmse", h.get("val_rmse", [np.nan])[-1]), grad_ratio=h["val_grad_ratio"][-1], grad_corr=h["val_grad_corr"][-1],
                   r_cloud=ph.get("abs_rbase_cloud", h["val_physbase_cloud"][-1]), r_obs=ph.get("abs_rbase_obs", h["val_physbase_obs"][-1]),
                   other=ph.get("other_form"), r_other_cloud=ph.get("abs_rbase_other_cloud", h.get("val_physother_cloud", [np.nan])[-1]),
                   r_other_obs=ph.get("abs_rbase_other_obs", h.get("val_physother_obs", [np.nan])[-1]),
                   clim_pix=v.get("climatology_hidden", {}).get("mae"), clim_doy=v.get("doy_climatology_hidden", {}).get("mae"),
                   n_hidden=ch.get("n"), minutes=(c.get("wall_s", np.nan) / 60) if c else np.nan,
                   peak_gb=c.get("peak_gpu_gb") if c else None, finished=m is not None, hist=h)
        rows.append(row)
    return rows


def spatial_rows(pattern="blk_*"):
    rows = []
    for rd in sorted(SPATIAL.glob(pattern)):
        r = load(rd)
        if r["metrics"] is None:
            continue
        v = r["metrics"]["val"]
        rows.append(dict(run=rd.name, val=v["mae"], rmse=v["rmse"], grad_ratio=v["grad_ratio"], grad_corr=v["grad_corr"], clim=v["clim_mae"],
                         n_hidden=v["n_hidden"], minutes=r["metrics"]["minutes"]))
    return rows


def fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    rows = cnn_rows()
    print("| run | phys | lam | feat | batch | crop | ep | val MAE final / best (ep) | RMSE | grad ratio | grad corr | |r| cloud / obs (train form) | |r| cloud / obs (other form) | min | peak GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run']} | {r['phys']} | {r['lam']} | {r['features']} | {r['batch']} | {r['crop']} | {r['epochs']} | {fmt(r['val_final'])} / {fmt(r['val_best'])} ({r['best_ep']}) | {fmt(r['rmse'])} | "
              f"{fmt(r['grad_ratio'], 3)} | {fmt(r['grad_corr'], 3)} | {fmt(r['r_cloud'])} / {fmt(r['r_obs'])} | {fmt(r['r_other_cloud'])} / {fmt(r['r_other_obs'])} ({r['other']}) | {fmt(r['minutes'], 0)} | {fmt(r['peak_gb'], 1)} |")
    if rows:
        print(f"\nclimatology on the same val hidden pixels: per-pixel mean {fmt(rows[0]['clim_pix'])}, day-of-year {fmt(rows[0]['clim_doy'])}; n hidden {rows[0]['n_hidden']}")
    print("\nspatial nODE blk runs (models/spatial, every 4th val day):")
    print("| run | val MAE | RMSE | grad ratio | grad corr | clim (DOY) | n hidden | min |")
    print("|---|---|---|---|---|---|---|---|")
    for r in spatial_rows():
        print(f"| {r['run']} | {fmt(r['val'])} | {fmt(r['rmse'])} | {fmt(r['grad_ratio'], 3)} | {fmt(r['grad_corr'], 3)} | {fmt(r['clim'])} | {r['n_hidden']} | {fmt(r['minutes'], 0)} |")

    if not rows:
        return
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    panels = [("val", "Val hidden-pixel MAE (ln CHL)", "MAE"), ("val_grad_ratio", "Val gradient ratio pred/truth on hidden pairs", "ratio"),
              ("val_grad_corr", "Val gradient correlation on hidden pairs", "corr"), ("train", "Train data loss (z units)", "loss"),
              ("val_physbase_cloud", "Val mean |r| under real clouds, training residual form", "ln CHL / day"),
              ("val_physother_cloud", "Val mean |r| under real clouds, the other residual form", "ln CHL / day")]
    for a_, (key, title, ylab) in zip(ax.flat, panels):
        for i, r in enumerate(rows):
            h = r["hist"]
            if key in h and len(h[key]) and h[key][0] is not None:
                a_.plot(np.arange(1, len(h[key]) + 1), h[key], "-o", ms=3, color=OKABE_ITO[i % 8], label=f"{r['run']} ({r['phys'][:4]}, lam {r['lam']})")
        a_.set(title=title, xlabel="epoch", ylabel=ylab); a_.grid(alpha=.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Black Sea 2 km, dilated CNN (train_pinn.py --store blk): per-epoch val curves (every 4th val day)")
    fig.tight_layout(); fig.savefig(FIG / "blk_cnn_curves.png", dpi=110); plt.close(fig)
    print("wrote", FIG / "blk_cnn_curves.png")


if __name__ == "__main__":
    main()
