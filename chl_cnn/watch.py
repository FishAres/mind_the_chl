"""Progress figure: val hidden-pixel MAE per epoch (left) and the physics residual under real clouds / on observed
pixels for PINN runs (right). Redrawn every --every seconds (history.json is rewritten after each epoch).

  python -m chl_cnn.watch                                  # all categories -> models/progress.png, once
  python -m chl_cnn.watch --cat pinn/cnn,pinn/unet --every 60 --out models/progress_pinn.png
  python -m chl_cnn.watch --cat pinn/cnn --glob 'pinn_v2_*'
Categories: cnn, unet, pinn/cnn, pinn/unet (see chl_cnn/runs.py). Interactive version: notebooks/progress.ipynb.
"""
import argparse, json, time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from .plotting import OKABE_ITO
from .runs import MODELS, CATEGORIES, list_runs, category_of


def draw_runs(run_dirs, axes=None, title_suffix=""):
    """Plot the given run folders on (ax_mae, ax_phys). Returns (rows, fig)."""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    else:
        fig = axes[0].figure
    ax, axp = axes
    rows, lo, hi = [], [], []
    for i, p in enumerate(run_dirs):
        h = json.load(open(p / "history.json")); name = p.name
        done = (p / "metrics.json").exists()
        col = OKABE_ITO[i % len(OKABE_ITO)]; ls = ["-", "--", ":"][(i // len(OKABE_ITO)) % 3]
        ep = range(1, len(h["val"]) + 1)
        ax.plot(ep, h["val"], color=col, ls=ls, lw=1.4 if not done else 1, alpha=1 if not done else .75,
                label=f"{category_of(p)}/{name}{'' if done else ' (running)'}  {len(h['val'])} ep, best {min(h['val']):.3f}")
        if "val_phys_cloud" in h:
            axp.plot(ep, h["val_phys_cloud"], color=col, ls=ls, lw=1.4, label=f"{name}: under real clouds")
            axp.plot(ep, h["val_phys_obs"], color=col, ls=ls, lw=0.8, alpha=.6, label=f"{name}: observed pixels")
        rows.append((category_of(p), name, len(h["val"]), min(h["val"]), "done" if done else "running"))
        v = sorted(h["val"]); lo.append(v[0]); hi.append(v[int(0.75 * (len(v) - 1))])
    ylim = (0.95 * min(lo), 1.05 * max(hi)) if lo else (0.1, 0.4)
    ax.set(xlabel="epoch", ylabel="val MAE on hidden pixels, ln(CHL)", ylim=ylim, title=f"Validation MAE by epoch{title_suffix}")
    ax.grid(alpha=.3)
    if ax.lines:
        ax.legend(fontsize=7)
    axp.set(xlabel="epoch", ylabel="val mean |r|, ln(CHL) per day", title="Transport residual on val days by epoch (PINN runs)")
    axp.grid(alpha=.3)
    if axp.lines:
        axp.legend(fontsize=6, ncol=2)
    return rows, fig


def draw(out=MODELS / "progress.png", cats=None, pattern="*"):
    runs = list_runs(cats, pattern)
    rows, fig = draw_runs(runs, title_suffix=f"  (updated {time.strftime('%H:%M:%S')})")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return rows


if __name__ == "__main__":
    matplotlib.use("Agg")                      # headless when run as a script; leave the notebook backend alone when imported
    ap = argparse.ArgumentParser(); ap.add_argument("--every", type=float, default=0)
    ap.add_argument("--cat", default=",".join(CATEGORIES), help="comma-separated categories: " + ", ".join(CATEGORIES))
    ap.add_argument("--glob", default="*", help="run-name glob within the categories")
    ap.add_argument("--out", default=str(MODELS / "progress.png")); a = ap.parse_args()
    while True:
        for cat, name, n, best, st in draw(Path(a.out), a.cat.split(","), a.glob):
            print(f"{cat:10s} {name:28s} {st:8s} epochs {n:3d}  best val {best:.4f}")
        if a.every <= 0:
            break
        time.sleep(a.every)
