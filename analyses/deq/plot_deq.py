"""Figures for the DEQ ladder (models/pinn/deq/*) vs the 12-epoch dilated CNN (models/pinn/cnn/cnn_12ep).

Reads history.json / metrics.json of each run that exists; writes
  analyses/deq/learning_curves.png   val hidden-pixel MAE by epoch, one line per run
  analyses/deq/iteration_curves.png  val MAE vs inference iterations (fixed-point | Anderson), one line per DEQ run
  analyses/deq/solver_stats.png      val solver iterations and mean relative residual by epoch per DEQ run
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analyses" / "deq"
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
RUNS = [("cnn_12ep", "cnn"), ("deq_unroll_k1", "deq"), ("deq_unroll_k6", "deq"),
        ("deq_and24_sn", "deq"), ("deq_and24", "deq"), ("deq_unroll_k12", "deq")]
YLIM = (0.1, 0.4)


def load(name, sub):
    d = ROOT / "models" / "pinn" / sub / name
    r = {"name": name, "dir": d}
    for f in ("history", "metrics", "config"):
        p = d / f"{f}.json"
        r[f] = json.load(open(p)) if p.exists() else None
    return r


runs = [load(n, s) for n, s in RUNS]
runs = [r for r in runs if r["history"] is not None]
col = {r["name"]: OKABE_ITO[i % len(OKABE_ITO)] for i, r in enumerate(runs)}

# 1. learning curves
fig, ax = plt.subplots(figsize=(7, 4.5))
for r in runs:
    v = r["history"]["val"]
    ax.plot(range(1, len(v) + 1), v, "o-", ms=3, color=col[r["name"]], label=r["name"])
ax.set(xlabel="epoch", ylabel="val hidden-pixel MAE (ln CHL)", title="Val hidden-pixel MAE by epoch: DEQ ladder vs 12-epoch CNN")
ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "learning_curves.png", dpi=120); plt.close(fig)

# 2. iteration curves
deq = [r for r in runs if r["metrics"] and "iteration_curve" in r["metrics"]["scores"]["val"]]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
n_days = None
for ax, solver, ttl in zip(axs, ("fixed", "anderson"), ("plain fixed-point iteration", "Anderson acceleration")):
    for r in deq:
        ic = r["metrics"]["scores"]["val"]["iteration_curve"]; n_days = ic.get("n_days", n_days)
        ks = sorted(int(k) for k in ic[solver]); ys = [ic[solver][str(k)] for k in ks]
        lab = r["name"]
        if max(ys) > YLIM[1] or min(ys) < YLIM[0]:
            lab += f" (leaves range: min {min(ys):.3f}, max {max(ys):.3g})"
        ax.plot(ks, ys, "o-", ms=3, color=col[r["name"]], label=lab)
    ax.set(xscale="symlog", xlabel="iterations at inference", title=ttl, ylim=YLIM)
    ax.grid(alpha=.3); ax.legend(fontsize=7)
axs[0].set(ylabel="val hidden-pixel MAE (ln CHL)")
fig.suptitle(f"Val hidden-pixel MAE vs inference iterations ({n_days} val days)")
fig.tight_layout(); fig.savefig(OUT / "iteration_curves.png", dpi=120); plt.close(fig)

# 3. solver stats by epoch
deq_h = [r for r in runs if r["history"].get("deq_eta") and r["history"]["deq_eta"][0] is not None]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for r in deq_h:
    h = r["history"]; ep = range(1, len(h["val"]) + 1)
    axs[0].plot(ep, h["deq_val_iters"], "o-", ms=3, color=col[r["name"]], label=r["name"])
    axs[1].plot(ep, h["deq_val_res"], "o-", ms=3, color=col[r["name"]], label=r["name"])
axs[0].set(xlabel="epoch", ylabel="mean val solver iterations", title="Val solver iterations by epoch")
axs[1].set(xlabel="epoch", ylabel="mean relative residual |f(z)-z| / |z|", yscale="log", title="Val mean relative residual by epoch")
for ax in axs:
    ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle("DEQ solver statistics on val days, by training epoch")
fig.tight_layout(); fig.savefig(OUT / "solver_stats.png", dpi=120); plt.close(fig)
print("runs plotted:", [r["name"] for r in runs], "| iteration curves:", [r["name"] for r in deq])
