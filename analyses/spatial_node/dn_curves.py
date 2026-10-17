"""D-net track (runs models/spatial/dn_*): per-epoch val hidden-pixel MAE, gradient ratio / correlation, beta and
anisotropy from history.json / metrics.json, plus a table. Groups: Q1 mlp lr x d-max, Q2 conv, Q3 feature ablation,
Q4 SST-residual MLP. Run: python analyses/spatial_node/dn_curves.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chl_cnn.runs import CATEGORIES
from chl_cnn.plotting import OKABE_ITO

FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
runs = {}
for p in sorted(CATEGORIES["spatial"].glob("dn_*/history.json")):
    if p.parent.name.startswith("dn_smoke"):
        continue
    c = json.load(open(p.parent / "config.json")); h = json.load(open(p))
    m = json.load(open(p.parent / "metrics.json")) if (p.parent / "metrics.json").exists() else None
    runs[p.parent.name] = dict(c=c, h=h, m=m)

GROUPS = {"Q1 MLP: lr-net x d-max (8 ep, stride 4)": lambda n: n.startswith("dn_mlp"),
          "Q2 conv D-net vs MLP (8 ep, stride 4)": lambda n: n.startswith("dn_conv") or n in ("dn_mlp_lr3e3_d4", "dn_mlpx_lr3e3_d4"),
          "Q3 MLP feature ablation (3 ep, stride 4)": lambda n: n.startswith("dn_abl"),
          "Q4 SST-residual MLP vs MLP (8 ep, stride 4)": lambda n: n.startswith("dn_resid") or n in ("dn_mlp_lr3e3_d4", "dn_mlpx_lr3e3_d4"),
          "Q5 full-stride run": lambda n: n.startswith("dn_final")}
print(f"{'run':22s} {'D':5s} {'xf':3s} {'lr_net':>7s} {'dmax':>5s} {'ep':>3s} {'str':>3s} {'MAE':>7s} {'best':>7s} {'RMSE':>7s} {'g.rat':>6s} {'g.cor':>6s} {'an.med':>7s} {'an.p90':>7s} {'beta':>7s} {'l1':>5s} {'l2':>5s} {'min':>5s}")
for n, r in runs.items():
    c, h, m = r["c"], r["h"], r["m"]
    v = m["val"] if m else None
    print(f"{n:22s} {c['dmodel']:5s} {str(c['extra_feats'])[0]:3s} {c['lr_net']:7.0e} {c['d_max']:5.0f} {c['epochs']:3d} {c['train_stride']:3d} "
          + (f"{v['mae']:7.4f} {m['best_val_mae']:7.4f} {v['rmse']:7.4f} {v['grad_ratio']:6.3f} {v['grad_corr']:6.3f} {v['aniso_median']:7.2f} {v['aniso_p90']:7.2f} "
             f"{m['params']['beta']:7.4f} {v['mean_l1']:5.2f} {v['mean_l2']:5.2f} {m['minutes']:5.1f}" if m else f"running: val {[round(x, 4) for x in h['val']]}"))

for gi, (title, sel) in enumerate(GROUPS.items()):
    names = [n for n in runs if sel(n)]
    if not names:
        continue
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    for i, n in enumerate(names):
        h = runs[n]["h"]; ep = list(range(len(h["val"])))
        col = OKABE_ITO[i % len(OKABE_ITO)]; ls = ["-", "--", ":", "-."][i // len(OKABE_ITO)]
        ax[0].plot(ep, h["val"], ls, color=col, marker="o", ms=3, label=n)
        ax[1].plot(ep, h["grad_ratio"], ls, color=col, marker="o", ms=3)
        ax[2].plot(ep, h["grad_corr"], ls, color=col, marker="o", ms=3)
        ax[3].plot(ep, [p["beta"] for p in h["params"]], ls, color=col, marker="o", ms=3)
    for a_, t, yl in [(ax[0], "Val hidden-pixel MAE per epoch (ln CHL)", "MAE"), (ax[1], "Gradient amplitude ratio pred/truth per epoch", "ratio"),
                      (ax[2], "Gradient correlation pred vs truth per epoch", "correlation"), (ax[3], "Learned beta (climatology mean reversion) per epoch", "beta")]:
        a_.set_title(t, fontsize=10); a_.set_xlabel("epoch (0 = init)"); a_.set_ylabel(yl)
    ax[3].set_yscale("log"); ax[0].legend(fontsize=7)
    fig.suptitle(f"{title}: val every 4th day", fontsize=11)
    out = FIG / f"dn_curves_{gi + 1}.png"; fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig); print("wrote", out)
