"""Summary of the spatial nODE runs (models/spatial/*): val hidden-pixel scores and learned parameters,
plus a figure of MAE / gradient ratio / gradient correlation per run against the references.
Run: python analyses/spatial_node/summary.py [--all]  (smoke runs are skipped unless --all)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chl_cnn.runs import CATEGORIES
from chl_cnn.plotting import OKABE_ITO

FIG = Path(__file__).resolve().parent / "figures"
show_all = "--all" in sys.argv


def fmt_params(p):
    out = []
    for k, v in p.items():
        if isinstance(v, dict):
            out.append(f"{k}=" + "/".join(f"{kk}:{vv:.3f}" for kk, vv in v.items()))
        else:
            out.append(f"{k}={v:.4g}")
    return " ".join(out)


rows = []
seen = set()
for p in sorted(list(CATEGORIES["spatial"].glob("*/metrics.json")) + list((CATEGORIES["spatial"].parent / "history" / "spatial").glob("*/metrics.json"))):
    name = p.parent.name
    if name in seen:
        continue
    seen.add(name)
    if not show_all and ("smoke" in name or name.endswith("_preds")):
        continue
    m = json.load(open(p)); c = json.load(open(p.parent / "config.json")); v = m["val"]
    rows.append(dict(run=name, dmodel=c["dmodel"], context=c.get("context", 0), lag=int(c.get("lagrangian", False)), alpha=c.get("alpha", 0.0),
                     epochs=0 if c.get("eval_only") else c["epochs"], stride=c["train_stride"], mae=v["mae"], best=m.get("best_val_mae", v["mae"]), rmse=v["rmse"], clim=v["clim_mae"],
                     grad_ratio=v["grad_ratio"], grad_corr=v["grad_corr"], aniso_med=v["aniso_median"], aniso_p90=v["aniso_p90"],
                     val_days=m["val_days"], n=v["n_hidden"], params=m["params"], cg=v.get("cg_iters", float("nan")), minutes=m.get("minutes", float("nan"))))
rows = [r for r in rows if r["val_days"] >= 50]
rows.sort(key=lambda r: r["mae"])
print(f"{'run':22s} {'D':5s} {'ctx':>3s} {'lag':>3s} {'ep':>2s} {'str':>3s} {'MAE':>6s} {'best':>6s} {'RMSE':>6s} {'g.ratio':>7s} {'g.corr':>6s} {'aniso med/p90':>13s} {'CG':>4s} {'min':>5s} {'val days':>8s}  params")
for r in rows:
    print(f"{r['run']:22s} {r['dmodel']:5s} {r['context']:3d} {r['lag']:3d} {r['epochs']:2d} {r['stride']:3d} {r['mae']:6.4f} {r['best']:6.4f} {r['rmse']:6.4f} {r['grad_ratio']:7.3f} {r['grad_corr']:6.3f} "
          f"{r['aniso_med']:6.2f}/{r['aniso_p90']:<6.1f} {r['cg']:4.0f} {r['minutes']:5.1f} {r['val_days']:8d}  {fmt_params(r['params'])}")
if rows:
    r0 = max(rows, key=lambda r: r["val_days"])
    print(f"climatology prior: {r0['clim']:.4f} on all val days ({r0['val_days']}, hidden px {r0['n']})")

fig, ax = plt.subplots(1, 3, figsize=(max(15, 0.5 * len(rows) + 6), 5))
names = [r["run"] for r in rows]; x = range(len(rows))
col = {"iso": OKABE_ITO[0], "sst": OKABE_ITO[1], "mlp": OKABE_ITO[2], "conv": OKABE_ITO[3]}
for a_, key, title, refs in [(ax[0], "mae", "Val hidden-pixel MAE (ln CHL)", [("DOY climatology", rows[0]["clim"] if rows else 0.25), ("dilated CNN (pinn_v2_lam0)", 0.1365)]),
                             (ax[1], "grad_ratio", "Gradient amplitude ratio pred/truth, hidden pairs", [("dilated CNN runs", 0.55)]),
                             (ax[2], "grad_corr", "Gradient correlation pred vs truth, hidden pairs", [("dilated CNN runs", 0.59)])]:
    a_.bar(x, [r[key] for r in rows], color=[col.get(r["dmodel"], OKABE_ITO[4]) for r in rows])
    for i, (lab, v) in enumerate(refs):
        a_.axhline(v, color="k", ls=["--", ":"][i], lw=1, label=lab)
    a_.set_xticks(list(x)); a_.set_xticklabels(names, rotation=60, ha="right", fontsize=7); a_.set_title(title, fontsize=10); a_.legend(fontsize=8)
ax[0].set_ylabel("MAE (ln CHL)"); ax[1].set_ylabel("ratio"); ax[2].set_ylabel("correlation")
fig.suptitle("Spatial nODE (steady-state anisotropic diffusion) runs sorted by val MAE, colour = D model (blue iso, orange SST tensor, green MLP, vermilion conv)")
fig.savefig(FIG / "runs_summary.png", dpi=120, bbox_inches="tight")
print("wrote", FIG / "runs_summary.png")
