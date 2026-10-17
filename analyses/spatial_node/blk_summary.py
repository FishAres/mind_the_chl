"""Markdown table of the Black Sea (blk_*) spatial nODE runs in models/spatial: val hidden-pixel scores, learned
parameters and run time, plus a bar figure of MAE / gradient ratio / gradient correlation per run.
Run: python analyses/spatial_node/blk_summary.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chl_cnn.runs import CATEGORIES
from chl_cnn.plotting import OKABE_ITO

FIG = Path(__file__).resolve().parent / "figures"
rows = []
for p in sorted(CATEGORIES["spatial"].glob("blk_*/metrics.json")):
    m = json.load(open(p)); c = json.load(open(p.parent / "config.json")); v = m["val"]; pr = m["params"]
    w = " ".join(f"{k}:{vv:.2f}" for k, vv in pr.get("w", {}).items() if int(k) > 0)
    rows.append(dict(run=p.parent.name, dmodel=c["dmodel"] + ("+x" if c.get("extra_feats") else "") + ("+cf" if c.get("ctx_feat") else ""),
                     context=c["context"], lag=("y" if c.get("lagrangian") else "n") + (f"({c['lag_max']:.0f})" if c.get("lagrangian") else ""),
                     epochs=0 if c.get("eval_only") else c["epochs"], stride=c["train_stride"], val_days=m["val_days"], mae=v["mae"], best=m["best_val_mae"],
                     rmse=v["rmse"], clim=v["clim_mae"], gr=v["grad_ratio"], gc=v["grad_corr"], am=v["aniso_median"], ap=v["aniso_p90"],
                     beta=pr["beta"], w=w, cg=v["cg_iters"], minutes=m["minutes"], n=v["n_hidden"]))
rows.sort(key=lambda r: r["mae"])
print("| run | dmodel | ctx | lag | ep | stride | val days | MAE final / best | RMSE | grad ratio | grad corr | aniso med / p90 | beta | w(+k) | CG | min |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['run']} | {r['dmodel']} | {r['context']} | {r['lag']} | {r['epochs']} | {r['stride']} | {r['val_days']} | {r['mae']:.4f} / {r['best']:.4f} | {r['rmse']:.4f} | "
          f"{r['gr']:.3f} | {r['gc']:.3f} | {r['am']:.1f} / {r['ap']:.1f} | {r['beta']:.4f} | {r['w']} | {r['cg']:.0f} | {r['minutes']:.0f} |")
if rows:
    print(f"\nDOY climatology MAE on the same hidden pixels: {rows[0]['clim']:.4f} (val days {rows[0]['val_days']}, hidden px {rows[0]['n']})")
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    names = [r["run"] for r in rows]; x = range(len(rows))
    col = {"iso": OKABE_ITO[0], "sst": OKABE_ITO[1], "mlp": OKABE_ITO[2], "conv": OKABE_ITO[3]}
    for a_, key, title, ref in [(ax[0], "mae", "Val hidden-pixel MAE (ln CHL), Black Sea", ("DOY climatology", rows[0]["clim"])),
                                (ax[1], "gr", "Gradient amplitude ratio pred/truth, hidden pairs", None),
                                (ax[2], "gc", "Gradient correlation pred vs truth, hidden pairs", None)]:
        a_.bar(x, [r[key] for r in rows], color=[col.get(r["dmodel"].split("+")[0], OKABE_ITO[4]) for r in rows])
        if ref: a_.axhline(ref[1], color="k", ls="--", lw=1, label=ref[0]); a_.legend(fontsize=8)
        a_.set_xticks(list(x)); a_.set_xticklabels(names, rotation=60, ha="right", fontsize=8); a_.set_title(title, fontsize=10)
    ax[0].set_ylabel("MAE (ln CHL)"); ax[1].set_ylabel("ratio"); ax[2].set_ylabel("correlation")
    fig.suptitle("Black Sea spatial nODE runs sorted by val MAE (colour = D model: blue iso, orange SST tensor, green MLP, vermilion conv)")
    fig.savefig(FIG / "blk_runs_summary.png", dpi=120, bbox_inches="tight"); print("wrote", FIG / "blk_runs_summary.png")
