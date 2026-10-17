"""Temporal-context track (runs models/spatial/tc_*): markdown table of every run and a figure of
val hidden-pixel MAE per run and the learned neighbour-day weights w_k per lag.
Run: python analyses/spatial_node/tc_summary.py"""
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
for p in sorted(CATEGORIES["spatial"].glob("tc_*/metrics.json")):
    m = json.load(open(p)); c = json.load(open(p.parent / "config.json")); h = json.load(open(p.parent / "history.json"))
    v = m["val"]; pr = m["params"]
    rows.append(dict(run=p.parent.name, dmodel=c["dmodel"] + ("+x" if c.get("extra_feats") else "") + ("+cf" if c.get("ctx_feat") else "") + (" lag" if c.get("lagrangian") else ""),
                     context=c["context"], alpha=("learn %.1f" % pr["alpha"]) if c.get("learn_alpha") else (c["alpha"] or "dir"),
                     epochs=c["epochs"], stride=c["train_stride"], val_days=m["val_days"], mae=v["mae"], best=m["best_val_mae"], rmse=v["rmse"],
                     gr=v["grad_ratio"], gc=v["grad_corr"], w=pr.get("w", {}), beta=pr["beta"], minutes=m["minutes"], hist=h["val"]))
print("| run | dmodel | ctx | alpha | ep | stride | val days | val MAE | best | RMSE | grad ratio | grad corr | w_k | beta | min |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    w = " ".join(f"{k}:{v:.3f}" for k, v in sorted(r["w"].items(), key=lambda kv: int(kv[0]))) or "-"
    print(f"| {r['run']} | {r['dmodel']} | {r['context']} | {r['alpha']} | {r['epochs']} | {r['stride']} | {r['val_days']} | {r['mae']:.4f} | {r['best']:.4f} | {r['rmse']:.4f} | {r['gr']:.3f} | {r['gc']:.3f} | {w} | {r['beta']:.4f} | {r['minutes']:.1f} |")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
x = list(range(len(rows))); names = [r["run"].replace("tc_", "") for r in rows]
base = {"iso": 0, "sst": 1, "mlp": 2, "conv": 3}
ax[0].bar(x, [r["mae"] for r in rows], color=[OKABE_ITO[base[r["dmodel"].split("+")[0].split(" ")[0]]] for r in rows])
ax[0].axhline(0.1365, color="k", ls="--", lw=1, label="dilated CNN (CHL t+-2)"); ax[0].axhline(0.167, color="k", ls=":", lw=1, label="dilated CNN (no context)")
ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=35, ha="right"); ax[0].set_ylabel("val hidden-pixel MAE (ln CHL)"); ax[0].legend(fontsize=8)
ax[0].set_title("Val hidden-pixel MAE by run (colour = D model: blue iso, orange sst, green mlp, yellow conv)", fontsize=9)
ax[0].set_ylim(0.12, max(r["mae"] for r in rows) + 0.01)
for i, r in enumerate(rows):
    if r["w"]:
        ks = sorted(int(k) for k in r["w"]); ax[1].plot(ks, [r["w"][str(k)] for k in ks], "o-", color=OKABE_ITO[i % 8], label=names[i])
ax[1].set_xlabel("lag k (days)"); ax[1].set_ylabel("learned weight w_k (softplus)"); ax[1].set_title("Learned neighbour-day weight w_k per lag, by run", fontsize=9); ax[1].legend(fontsize=7)
for i, r in enumerate(rows):
    ax[2].plot(range(len(r["hist"])), r["hist"], "o-", color=OKABE_ITO[i % 8], label=names[i], ms=3)
ax[2].set_xlabel("epoch (0 = init)"); ax[2].set_ylabel("val hidden-pixel MAE (ln CHL)"); ax[2].set_title("Val hidden-pixel MAE per epoch, by run", fontsize=9); ax[2].legend(fontsize=7)
fig.suptitle("Temporal-context runs of the spatial steady-state model (val days every val-stride, hidden pixels)", fontsize=10)
fig.savefig(FIG / "tc_runs_summary.png", dpi=120, bbox_inches="tight")
print("wrote", FIG / "tc_runs_summary.png")
