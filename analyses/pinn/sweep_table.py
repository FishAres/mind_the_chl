"""Table of runs matching a name glob: val/test hidden-pixel scores, gradient metrics at the best epoch.

  python analyses/pinn/sweep_table.py 's1_*'           # default glob 's1_*'
  python analyses/pinn/sweep_table.py 'pinn_v3_*' --sort mae
Writes analyses/pinn/figures/sweep_<glob>.png (val MAE curves) and prints a markdown table.
"""
import argparse, json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chl_cnn.runs import list_runs
from chl_cnn.plotting import OKABE_ITO

ap = argparse.ArgumentParser(); ap.add_argument("glob", nargs="?", default="s1_*"); ap.add_argument("--sort", default="mae")
a = ap.parse_args()
rows = []
for p in list_runs(None, a.glob):
    h = json.load(open(p / "history.json")); done = (p / "metrics.json").exists()
    be = h["val"].index(min(h["val"]))
    r = dict(run=p.name, mae=min(h["val"]), best_ep=be + 1, epochs=len(h["val"]), done=done,
             grad_ratio=h.get("val_grad_ratio", [None] * len(h["val"]))[be], grad_corr=h.get("val_grad_corr", [None] * len(h["val"]))[be],
             rmse=None, test_mae=None)
    if done:
        m = json.load(open(p / "metrics.json")); v = m["scores"]["val"]["cnn_hidden"]; t = m["scores"]["test"]["cnn_hidden"]
        r.update(mae=v["mae"], rmse=v["rmse"], test_mae=t["mae"])
    rows.append((p, h, r))
rows.sort(key=lambda x: x[2][a.sort] if x[2][a.sort] is not None else 9)
f = lambda x, d=4: "" if x is None else f"{x:.{d}f}"
print("| run | val MAE | val RMSE | test MAE | grad ratio | grad corr | best ep | epochs | state |")
print("|---|---|---|---|---|---|---|---|---|")
for _, _, r in rows:
    print(f"| {r['run']} | {f(r['mae'])} | {f(r['rmse'])} | {f(r['test_mae'])} | {f(r['grad_ratio'],3)} | {f(r['grad_corr'],3)} | {r['best_ep']} | {r['epochs']} | {'done' if r['done'] else 'running'} |")
fig, ax = plt.subplots(figsize=(9, 5))
for i, (p, h, r) in enumerate(rows):
    ax.plot(range(1, len(h["val"]) + 1), h["val"], color=OKABE_ITO[i % 8], ls=["-", "--", ":"][i // 8 % 3], label=f"{r['run']}  best {r['mae']:.4f}")
ax.set(xlabel="epoch", ylabel="val MAE on hidden pixels, ln(CHL)", title=f"Validation MAE by epoch, runs matching {a.glob}")
ax.set_ylim(top=min(0.2, ax.get_ylim()[1])); ax.grid(alpha=.3); ax.legend(fontsize=8)
out = Path(__file__).parent / "figures" / f"sweep_{a.glob.replace('*', 'x')}.png"; out.parent.mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=110); print("wrote", out)
