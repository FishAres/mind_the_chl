"""Black Sea block-count and window sweep (blk_t3d_c4_b{4..7}_s12, blk_t3d_c3_b8_s12; 2026-10-16).

Design: every sweep run uses the 20-epoch cosine schedule of the control blk_t3d_c4_gdl100_lam0 (8 blocks, +-4 days) and stops
after epoch 12 (--stop-epoch 12). The control is not retrained: its history.json epochs 1-12 (before its legacy resume at
epoch 15) are the comparison. Test MAE exists only for the sweep runs (the control's test score is from epoch 20).
blk_t3d_c4_b8_12ep (a killed 12-epoch-schedule run) is excluded.

Data: models/pinn/t3d/<run>/{config,history,metrics}.json.
Table: blocks, dilations, receptive field (px, km at 2 km), window, best val hidden-pixel MAE within epochs 1-12 + epoch,
val at epoch 12, test MAE (sweep runs: model.pt = best val epoch after --ckpt-from 5), grad ratio / corr at the best epoch,
s/epoch, peak GPU GB, n_params.
Receptive field: 1 + 2 (3x3 input conv) + 2 x t3d_depth (3x3 spatial kernels of the Conv3d stem) + sum over blocks of 2 x 2d.
Figure: figures/blk_block_sweep.png (left: val at epoch 12 vs number of blocks, +-4 days, control at 8; right: val by epoch 1-12).
Usage: python analyses/pinn/blk_block_sweep.py [--bullet RUN]   (from the project root)
"""
import argparse
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

CONTROL = "blk_t3d_c4_gdl100_lam0"
RUNS = ["blk_t3d_c4_b4_s12", "blk_t3d_c4_b5_s12", "blk_t3d_c4_b6_s12", "blk_t3d_c4_b7_s12", CONTROL, "blk_t3d_c3_b8_s12"]
OUT = Path(__file__).resolve().parent / "figures"
KM_PER_PX = 2.0
N_EP = 12
DEFAULT_DIL = [1, 2, 4, 8, 16, 1, 2, 4]


# ---- data
def receptive_field(dil, stem_depth):
    return 1 + 2 + 2 * stem_depth + sum(4 * d for d in dil)


def load(run):
    d = ROOT / "models" / "pinn" / "t3d" / run
    if not (d / "history.json").exists():
        return None
    h = json.loads((d / "history.json").read_text())
    v = np.asarray(h["val"][:N_EP], float)
    r = dict(run=run, control=run == CONTROL, hist_val=v.tolist(), done=(d / "metrics.json").exists())
    if not r["done"]:
        return r
    c = json.loads((d / "config.json").read_text())
    dil = c.get("dilations") or DEFAULT_DIL
    if isinstance(dil, str):
        dil = [int(x) for x in dil.split(",")]
    b = int(np.nanargmin(v)); rf = receptive_field(dil, c["t3d_depth"])
    r.update(blocks=len(dil), dilations=",".join(map(str, dil)), rf_px=rf, rf_km=rf * KM_PER_PX, window=f"+-{c['context']}",
             n_epochs_seen=len(v), best_val=float(v[b]), best_epoch=b + 1, val_ep12=(float(v[N_EP - 1]) if len(v) >= N_EP else None),
             grad_ratio=h["val_grad_ratio"][b], grad_corr=h["val_grad_corr"][b],
             s_per_epoch=float(np.mean(h["epoch_s"][:N_EP])), peak_gb=c["peak_gpu_gb"], n_params=c["n_params"],
             nan_val=int(np.sum(~np.isfinite(v))), legacy=c.get("legacy_resume"), resumes=len(c.get("resume_events") or []))
    if not r["control"]:
        m = json.loads((d / "metrics.json").read_text())["scores"]
        r.update(test=m["test"]["cnn_hidden"]["mae"], ckpt_epoch=c["best_epoch"], n_val=m["val"]["cnn_hidden"]["n"],
                 n_test=m["test"]["cnn_hidden"]["n"])
    return r


# ---- reporting
def bullet(r):
    s = (f"- {r['run']} ({r['blocks']} blocks, dilations {r['dilations']}, receptive field {r['rf_px']} px = {r['rf_km']:.0f} km, "
         f"window {r['window']} days): best val within epochs 1-12 {r['best_val']:.4f} (epoch {r['best_epoch']}), val at epoch 12 "
         f"{r['val_ep12']:.4f}")
    if r["control"]:
        s += " (control, from its 20-epoch history; no epoch-12 test score)"
    else:
        s += f"; test {r['test']:.4f} (checkpoint epoch {r['ckpt_epoch']})"
    s += (f"; grad ratio {r['grad_ratio']:.3f}, corr {r['grad_corr']:.3f} at the best epoch; {r['s_per_epoch']:.0f} s/epoch; "
          f"peak {r['peak_gb']:.1f} GB; {r['n_params']:,} params"
          + (f"; n hidden px val {r['n_val']:,} / test {r['n_test']:,}" if not r["control"] else "")
          + (f"; resumed {r['resumes']}x" if r["resumes"] and not r["control"] else "")
          + (f"; {r['nan_val']} NaN val epochs" if r["nan_val"] else "") + ".")
    return s


def label(r):
    nb = r.get("blocks") or (8 if r["control"] else int(r["run"].split("_b")[1].split("_")[0]))
    win = "+-3 d" if "_c3_" in r["run"] else "+-4 d"
    return f"{nb} blocks, {win}" + (" (control, 20-ep run)" if r["control"] else "")


def figure(rows):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    c4 = sorted([r for r in rows if r.get("done") and r["window"] == "+-4" and r["val_ep12"] is not None], key=lambda r: r["blocks"])
    if c4:
        a1.plot([r["blocks"] for r in c4], [r["val_ep12"] for r in c4], "o-", color=OKABE_ITO[0], label="val at epoch 12 (+-4 days)")
        for r in c4:
            a1.annotate(f"RF {r['rf_px']} px", (r["blocks"], r["val_ep12"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
        ctrl = [r for r in c4 if r["control"]]
        if ctrl:
            a1.plot([8], [ctrl[0]["val_ep12"]], "s", ms=9, mfc="none", color=OKABE_ITO[3], label="control (20-epoch run, epoch 12)")
    c3 = [r for r in rows if r.get("done") and r["window"] == "+-3" and r["val_ep12"] is not None]
    for r in c3:
        a1.plot([r["blocks"]], [r["val_ep12"]], "^", ms=8, color=OKABE_ITO[2], label="val at epoch 12 (+-3 days)")
    a1.set(xlabel="residual blocks in the dilated backbone", ylabel="val hidden-pixel MAE at epoch 12, ln CHL",
           title="Black Sea: hidden-pixel MAE vs residual blocks (12 epochs)", xticks=[4, 5, 6, 7, 8])
    a1.grid(alpha=.3); a1.legend(fontsize=8)
    for i, r in enumerate(rows):
        a2.plot(np.arange(1, len(r["hist_val"]) + 1), r["hist_val"], marker="o", ms=3, color=OKABE_ITO[i % len(OKABE_ITO)],
                ls="--" if (r["control"] or "_c3_" in r["run"]) else "-", lw=2.2 if r["control"] else 1.4, label=label(r))
    a2.set(xlabel="epoch (20-epoch cosine schedule, stopped at 12)", ylabel="val hidden-pixel MAE, ln CHL",
           title="Black Sea: val hidden-pixel MAE by epoch", xticks=range(1, N_EP + 1))
    a2.grid(alpha=.3); a2.legend(fontsize=8)
    OUT.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(OUT / "blk_block_sweep.png", dpi=120); plt.close(fig)
    print("wrote", OUT / "blk_block_sweep.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bullet"); ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    if a.bullet:
        print(bullet(load(a.bullet))); return
    rows = [r for r in (load(x) for x in RUNS) if r is not None]
    for r in rows:
        if r.get("done"):
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "hist_val"}))
    if not a.no_fig:
        figure(rows)


if __name__ == "__main__":
    main()
