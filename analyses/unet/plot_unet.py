"""Val hidden-pixel MAE by epoch for the unet_c* campaign runs plus two reference curves.
Usage: python analyses/unet/plot_unet.py  -> analyses/unet/learning_curves.png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2", "#CC79A7", "#F0E442", "#000000"]
RUNS = [  # (label, history path, colour, linestyle)
    ("c1 d3w64 ctx2 (20 ep)", "models/pinn/unet/unet_c1_d3w64", OKABE_ITO[0], "-"),
    ("c2 + ctx4 (20 ep)", "models/pinn/unet/unet_c2_ctx4", OKABE_ITO[1], "-"),
    ("c3 + ctx4, lr 2e-3 (20 ep)", "models/pinn/unet/unet_c3_lr2e3", OKABE_ITO[2], "-"),
    ("c4 + ctx4, batch 4 (20 ep)", "models/pinn/unet/unet_c4_b4", OKABE_ITO[3], "-"),
    ("ref: dilated CNN pinn_v2_lam0 (60 ep)", "models/pinn/cnn/pinn_v2_lam0", OKABE_ITO[7], "--"),
    ("ref: U-Net unet_mtcg_lam0 (30 ep)", "models/history/pinn/unet/unet_mtcg_lam0", OKABE_ITO[4], ":"),
]


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, rel, col, ls in RUNS:
        p = ROOT / rel / "history.json"
        if not p.exists():
            print("missing", p)
            continue
        val = json.load(open(p))["val"]
        ax.plot(range(1, len(val) + 1), val, ls, color=col, lw=1.8, label=f"{label}: best {min(val):.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val hidden-pixel MAE (ln CHL)")
    ax.set_title("Val hidden-pixel MAE by epoch: U-Net configurations vs references")
    ax.set_ylim(0.13, 0.20)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out = ROOT / "analyses/unet/learning_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
