"""Val hidden-pixel MAE and lr per epoch for the learning-rate-schedule runs (lr_*) of the conv D-net, t+-4 context,
train stride 2. Prints the summary table (markdown) and writes figures/lr_curves.png.

  python analyses/spatial_node/lr_curves.py [--runs lr_cos lr_const ...]
history.json: index 0 is the initial (untrained) evaluation, index e the state after epoch e; "lr" is the net lr that the
optimiser holds after that epoch's scheduler step (i.e. the lr of the NEXT epoch).
"""
import argparse
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chl_cnn.plotting import OKABE_ITO

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "models" / "history" / "spatial"
MODELS = ROOT / "models" / "spatial"
FIG = Path(__file__).resolve().parent / "figures"

# run -> (label, schedule, lr, lr_net, continues_from)
RUNS = {
    "lr_cos": ("cosine 3e-2/3e-3", "cosine", 3e-2, 3e-3, None),
    "lr_const": ("const 3e-2/3e-3", "none", 3e-2, 3e-3, None),
    "lr_const_low": ("const 1e-2/1e-3", "none", 1e-2, 1e-3, None),
    "lr_const_high": ("const 3e-2/1e-2", "none", 3e-2, 1e-2, None),
    "lr_cos12": ("cosine 12 ep", "cosine", 3e-2, 3e-3, None),
    "lr_const_cont": ("const, continued 6+6 ep", "none", None, None, None),  # placeholder, filled by --cont
}
REF = {"mt_long_conv_ctx4": "cosine 12 ep, stride 1 (ref)"}


def load(run):
    for base in (HIST, MODELS):
        f = base / run / "history.json"
        if f.exists():
            h = json.load(open(f))
            m = MODELS / run / "metrics.json"
            met = json.load(open(m)) if m.exists() else None
            return h, met
    return None, None


def row(run, h, met, sched, lr, lr_net, offset=0):
    val = h["val"][1:]
    fin, best = val[-1], min(val)
    gr = h["grad_ratio"][-1]; gc = h["grad_corr"][-1]
    beta = h["params"][-1]["beta"]
    mins = met["minutes"] if met else sum(h["epoch_time"][1:]) / 60
    return f"| {run} | {sched} | {lr:g} | {lr_net:g} | {offset + len(val)} | {', '.join(f'{v:.4f}' for v in val)} | {fin:.4f} | {best:.4f} | {gr:.3f} | {gc:.3f} | {beta:.5f} | {mins:.1f} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[r for r in RUNS if r != "lr_const_cont"])
    ap.add_argument("--cont", nargs=2, metavar=("BASE", "CONT"), default=None, help="continued run: CONT was initialised from BASE; plotted as one 12-epoch curve")
    a = ap.parse_args()

    print("| run | sched | lr | lr-net | epochs | val MAE per epoch | final | best | grad ratio | grad corr | beta final | minutes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    k = 0
    for run in a.runs:
        h, met = load(run)
        if h is None or len(h["val"]) < 2:
            print(f"| {run} | (missing or no completed epoch) |")
            continue
        label, sched, lr, lr_net, _ = RUNS.get(run, (run, "?", float("nan"), float("nan"), None))
        print(row(run, h, met, sched, lr, lr_net))
        ep = list(range(1, len(h["val"])))
        c = OKABE_ITO[k % len(OKABE_ITO)]; k += 1
        ax1.plot(ep, h["val"][1:], "o-", color=c, label=label)
        ax2.plot(ep, h["lr"][:-1], "o-", color=c, label=label)  # lr used during epoch e
    if a.cont:
        base, cont = a.cont
        hb, mb = load(base); hc, mc = load(cont)
        if hb is not None and hc is not None:
            lab, sched, lr, lr_net, _ = RUNS[base]
            print(row(cont, hc, mc, sched, lr, lr_net, offset=len(hb["val"]) - 1))
            val = hb["val"][1:] + hc["val"][1:]
            ep = list(range(1, len(val) + 1))
            c = OKABE_ITO[k % len(OKABE_ITO)]; k += 1
            nb = len(hb["val"]) - 1  # draw only the continuation (from the base run's last epoch) so the base curve stays visible
            ax1.plot(ep[nb - 1:], val[nb - 1:], "s--", color=c, label=f"{lab}, continued ({cont})")
            ax2.plot(ep[nb:], hc["lr"][:-1], "s--", color=c)
    for run, label in REF.items():
        h, _ = load(run)
        if h is None:
            continue
        ep = list(range(1, len(h["val"])))
        ax1.plot(ep, h["val"][1:], ":", color="#000000", label=label)
        ax2.plot(ep, h["lr"][:-1], ":", color="#000000")
    ax1.set_ylabel("val hidden-pixel MAE (ln CHL)")
    ax1.set_title("Val hidden-pixel MAE per epoch by learning-rate schedule (conv D, t+-4, train stride 2)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_ylabel("D-net lr during epoch"); ax2.set_xlabel("epoch"); ax2.set_yscale("log"); ax2.grid(alpha=0.3)
    ax2.set_title("D-net learning rate used in each epoch (scalar lr = 10x)")
    FIG.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / "lr_curves.png", dpi=130)
    print("wrote", FIG / "lr_curves.png")


if __name__ == "__main__":
    main()
