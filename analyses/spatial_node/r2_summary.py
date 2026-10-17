"""Round-2 Black Sea spatial nODE runs (models/spatial/r2_*): markdown table of the final metrics, CG statistics and learned
scalars, plus two figures: val MAE per epoch and beta / w(+-1) per epoch (does --norm-d stop the drift?).
Usage: python analyses/spatial_node/r2_summary.py [run ...]   (default: r2_* plus blk_conv_ctx4_cf and blk_final_conv_ctx4_cf)"""
import sys, json, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chl_cnn.plotting import OKABE_ITO

ROOT = Path(__file__).resolve().parents[2]; M = ROOT / "models/spatial"; FIG = Path(__file__).resolve().parent / "figures"; FIG.mkdir(exist_ok=True)
FLAGS = ["norm_d", "real_depth", "sal_feat", "river_feat", "extra_feats"]


def row(run):
    d = M / run
    if not (d / "metrics.json").exists():
        return None
    c, m, h = json.load(open(d / "config.json")), json.load(open(d / "metrics.json")), json.load(open(d / "history.json"))
    v, p = m["val"], m["params"]
    fl = " ".join("--" + f.replace("_", "-") for f in FLAGS if c.get(f)) or "-"
    w = p.get("w", {}); wtxt = "/".join(f"{0.5 * (w.get(str(k), 0) + w.get(str(-k), 0)):.4f}" for k in (1, 2, 3, 4))
    cap = v.get("cg_cap_batches"); cg = f"{v['cg_iters']:.0f}" + ("" if cap is None else f" ({cap}/{v['cg_batches']} capped, rel max {v['cg_rel_max']:.0e})")
    tcg = h["cg"][-1] if h.get("cg") else None
    train_cg = "-" if not tcg else f"fwd {tcg['train_fwd_it'] / max(tcg['train_n'], 1):.0f} cap {tcg['train_fwd_cap']}/{tcg['train_n']} rel {tcg['train_fwd_rel']:.0e}; bwd cap {tcg['train_bwd_cap']}/{tcg['train_n']} rel {tcg['train_bwd_rel']:.0e}"
    return dict(run=run, flags=fl, dmodel=c["dmodel"], epochs=c["epochs"], stride=c["train_stride"], sched=c.get("sched"), maxiter=c.get("maxiter"), chl_max=c.get("chl_max", 0),
                mae=v["mae"], best=m["best_val_mae"], rmse=v["rmse"], clim=v["clim_mae"], gr=v["grad_ratio"], gc=v["grad_corr"], am=v["aniso_median"], ap=v["aniso_p90"],
                beta=p["beta"], w=wtxt, cg=cg, train_cg=train_cg, minutes=m["minutes"], hist=h)


def main(runs):
    rows = [r for r in (row(x) for x in runs) if r]
    print("| run | flags | dmodel | ep | stride | sched/maxiter/chl-max | val MAE final | best | clim (same px) | RMSE | grad ratio | grad corr | aniso med | p90 | beta | w(+-1/2/3/4) | CG eval iters (cap) | train CG last epoch | min |")
    print("|---||---||---||---||---||---||---||---||---||---||---||---||---||---||---||---||---||---||---|")
    for r in rows:
        print(f"| {r['run']} | {r['flags']} | {r['dmodel']} | {r['epochs']} | {r['stride']} | {r['sched']}/{r['maxiter']}/{r['chl_max']:g} | {r['mae']:.4f} | {r['best']:.4f} | {r['clim']:.4f} | {r['rmse']:.4f} | {r['gr']:.3f} | {r['gc']:.3f} | "
              f"{r['am']:.2f} | {r['ap']:.1f} | {r['beta']:.5f} | {r['w']} | {r['cg']} | {r['train_cg']} | {r['minutes']:.0f} |")
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, r in enumerate(rows):
        h = r["hist"]; col = OKABE_ITO[i % len(OKABE_ITO)]; ep = range(len(h["val"]))
        ax[0].plot(ep, h["val"], "o-", color=col, label=r["run"])
        ax[1].plot(ep, [q["beta"] for q in h["params"]], "o-", color=col, label=r["run"])
        ax[2].plot(ep, [0.5 * (q.get("w", {}).get("1", 0) + q.get("w", {}).get("-1", 0)) for q in h["params"]], "o-", color=col, label=r["run"])
    ax[0].set(title="Val hidden-pixel MAE (ln CHL) per epoch", xlabel="epoch (0 = init)", ylabel="val MAE")
    ax[1].set(title="Learned beta (climatology weight) per epoch", xlabel="epoch (0 = init)", ylabel="beta", yscale="log")
    ax[2].set(title="Learned neighbour-day weight w(+-1) per epoch", xlabel="epoch (0 = init)", ylabel="w(+-1)", yscale="log")
    ax[0].legend(fontsize=7)
    fig.suptitle("Black Sea spatial nODE round 2: training curves and scalar parameters"); fig.tight_layout()
    fig.savefig(FIG / "r2_curves.png", dpi=110); print("wrote", FIG / "r2_curves.png")


if __name__ == "__main__":
    runs = sys.argv[1:] or ["blk_conv_ctx4_cf", "blk_final_conv_ctx4_cf"] + sorted(Path(p).parent.name for p in glob.glob(str(M / "r2_*/metrics.json")))
    main(runs)
