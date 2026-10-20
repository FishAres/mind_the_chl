"""Check of the learned per-band velocity factors (test 2 const runs gave a ~ 0.72 / 0.03 / 0.36 for < 20 / 20-50 /
50-150 km): no-training grid over a_b in A_VALS for each band, damping fixed at tau = TAU (the const runs' last-5-epoch
values), B held fixed (no source). Pixel MAE on the test-1 common mask intersected with validity of every grid point
(one fixed px set for all combos). Output analyses/tracer/results/band_check_a.{npz,md}, figure band_check_a.png.
Run under flock ~/mind_the_chl_gap/logs/gpu.lock.
"""
import sys, json, time, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

RES = ROOT / "analyses/tracer/results"; FIG = ROOT / "analyses/tracer/figures"
A_VALS = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
TAU = [0.97, 2.5, 9.3]
K = 5


def gpu():
    import torch
    from analyses.tracer import band_train as BT
    from chl_cnn import train_tracer as TT, tracer as TR, tracer_band as TB
    t0 = time.time(); dev = torch.device("cuda")
    d = BT.load_cpu(); dg = BT.to_dev(d, dev); H, W = d["ocean"].shape
    va_s = TT.starts(d["times"], d["va"], K)[::2]
    it = TR.Integrator(H, W, dev, n_sub=4, vmax=10.0)
    combos = list(itertools.product(range(len(A_VALS)), repeat=3))
    ci = torch.as_tensor(np.array(combos), device=dev)
    acc = torch.zeros(len(va_s), K, len(combos), 2, dtype=torch.float64, device=dev)
    w_all = torch.exp(-torch.arange(1, K + 1, device=dev)[:, None].float() / torch.as_tensor(TAU, device=dev)[None])
    with torch.no_grad():
        for b0 in range(0, len(va_s), 16):
            bt = va_s[b0:b0 + 16]; nb = len(bt)
            days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev); fl = days.reshape(-1); t = days[:, 0]
            ok = dg["CUROK"][fl].float().view(nb, K + 1, H, W)
            U1 = torch.stack([dg["U"][fl].float() * dg["CPDX"], dg["V"][fl].float() * dg["CPDY"]], 1).view(nb, K + 1, 2, H, W)
            Bt = dg["BL"][t]; A, mA = TB.decompose(dg["Y"][t], dg["OBS"][t], Bt)
            for k in range(1, K + 1):
                truth = dg["Y"][days[:, k]]
                S = torch.zeros(len(A_VALS), 3, nb, H, W, device=dev)
                cm = dg["MASK"][b0:b0 + nb, k - 1].clone()
                for i, a_ in enumerate(A_VALS):
                    pos, val = it.departure(U1 * a_, ok, k)
                    s, v = TB.band_sample(it, A, mA, pos)
                    S[i] = s.permute(1, 0, 2, 3); cm &= val & v
                cf = cm.double(); Bz = torch.nan_to_num(Bt); w = w_all[k - 1]
                for c0 in range(0, len(combos), 64):
                    cc = ci[c0:c0 + 64]
                    pred = Bz[None] + w[0] * S[cc[:, 0], 0] + w[1] * S[cc[:, 1], 1] + w[2] * S[cc[:, 2], 2]
                    e = (pred - truth[None]).abs().double() * cf[None]
                    acc[b0:b0 + nb, k - 1, c0:c0 + len(cc), 0] = e.sum((2, 3)).T
                    acc[b0:b0 + nb, k - 1, c0:c0 + len(cc), 1] = cf.sum((1, 2))[:, None]
            print(f"  {b0 + nb}/{len(va_s)} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(RES / "band_check_a.npz", acc=acc.cpu().numpy(), combos=np.array(combos), a_vals=np.array(A_VALS), tau=np.array(TAU), starts=va_s)
    print(f"wall {time.time() - t0:.0f}s")


def report():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from chl_cnn.plotting import OKABE_ITO as C
    z = np.load(RES / "band_check_a.npz"); acc = z["acc"].sum(0); combos = [tuple(c) for c in z["combos"]]; av = list(z["a_vals"])
    mae = acc[..., 0] / acc[..., 1]; n = acc[:, 0, 1]
    ml = mae.mean(0)
    best = combos[int(np.argmin(ml))]
    bl = [combos[int(np.argmin(mae[k]))] for k in range(K)]
    ix = lambda a: combos.index(tuple(av.index(x) for x in a))
    L = ["# Per-band velocity factor grid (tau fixed 0.97 / 2.5 / 9.3 d, B held, no source), test-1 mask & all grid points valid", "",
         f"n px leads 1-5: {', '.join(f'{int(x):,}' for x in n)}", "",
         f"Mean-lead MAE optimum a (< 20, 20-50, 50-150): {tuple(av[i] for i in best)}; per lead: {[tuple(av[i] for i in b) for b in bl]}", "",
         "| a (< 20, 20-50, 50-150) | l1 | l2 | l3 | l4 | l5 | mean |", "|---|---|---|---|---|---|---|"]
    for a in [(0.25, 0.25, 0.25), (0.7, 0.0, 0.4), (0.7, 0.1, 0.4), (0.7, 0.25, 0.4), (0.25, 0.0, 0.25), (0.0, 0.0, 0.0), tuple(av[i] for i in best)]:
        q = ix(a); L.append(f"| {a} | " + " | ".join(f"{mae[k, q]:.4f}" for k in range(K)) + f" | {ml[q]:.4f} |")
    (RES / "band_check_a.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for b in range(3):
        for k, c in zip(range(K), C):
            other = list(best)
            ys = []
            for i in range(len(av)):
                other[b] = i; ys.append(mae[k, combos.index(tuple(other))] - mae[k, combos.index(best)])
            axes[b].plot(av, ys, marker="o", color=c, label=f"lead {k + 1}")
        axes[b].set(xlabel=f"a for band {['< 20', '20-50', '50-150'][b]} km (others at the mean-lead optimum)", ylabel="MAE minus optimum MAE (ln CHL)",
                    title=f"MAE vs a, band {['< 20', '20-50', '50-150'][b]} km")
        axes[b].grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "band_check_a.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    {"gpu": gpu, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
