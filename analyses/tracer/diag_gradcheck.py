"""Test 3a check: training objective and its autograd gradient vs a for the scalar tracer a x u_rea, all train starts.
For each a on a grid: one pass over the train starts in fixed batches of 16 (permutation seed 0, as one epoch of
tracer_diag.train_scalar); per batch the training loss (pixel L1 mean per lead over the batch's own valid set, mean of
leads 1-5) and d loss / d a by autograd (a a leaf tensor; same integrator, n_sub 4, bilinear). Reports the batch-mean
loss, batch-mean gradient, and the pooled own-set MAE (mean of leads). GPU; gpu.lock. -> results/diag_gradcheck.json
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

RES = Path(__file__).resolve().parent / "results"
K = 5
GRID = [0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]


def main():
    from chl_cnn import tracer as TR, tracer_diag as TD, train_tracer as TT
    dev = torch.device("cuda")
    t0 = time.time()
    a_, d = TD.setup(dev, K)
    H, W = d["ocean"].shape
    tr = TT.starts(d["times"], d["tr"], K)
    it = TR.Integrator(H, W, dev, n_sub=4, vmax=10.0)
    vel = TD.make_vel(d, "rea", 0, K)
    perm = np.random.default_rng(0).permutation(len(tr)); nb = len(perm) // 16
    out = dict(grid=GRID, n_train=int(len(tr)), n_batches=nb, loss=[], grad=[], pooled=[], pooled_lead=[])
    for al in GRID:
        a = torch.tensor(al, device=dev, requires_grad=True)
        L, G = [], []
        acc = np.zeros((K, 2))
        for bi in range(nb):
            bt = tr[perm[bi * 16:(bi + 1) * 16]]
            days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev)
            U, ok = vel(days)
            ls = []
            for k in range(1, K + 1):
                pos, v = it.departure(U * a, ok, k)
                pr, pv = it.depart_sample(d["Y"][days[:, 0]], d["OBS"][days[:, 0]], pos)
                m = v & pv & d["OBS"][days[:, k]]
                e = (pr - d["Y"][days[:, k]]).abs()[m]
                ls.append(e.mean())
                acc[k - 1] += [float(e.detach().double().sum()), float(m.sum())]
            loss = torch.stack(ls).mean()
            g, = torch.autograd.grad(loss, a)
            L.append(float(loss)); G.append(float(g))
        out["loss"].append(float(np.mean(L))); out["grad"].append(float(np.mean(G)))
        pl = acc[:, 0] / acc[:, 1]
        out["pooled"].append(float(pl.mean())); out["pooled_lead"].append(pl.tolist())
        print(f"a {al:.3f}  batch-mean loss {np.mean(L):.6f}  grad {np.mean(G):+.5f}  pooled own MAE mean lead {pl.mean():.6f}  ({time.time() - t0:.0f}s)", flush=True)
    (RES / "diag_gradcheck.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def fd_check(a0=0.2, delta=0.01, n_batches=60):
    """Autograd d loss / d a vs central finite differences at a0 on the first n_batches batches (same permutation),
    for (i) the training loss on each a's own valid set and (ii) the loss on a mask held fixed across a0 - delta,
    a0, a0 + delta (own(a0 - delta) & own(a0) & own(a0 + delta)). Pixel-pooled per lead (sum over batches), then mean
    of leads. -> results/diag_gradcheck_fd.json"""
    from chl_cnn import tracer as TR, tracer_diag as TD, train_tracer as TT
    dev = torch.device("cuda")
    a_, d = TD.setup(dev, K)
    H, W = d["ocean"].shape
    tr = TT.starts(d["times"], d["tr"], K)
    it = TR.Integrator(H, W, dev, n_sub=4, vmax=10.0)
    vel = TD.make_vel(d, "rea", 0, K)
    perm = np.random.default_rng(0).permutation(len(tr))
    S = np.zeros((3, K, 2)); SF = np.zeros((3, K, 2)); G = np.zeros((K, 2)); GF = np.zeros((K, 2))
    n_change = np.zeros(K)
    for bi in range(n_batches):
        bt = tr[perm[bi * 16:(bi + 1) * 16]]
        days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev)
        U, ok = vel(days)
        for k in range(1, K + 1):
            res = []
            for j, al in enumerate((a0 - delta, a0, a0 + delta)):
                a = torch.tensor(al, device=dev, requires_grad=(j == 1))
                pos, v = it.departure(U * a, ok, k)
                pr, pv = it.depart_sample(d["Y"][days[:, 0]], d["OBS"][days[:, 0]], pos)
                m = v & pv & d["OBS"][days[:, k]]
                res.append((a, (pr - d["Y"][days[:, k]]).abs(), m))
            mf = res[0][2] & res[1][2] & res[2][2]
            n_change[k - 1] += float((res[0][2] ^ res[2][2]).sum())
            for j, (a, e, m) in enumerate(res):
                S[j, k - 1] += [float(e[m].detach().double().sum()), float(m.sum())]
                SF[j, k - 1] += [float(e[mf].detach().double().sum()), float(mf.sum())]
            a, e, m = res[1]
            g_own, = torch.autograd.grad(e[m].sum(), a, retain_graph=True)
            g_fix, = torch.autograd.grad(e[mf].sum(), a)
            G[k - 1] += [float(g_own), float(m.sum())]; GF[k - 1] += [float(g_fix), float(mf.sum())]
    L = S[..., 0] / S[..., 1]; LF = SF[..., 0] / SF[..., 1]
    out = dict(a0=a0, delta=delta, n_batches=n_batches,
               fd_own=((L[2] - L[0]) / (2 * delta)).tolist(), autograd_own=(G[:, 0] / G[:, 1]).tolist(),
               fd_fixed=((LF[2] - LF[0]) / (2 * delta)).tolist(), autograd_fixed=(GF[:, 0] / GF[:, 1]).tolist(),
               n_own=S[1, :, 1].tolist(), n_fixed=SF[1, :, 1].tolist(), n_mask_changed=n_change.tolist(),
               note="own: autograd of the pooled mean on a0's own set (mask held at a0) vs FD of the pooled own-set MAE (mask varies)")
    for k_ in ("fd_own", "autograd_own", "fd_fixed", "autograd_fixed", "n_own", "n_mask_changed"):
        print(k_, np.round(out[k_], 5))
    (RES / "diag_gradcheck_fd.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fd":
    fd_check()
