"""Step 2 (tr_src, tr_src_s1, saved best epoch) re-scored on exactly the px of each test-2 epoch evaluation:
test-1 common mask & band model (trb_*) valid at that epoch, for the best epoch and the last 5 epochs of each run.
Per-start sums [sum|e|, n, band(20-70 km) sum|e|, band n] for predictors model and the two step-2 runs; the model sums
must equal the trainer's perday.npz (checked). Output analyses/tracer/results/band_step2_rescore.npz.
Run under flock ~/mind_the_chl_gap/logs/gpu.lock.
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from analyses.tracer import band_train as BT, band_test1 as B1
from chl_cnn import train_tracer as TT, tracer as TR, tracer_band as TB
from chl_cnn.runs import CATEGORIES

RUNS = ["trb_const_s0", "trb_const_s1", "trb_phys_s0", "trb_phys_s1"]
K = 5


def main():
    t0 = time.time(); dev = torch.device("cuda")
    d = BT.load_cpu(); dg = BT.to_dev(d, dev)
    H, W = d["ocean"].shape; NF = len(TT.SRC_CH)
    va_s = TT.starts(d["times"], d["va"], K)[::2]
    tau, _ = TB.tau_measured()
    jobs = []                                     # (run, epoch, model)
    for r in RUNS:
        p = CATEGORIES["tracer"] / r
        if not (p / "metrics.json").exists():
            continue
        m = json.load(open(p / "metrics.json")); cfg = json.load(open(p / "config.json"))
        E = len(json.load(open(p / "history.json"))["epoch"])
        for e in sorted(set([m["best_epoch"]] + list(range(max(1, E - 4), E + 1)))):
            mdl, _ = BT.build(cfg, d, dev)
            mdl.load_state_dict(torch.load(p / f"ep{e}.pt", map_location=dev)["model"]); mdl.eval()
            jobs.append((r, e, mdl))
    eng = BT.Engine(dg, jobs[0][2], tau, dev)
    it, bp = eng.it, eng.bp
    S2 = {r: B1.load_vel(dg, r, dev) for r in ["tr_src", "tr_src_s1"]}
    ns = len(va_s)
    acc = np.zeros((len(jobs), ns, K, 3, 4))
    with torch.no_grad():
        for b0 in range(0, ns, 16):
            bt = va_s[b0:b0 + 16]; nb = len(bt)
            days = torch.as_tensor(bt[:, None] + np.arange(K + 1)[None], device=dev); fl = days.reshape(-1); t = days[:, 0]
            ok = dg["CUROK"][fl].float().view(nb, K + 1, H, W)
            s2p = {}
            for r, (vm, src, _, _) in S2.items():
                g = lambda X: X[fl].float()
                ue, ve = vm(dg["STATIC"], dg["DS"][fl], dg["DC"][fl], g(dg["U"]), g(dg["V"]), g(dg["UG"]), g(dg["VG"]), g(dg["TX"]), g(dg["TY"]), dg["OCf"])
                U = torch.stack([ue * dg["CPDX"], ve * dg["CPDY"]], 1).view(nb, K + 1, 2, H, W)
                s2p[r] = []
                for k in range(1, K + 1):
                    path = []
                    pos, val = it.departure(U, ok, k, path=path)
                    c, pv = it.depart_sample(dg["Y"][t], dg["OBS"][t], pos)
                    Fd = dg["FEAT"][fl].float().view(nb, K + 1, NF, H, W)
                    for p, j, w in reversed(path):
                        smp = it.sample(torch.cat([Fd[:, j], Fd[:, j + 1], dg["SSTATIC"][None].expand(nb, -1, -1, -1)], 1), p)
                        ds = ((1 - w) * dg["DS"][days[:, j]] + w * dg["DS"][days[:, j + 1]]).view(nb, 1, 1, 1).expand(nb, 1, H, W)
                        dc = ((1 - w) * dg["DC"][days[:, j]] + w * dg["DC"][days[:, j + 1]]).view(nb, 1, 1, 1).expand(nb, 1, H, W)
                        x = torch.cat([(1 - w) * smp[:, :NF] + w * smp[:, NF:2 * NF], smp[:, 2 * NF:], ds, dc, ((c - dg["CMU"]) / dg["CSD"])[:, None]], 1)
                        c = c + 0.25 * src(x)
                    s2p[r].append(c)
            for q, (r, e, mdl) in enumerate(jobs):
                eng.m = mdl
                preds, _, _ = eng.forward(days, grad=False)
                for k in range(1, K + 1):
                    truth = dg["Y"][days[:, k]]
                    pm, vm_ = preds[k - 1]
                    cm = dg["MASK"][b0:b0 + nb, k - 1] & vm_
                    cf = cm.double(); bt_t, vb = bp(truth, cm); vbf = vb.double()
                    for i, x in enumerate([pm, s2p["tr_src"][k - 1], s2p["tr_src_s1"][k - 1]]):
                        e_ = x - truth; bpp, _ = bp(x, cm)
                        acc[q, b0:b0 + nb, k - 1, i] = torch.stack([(e_.abs().double() * cf).sum((1, 2)), cf.sum((1, 2)), ((bpp - bt_t).abs().double() * vbf).sum((1, 2)), vbf.sum((1, 2))], 1).cpu().numpy()
            print(f"  {b0 + nb}/{ns} {time.time() - t0:.0f}s", flush=True)
    chk = {}
    for q, (r, e, _) in enumerate(jobs):
        pd_ = np.load(CATEGORIES["tracer"] / r / "perday.npz")["acc"][e, :, :, 0]
        chk[f"{r}@{e}"] = float(np.abs(pd_ - acc[q, :, :, 0]).max())
    np.savez_compressed(ROOT / "analyses/tracer/results/band_step2_rescore.npz", acc=acc, jobs=np.array([f"{r}@{e}" for r, e, _ in jobs]),
                        preds=np.array(["model", "tr_src", "tr_src_s1"]), starts=va_s)
    print("max |model sums - perday.npz|:", chk, f"wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
