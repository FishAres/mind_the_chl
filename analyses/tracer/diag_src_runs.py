"""Test 3b driver: runs chl_cnn.train_tracer.main unchanged, with optional in-process patches (no edits to tracer.py /
train_tracer.py):
  --wd X        torch.optim.Adam -> AdamW with decoupled weight decay X on the source-net parameter group only (other
                groups weight_decay 0, i.e. plain Adam).
  --dropout p   tracer.SourceModel replaced by a subclass with nn.Dropout(p) after each hidden GELU (active in train
                mode; train_tracer's evaluate() calls src.eval()).
Always: tracer.grad_acc is wrapped to record, for predictor "model" (first of PRED0, 5 predictors; no --ref-run), the
per-start-day sum |pred - truth| and n on the common set for every lead, in call order (evaluate: batches of 16 starts x
leads 1..K x predictors). Saved to models/tracer/<run>/perday_val.npz (e, n: (n_evals, n_val_starts, K); evals = init,
epochs 1..E, final) and checked against history.json val_model_common.
Usage: python -m analyses.tracer.diag_src_runs [--wd X] [--dropout p] -- <train_tracer args>
"""
import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn


def main():
    argv = sys.argv[1:]
    sep = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args(argv[:sep]); targv = argv[sep + 1:]
    from chl_cnn import tracer as TR, train_tracer as TT
    from chl_cnn.runs import CATEGORIES
    src_ids = set()

    Base = TR.SourceModel

    class SourceModelDiag(Base):
        def __init__(self, in_ch, hidden=32, g_max=0.5):
            super().__init__(in_ch, hidden, g_max)
            if args.dropout > 0:
                old = self.net
                self.net = nn.Sequential(old[0], old[1], nn.Dropout(args.dropout), old[2], old[3], nn.Dropout(args.dropout), old[4])
            src_ids.update(id(p) for p in self.parameters())
    TR.SourceModel = SourceModelDiag

    if args.wd > 0:
        def adam_with_src_wd(groups, *a, **k):
            for g in groups:
                g["weight_decay"] = args.wd if all(id(p) in src_ids for p in g["params"]) else 0.0
            n_wd = sum(1 for g in groups if g["weight_decay"] > 0)
            assert n_wd == 1, n_wd
            print(f"AdamW: weight decay {args.wd} on {n_wd} group (source)", flush=True)
            return torch.optim.AdamW(groups, *a, **k)
        TT.torch.optim.Adam = adam_with_src_wd           # process-local patch; only train_tracer builds an optimiser here

    rec = []
    calls = [0]
    orig = TR.grad_acc

    def hooked(pred, truth, mask, lags=(1, 4, 16)):
        if calls[0] % len(TT.PRED0) == 0:
            m = mask.double()
            e = torch.where(mask, (pred.double() - truth.double()).abs(), 0.0)
            rec.append((e.sum((1, 2)).cpu().numpy(), m.sum((1, 2)).cpu().numpy()))
        calls[0] += 1
        return orig(pred, truth, mask, lags)
    TR.grad_acc = hooked

    TT.main(targv)

    a = TT.parse(targv)
    K = a.leads
    out = CATEGORIES["tracer"] / a.run
    hist = json.load(open(out / "history.json"))
    nval = len(json.load(open(out / "metrics.json"))["best"]["val_days"])
    nb = (nval + 15) // 16
    per_eval = nb * K
    assert len(rec) % per_eval == 0, (len(rec), per_eval)
    ne = len(rec) // per_eval
    E = np.zeros((ne, nval, K)); N = np.zeros((ne, nval, K))
    for j, (e, n) in enumerate(rec):
        ev, b, k = j // per_eval, (j // K) % nb, j % K
        E[ev, b * 16:b * 16 + len(e), k] = e; N[ev, b * 16:b * 16 + len(n), k] = n
    mae = E.sum(1) / N.sum(1)
    hv = np.array(hist["val_model_common"])
    diff = float(np.abs(mae[1:1 + len(hv)] - hv).max())
    print(f"per-day record: {ne} evals; max |pooled - history val_model_common| = {diff:.2e}", flush=True)
    np.savez_compressed(out / "perday_val.npz", e=E, n=N, check_max_abs_diff=diff, evals=np.array(["init"] + [f"epoch{i}" for i in range(1, len(hv) + 1)] + (["final"] if ne > len(hv) + 1 else [])))


if __name__ == "__main__":
    main()
