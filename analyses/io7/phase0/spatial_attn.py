"""Phase 0 item 2 (data generation, GPU): spatial self-attention weights of a --tokdec-spatial model, binned by patch distance and |dSST|.

Eval-only pass of chl_cnn.train_pinn with the tokdec recorder. No edit to chl_cnn/tokdec.py: this script wraps _SpatialBlock.forward at
class level (in this process only) to keep each block's input; on the selected val days the weights are recomputed explicitly as
softmax(q k^T / sqrt(d_head) + relative-position bias), the same expression F.scaled_dot_product_attention evaluates in the block.
Selected days: every --every-th scored val position (default 10: the 146 io6 diag days). Every ocean patch is a query.
Accumulated per block and head (float64):
  sum_w[blk, head, dist_bin, sst_bin]   sum of weights over (query, key) pairs; n_pairs[dist_bin, sst_bin] pair counts (same for all heads)
  dist = Euclidean distance between patch centres in patches (1 patch = 4 px = 1 deg); |dSST| = |patch-mean day-t SST(query) -
  patch-mean day-t SST(key)| in K (ocean px of the patch; trainer z-score x feature_stats sst std)
  rel_sum_w[blk, rel_offset] / rel_n[rel_offset]   head-mean weight summed per relative patch offset (dy, dx) -> mean-weight map
Saved to ~/mind_the_chl_gap/data/io7/<run>_spattn.npz.
Usage (project root): python -m analyses.io7.phase0.spatial_attn --run NAME --n-val 1458 --n-test 1463 [--every 10] -- <train_pinn args incl. --eval-only>
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chl_cnn import tokdec, train_pinn
from chl_cnn.runs import find_run

DATA = Path.home() / "mind_the_chl_gap" / "data" / "io7"
DIST_EDGES = np.array([0, 0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, np.inf])          # patches: 0 | 1 | 2 | 3-4 | 5-8 | 9-16 | 17-32 | 33+
SST_EDGES_K = np.array([0, 0.125, 0.375, 0.625, 0.875, 1.125, 1.625, 2.125, 3.125, np.inf])   # as analyses/io6_kriging/io6_eval.py

_INPUTS = []
_orig = tokdec._SpatialBlock.forward


def _fwd(self, x, rel_idx):
    _INPUTS.append((self, x.detach(), rel_idx))
    return _orig(self, x, rel_idx)


@torch.no_grad()
def weights(blk, x, rel_idx):
    """(b, heads, Np, Np) float32 attention weights of one _SpatialBlock for inputs x (b, Np, d)."""
    B, Np, d = x.shape
    q, k, _ = blk.qkv(blk.n1(x.float())).view(B, Np, 3, blk.h, d // blk.h).permute(2, 0, 3, 1, 4)
    bias = blk.rel[rel_idx].permute(2, 0, 1).unsqueeze(0).float()
    s = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) / math.sqrt(d // blk.h) + bias
    return torch.softmax(s, -1)


class Rec:
    def __init__(self, n_val, every, sst_std):
        self.n_val, self.every, self.sst_std, self.k, self.acc, self.n_days = n_val, every, sst_std, 0, None, 0

    def setup(self, net, dev):
        py, px = (net.patch_idx // net.nPx).float(), (net.patch_idx % net.nPx).float()
        dist = torch.sqrt((py.view(-1, 1) - py.view(1, -1)) ** 2 + (px.view(-1, 1) - px.view(1, -1)) ** 2)
        self.dbin = (torch.bucketize(dist, torch.as_tensor(DIST_EDGES[1:-1], device=dev, dtype=torch.float32), right=False)).view(-1)
        self.nD, self.nS = len(DIST_EDGES) - 1, len(SST_EDGES_K) - 1
        self.sedges = torch.as_tensor(SST_EDGES_K[1:-1] / self.sst_std, device=dev, dtype=torch.float32)
        nb, h, nrel = len(net.sp_blocks), net.h, net.sp_blocks[0].rel.shape[0]
        self.acc = dict(sum_w=torch.zeros(nb, h, self.nD * self.nS, device=dev, dtype=torch.float64),
                        n_pairs=torch.zeros(self.nD * self.nS, device=dev, dtype=torch.float64),
                        rel_sum_w=torch.zeros(nb, nrel, device=dev, dtype=torch.float64),
                        rel_n=torch.zeros(nrel, device=dev, dtype=torch.float64))
        self.meta = dict(nPy=net.nPy, nPx=net.nPx, Np=net.Np, heads=h, n_blocks=nb, patch=net.P)

    @torch.no_grad()
    def __call__(self, d):
        net = d["net"]
        ins, _INPUTS[:] = list(_INPUTS), []
        B = d["x3d"].shape[0]
        nb = min(B, max(self.n_val - self.k, 0))
        sel = [i for i in range(nb) if (self.k + i) % self.every == 0]
        self.k += B
        if not sel:
            return
        dev = d["x3d"].device
        if self.acc is None:
            self.setup(net, dev)
        assert len(ins) == len(net.sp_blocks), len(ins)
        # patch-mean day-t SST (z units) over ocean px
        sst = d["x3d"][sel, net.sst, net.ti].float()                                     # (b, H, W)
        oc = net.ocean_p[:net.H, :net.W]
        pad = lambda z: F.pad(z, (0, net.Wp - net.W, 0, net.Hp - net.H))
        num = F.avg_pool2d(pad(sst * oc).unsqueeze(1), net.P).flatten(1)[:, net.patch_idx]
        den = F.avg_pool2d(pad(oc.expand_as(sst)).unsqueeze(1), net.P).flatten(1)[:, net.patch_idx]
        ps = num / den                                                                   # (b, Np)
        for b_, i in enumerate(sel):
            ds = (ps[b_].view(-1, 1) - ps[b_].view(1, -1)).abs().view(-1)
            cell = self.dbin * self.nS + torch.bucketize(ds, self.sedges)
            self.acc["n_pairs"] += torch.bincount(cell, minlength=self.nD * self.nS).double()
            rel = ins[0][2].reshape(-1)
            self.acc["rel_n"] += torch.bincount(rel, minlength=self.acc["rel_n"].numel()).double()
            for j, (blk, x, rel_idx) in enumerate(ins):
                w = weights(blk, x[i:i + 1], rel_idx)[0].reshape(blk.h, -1)            # (h, Np*Np)
                for hh in range(blk.h):
                    self.acc["sum_w"][j, hh] += torch.bincount(cell, weights=w[hh].double(), minlength=self.nD * self.nS)
                self.acc["rel_sum_w"][j] += torch.bincount(rel, weights=w.mean(0).double(), minlength=self.acc["rel_n"].numel())
            self.n_days += 1


def main():
    argv = sys.argv[1:]
    i = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--n-val", type=int, required=True)
    p.add_argument("--n-test", type=int, required=True)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--out", default=None, help="output stem (default: the --init-from run)")
    a = p.parse_args(argv[:i])
    targs = argv[i + 1:]
    src = targs[targs.index("--init-from") + 1]
    sst_std = float(json.loads((find_run(src) / "config.json").read_text())["feature_stats"]["sst"]["std"])
    tokdec._SpatialBlock.forward = _fwd
    rec = Rec(a.n_val, a.every, sst_std)
    tokdec.HOOKS["recorder"] = rec
    train_pinn.main(["--run", a.run] + targs)
    assert rec.acc is not None and rec.n_days == len(range(0, a.n_val, a.every)) and rec.k == a.n_val + a.n_test, (rec.n_days, rec.k)
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"{a.out or src}_spattn.npz"
    np.savez(out, dist_edges=DIST_EDGES, sst_edges_k=SST_EDGES_K, sst_std=sst_std, every=a.every, n_days=rec.n_days,
             **{k: v.cpu().numpy() for k, v in rec.acc.items()}, **{k: np.array(v) for k, v in rec.meta.items()})
    print(f"recorded {rec.n_days} val days -> {out}", flush=True)


if __name__ == "__main__":
    main()
