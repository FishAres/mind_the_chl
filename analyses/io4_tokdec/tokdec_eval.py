"""Eval-only runs of a --t3d-stem tokdec model with the chl_cnn/tokdec.py recorder (io4 track; 2026-10-23). Data generation only.

Records, over every scored val day (the first n_val forwards of the eval pass, in the order of va[::val_stride]):
  pool_w   (n_val, Np, T) float16 pooling weights (mean over heads) of each ocean patch;
  visfrac  (n_val, Np, T) float16 visible fraction (visible / ocean px) of each patch on each window day, after donor masking;
and for the val days at positions --positions: emb, tok, pool_w, dec (decoded context channels), the stem input's visible mask,
and with --tokdec-blend the per-pixel day weights w, blend, mass, entropy.
Saved to ~/mind_the_chl_gap/data/io4/<run>_tokdec.npz (patch_idx, nPy, nPx for mapping patches to the grid).

Usage (project root): python -m analyses.io4_tokdec.tokdec_eval --run NAME --positions 0 5 --n-val 1458 -- <train_pinn args incl. --eval-only>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from chl_cnn import tokdec, train_pinn
from chl_cnn.ptok import _patch_sum

DATA = Path.home() / "mind_the_chl_gap" / "data" / "io4"
KEYS = ("emb", "tok", "pool_w", "dec", "w", "blend", "mass", "ent")


class Rec:
    def __init__(self, positions, n_val):
        self.pos, self.n_val, self.k, self.out = set(positions), n_val, 0, {}
        self.pool, self.visf, self.meta = [], [], None

    def __call__(self, d):
        B = d["pool_w"].shape[0]
        if self.k >= self.n_val:
            return
        m = d["net"]
        vis = d["x3d"][:, -1].float()
        visp = torch.nn.functional.pad(vis, (0, m.Wp - m.W, 0, m.Hp - m.H))
        vf = (_patch_sum(visp, m.P) / _patch_sum(m.ocean_p, m.P).clamp(min=1)).flatten(2)[..., m.patch_idx].transpose(1, 2)   # (B, Np, T)
        nb = min(B, self.n_val - self.k)
        self.pool.append(d["pool_w"][:nb].detach().float().cpu().numpy().astype("float16"))
        self.visf.append(vf[:nb].cpu().numpy().astype("float16"))
        if self.meta is None:
            self.meta = dict(patch_idx=m.patch_idx.cpu().numpy(), nPy=m.nPy, nPx=m.nPx, P=m.P)
        for i in range(nb):
            if self.k + i in self.pos:
                self.out[self.k + i] = {k: d[k][i].detach().float().cpu().numpy() for k in KEYS if k in d}
                self.out[self.k + i]["vis"] = (d["x3d"][i, -1] > 0.5).cpu().numpy()
        self.k += nb


def main():
    argv = sys.argv[1:]
    i = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--positions", type=int, nargs="+", required=True)
    p.add_argument("--n-val", type=int, required=True, help="number of scored val days (len(va[::val_stride]))")
    a = p.parse_args(argv[:i])
    rec = Rec(a.positions, a.n_val)
    tokdec.HOOKS["recorder"] = rec
    train_pinn.main(["--run", a.run] + argv[i + 1:])
    assert rec.k == a.n_val and set(rec.out) == set(a.positions), (rec.k, sorted(rec.out))
    DATA.mkdir(parents=True, exist_ok=True)
    flat = {f"{k}__{pos}": v for pos, d in rec.out.items() for k, v in d.items()}
    np.savez_compressed(DATA / f"{a.run}_tokdec.npz", positions=np.array(sorted(rec.out)), pool_w_all=np.concatenate(rec.pool),
                        visfrac_all=np.concatenate(rec.visf), **rec.meta, **flat)
    print(f"recorded positions {sorted(rec.out)}, pooling weights of {rec.k} val days", flush=True)


if __name__ == "__main__":
    main()
