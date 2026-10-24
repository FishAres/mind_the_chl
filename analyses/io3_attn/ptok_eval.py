"""Eval-only runs of a --t3d-stem patch model with the chl_cnn/ptok.py recorder (io3 track, Part B; 2026-10-23). Data generation only.

Records, for the val days at positions --positions of the scored val list (va[::val_stride]), the patch tokens (after self-attention),
the token input statistics, the per-pixel day weights (T days + "no day"), the score terms (patch score interpolated, pixel term), the
blend / mass / entropy channels and the stem input's visible mask. Saved to ~/mind_the_chl_gap/data/io3/<run>_ptok.npz.

Usage (project root): python -m analyses.io3_attn.ptok_eval --run NAME --positions 0 5 --eval-batch 8 -- <train_pinn args incl. --eval-only>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from chl_cnn import ptok, train_pinn

DATA = Path.home() / "mind_the_chl_gap" / "data" / "io3"
KEYS = ("tok", "stats", "w", "blend", "mass", "ent", "s_patch", "s_pixel", "conf", "value")


class Rec:
    def __init__(self, positions, n_val):
        self.pos, self.n_val, self.k, self.out = set(positions), n_val, 0, {}

    def __call__(self, d):
        B = d["w"].shape[0]
        if self.k >= self.n_val:
            return
        for i in range(B):
            if self.k + i in self.pos:
                self.out[self.k + i] = {k: d[k][i].detach().float().cpu().numpy() for k in KEYS if k in d}
                self.out[self.k + i]["vis"] = (d["x3d"][i, -1] > 0.5).cpu().numpy()
        self.k += B


def main():
    argv = sys.argv[1:]
    i = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--positions", type=int, nargs="+", required=True)
    p.add_argument("--n-val", type=int, required=True, help="number of scored val days (len(va[::val_stride]))")
    a = p.parse_args(argv[:i])
    rec = Rec(a.positions, a.n_val)
    ptok.HOOKS["recorder"] = rec
    train_pinn.main(["--run", a.run] + argv[i + 1:])
    assert rec.k == a.n_val and set(rec.out) == set(a.positions), (rec.k, sorted(rec.out))
    DATA.mkdir(parents=True, exist_ok=True)
    flat = {f"{k}__{pos}": v for pos, d in rec.out.items() for k, v in d.items()}
    np.savez_compressed(DATA / f"{a.run}_ptok.npz", positions=np.array(sorted(rec.out)), **flat)
    print(f"recorded positions {sorted(rec.out)}", flush=True)


if __name__ == "__main__":
    main()
