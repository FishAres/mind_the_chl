"""Eval-only recorder for --wm-attn models: analyses/io6_kriging/io6_eval.py with the water-mass attention records added (wm: the dv + 4 stem
channels, wm_pool: the key pool's candidate indices, -1 = invalid slot, wm_nvalid). Same CLI as io6_eval; writes <data io6>/<run>_io6.npz.
Used by analyses/io7/wmattn/leak_wmattn.py."""
from analyses.io6_kriging import io6_eval as E

E.KEYS = tuple(E.KEYS) + ("wm", "wm_pool", "wm_nvalid")

if __name__ == "__main__":
    E.main()
