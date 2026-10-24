"""Eval-only run of a --t3d-stem patch model with chl_cnn/ptok.py HOOKS["no_blend"] on (io4 track Job 1, 2026-10-23).

The backbone gets blend 0, weight mass 0 and entropy 0 (all weight on the "no day" slot); FiLM and day t's channels unchanged.
Indicative only: the backbone never saw these inputs in training (distribution shift).

Usage (project root): python -m analyses.io4_tokdec.eval_hooks --no-blend -- <train_pinn args incl. --run, --eval-only --init-from>
Without --no-blend this is a plain eval-only run (the reproduction check).
"""
import argparse
import sys

from chl_cnn import ptok, train_pinn


def main():
    argv = sys.argv[1:]
    i = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-blend", action="store_true")
    a = p.parse_args(argv[:i])
    ptok.HOOKS["no_blend"] = a.no_blend
    print(f"ptok HOOKS: {ptok.HOOKS}", flush=True)
    train_pinn.main(argv[i + 1:])


if __name__ == "__main__":
    main()
