"""Score runs with analyses/io_improve/score.py, writing to analyses/io3_attn/results/<run>.json (io3 track).

Usage (project root): python -m analyses.io3_attn.score_runs RUN [RUN ...]
"""
import sys
from pathlib import Path

from analyses.io_improve import score

score.RES = Path(__file__).resolve().parent / "results"

if __name__ == "__main__":
    for r in sys.argv[1:]:
        score.score_run(r)
