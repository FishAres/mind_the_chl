"""Leak test of the water-mass attention model (io7_wmattn): analyses/io3_attn/leak_ptok.py's four eval-only cases (base; +1 on day t's
hidden px; +1 on the t+2 px the donor mask hides; positive control +1 on every observed px of t+2) on a --wm-attn model, recording the
token-decoder tensors, the blend, and the stem's channels and key pool (analyses/io7/wmattn/wm_eval.py).
Passes (exit 0) only if every recorded tensor, the key pool included, is bit-identical in the hidden and masked_t+2 cases, and the positive
control changes the prediction on hidden px and the stem channels. Writes analyses/io7/wmattn/results/leak_<RUN>.json.
Usage (project root): python -m analyses.io7.wmattn.leak_wmattn --init-from RUN --base-args "<trainer flags incl. --wm-attn>"
"""
import json
import sys
from pathlib import Path

from analyses.io3_attn import leak_ptok as LP

HERE = Path(__file__).resolve().parent
NAMES = {"emb": "patch_embeddings", "tok": "tokens", "pool_w": "pool_weights", "pooled": "spatial_pooled_tokens", "dec": "decoded",
         "w": "day_weights", "blend": "blend", "mass": "mass", "ent": "entropy", "wm": "wm_channels", "wm_pool": "wm_key_pool",
         "wm_nvalid": "wm_n_valid_keys"}
LP.STEMS["wmattn"] = ("analyses.io7.wmattn.wm_eval", "io7_leak_", "_io6.npz", Path.home() / "mind_the_chl_gap" / "data" / "io6",
                      HERE / "results", NAMES)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    run = argv[argv.index("--init-from") + 1]
    LP.main(argv + ["--stem", "wmattn"])
    r = json.loads((HERE / "results" / f"leak_{run}.json").read_text())
    pc = r["cases"]["obs_t+2"]
    missing = [k for k in ("max_abs_change_wm_channels", "max_abs_change_wm_key_pool") if k not in r["cases"]["hidden"]]
    ok = r["no_leak"] and pc["max_abs_change_pred_hidden"] > 0 and pc.get("max_abs_change_wm_channels", 0) > 0 and not missing
    print(f"leak_wmattn {run}: no_leak {r['no_leak']}, positive control pred {pc['max_abs_change_pred_hidden']:.4g}, "
          f"wm channels {pc.get('max_abs_change_wm_channels', float('nan')):.4g}, missing records {missing} -> {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
