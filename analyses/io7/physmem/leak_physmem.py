"""Leak test of a --physmem model (io7_physmem), as analyses/io3_attn/leak_ptok.py --stem tokdec_spatial, with cases inside and beyond the
blend's +-8 days. Eval-only runs (each under the GPU lock) with the input copy of ln CHL changed (train_pinn.py --leak-*), recorded by
analyses/io6_kriging/io6_eval.py (patch embeddings, tokens, pooling weights, spatial pooled tokens, decoded channels, blend w / blend / mass /
entropy):
  base         no change
  hidden       day t: +1 on day t's fixed hidden px                                   -> expect 0 change everywhere
  masked_t+2   day t+2: +1 on the observed px the donor (persist) rule hides from t      -> expect 0 change
  masked_t+12  day t+12 (token window only, beyond the blend), same rule                  -> expect 0 change
  masked_t-12  day t-12, same rule                                                        -> expect 0 change
  obs_t+2      day t+2: +1 on every observed px                                         -> positive control: prediction changes
  obs_t+12     day t+12: +1 on every observed px                                        -> positive control: tokens and prediction change
t = the val day with the most hidden px among every 20th val day (as leak_ptok.py), --val-stride 20.
Offsets +-12 are in the window with and without --physmem-stride2.

Usage (project root): python -m analyses.io7.physmem.leak_physmem --init-from RUN --base-args "<trainer flags>"
Writes analyses/io7/physmem/results/leak_<RUN>.json; exits 1 on a leak or a dead positive control.
Store: --store / --crop / --val-years / --test-years are read from --base-args (analyses/blk7/storectx.py; blk7 2026-10-26), so the
Black Sea 4 km store (--store blk4 --crop) works; the IO default is unchanged.
--far K (blk8, 2026-10-26; default 12 = unchanged): the far offsets are +-K (cases masked_t+K, masked_t-K, obs_t+K); use K <= the window
(--far 6 for the +-8 window of blk8_spatial_bsmem).
PHYSMEM_LEAK_CMD (env, default 'flock <gpu.lock> python'): the command prefix of the eval runs (CPU dry-run testing only).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from analyses.blk7 import storectx
from chl_cnn.runs import find_run

HERE = Path(__file__).resolve().parent
LOCK = Path.home() / "mind_the_chl_gap" / "logs" / "gpu.lock"
DATA = Path.home() / "mind_the_chl_gap" / "data" / "io6"
STRIDE = 20
NAMES = {"emb": "patch_embeddings", "tok": "tokens", "pool_w": "pool_weights", "pooled": "spatial_pooled_tokens", "dec": "decoded",
         "w": "day_weights", "blend": "blend", "mass": "mass", "ent": "entropy"}
NEG = ["hidden", "masked_t+2", "masked_t+12", "masked_t-12"]
POS = ["obs_t+2", "obs_t+12"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-from", required=True)
    p.add_argument("--base-args", required=True)
    p.add_argument("--far", type=int, default=12, help="far offset K (cases masked_t+-K, obs_t+K)")
    a = p.parse_args(argv)
    far = a.far
    neg_cases = ["hidden", "masked_t+2", f"masked_t+{far}", f"masked_t-{far}"]; pos_cases = ["obs_t+2", f"obs_t+{far}"]
    sc = storectx.from_args(a.base_args)
    times, va, hid_va = sc["times"], sc["va"], sc["hid_va"]
    del sc
    cand = np.arange(0, len(va), STRIDE)
    k = cand[np.argmax(hid_va[cand].reshape(len(cand), -1).sum(1))]
    t = int(va[k]); pos = int(k // STRIDE); n_val = len(va[::STRIDE])
    print(f"leak day t = {str(times[t])[:10]} (store index {t}, scored position {pos}), {int(hid_va[k].sum())} hidden px", flush=True)
    cases = {"base": [], "hidden": ["--leak-what", "hidden"]}
    for o in (2, far, -far):
        cases[f"masked_t{o:+d}"] = ["--leak-what", "masked", "--leak-offset", str(o)]
    for o in (2, far):
        cases[f"obs_t{o:+d}"] = ["--leak-what", "observed", "--leak-offset", str(o)]
    out = {}
    for name, extra in cases.items():
        run = f"io7_leak_{a.init_from}_{name.replace('+', 'p').replace('-', 'm')}"
        lk = ["--leak-day", str(t), "--leak-delta", "0" if name == "base" else "1.0"] + extra
        cmd = os.environ.get("PHYSMEM_LEAK_CMD", f"flock {LOCK} python").split() + ["-W", "ignore", "-m", "analyses.io6_kriging.io6_eval", "--run", run, "--positions", str(pos),
               "--n-val", str(n_val), "--"] + shlex.split(a.base_args) + \
              ["--eval-only", "--init-from", a.init_from, "--val-stride", str(STRIDE), "--no-return", "--fig-every", "0"] + lk
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print("\n".join(l for l in r.stdout.splitlines() if "LEAK" in l or "VAL" in l or "recorded" in l or "physmem:" in l), flush=True)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f"{run} failed")
        z = np.load(DATA / f"{run}_io6.npz")
        pr = np.load(find_run(run) / "preds.npz"); i = int(np.flatnonzero(pr["idx"] == t)[0])
        out[name] = dict({kk: z[f"{kk}__{pos}"] for kk in NAMES if f"{kk}__{pos}" in z}, pred=pr["pred"][i].astype("float32"), hid=pr["hidden"][i])
    hid = out["base"]["hid"]
    assert np.array_equal(hid, hid_va[k])
    assert "tok" in out["base"] and "w" in out["base"], "recorder keys missing"
    res = dict(init_from=a.init_from, day=str(times[t])[:10], store_index=t, n_hidden=int(hid.sum()),
               n_token_days=int(out["base"]["tok"].shape[1]), n_blend_slots=int(out["base"]["w"].shape[0]), cases={})
    for name in cases:
        if name == "base":
            continue
        dp = np.abs(out[name]["pred"] - out["base"]["pred"])[hid]
        res["cases"][name] = {f"max_abs_change_{NAMES[kk]}": float(np.abs(out[name][kk] - out["base"][kk]).max()) for kk in NAMES if kk in out["base"]}
        res["cases"][name].update(max_abs_change_pred_hidden=float(np.nanmax(dp)), mean_abs_change_pred_hidden=float(np.nanmean(dp)))
        print(name, json.dumps(res["cases"][name]), flush=True)
    zero = lambda c: all(v == 0 for kk, v in c.items() if kk.startswith("max_abs_change_"))
    res["no_leak"] = bool(all(zero(res["cases"][c]) for c in neg_cases))
    res["controls_ok"] = bool(all(res["cases"][c]["max_abs_change_pred_hidden"] > 0 for c in pos_cases) and res["cases"][f"obs_t+{far}"]["max_abs_change_tokens"] > 0)
    print(("NO LEAK" if res["no_leak"] else "LEAK DETECTED") + ("; positive controls change" if res["controls_ok"] else "; POSITIVE CONTROL UNCHANGED"), flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"leak_{a.init_from}.json").write_text(json.dumps(res, indent=1))
    sys.exit(0 if res["no_leak"] and res["controls_ok"] else 1)


if __name__ == "__main__":
    main()
