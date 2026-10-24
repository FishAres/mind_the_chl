"""Leak test of the patch-token stem (io3 track, Part B; follows analyses/io_improve/leak_test.py).

Eval-only runs (each under the GPU lock) of a trained --t3d-stem patch model with the input copy of ln CHL changed
(train_pinn.py --leak-day / --leak-offset / --leak-what / --leak-delta), recording the tokens and day weights of target day t
(analyses/io3_attn/ptok_eval.py):
  base      no change
  hidden    day t: +1.0 on day t's fixed hidden pixels            -> expect 0 change in tokens and in the prediction on hidden px
  masked_t+2  day t+2: +1.0 on the observed px the donor (persist) mask hides from t -> expect 0 change (context values, incl. the
            --ptok-value lavg local averages, use only px visible after donor masking)
  obs_t+2   day t+2: +1.0 on every observed pixel                  -> positive control: tokens and prediction change
t = the val day with the most hidden pixels among every 20th val day (as leak_test.py), --val-stride 20.

Usage (project root): python -m analyses.io3_attn.leak_ptok --init-from RUN --base-args "<trainer flags>"
Writes analyses/io3_attn/results/leak_<RUN>.json.

--stem tokdec (io4 track, 2026-10-23): the same four cases on a --t3d-stem tokdec model, recorded by analyses/io4_tokdec/tokdec_eval.py
(patch embeddings, output tokens, pooling weights, decoded context channels, and with --tokdec-blend the per-pixel day weights and
blend / mass / entropy); runs io4_leak_<RUN>_<case>; writes analyses/io4_tokdec/results/leak_<RUN>.json. The default (--stem patch)
is unchanged.

--stem krig / tokdec_spatial (io6 track, 2026-10-24): the same four cases on a --tokdec-krig / --tokdec-spatial model, recorded by
analyses/io6_kriging/io6_eval.py: patch embeddings, output tokens, pooling weights, decoded channels, and the kriging output channels with the
visible-key counts (krig) or the spatially attended pooled tokens with the blend weights / blend / mass / entropy (tokdec_spatial); runs
io6_leak_<RUN>_<case>; writes analyses/io6_kriging/results/leak_<RUN>.json.
"""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

from chl_cnn import data as D, store as S, clouds as C
from chl_cnn.runs import find_run

HERE = Path(__file__).resolve().parent
LOCK = Path.home() / "mind_the_chl_gap" / "logs" / "gpu.lock"
DATA = Path.home() / "mind_the_chl_gap" / "data" / "io3"
STRIDE = 20
# stem -> (eval module, run prefix, recorder file suffix, data dir, results dir, {recorded key: name in the json})
STEMS = dict(
    patch=("analyses.io3_attn.ptok_eval", "io3_leak_", "_ptok.npz", DATA, HERE / "results",
           {"tok": "tokens", "stats": "token_stats", "w": "day_weights"}),
    tokdec=("analyses.io4_tokdec.tokdec_eval", "io4_leak_", "_tokdec.npz", Path.home() / "mind_the_chl_gap" / "data" / "io4",
            HERE.parent / "io4_tokdec" / "results",
            {"emb": "patch_embeddings", "tok": "tokens", "pool_w": "pool_weights", "dec": "decoded", "w": "day_weights", "blend": "blend",
             "mass": "mass", "ent": "entropy"}),
    krig=("analyses.io6_kriging.io6_eval", "io6_leak_", "_io6.npz", Path.home() / "mind_the_chl_gap" / "data" / "io6",
          HERE.parent / "io6_kriging" / "results",
          {"emb": "patch_embeddings", "tok": "tokens", "pool_w": "pool_weights", "dec": "decoded", "krig": "kriging_channels", "n_keys": "n_keys"}),
    tokdec_spatial=("analyses.io6_kriging.io6_eval", "io6_leak_", "_io6.npz", Path.home() / "mind_the_chl_gap" / "data" / "io6",
                    HERE.parent / "io6_kriging" / "results",
                    {"emb": "patch_embeddings", "tok": "tokens", "pool_w": "pool_weights", "pooled": "spatial_pooled_tokens", "dec": "decoded",
                     "w": "day_weights", "blend": "blend", "mass": "mass", "ent": "entropy"}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-from", required=True)
    p.add_argument("--base-args", required=True)
    p.add_argument("--stem", choices=list(STEMS), default="patch")
    a = p.parse_args(argv)
    module, prefix, suffix, data, res_dir, names = STEMS[a.stem]
    d = S.load_store(); times = d["time"]
    obs = np.isfinite(D.log_chl(np.asarray(d["chl_l3"]))) & d["ocean"]
    tr, va, te = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    hid_va, _ = C.fixed_hidden(obs, va, seed=0, times=times)
    cand = np.arange(0, len(va), STRIDE)
    k = cand[np.argmax(hid_va[cand].reshape(len(cand), -1).sum(1))]
    t = int(va[k]); pos = int(k // STRIDE); n_val = len(va[::STRIDE])
    print(f"leak day t = {str(times[t])[:10]} (store index {t}, scored position {pos}), {int(hid_va[k].sum())} hidden px", flush=True)
    cases = {"base": [], "hidden": ["--leak-what", "hidden"], "masked_t+2": ["--leak-what", "masked", "--leak-offset", "2"], "obs_t+2": ["--leak-what", "observed", "--leak-offset", "2"]}
    out = {}
    for name, extra in cases.items():
        run = f"{prefix}{a.init_from}_{name.replace('+', 'p')}"
        lk = ["--leak-day", str(t), "--leak-delta", "0" if name == "base" else "1.0"] + extra
        cmd = ["flock", str(LOCK), "python", "-W", "ignore", "-m", module, "--run", run, "--positions", str(pos),
               "--n-val", str(n_val), "--"] + shlex.split(a.base_args) + \
              ["--eval-only", "--init-from", a.init_from, "--val-stride", str(STRIDE), "--no-return", "--fig-every", "0"] + lk
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print("\n".join(l for l in r.stdout.splitlines() if "LEAK" in l or "VAL" in l or "recorded" in l), flush=True)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f"{run} failed")
        z = np.load(data / f"{run}{suffix}")
        pr = np.load(find_run(run) / "preds.npz"); i = int(np.flatnonzero(pr["idx"] == t)[0])
        out[name] = dict({k: z[f"{k}__{pos}"] for k in names if f"{k}__{pos}" in z}, pred=pr["pred"][i].astype("float32"), hid=pr["hidden"][i])
    hid = out["base"]["hid"]
    assert np.array_equal(hid, hid_va[k])
    res = dict(init_from=a.init_from, day=str(times[t])[:10], store_index=t, n_hidden=int(hid.sum()), n_tokens=int(out["base"]["tok"].shape[0] * out["base"]["tok"].shape[1]), cases={})
    for name in cases:
        if name == "base":
            continue
        dp = np.abs(out[name]["pred"] - out["base"]["pred"])[hid]
        res["cases"][name] = {f"max_abs_change_{names[k]}": float(np.abs(out[name][k] - out["base"][k]).max()) for k in names if k in out["base"]}
        res["cases"][name].update(max_abs_change_pred_hidden=float(np.nanmax(dp)), mean_abs_change_pred_hidden=float(np.nanmean(dp)))
        print(name, json.dumps(res["cases"][name]), flush=True)
    h = res["cases"]["hidden"]
    zero = lambda c: all(v == 0 for k, v in c.items() if k.startswith("max_abs_change_"))
    res["no_leak"] = bool(zero(h) and zero(res["cases"]["masked_t+2"]))
    print("NO LEAK" if res["no_leak"] else "LEAK DETECTED", flush=True)
    res_dir.mkdir(exist_ok=True)
    (res_dir / f"leak_{a.init_from}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
