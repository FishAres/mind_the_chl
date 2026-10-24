"""Leak test of the --memory channels (io2 track items 5 / 6; plan: analyses/io_improve/PLAN.md).

Runs the trainer in --eval-only mode on a trained --memory model four times (each under the GPU lock), with the input copy of
ln CHL changed before anything is computed (train_pinn.py --leak-day / --leak-offset / --leak-what / --leak-delta):
  base       no change
  hidden     day t: +1.0 on day t's fixed hidden pixels                         -> expect 0 change (no leak)
  obs_t+6    day t+6 (outside the +-4 window): +1.0 on every observed pixel       -> positive control: memory changes
  obs_t+2    day t+2 (inside the window): +1.0 on every observed pixel            -> positive control: 8-day median and inputs change
and compares, for target day t, the 8 memory channels (+ the 30-day anomaly), and the prediction on t's hidden pixels.
t = the val day with the most hidden pixels among every 20th val day (the runs score every 20th val / test day, --val-stride 20).

Usage (project root): python -m analyses.io_improve.leak_test --init-from io2_mem_s30 [--extra "--residual-base anom"]
Writes analyses/io_improve/results/leak_<init-from>.json.
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
STRIDE = 20


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-from", required=True)
    p.add_argument("--base-args", required=True, help="the trainer flags of the model (BASE + its options)")
    p.add_argument("--no-lock", action="store_true", help="debug only: run without the GPU lock")
    a = p.parse_args(argv)
    d = S.load_store(); times = d["time"]
    obs = np.isfinite(D.log_chl(np.asarray(d["chl_l3"]))) & d["ocean"]
    tr, va, te = D.split_by_years(times, D.VAL_YEARS, D.TEST_YEARS)
    hid_va, _ = C.fixed_hidden(obs, va, seed=0, times=times)
    cand = np.arange(0, len(va), STRIDE)
    k = cand[np.argmax(hid_va[cand].reshape(len(cand), -1).sum(1))]
    t = int(va[k])
    print(f"leak day t = {str(times[t])[:10]} (store index {t}), {int(hid_va[k].sum())} hidden px", flush=True)
    cases = {"base": [], "hidden": ["--leak-what", "hidden"], "obs_t+6": ["--leak-what", "observed", "--leak-offset", "6"],
             "obs_t+2": ["--leak-what", "observed", "--leak-offset", "2"]}
    res = dict(init_from=a.init_from, day=str(times[t])[:10], store_index=t, n_hidden=int(hid_va[k].sum()), cases={})
    out = {}
    for name, extra in cases.items():
        run = f"io2_leak_{a.init_from}_{name.replace('+', 'p')}"
        lk = ["--leak-day", str(t)] + (["--leak-delta", "0"] if name == "base" else ["--leak-delta", "1.0"]) + extra
        cmd = ([] if a.no_lock else ["flock", str(LOCK)]) + ["python", "-W", "ignore", "-m", "chl_cnn.train_pinn", "--run", run] + shlex.split(a.base_args) + \
              ["--eval-only", "--init-from", a.init_from, "--val-stride", str(STRIDE), "--no-return", "--fig-every", "0"] + lk
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print("\n".join(l for l in r.stdout.splitlines() if "LEAK" in l or "VAL" in l), flush=True)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f"{run} failed")
        rd = find_run(run)
        z = np.load(rd / "preds.npz"); i = int(np.flatnonzero(z["idx"] == t)[0])
        m = np.load(rd / "leak_memory.npz")
        out[name] = dict(pred=z["pred"][i].astype("float32"), ch=m["channels"], anom=m["anom"], hid=m["hidden"])
    hid = out["base"]["hid"]
    assert np.array_equal(hid, hid_va[k])
    for name in cases:
        if name == "base":
            continue
        dch = np.abs(out[name]["ch"] - out["base"]["ch"]).reshape(8, -1).max(1)
        dp = np.abs(out[name]["pred"] - out["base"]["pred"])[hid]
        res["cases"][name] = dict(max_abs_change_memory_channels=dch.tolist(), max_abs_change_anom=float(np.abs(out[name]["anom"] - out["base"]["anom"]).max()),
                                  max_abs_change_pred_hidden=float(np.nanmax(dp)), mean_abs_change_pred_hidden=float(np.nanmean(dp)), n_hidden=int(hid.sum()))
        print(f"{name:8s} memory channels max|change| {np.round(dch, 4).tolist()}  anom {res['cases'][name]['max_abs_change_anom']:.4f}  "
              f"pred on t's hidden px max|change| {np.nanmax(dp):.5f} mean {np.nanmean(dp):.5f}", flush=True)
    ok = res["cases"]["hidden"]["max_abs_change_pred_hidden"] == 0 and max(res["cases"]["hidden"]["max_abs_change_memory_channels"]) == 0 \
        and res["cases"]["hidden"]["max_abs_change_anom"] == 0
    res["no_leak"] = bool(ok)
    print("NO LEAK" if ok else "LEAK DETECTED", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"leak_{a.init_from}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
