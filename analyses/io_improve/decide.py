"""Pre-registered decision rules of the io2_* track (analyses/io_improve/PLAN.md sections 3-4), applied mechanically.

Subcommands:
  band          print the seed band (S30, S60, SR) and the test spreads as JSON
  item7-flags   print the extra flags of the item-7 run (--memory --attn-query-memory if item 5 helped, else nothing)
  tattn15       exit 0 if the +-15-day attention run is allowed (smoke peak reserved <= 22 GB and s/batch <= 2x the baseline smoke)
  stageB        write the stage-B job files (seed restarts, ensemble scoring, full-length runs of helping options, combination,
                the item-4 decision job) into the queue's jobs/ folder; log the decisions to results/decisions.json
  item4         write the item-4 jobs (train+val final fit) and the report / END jobs
Every decision is appended to results/decisions.json with its inputs.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from chl_cnn.runs import find_run

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LOGS = Path.home() / "mind_the_chl_gap" / "logs" / "io_improve"
JOBS = LOGS / "jobs"
BASELINE = "t3d_dil_c4_60ep_gdl100"
SEEDS = {0: BASELINE, 1: "io2_seed1", 2: "io2_seed2"}
RESTARTS = {"io2_r_plat": "--continue-weights-only --ckpt-from 0 --lr 1e-3 --sched plateau --plateau-patience 5 --patience 15 --epochs 60",
            "io2_r_cos": "--continue-weights-only --ckpt-from 0 --lr 3e-4 --sched cosine --epochs 20"}
OPTIONS = {   # short run -> (full-length name, flags beyond BASE)
    "io2_w96_s30": ("io2_w96", "--width 96"),
    "io2_mem_s30": ("io2_mem", "--memory"),
    "io2_anom_s30": ("io2_anom", "--memory --residual-base anom"),
    "io2_tattn8_s30": ("io2_tattn8", "--t3d-stem attn --context 8 --feat-context 8 --batch 4 --eval-batch 8"),   # batch 4: PLAN section 6, 2026-10-22
    "io2_tattn15_s30": ("io2_tattn15", "--t3d-stem attn --context 15 --feat-context 15 --batch 4 --eval-batch 8"),
}
PROVISIONAL = 0.001
HEADER = "source ~/mind_the_chl_gap/logs/io_improve/common.sh\n"


def done(run):
    try:
        return (find_run(run) / "metrics.json").exists()
    except FileNotFoundError:
        return False


def hist(run):
    return json.loads((find_run(run) / "history.json").read_text())


def cfg(run):
    return json.loads((find_run(run) / "config.json").read_text())


def val_at(run, N, ckpt_from=10):
    """Lowest val over epochs ckpt_from+1..N (the kept model), and the raw val at epoch N."""
    v = hist(run)["val"]
    if len(v) < N:
        return None, None
    return float(min(v[ckpt_from:N])), float(v[N - 1])


def restart_best(run):
    v = hist(run)["val"]
    return float(min(v)), int(np.argmin(v)) + 1


def test_mae(run):
    return json.loads((find_run(run) / "metrics.json").read_text())["scores"]["test"]["cnn_hidden"]["mae"]


def item1_winner():
    if not all(done(r) for r in RESTARTS):
        return None
    best = {r: restart_best(r)[0] for r in RESTARTS}
    return min(best, key=best.get)


def band():
    out = dict(provisional=False)
    if all(done(s) for s in SEEDS.values()):
        for N in (30, 60):
            v = [val_at(s, N)[0] for s in SEEDS.values()]
            out[f"S{N}"] = max(v) - min(v); out[f"val{N}"] = v
        t = [test_mae(s) for s in SEEDS.values()]
        out["test_spread_60"] = max(t) - min(t); out["test60"] = t
    else:
        out.update(provisional=True, S30=PROVISIONAL, S60=PROVISIONAL)
    w = item1_winner()
    rs = [w, "io2_seed1_r", "io2_seed2_r"] if w else []
    if rs and all(done(r) for r in rs):
        v = [restart_best(r)[0] for r in rs]; t = [test_mae(r) for r in rs]
        out.update(SR=max(v) - min(v), valR=v, test_spread_R=max(t) - min(t), testR=t, restarted=rs)
    return out


def helps(run, N=30):
    b = band()
    v, raw = val_at(run, N)
    vb, rawb = val_at(BASELINE, N)
    thr = b[f"S{N}"]
    return dict(run=run, N=N, val=v, val_raw=raw, base=vb, base_raw=rawb, delta=v - vb, threshold=thr, provisional=b["provisional"],
                helps=bool(v - vb < -thr))


def log(entry):
    RES.mkdir(exist_ok=True)
    f = RES / "decisions.json"
    d = json.loads(f.read_text()) if f.exists() else []
    d.append(dict(entry, time=time.strftime("%Y-%m-%d %H:%M:%S")))
    f.write_text(json.dumps(d, indent=1))


def write_job(name, body):
    p = JOBS / f"{name}.sh"
    if p.exists():
        print(f"job {name} exists; not rewritten"); return
    p.write_text(HEADER + body)
    print(f"wrote job {name}:\n{body}")


def smoke_stats(run):
    """(peak reserved GB, seconds per training batch) from a smoke run's config / history (max-batches runs)."""
    c = cfg(run); h = hist(run)
    return c["peak_gpu_reserved_gb"], c["max_batches"], h["epoch_s"]


def cmd_item7_flags():
    if not done("io2_mem_s30"):
        raise SystemExit("io2_mem_s30 not finished: item 7 cannot be configured yet")
    h = helps("io2_mem_s30")
    flags = "--memory --attn-query-memory" if h["helps"] else ""
    log(dict(decision="item7 inputs", rule="PLAN 4.4: memory features only if item 5 helped", item5=h, flags=flags))
    print(flags)


def cmd_tattn15():
    try:
        r15 = cfg("io2_smoke_tattn15")
    except FileNotFoundError:
        log(dict(decision="tattn15", allowed=False, reason="smoke missing")); raise SystemExit(1)
    t15 = json.loads((find_run("io2_smoke_tattn15") / "history.json").read_text())["epoch_s"][-1]
    tb = json.loads((find_run("io2_smoke_base") / "history.json").read_text())["epoch_s"][-1]
    ok = r15["peak_gpu_reserved_gb"] <= 22.0 and t15 <= 2.0 * tb
    log(dict(decision="tattn15", rule="PLAN 4.4: run +-15 only if <= 22 GB reserved at batch 8 and <= 2x baseline time",
             peak_reserved_gb=r15["peak_gpu_reserved_gb"], smoke_epoch_s=t15, base_smoke_epoch_s=tb, allowed=bool(ok)))
    print("allowed" if ok else "not allowed")
    raise SystemExit(0 if ok else 1)


def cmd_stageB():
    w = item1_winner()
    assert w, "item 1 not finished"
    b = band()
    wb, we = restart_best(w)
    log(dict(decision="item1 winner", rule="PLAN 4.1: lower best val of io2_r_plat / io2_r_cos",
             vals={r: restart_best(r) for r in RESTARTS}, winner=w, band=b,
             winner_delta=wb - 0.130165, winner_helps_vs_S60=bool(wb - 0.130165 < -b["S60"])))
    R = RESTARTS[w]
    for s in (1, 2):
        write_job(f"110_item2_seed{s}_r", f"train io2_seed{s}_r --seed {s} --continue-from io2_seed{s} {R}\nscore io2_seed{s}_r\n")
    write_job("112_item2_ensemble", f"score --ensemble ens3={w},io2_seed1_r,io2_seed2_r ens3pre=io2_base_eval,io2_seed1,io2_seed2\n")
    # tier-2 options at 30 epochs
    res = {r: helps(r) for r in OPTIONS if done(r)}
    log(dict(decision="tier-2 at epoch 30", rule="PLAN 3: helps if delta < -S30", results=res))
    helping = [r for r, h in res.items() if h["helps"]]
    if "io2_mem_s30" in helping and "io2_anom_s30" in helping:          # 6 contains 5: keep the lower-val one
        drop = "io2_mem_s30" if res["io2_mem_s30"]["val"] > res["io2_anom_s30"]["val"] else "io2_anom_s30"
        helping.remove(drop)
    if "io2_tattn8_s30" in helping and "io2_tattn15_s30" in helping:
        drop = "io2_tattn8_s30" if res["io2_tattn8_s30"]["val"] > res["io2_tattn15_s30"]["val"] else "io2_tattn15_s30"
        helping.remove(drop)
    item7_flags = ""
    for e in json.loads((RES / "decisions.json").read_text()):
        if e["decision"] == "item7 inputs":
            item7_flags = e["flags"]
    full = []
    for i, r in enumerate(helping):
        name, flags = OPTIONS[r]
        if "tattn" in r:
            flags = f"{flags} {item7_flags}".strip()
        full.append((name, flags))
        write_job(f"12{i}a_full_{name}", f"train {name} {flags}\nscore {name}\n")
        write_job(f"12{i}b_full_{name}_r", f"train {name}_r {flags} --continue-from {name} {R}\nscore {name}_r\n")
    combo = None
    if len(full) >= 2:
        toks = []                                                      # union of flags, each once, in order
        for _, f in full:
            for t in re.findall(r"--\S+(?: (?!--)\S+)*", f):
                if t not in toks:
                    toks.append(t)
        # --t3d-stem attn with --context K overrides BASE's --context 4 (later flags win)
        combo = ("io2_combo", " ".join(toks))
        write_job("130a_full_io2_combo", f"train io2_combo {combo[1]}\nscore io2_combo\n")
        write_job("130b_full_io2_combo_r", f"train io2_combo_r {combo[1]} --continue-from io2_combo {R}\nscore io2_combo_r\n")
    log(dict(decision="stage B full-length runs", rule="PLAN 4.2-4.3", helping=helping, full=full, combo=combo))
    write_job("190_item4_decide", "python -W ignore -m analyses.io_improve.decide item4 2>&1 | tee -a $L/decide.log\n")


def cmd_item4():
    w = item1_winner(); R = RESTARTS[w]
    st = [e for e in json.loads((RES / "decisions.json").read_text()) if e["decision"] == "stage B full-length runs"][-1]
    cands = {}

    def sc(run):
        f = RES / f"{run}.json"
        return json.loads(f.read_text())["val"]["mae"] if f.exists() else None
    cands["baseline recipe"] = dict(val=sc(w), runs=[(0, BASELINE, w)], flags="")
    for name, flags in st["full"] + ([st["combo"]] if st["combo"] else []):
        if done(f"{name}_r"):
            cands[name] = dict(val=sc(f"{name}_r"), runs=[(0, name, f"{name}_r")], flags=flags)
    cands["ensemble (baseline recipe, 3 seeds)"] = dict(val=sc("ens3_mean"), runs=[(0, BASELINE, w), (1, "io2_seed1", "io2_seed1_r"),
                                                                                  (2, "io2_seed2", "io2_seed2_r")], flags="")
    ok = {k: v for k, v in cands.items() if v["val"] is not None}
    best = min(ok, key=lambda k: ok[k]["val"])
    singles = {k: v["val"] for k, v in ok.items() if not k.startswith("ensemble")}
    log(dict(decision="item4 recipe", rule="PLAN 4.5: lowest val after restart; the ensemble wins only if below every single",
             candidates={k: v["val"] for k, v in cands.items()}, chosen=best))
    c = ok[best]
    jobs = []
    for seed, src60, srcr in c["runs"]:
        E1 = cfg(src60)["best_epoch"]; E2 = restart_best(srcr)[1]
        rr = R.replace("--continue-weights-only --ckpt-from 0", "")
        if "plateau" in rr:                                     # replay the val-driven lr sequence; no val needed
            rr = re.sub(r"--sched plateau --plateau-patience \d+ --patience \d+", f"--sched none --lr-replay {srcr}", rr)
        n1, n2 = f"io2_final_s{seed}", f"io2_final_s{seed}_r"
        log(dict(decision=f"item4 epochs seed {seed}", source=src60, E1=E1, restart_source=srcr, E2=E2))
        write_job(f"20{seed}a_item4_{n1}", f"train {n1} {c['flags']} --seed {seed} --train-on-val --no-select --stop-epoch {E1}\nscore {n1}\n")
        write_job(f"20{seed}b_item4_{n2}", f"train {n2} {c['flags']} --seed {seed} --train-on-val --no-select --continue-from {n1} "
                                           f"--continue-weights-only --ckpt-from 0 {rr} --stop-epoch {E2}\nscore {n2}\n")
        jobs.append(n2)
    if len(jobs) > 1:
        write_job("209_item4_ensemble", f"score --ensemble ens3final={','.join(jobs)}\n")
    write_job("990_report", "python -W ignore -m analyses.io_improve.report 2>&1 | tee -a $L/report.log\n"
                            "python -m chl_cnn.runs_index 2>&1 | tail -2\ntouch $L/jobs/END\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "band":
        print(json.dumps(band(), indent=1))
    elif cmd == "item7-flags":
        cmd_item7_flags()
    elif cmd == "tattn15":
        cmd_tattn15()
    elif cmd == "stageB":
        cmd_stageB()
    elif cmd == "item4":
        cmd_item4()
    else:
        raise SystemExit(__doc__)
