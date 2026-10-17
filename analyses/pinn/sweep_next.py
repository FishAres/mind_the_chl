"""Pick the next sweep stage from finished runs and write its queue lines for scripts/train_queue.sh.

  python analyses/pinn/sweep_next.py            # after sweep 1: writes models/sweep2.txt (combinations of the winners)
  python analyses/pinn/sweep_next.py            # after sweep 2: writes models/sweep3.txt (best config, 60 epochs)
Rules: the noise floor is |s1_base - s1_seed1| (at least 0.001). A sweep-1 change "wins" if it beats s1_base by more
than the floor. Sweep 2 = every pair of winners plus all winners together (when 3+). Sweep 3 = the best of all s1/s2
configs by val MAE, trained for 60 epochs with best-val checkpointing, plus a second seed of it.
Lines are `name args...`; scripts/train_queue.sh appends them to the sweep base recipe S1 (minus --epochs for sweep 3).
"""
import itertools, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chl_cnn.runs import list_runs, MODELS

CHANGES = {"gdl3": "--grad-weight 3", "gdl10": "--grad-weight 10", "w96": "--width 96", "ctx3": "--context 3",
           "lr2e3": "--lr 2e-3", "drop01": "--dropout 0.1", "huber": "--loss huber"}
EXCLUSIVE = [{"gdl3", "gdl10"}]                      # never combine two values of the same flag


def mae(p):
    return json.load(open(p / "metrics.json"))["scores"]["val"]["cnn_hidden"]["mae"]


def done(glob):
    return {p.name: mae(p) for p in list_runs(None, glob) if (p / "metrics.json").exists()}


s1 = done("s1_*")
need = {"s1_base", "s1_seed1"} | {f"s1_{k}" for k in CHANGES}
missing = need - set(s1)
if missing:
    sys.exit(f"sweep 1 not finished, missing: {sorted(missing)}")
floor = max(abs(s1["s1_base"] - s1["s1_seed1"]), 0.001)
gain = {k: s1["s1_base"] - s1[f"s1_{k}"] for k in CHANGES}
winners = [k for k, g in sorted(gain.items(), key=lambda kv: -kv[1]) if g > floor]
print(f"s1_base {s1['s1_base']:.4f}  seed1 {s1['s1_seed1']:.4f}  noise floor {floor:.4f}")
for k, g in sorted(gain.items(), key=lambda kv: -kv[1]):
    print(f"  {k:7s} gain {g:+.4f} {'WIN' if k in winners else ''}")

s2 = done("s2_*")
if not s2:
    combos = [c for c in itertools.combinations(winners, 2) if not any(set(c) >= e for e in EXCLUSIVE)]
    if len(winners) >= 3 and not any(set(winners) >= e for e in EXCLUSIVE):
        combos.append(tuple(winners))
    if len(winners) >= 3:
        for e in EXCLUSIVE:
            if set(winners) >= e:
                best = min(e, key=lambda k: s1[f"s1_{k}"]); combos.append(tuple(k for k in winners if k not in e - {best}))
    lines = [f"s2_{'_'.join(c)} {' '.join(CHANGES[k] for k in c)}" for c in dict.fromkeys(combos)]
    # the gradient-loss weight is a dose: if the larger dose won, extrapolate it (alone and with the other winners)
    if "gdl10" in winners and gain["gdl10"] > gain["gdl3"]:
        others = [k for k in winners if k not in ("gdl3", "gdl10")]
        for w in (30, 100):
            lines.append(f"s2_gdl{w} --grad-weight {w}")
            if others:
                lines.append(f"s2_gdl{w}_{'_'.join(others)} --grad-weight {w} {' '.join(CHANGES[k] for k in others)}")
    if not lines:
        lines = [f"# no combinations: winners = {winners}"]
    out = MODELS / "sweep2.txt"; out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len([l for l in lines if not l.startswith('#')])} runs):"); print(open(out).read())
else:
    allr = {**{k: v for k, v in s1.items()}, **s2}
    best = min(allr, key=allr.get)
    args = " ".join(CHANGES[k] if k in CHANGES else f"--grad-weight {k[3:]}" for k in best.split("_")[1:] if k in CHANGES or k.startswith("gdl"))
    lines = [f"s3_best {args}", f"s3_best_seed1 {args} --seed 1"]
    out = MODELS / "sweep3.txt"; out.write_text("\n".join(lines) + "\n")
    print(f"best so far {best} {allr[best]:.4f}; wrote {out}:"); print(open(out).read())
