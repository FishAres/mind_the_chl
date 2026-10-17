"""One index of every training run, live (models/<category>/<run>) and archived (models/history/<category>/<run>):
what was trained, with which non-default settings, and the val / test hidden-pixel MAE. Writes models/RUNS.md, the
only file under models/ tracked by git. The narrative of what each round taught us is analyses/FINDINGS.md.
Usage: python -m chl_cnn.runs_index        (re-run after new experiments; reads config/metrics/history.json only)
"""
import json
import time
from pathlib import Path

from .runs import MODELS, CATEGORIES
from .archive_runs import HIST, val_test, settings

STORE = {"io": "Indian Ocean", "blk": "Black Sea 2 km", "blk4": "Black Sea 4 km"}


def _row(cat, run, where):
    c = json.load(open(run / "config.json"))
    mp, hp = run / "metrics.json", run / "history.json"
    m = json.load(open(mp)) if mp.exists() else None
    h = json.load(open(hp)) if hp.exists() else {}
    if cat == "ude":                                         # daily UDE: val MAE averaged over forecast leads
        v, t = (m or {}).get("best_val_mae_mean_leads"), None
    else:
        v, t, _, _ = val_test(cat, m, c) if m else (None, None, None, None)
    if v is None and h.get("val"):
        v = min(x for x in h["val"] if x is not None)
    ep = len(h.get("val", [])) or None
    date = time.strftime("%Y-%m-%d", time.localtime((mp if m else run / "config.json").stat().st_mtime))
    status = "done" if m else "unfinished"
    if where == "live" and (run / "model.pt").exists():
        status += ", weights"
    region = STORE.get(c.get("store", "io"), c.get("store"))
    return dict(cat=cat, run=run.name, date=date, region=region, settings=settings(cat, c) if cat != "ude" else "", ep=ep, val=v, test=t, status=status)


def collect():
    rows = {}
    for where, root in (("live", None), ("history", HIST)):
        for cat, d in CATEGORIES.items():
            d = d if root is None else root / cat
            if not d.exists():
                continue
            for run in sorted(p for p in d.iterdir() if (p / "config.json").exists()):
                key = (cat, run.name)
                if where == "history" and key in rows:
                    continue
                try:
                    rows[key] = _row(cat, run, where)
                except Exception as e:                            # never let one odd run break the index
                    rows[key] = dict(cat=cat, run=run.name, date="", region="", settings=f"(unreadable: {type(e).__name__})", ep=None, val=None, test=None, status=where)
    return rows


def main():
    rows = collect()
    fmt = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else ""
    out = ["# Training runs: index", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m chl_cnn.runs_index` ({len(rows)} runs). "
           "What each round found, and why it was run: [analyses/FINDINGS.md](../analyses/FINDINGS.md).", "",
           "- **val / test MAE**: hidden-pixel MAE in ln CHL (fake clouds from a donor day). Best val epoch; test at that epoch. "
           "Only comparable within one region and one `val_stride` / `donor_exclude` setting.",
           "- **settings**: non-default flags only (see the trainer docstrings). `stop_epoch=N`: the schedule spans `epochs` but training stopped at N.",
           "- **status**: `weights` = model.pt still on disk (run folder under models/<category>/); otherwise only config/history/metrics are kept, in models/history/.", ""]
    for cat in CATEGORIES:
        rs = sorted((r for r in rows.values() if r["cat"] == cat), key=lambda r: (r["region"], r["date"], r["run"]))
        if not rs:
            continue
        out += [f"## {cat} ({len(rs)} runs)", "", "| run | region | settings | val MAE | test MAE | epochs | finished | status |", "|---|---|---|---|---|---|---|---|"]
        out += [f"| {r['run']} | {r['region']} | {r['settings']} | {fmt(r['val'])} | {fmt(r['test'])} | {r['ep'] or ''} | {r['date']} | {r['status']} |" for r in rs]
        out.append("")
    (MODELS / "RUNS.md").write_text("\n".join(out))
    print(f"{len(rows)} runs -> {MODELS / 'RUNS.md'}")


if __name__ == "__main__":
    main()
