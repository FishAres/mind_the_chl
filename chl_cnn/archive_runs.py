"""Archive training runs: copy the lightweight record of every run (config.json, history.json, metrics.json, figures/,
and for spatial runs the tiny model.pt) to models/history/<category>/<run>/, write models/history/RUNS.md (one table
per category: run, date, key settings, val/test MAE), then delete the live run folder unless it is in KEEP (reference
runs whose weights/predictions the analysis scripts use). Idempotent: re-run after new experiments.
Usage: python -m chl_cnn.archive_runs [--dry-run]
"""
import json
import shutil
import sys
import time
from pathlib import Path

from .runs import MODELS, CATEGORIES

HIST = MODELS / "history"
KEEP = {"cnn/perf_dilated_vis01", "pinn/cnn/pinn_v2_lam0", "spatial/mt_long_conv_ctx4", "spatial/mt_final_conv_ctx4",
        "spatial/tc_sst_ctx2", "spatial/viz_conv_ctx4"}
LIGHT = ["config.json", "history.json", "metrics.json"]
SPATIAL_DEFAULTS = dict(context=0, ctx_feat=False, extra_feats=False, lagrangian=False, alpha=0.0, source=False, train_stride=1, dkw="{}", seed=0, init_from=None, eval_only=False)


def val_test(cat, m, c):
    if cat == "spatial":
        return m.get("best_val_mae", m["val"]["mae"]), None, m["val"].get("grad_ratio"), m["val"].get("grad_corr")
    s = m.get("scores", m)
    v = s.get("val", {}).get("cnn_hidden", {}).get("mae"); t = s.get("test", {}).get("cnn_hidden", {}).get("mae")
    g = s.get("val", {}).get("physics", {})
    return c.get("best_val_mae", v), t, g.get("grad_ratio"), g.get("grad_corr")


def settings(cat, c):
    if cat == "spatial":
        parts = [c.get("dmodel", "?")]
        for k, d in SPATIAL_DEFAULTS.items():
            if c.get(k, d) != d:
                parts.append(f"{k}={c[k]}" if not isinstance(c[k], bool) else k)
        parts.append(f"ep={c.get('epochs')}")
        return " ".join(parts)
    parts = [c.get("model", "?"), f"feat={len(c.get('features', []))}", f"ctx={c.get('context')}"]
    for k in ["lam", "loss", "vis_weight", "grad_weight", "width", "spectral", "donor_window", "unet_depth", "epochs"]:
        if k in c and c[k] not in (None, 0, 0.0, False, "l1") and not (k == "unet_depth" and c.get("model") != "unet"):
            parts.append(f"{k}={c[k]}")
    for k in ["year_input", "no_chl_input", "no_chl_today", "cloud_input", "src_clim", "src_ekman", "diff_fill", "feat_today_only", "doy_input", "land_input"]:
        if c.get(k):
            parts.append(k if c[k] is True else f"{k}={c[k]}")
    if c.get("phys_weight", "none") != "none":
        parts.append(f"phys_weight={c['phys_weight']}")
    # flags added after 2026-10-10 (store, 3D stem, windows, schedule, warm starts)
    if c.get("store", "io") != "io":
        parts.insert(0, f"store={c['store']}")
    if c.get("model") == "temporal3d" and c.get("dilations"):
        parts.append(f"blocks={len(c['dilations'])}")
    if c.get("context_before") is not None and (c.get("context_before"), c.get("context_after")) != (c.get("context"), c.get("context")):
        parts.append(f"window=t-{c['context_before']}..t+{c['context_after']}")
    for k, d in [("phys", "none"), ("context_clouds", "persist"), ("donor_exclude", 0), ("batch", None), ("lr", 1e-3),
                 ("stop_epoch", 0), ("continue_from", None), ("init_from", None), ("fake_mask", "donor"), ("recipe", "none")]:
        if k in c and c[k] not in (d, None):
            parts.append(f"{k}={c[k]}")
    if c.get("recipe", "none") != "none" and c.get("sched", "none") != "none":     # recipe runs: constant lr unless a schedule was asked for
        parts.append(f"sched={c['sched']}")
    return " ".join(parts)


def main(dry=False):
    rows = {}
    for cat, d in CATEGORIES.items():
        for run in sorted(p for p in d.iterdir() if p.is_dir()) if d.exists() else []:
            cfg = run / "config.json"
            if not cfg.exists():
                continue
            c = json.load(open(cfg)); m = json.load(open(run / "metrics.json")) if (run / "metrics.json").exists() else None
            dest = HIST / cat / run.name
            if not dry:
                dest.mkdir(parents=True, exist_ok=True)
                for f in LIGHT + (["model.pt"] if cat == "spatial" else []):
                    if (run / f).exists():
                        shutil.copy2(run / f, dest / f)
                if (run / "figures").exists():
                    shutil.copytree(run / "figures", dest / "figures", dirs_exist_ok=True)
                if (run / "dumps").exists() and f"{cat}/{run.name}" not in KEEP:
                    shutil.copytree(run / "dumps", dest / "dumps", dirs_exist_ok=True)
            date = time.strftime("%Y-%m-%d %H:%M", time.localtime((run / "metrics.json").stat().st_mtime if m else cfg.stat().st_mtime))
            v, t, gr, gc = val_test(cat, m, c) if m else (None, None, None, None)
            rows.setdefault(cat, []).append((run.name, date, settings(cat, c), v, t, gr, gc, "kept" if f"{cat}/{run.name}" in KEEP else ("unfinished" if m is None else "")))
            if not dry and f"{cat}/{run.name}" not in KEEP:
                shutil.rmtree(run)
    # loose files at the top level
    misc = HIST / "misc"
    for f in list(MODELS.glob("*.png")) + list(MODELS.glob("*.log")) + list(MODELS.glob("*.txt")):
        if not dry:
            misc.mkdir(exist_ok=True); shutil.move(str(f), misc / f.name)
    fmt = lambda x: "" if x is None else f"{x:.4f}"
    out = ["# Training runs: history index", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} by chl_cnn/archive_runs.py. Each run folder here holds config.json, history.json "
           "(per-epoch curves), metrics.json and figures/. Weights and predictions were deleted except for the runs marked 'kept', which stay under models/<category>/. "
           "Val MAE = hidden-pixel MAE (ln CHL) on the val years; spatial runs with train_stride > 1 or val_stride > 1 were scored on every k-th day (see config.json). "
           "Settings list only the non-default flags; see the trainer docstrings for meanings.", ""]
    for cat, rs in rows.items():
        out += [f"## {cat} ({len(rs)} runs)", "", "| run | finished | settings | val MAE | test MAE | grad ratio | grad corr | note |", "|---|---|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: r[1]):
            out.append(f"| {r[0]} | {r[1]} | {r[2]} | {fmt(r[3])} | {fmt(r[4])} | {fmt(r[5])} | {fmt(r[6])} | {r[7]} |")
        out.append("")
    HIST.mkdir(exist_ok=True)
    (HIST / "RUNS.md").write_text("\n".join(out))
    print("\n".join(out[:4])); print(f"... {sum(len(r) for r in rows.values())} runs indexed -> {HIST / 'RUNS.md'}" + (" (dry run, nothing moved)" if dry else ""))


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
