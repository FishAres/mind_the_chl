"""Append the io5 entries to analyses/FINDINGS.md from the generated tables (observations only; run by the io5 queue).
Idempotent: an entry whose heading is already in FINDINGS.md is not appended again.
Usage (project root): python -m analyses.io5_ablation.findings --job 1|2
"""
import argparse
import json
import time
from pathlib import Path

from chl_cnn.runs import find_run

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FINDINGS = HERE.parent / "FINDINGS.md"
H1 = "## Global input ablation of three IO gap-fillers (io5 Job 1; eval-only, val; chl_cnn/ablate.py, train_pinn.py --ablate, analyses/io5_ablation/)"
H2 = "## Physics loss λ 0.3 on the token-decoder + blend recipe (io5 Job 2, io5_tokdec_blend_up0p3; analyses/io5_ablation/report_job2.py)"


def section(md, title):
    """Lines of the markdown section '## title' (without its heading)."""
    out, on = [], False
    for ln in md.splitlines():
        if ln.startswith("## "):
            on = ln[3:].startswith(title)
            continue
        if on:
            out.append(ln)
    return "\n".join(out).strip()


def job1():
    md = (RES / "job1_tables.md").read_text()
    s = json.loads((RES / "job1_summary.json").read_text())
    rc = json.loads((find_run("io5_regcheck_default_eval") / "metrics.json").read_text())["scores"]
    chk = [ln for ln in md.splitlines() if ln.startswith("Checks:")]
    L = [f"{H1}", f"- Logged {time.strftime('%Y-%m-%d %H:%M')} by the io5 queue (logs ~/mind_the_chl_gap/logs/io5/).",
         "- Models: conv ±4 `t3d_dil_c4_60ep_gdl100`, `io3_ptok4` (patch tokens, arm A), `io4_tokdec_blend`. Val, all 1,458 days, fixed hidden px; "
         "each model's Δ is against its own eval-only reproduction (io5_eval_<model>_base).",
         "- Groups, one at a time, on every window day (context days and day t): SST, salinity, MLD, SLA, currents (u + v), winds (u + v), air temp, "
         "all 9 physics; context-day CHL (value and visible mask 0 on every day ≠ t); day-t CHL (value and mask 0 on day t); static channels: "
         "DOY climatology, lat/lon, depth, year.",
         "- Replacements: physics `mean` = 0 in the z-scored input (the training mean); `doyclim` = the field's per-pixel day-of-year climatology over "
         "the training days (circular ±15-day window, NaN-aware), z-scored with the trainer's feature stats, NaN → 0 (cache "
         "~/mind_the_chl_gap/data/io5/doyclim_io.npz). Static channels → their mean over ocean px of the training days.",
         "- Assumptions:",
         "  - `clim` replaces the static climatology channel and the per-day climatology the token stems subtract from CHL (clim3d). "
         "`clim_static` replaces the static channel only (token models).",
         "  - The --residual output base (climatology added to the output) is not an input and is unchanged. The sin/cos DOY channels were not ablated.",
         "  - Window days missing from the store stay 0 under doyclim, as the trainer gives them.",
         "  - Targets, hidden / observed masks and the residual diagnostics are untouched.",
         f"- Checks: `--eval-only --init-from t3d_dil_c4_60ep_gdl100` (default flags, after the edit) val / test "
         f"{rc['val']['cnn_hidden']['mae']:.6f} / {rc['test']['cnn_hidden']['mae']:.6f}. Each ablation run checks its first val batch: "
         "ablated slots hold the replacement (doyclim against an independent numpy reference) and every other input value is unchanged (exact). "
         + (chk[0].replace("Checks: ", "") if chk else ""),
         "", section(md, "Unablated baselines"), "",
         "ΔMAE (val hidden-px MAE, ln CHL; ablated − unablated), training-mean replacement:", "", section(md, "ΔMAE, training-mean"), "",
         "ΔMAE, DOY-climatology replacement (physics groups):", "", section(md, "ΔMAE, DOY-climatology"), "",
         f"- Groups with the largest mean ΔMAE over the three models (mean mode): {', '.join(s['top4_mean'])}.",
         "- Full per-model tables (RMSE, bias, gradient ratio / corr at lag 1, ΔMAE by distance and two-sided age bin, n): "
         "analyses/io5_ablation/results/job1_tables.md; rows in job1_table.csv.",
         "- Figures: analyses/io5_ablation/figures/job1_heatmap_mean.png, job1_bars.png, job1_dist_bins.png.",
         "- Caveat: an ablation shows what a trained model relies on at inference. It includes distribution shift (the model never saw the replaced "
         "values in training), and correlated inputs can stand in for each other. It is not a causal contribution of the input.",
         "- Interpretation: not written by the queue (pending review).", ""]
    return "\n".join(L)


def job2():
    md = (RES / "job2_tables.md").read_text()
    c = json.loads((find_run("io5_tokdec_blend_up0p3") / "config.json").read_text())
    body = md.split("\n", 2)[2].strip()
    L = [f"{H2}", f"- Logged {time.strftime('%Y-%m-%d %H:%M')} by the io5 queue.",
         f"- Recipe: io4_tokdec_blend flags (batch 8, 60-epoch cosine, lr 1e-3, wd 1e-4, --ckpt-from 10, seed 0) + `--phys upwind --lam 0.3` "
         f"(phys warm-up {c['phys_warmup']} epochs, L1 residual, K {c['K']:g} m²/s, as earlier IO λ 0.3 runs). Single seed. Test scored once, at the "
         "checkpoint selected on val over epochs 11-60.",
         "- At 30 epochs only the per-epoch history exists (val MAE, RMSE, gradient lag 1, residual); lags 2 / 4, bins and test are for the selected checkpoint.",
         "", body, "",
         "- Figures: analyses/io5_ablation/figures/job2_val_curves.png, job2_residual.png, job2_bins.png. Smoke test (1 epoch): io5_smoke_up0p3, "
         "log ~/mind_the_chl_gap/logs/io5/io5_smoke_up0p3.log.",
         "- Interpretation: not written by the queue (pending review). Single seed; earlier S60 between conv seeds was 0.00024.", ""]
    return "\n".join(L).replace("\n## ", "\n### ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=int, choices=[1, 2], required=True)
    a = ap.parse_args()
    head, txt = (H1, job1()) if a.job == 1 else (H2, job2())
    cur = FINDINGS.read_text()
    if head in cur:
        print("already in FINDINGS.md:", head)
    else:
        FINDINGS.write_text(cur.rstrip("\n") + "\n\n" + txt)
        print("appended:", head)
