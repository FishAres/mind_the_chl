# Handoff: back to gap filling (2026-10-20)

For a new session picking up the Black Sea gap-filler. Nothing is running; the GPU is free.

## State

| | |
|---|---|
| focus | gap filling on the Black Sea 4 km store (`--store blk4 --crop`); the Indian Ocean work is finished |
| best run | `models/pinn/t3d/blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50`: 3D stem, λ 0.3, 50 epochs + two warm restarts (`--continue-from`), weights on disk |
| best scores | val / test MAE 0.1498 / 0.1430; gradient ratio 0.54, gradient corr 0.66 (val, 1 px pairs) |
| tried since, none better | `--geo-static` (neutral), width 96 (0.1527 / 0.1480 at 50 epochs, no restarts yet), post-hoc sharpening (test −0.0002), 4 km beats 2 km on 2 km truth (−0.0086) |
| shelved | CHL tracer project, see [analyses/tracer/TRACER.md](../analyses/tracer/TRACER.md) |
| decided next step | none. Candidates are in [BLACK_SEA.md section 6](../analyses/BLACK_SEA.md); none has been approved yet. |

## Read first

1. [analyses/BLACK_SEA.md](../analyses/BLACK_SEA.md): results, error structure, gradient diagnosis, open questions.
2. [analyses/FINDINGS.md](../analyses/FINDINGS.md): dated log of every experiment. Search before planning a run.
3. [models/RUNS.md](../models/RUNS.md): every run with settings and scores (`python -m chl_cnn.runs_index`).
4. [README.md](../README.md): set-up, the training recipe, flags.

## How the user works

- Delegate experiment tracks and monitoring to subagents; the main session designs, checks and summarises.
- Run one GPU job at a time: `flock ~/mind_the_chl_gap/logs/gpu.lock ...`, inside tmux.
- Don't retrain controls. Train new options on the existing run's schedule with `--stop-epoch N` and compare against
  its `history.json` at the same epoch.
- Report gradient ratio and correlation (and their scale dependence) with MAE. Gradients at 20–70 km matter for
  fisheries use, and the user expects physics to help sharpen them.
- Follow `~/shared/CLAUDE.md` for analysis and reporting:
  - observations first, with interpretation in its own section;
  - n with every number;
  - colourblind-friendly plots (Okabe-Ito, `chl_cnn/plotting.py`) with titles that make no claims;
  - concise tables, effect first and n last.
- Keep data on local disk (`~/mind_the_chl_gap/data/`); the shared s3 mount is slow.

## Environment gotchas

- **Idle stop.** The SageMaker space stops after 60 idle minutes; tmux jobs don't count as activity. Before any job
  longer than an hour, start `tmux new -d -s keepalive ~/mind_the_chl_gap/logs/keepalive.sh`.
  - It only keeps the instance up while a `python -m chl_cnn.*` or `python -m analyses.*` process runs, so launch jobs
    that way.
  - The pattern was widened on 2026-10-20. Before that, `analyses.*` jobs were unprotected, and one run was lost to an
    idle stop on 10-19.
- **After an instance restart.** tmux sessions are gone; restart keepalive and any queue.
  - Packages live in `~/.local`, with numpy pinned at 1.26.4.
  - tmux is `~/.local/bin/tmux`, a symlink into the conda env `~/.local/tmux-env`.
  - Check with `python -c "import xarray, zarr"` and `which tmux`. Reinstall steps are in the README.
- **Plot orientation.** Black Sea stores run south to north, so plot with `origin="lower"`.
- **Notebook context panels.** Fixed 2026-10-23: `chl_cnn/viz.draw_day` now picks the t±1 panels by calendar day, as
  the trainer does (`train2.neighbour_index`), and labels a date missing from the store.

## Unapproved candidate next steps (from the 10-17 to 10-20 sessions)

- Black Sea seed noise. Every Δ so far is a single seed.
- Copernicus L4 product scored on the same hidden pixels, as an external benchmark.
- Width 96 with the warm restarts. More blocks, the ±3-day window, high-CHL loss weighting.
- Warped context channels: context days moved into day t's frame with 0.25 × currents, optionally split into
  < 20 / 20–50 / 50–150 km bands with fade-time confidence. The tracer results suggest structure at 20–50 km barely
  moves, so the gain may be small.
- Whether the gradient ratio is MAE shrinkage. If it is, the fix is a loss change or sampled (ensemble / generative)
  outputs.
- New store inputs not yet used by the gap-filler: rsntds (light) and friver (rivers).
- Retrieval noise floor from OC-CCI per-pixel RMSD or SENSORMASK.
- A fair 2 km comparison: the 20-epoch 4 km run against the 20-epoch 2 km run.
