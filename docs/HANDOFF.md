# Handoff: Indian Ocean gap filler, attention stems (2026-10-24)

For a new session. Nothing is running and the GPU is free; only tmux `keepalive` is up. The previous handoff (Black Sea,
2026-10-20) is in [HANDOFF_2026-10-20_blacksea.md](HANDOFF_2026-10-20_blacksea.md).

The last session used a lot of tokens. Keep the main session lean: delegate builds, queues and monitoring to subagents,
read logs with `tail`/`grep` instead of whole files, and keep reports short.

## State

| | |
|---|---|
| focus | IO store (`--store io`): better gap filling via attention over ±8 context days |
| best single model | `io6_tokdec_blend_spatial`, val / test MAE 0.1271 / 0.1275 (single seed) |
| next best | `io4_tokdec_blend` 0.1276 / 0.1280; conv 3-seed ensemble 0.1279 / 0.1279 |
| conv seed noise | conv ±4, 3 seeds: val 0.1301–0.1304, S60 0.0002 |
| decided next step | none; nothing has been approved |
| interpretation | the io5 Job 2 and io6 entries in FINDINGS.md are observations only; the user has not reviewed them yet |

### IO models (60 epochs, batch 8, single seed unless noted)

| run | val MAE | test MAE | grad ratio / corr 1 px | s/epoch |
|---|---|---|---|---|
| io6_tokdec_blend_spatial (io4 blend + 2 spatial attention blocks on tokens) | 0.1271 | 0.1275 | 0.546 / 0.629 | 128 |
| io4_tokdec_blend (token decoder + own-pixel blend) | 0.1276 | 0.1280 | 0.548 / 0.631 | 108 |
| io5_tokdec_blend_up0p3 (io4_tokdec_blend + physics λ 0.3) | 0.1276 | 0.1286 | 0.545 / 0.630 | — |
| io4_tokdec (tokens only) | 0.1282 | 0.1287 | — | — |
| io3_ptok4 (patch tokens, arm A) | 0.1282 | 0.1286 | — | 80 |
| io6_tokdec_krig (learned kriging stem replaces blend) | 0.1283 | 0.1294 | 0.546 / 0.632 | 145 |
| io2_tattn8 (pixel attention ±8, batch 4) | 0.1286 | 0.1272 | — | 286 |
| io6_tokdec_krig_signed (kriging + signed head pair) | 0.1291 | 0.1297 | 0.544 / 0.631 | 159 |
| conv ±4 (t3d_dil_c4_60ep_gdl100) | 0.1302 | 0.1305 | — | — |

io6 val MAE split by whether a hidden pixel has a visible kriging key (n = 146 diag days, 487,572 hidden px, 417,788 with ≥ 1 key):

| | ≥ 1 key | no key |
|---|---|---|
| io6_tokdec_blend_spatial | 0.1101 | 0.2425 |
| io4_tokdec_blend | 0.1104 | 0.2448 |
| io6_tokdec_krig | 0.1109 | 0.2471 |
| io6_tokdec_krig_signed | 0.1113 | 0.2509 |
| conv seed 0 | 0.1115 | 0.2577 |

## Read first

1. [analyses/FINDINGS.md](../analyses/FINDINGS.md): last four entries (io4, io5 Job 1 ablation, io5 Job 2, io6).
   Search it before planning a run.
2. [models/RUNS.md](../models/RUNS.md): every run with settings and scores (`python -m chl_cnn.runs_index`).
3. [analyses/io6_kriging/DESIGN.md](../analyses/io6_kriging/DESIGN.md) and `analyses/io6_kriging/figures/` (variogram,
   weight vs |ΔSST|, spread vs error, learned params).
4. `notebooks/attention_explorer.ipynb`: per-day attention / blend / integrated-gradient panels for 6 IO models,
   24 preset days (`analyses/attn_viz/explorer_days.json`). Caches in `analyses/attn_viz/explorer_cache/`; rebuild with
   `python -W ignore -m analyses.attn_viz.build_explorer --run <run>` (CPU, ~18 GB RAM, 8–27 min).
5. Bridge doc with claude.ai (CHL gap-filling project doc; ids in the `claude-bridge` skill). Log runs to #17+.
6. [README.md](../README.md): set-up, the training recipe, flags.

## Code added 2026-10-23..24

- `chl_cnn/tokdec.py` (`--t3d-stem tokdec`, `--tokdec-blend`, `--tokdec-spatial N`).
- `chl_cnn/krig.py`, `chl_cnn/krig_triton.py` (`--tokdec-krig`, `--krig-*`). Triton kernels default to TF32
  (`--krig-prec ieee` for fp32); PyTorch reference via `--krig-impl torch`. `--krig-advect` exists on the torch path only,
  never run. Triton needs `cuda.h`; `krig.py` sets CPATH to TensorFlow's CUDA headers when CPATH is unset.
- `chl_cnn/ablate.py` / `train_pinn.py --ablate` (io5 input ablation).
- `chl_cnn/attn_explorer.py`, `analyses/attn_viz/build_explorer.py`.
- Queues: `~/mind_the_chl_gap/logs/{io5,io6}/queue.sh`. Backups of edited files: `logs/io6/*.pre_io6.py.bak`,
  `logs/io5/train_pinn.pre_io5.py.bak`.
- Fixed 2026-10-24: io6 queue checked `<run>_diag.npz` but `io6_eval.py` wrote `<run>_rec_diag.npz`. The recorder
  pass was flagged failed and the key split and 5 figures were missing. Files renamed, `queue.sh` patched, scoring /
  report / FINDINGS entry regenerated. Pre-fix results in `analyses/io6_kriging/results_pre_recfix/`.

## How the user works

- Delegate experiment tracks and monitoring to subagents; the main session designs, checks and summarises.
- Run one GPU job at a time: `flock ~/mind_the_chl_gap/logs/gpu.lock ...`, inside tmux.
- Don't retrain controls. Train new options on the existing run's schedule with `--stop-epoch N` and compare against
  its `history.json` at the same epoch.
- Report gradient ratio and correlation (and their scale dependence) with MAE. Gradients at 20–70 km matter for
  fisheries use, and the user expects physics to help sharpen them.
- Follow `~/shared/CLAUDE.md`: observations first, interpretation in its own section; n with every number;
  colourblind-friendly plots (Okabe-Ito, `chl_cnn/plotting.py`) with titles that make no claims; concise tables,
  effect first and n last.
- Write anything the user needs to paste into a file under `~/`; they can't copy from the console.
- Keep data on local disk (`~/mind_the_chl_gap/data/`); the shared s3 mount is slow.

## Environment gotchas

- **Idle stop.** The SageMaker space stops after 60 idle minutes; tmux jobs don't count. Before any job longer than an
  hour, check `~/.local/bin/tmux ls` shows `keepalive`, else start
  `~/.local/bin/tmux new -d -s keepalive ~/mind_the_chl_gap/logs/keepalive.sh`. It only counts `python -m chl_cnn.*` or
  `python -m analyses.*` processes, so launch jobs that way.
- **After an instance restart.** tmux sessions are gone; restart keepalive and any queue (queues skip finished steps
  and resume from `resume.pt`). Packages live in `~/.local`, numpy pinned at 1.26.4; tmux is `~/.local/bin/tmux`.
- **Host RAM is ~30 GB.** Explorer cache builds (~18 GB) and recorder passes can OOM when run beside other jobs.
- **IO SST is quantised to 0.25 K**: 78 % of neighbouring-pixel differences are exactly 0.
- **IO.zarr CHL**: only `CHL_cmes-level3` is observations; `CHL` and `CHL_cmes-gapfree` are interpolation products.
- **Black Sea plots** need `origin="lower"` (stores run south to north).

## Parked: Black Sea

Best run `blk4_t3d_c4_gdl100_sl_lam0p3_b16_50ep_cont50_lr1e-4_cont50`, val / test 0.1498 / 0.1430. Unapproved
candidates are in [HANDOFF_2026-10-20_blacksea.md](HANDOFF_2026-10-20_blacksea.md) and
[BLACK_SEA.md section 6](../analyses/BLACK_SEA.md).
