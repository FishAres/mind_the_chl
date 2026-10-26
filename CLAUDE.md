# mind_the_chl_gap: repo map (read at session start; keep ≤ 60 lines)

CHL gap filling. Focus: Indian Ocean store (`--store io`). Black Sea parked. Workspace rules: ~/shared/CLAUDE.md.
Working rules (token efficiency): bridge CHL doc Brief → "Working rules for CC sessions" (#25). No polling, narrow reads.

## Start here
- docs/HANDOFF.md: state, model table, code map, gotchas.
- analyses/FINDINGS.md: dated experiment log. grep headers (`grep -n "^## " analyses/FINDINGS.md | tail`), read one entry.
- models/RUNS.md: every run with settings + scores (regenerate: `python -m chl_cnn.runs_index`).
- Bridge with claude.ai: `claude-bridge` skill (CHL gap-filling project doc).

## Code
- chl_cnn/train_pinn.py: training/eval entry point (`python -W ignore -m chl_cnn.train_pinn --run <name> ...`, `--eval-only --init-from <run>`).
- chl_cnn/tokdec.py: token-decoder stem (`--t3d-stem tokdec`, `--tokdec-blend`, `--tokdec-spatial N`).
- chl_cnn/krig.py, krig_triton.py: learned kriging (`--tokdec-krig`, diagnostic only since #25).
- chl_cnn/ptok.py (patch tokens, io3), tattn.py (pixel attention, io2), ablate.py (input ablation, io5).
- io7 modules (2026-10-24): physics memory, water-mass attention, analog tokens: see analyses/io7/*/DESIGN.md.
- chl_cnn/plotting.py: Okabe-Ito palette; use it for every figure.

## Runs and data
- Runs: models/pinn/t3d/<run>/ (metrics.json, history.json, preds.npz, model.pt, config.json). Not in git.
- Stores, recorder data: ~/mind_the_chl_gap/data/ (not in git). Scores: metrics.json → scores.{val,test}.cnn_hidden.mae.
- Naming: io<N>_<idea> per track (io6 = kriging/spatial, io7 = #25 plan). Seeds: suffix _s<k>.
- Recipe: batch 8, 60-epoch cosine, lr 1e-3, wd 1e-4, single seed. Flags: ~/mind_the_chl_gap/logs/io7/common.sh (BASE, TOKDEC, SPATIAL).

## Scoring
- Val/test hidden-px MAE (fake clouds); report gradient ratio/corr (lags 1/2/4 px) with MAE.
- Key split: ≥ 1 visible kriging-stencil key vs no key (analyses/io6_kriging/). Real-cloud reweighted score: analyses/io7/.
- Per-run post-processing: `bash ~/mind_the_chl_gap/logs/io7/postrun.sh <run>` (results JSON, FINDINGS entry, RUNS.md).
- Noise: conv seeds S60 = 0.00024. Decision rule (#25, amended #29): compare against the mean and range of the base seeds, not one seed.

## Queues (GPU: one A10G, 23 GB; one job at a time)
- Finish writing and smoke-testing ALL code (incl. GPU smokes, leak tests) BEFORE starting a queue. Don't build while a queue
  trains: agents' CPU tests then compete with training for the 30 GB RAM, and GPU checks wait hours behind the lock (io7, 2026-10-24).
- io7 queue: ~/mind_the_chl_gap/logs/io7/ (runner.sh header: format). jobs.tsv = id, prio, deps, command; markers done/ failed/;
  errors.log; queue.log. Reorder by editing prio. Relaunch: `~/.local/bin/tmux new -d -s io7_queue 'bash ~/mind_the_chl_gap/logs/io7/runner.sh'`.
- GPU jobs take `flock ~/mind_the_chl_gap/logs/gpu.lock`. Keep tmux `keepalive` up (SageMaker 60-min idle stop):
  `~/.local/bin/tmux new -d -s keepalive ~/mind_the_chl_gap/logs/keepalive.sh`.
- Status in one command: `tail -n 5 ~/mind_the_chl_gap/logs/io7/queue.log; cat ~/mind_the_chl_gap/logs/io7/errors.log`.
- blk7 queue (2026-10-26): 5 IO attention configs on `--store blk4` (~9 h), same layout in ~/mind_the_chl_gap/logs/blk7/ (session
  blk7_queue); `train_oom` retries a CUDA OOM once at the next smaller batch (`<run>_b<k>`). Notes: analyses/blk7/NOTES.md.
- q8 queue (2026-10-26, #29): io7_physmem_s1, io8_physmem_rwloss, blk8_spatial_p12, blk8_spatial_bsmem (~9 h), ~/mind_the_chl_gap/logs/q8/ (session q8_queue; starts tmux autostop when done). Notes: analyses/io8/NOTES.md.

## Gotchas
- Don't retrain controls: compare new options at the same epoch against the existing run's history.json (`--stop-epoch`).
- Only CHL_cmes-level3 is observations; CHL and CHL_cmes-gapfree are interpolation products (leak).
- Triton needs cuda.h; krig.py sets CPATH when unset. Default path regression check: logs/io7/regcheck.sh.
- git: results JSON/CSV, models/, data and logs are ignored; figures, code and docs are tracked.

## Token Economy
- User smaller models for agents where possible
- Create and re-use files or skills for token-heavy tasks: e.g. a visualization, training model, adding something to the queue - these can be running a short script or making a short edit somewhere