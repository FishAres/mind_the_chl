# io7_analog: analog tokens (bridge plan #25, Phase 1 run 3)

Base: `io6_tokdec_blend_spatial` (val 0.1271). Change: each pooled day-t patch token cross-attends to the same patch on K
training-year days within ±30 days of season. Code: `chl_cnn/analog.py`; flags `--analog --analog-k 32` (train_pinn.py).

## Design

| choice | decision | why |
|---|---|---|
| bank entry | per (training day, ocean patch): 16 observed CHL z-anomalies (vs DOY climatology, 0 where unobserved), 16-px observed mask, 9 physics patch means (z-scored) = 41 floats | fixed and cheap: built once at start-up from the in-memory GPU tensors (5,540 days x 1,164 patches x 41, fp16, 0.53 GB; seconds). On-the-fly encoding with the patch encoder would rerun the 12-ch Conv on K days per sample |
| bank days | training years only (16 years; asserted vs val / test lists) | val / test data never enter |
| candidates for day t | circular DOY distance <= 30, year != year(t), \|d - t\| > 30 calendar days | own-year exclusion = leakage guard and no copying of neighbouring days; the gap stops adjacent-year days near Jan 1 (which sit in t's ±8 context, masked) from appearing unmasked. Same rule for train / val / test. 851-965 candidates per day |
| K | 32 drawn without replacement (fallback 16) | ~3.5 % of candidates; ~300k keys per batch of 8 |
| draws | training: new draw per sample per step (own numpy RNG); eval: fixed per day, `default_rng([analog_seed, t])` | eval is deterministic and independent of batch composition |
| encoder | [entry, dDOY/30, \|dDOY\|/30] -> Linear 43->128, GELU, Linear, LayerNorm -> K, V; query = W_q LayerNorm(pooled) | 4 heads, the stem's width |
| masking | analog patches with no observed px masked; learned null key / value (zero init) | a patch can attend to nothing |
| output | zero-init Linear(D, D), added to the pooled token before the 2 spatial blocks | starts as io6 exactly (unit-tested bit-identical) |
| hook | forward hook on `TokDecNet.po`; `predict_z` passes the batch's store days via `model.analog.set_days(bi)` | tokdec.py untouched |

Bank not saved in model.pt (rebuilt each run, deterministic). Training draw RNG is not in resume.pt: a resumed run's draws differ.
`--train-on-val` with `--analog` is refused (bank assertion).

## Shared-file edits (train_pinn.py, all gated by --analog)
parser (`AN.add_args`, `AN.check_args`); `RESUME_NEW_DEFAULTS.update(...)`; model build (bank + attach); `predict_z` (set_days).
Backup: `~/mind_the_chl_gap/logs/io7/train_pinn.pre_analog.py.bak`.

## Tests
- CPU: `python -m chl_cnn.test_analog` (19 checks: bank membership, own-year / gap / season rules, eval determinism and batch-order
  independence, val-data perturbation leaves the bank unchanged, init output == plain model, forward / backward).
- Real split: `python -m chl_cnn.analog` (0 rule violations over all 8,461 days).
- Queue: `leak_analog` (above + io6 leak test on a 20-step init run), `smoke_analog` (2 epochs, OOM -> K 16).

## Cost estimate
Bank 0.53 GB; extra activations ~1 GB at batch 8, K 32 -> ~13 GB allocated (io6 11.2). Extra compute ~0.1 TFLOP / step;
expect ~135-145 s/epoch vs 128 (~2.4 h for 60 epochs). Not measured yet: the smoke job logs peak memory and s/epoch.

## Open questions
- Whether 32 random draws of ~900 are enough, versus a physics-similarity preselection (not built).
- Whether a val-day analog set differs in quality from a training-day set (val candidates ~914 vs train ~854).
