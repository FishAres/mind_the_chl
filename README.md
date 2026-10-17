# mind the chl gap: filling cloud gaps in satellite chlorophyll

Deep networks that fill cloud gaps in daily L3 chlorophyll (ln CHL), using neighbouring days and physical covariates
(SST, salinity, mixed-layer depth, sea level, currents, wind). Two regions: the Indian Ocean (0.25°, 1997–2020, from
[geo-smart/mind-the-chl-gap](https://github.com/geo-smart/mind-the-chl-gap)) and the Black Sea (Copernicus 1 km CHL
arranged at 2 km and 4 km, 2014–2024).

**Score.** Observed pixels are hidden with a real cloud mask from another day (a "donor" day), and the model is scored
by the mean absolute error on those hidden pixels (MAE in ln CHL). Val and test sets are whole years interleaved
with the training years.

## Where to look first

| | |
|---|---|
| [analyses/FINDINGS.md](analyses/FINDINGS.md) | Dated log of every experiment round and what it showed. Read before planning a run. |
| [models/RUNS.md](models/RUNS.md) | Every training run: settings, val/test MAE, epochs, whether its weights are on disk. Regenerate with `python -m chl_cnn.runs_index`. |
| [analyses/BLACK_SEA.md](analyses/BLACK_SEA.md) | Black Sea results so far and open questions. |
| [analyses/forecast/MODELS.md](analyses/forecast/MODELS.md) | Schematics of the gap-fill, forecast and nowcast set-ups ([PDF](analyses/forecast/MODELS.pdf)). |
| [analyses/IDEAS.md](analyses/IDEAS.md) | Candidate directions. |
| [notebooks/progress.ipynb](notebooks/progress.ipynb) | Live training curves and per-day prediction maps for any run (ipywidgets). |

## Layout

| path | contents |
|---|---|
| `chl_cnn/` | The package. Trainers: `train_pinn.py` (main: dilated CNN / U-Net / DEQ / 3D temporal stem, optional physics loss), `train_spatial.py` (spatial nODE), `train_ude.py` (daily UDE), `train.py` / `train2.py` (early CNNs). Models: `models.py`, `deq.py`, `spatial_node.py`, `ude.py`. Data: `store.py` (IO), `store_blk.py` (Black Sea 2 km), `store_blk4.py` (4 km), `clouds.py` (fake clouds), `data.py`. Physics: `physics.py`, `priors.py`. Runs: `runs.py`, `runs_index.py`, `archive_runs.py`, `watch.py`, `viz.py`. |
| `analyses/<topic>/` | One folder per study: scripts that read run outputs, their `figures/` and `results/`. |
| `notebooks/` | Results notebooks (outputs kept), `progress.ipynb`, literature notes (`*_literature_notes.md`) and dataset surveys. |
| `scripts/` | Downloads and data arranging (`fetch_data.py`, `fetch_cmems.py`, `fetch_cmems_all.sh`, `fetch_occci.py`), the IO EDA (`IO_EDA.ipynb`, `io_summaries.py`, `io_viewer.py`), the old training queue `train_queue.sh`. |
| `docs/` | [data_sources.md](docs/data_sources.md) (buckets, Copernicus / OC-CCI pulls); `archive/` (the 2026-10-03 project brief, deliberation notes and old README). |
| `models/` | Not in git. `models/<category>/<run>/`: `config.json`, `history.json`, `metrics.json`, `model.pt`, `preds.npz`, `figures/`. Archived runs keep only the JSON files, in `models/history/`. |

## Setup in a new SageMaker session

```bash
cd ~/shared/ecology/mind_the_chl_gap
pip install --user xarray dask zarr gcsfs netCDF4 "numpy==1.26.4"   # keep numpy pinned; --user survives restarts
python scripts/fetch_data.py                 # IO.zarr, ~24.7 GB -> ~/mind_the_chl_gap/data/, ~2.5 min
python -m chl_cnn.store                      # IO training store (~5 GB)
scripts/fetch_cmems_all.sh blk 2014 2024 2   # Black Sea CHL + physics at 2 km (needs `copernicusmarine login`)
python -m chl_cnn.store_blk                  # Black Sea 2 km store
python -m chl_cnn.store_blk4                 # Black Sea 4 km store (2x2 average of the 2 km store)
```

Data live on local disk under `~/mind_the_chl_gap/data/`, not in `~/shared` (s3fs is slow for many small files).
`tmux` is not on the base image: install it once with `conda create -p ~/.local/tmux-env -c conda-forge tmux` and
link `~/.local/tmux-env/bin/tmux` into `~/.local/bin/`.

## Training

Current Black Sea recipe (3D temporal stem, 8 residual blocks, ±4 days, gradient loss, semi-Lagrangian physics loss):

```bash
python -m chl_cnn.train_pinn --run my_run --store blk4 --crop --feat-device gpu --val-stride 4 --eval-batch 8 \
  --features sst so mlotst sla u_curr v_curr u_wind v_wind \
  --model temporal3d --t3d-backbone dilated --context 4 --feat-context 4 --grad-weight 100 --clim --coords --residual \
  --vis-weight 0.1 --year-input --donor-window 15 --batch 16 --epochs 50 --ckpt-from 10 --phys semi-lagrangian --lam 0.3
```

Useful flags (full list in the `train_pinn.py` docstring): `--return` (default on: resume an interrupted run from its
best epoch), `--continue-from RUN` (warm restart from another run's best state), `--stop-epoch N` (stop early but keep
the `--epochs` schedule, for short comparisons with an existing run), `--context-before/--context-after` (forecast and
nowcast windows), `--donor-exclude D`, `--eval-only --init-from RUN`.

Run one trainer at a time (`flock ~/mind_the_chl_gap/logs/gpu.lock ...`) inside tmux. SageMaker stops an idle instance
after 60 min. A keepalive loop that touches `/tmp/.sagemaker-last-active-timestamp` while a trainer runs is in
`~/mind_the_chl_gap/logs/keepalive.sh`.

## Gotchas

- `xr.open_zarr(path)` uses dask; `chunks=None` loads whole variables (2.7 GB each for IO) and can crash a kernel.
- IO.zarr: ERA5 is missing whole months in 1979–1992 and 2021–22. `mlotst` for 1979–1992 is a copy of `u_wind`. 43 lake pixels are not flagged as land.
- The Black Sea stores run south to north (latitude ascending), so use `origin="lower"` when plotting them.
- Widget cells need a live Jupyter or VS Code front end.
- Scores are only comparable within one region and one `--val-stride` / `--donor-exclude` setting.
