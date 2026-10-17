#!/bin/bash
# Resumable training queue. Run from a tmux session on the SageMaker terminal:
#   tmux new -s train
#   cd ~/shared/ecology/mind_the_chl_gap && scripts/train_queue.sh            # all pending runs below
#   scripts/train_queue.sh unet                                                # only the runs whose name contains 'unet'
# A run is skipped if models/<category>/<run>/metrics.json exists (done) or a trainer with that --run is already alive.
# Log: models/queue.log (appended). Progress figure: models/progress_long.png (refresher started below if not running).
# Detach with Ctrl-b d, reattach with  tmux attach -t train.

cd "$(dirname "$0")/.."                                                # repo root
LOG=models/queue.log
# the SageMaker image loses pip packages on restart: restore them before training
python -c "import xarray, zarr, dask, netCDF4" 2>/dev/null || pip install -q xarray dask zarr gcsfs netCDF4 copernicusmarine "numpy==1.26.4" 2>&1 | grep -v -i "requires\|incompatib\|resolver"
FILTER=${1:-}
ALL="sst so air_temp u_wind v_wind sla mlotst u_curr v_curr"
FULL="--features $ALL --clim --coords --residual --context 2 --vis-weight 0.1 --loss l1"

run() {  # run <name> <args...>
  NAME=$1; shift
  [[ -n "$FILTER" && "$NAME" != *"$FILTER"* ]] && return
  if compgen -G "models/*/$NAME/metrics.json" > /dev/null || compgen -G "models/*/*/$NAME/metrics.json" > /dev/null; then echo "--- $NAME done, skipping" | tee -a $LOG; return; fi
  if pgrep -f -- "--run $NAME( |$)" > /dev/null; then echo "--- $NAME already running elsewhere, skipping" | tee -a $LOG; return; fi
  echo "=== $NAME start $(date)" | tee -a $LOG
  python -m chl_cnn.train_pinn --run "$NAME" "$@" 2>&1 | grep --line-buffered -v -E "^(E0000|W0000)|Warning|warn" | tee -a $LOG
}

pgrep -f "chl_cnn.watch" > /dev/null || (nohup python -m chl_cnn.watch --cat pinn/cnn,pinn/unet --out models/progress_pinn.png --every 60 > /dev/null 2>&1 &)

# --- physics-only at t: CHL context t+-1, physical features at t-1, t, t+1, no CHL image at t (12 epochs)
NT="--features $ALL --clim --coords --residual --context 1 --feat-context 1 --no-chl-today --epochs 12 --fig-every 5"
run nochl_today_lam0   $NT --lam 0
run nochl_today_lam0p3 $NT --lam 0.3

# --- U-Net PINNs with the full CNN-PINN input set, ~1.9 M params, 30 epochs, best-val checkpoint from epoch 10
UN="$FULL --model unet --epochs 30 --ckpt-from 10 --fig-every 10"
run unet_d2w64_lam0   $UN --width 64 --unet-depth 2 --lam 0
run unet_d2w64_lam0p3 $UN --width 64 --unet-depth 2 --lam 0.3
run unet_d3w32_lam0   $UN --width 32 --unet-depth 3 --lam 0
run unet_d3w32_lam0p3 $UN --width 32 --unet-depth 3 --lam 0.3

# --- training-data variants, dilated CNN, 12 epochs:  scripts/train_queue.sh season   /   scripts/train_queue.sh year
# (superseded by the v2 runs, which use both at 60 epochs; never run. Uncomment to run them.)
#TD="$FULL --epochs 12 --fig-every 5"
#run pinn_season15_lam0   $TD --donor-window 15 --lam 0        # training fake clouds from the same season (+-15 days of year)
#run pinn_season15_lam0p3 $TD --donor-window 15 --lam 0.3
#run pinn_year_lam0       $TD --year-input --lam 0             # calendar year as a constant input channel
#run pinn_year_lam0p3     $TD --year-input --lam 0.3

# --- v2 CNN PINNs: year channel + seasonal donors, 60 epochs:  scripts/train_queue.sh v2
V2="$FULL --year-input --donor-window 15 --epochs 60 --ckpt-from 10 --fig-every 10"
run pinn_v2_lam0   $V2 --lam 0
run pinn_v2_lam0p3 $V2 --lam 0.3
run pinn_v2_lam1p0 $V2 --lam 1.0

# --- v3: spectral blocks and gradient-difference loss on the v2 recipe, 20 epochs:  scripts/train_queue.sh v3
V3="$FULL --year-input --donor-window 15 --epochs 20 --ckpt-from 8 --fig-every 10 --lam 0"
run pinn_v3_spec      $V3 --spectral 4                       # FNO spectral path in the first 4 blocks (16 modes, 16 ch)
run pinn_v3_gdl       $V3 --grad-weight 1                    # L1 + gradient-difference loss on hidden pixels
run pinn_v3_spec_gdl  $V3 --spectral 4 --grad-weight 1
run pinn_v3_ctrl      $V3                                    # v2 recipe with the same 20-epoch schedule (control, last)

# --- sweep 1: 12-epoch screen around the v2 recipe (lam 0):  scripts/train_queue.sh s1_
#     s1_base / s1_seed1 give the seed noise floor; the rest change one thing each. ~9 min per run.
S1="$FULL --year-input --donor-window 15 --epochs 12 --ckpt-from 6 --fig-every 6 --lam 0"
run s1_base    $S1
run s1_seed1   $S1 --seed 1
run s1_gdl3    $S1 --grad-weight 3
run s1_gdl10   $S1 --grad-weight 10
run s1_w96     $S1 --width 96
run s1_ctx3    $S1 --context 3
run s1_lr2e3   $S1 --lr 2e-3
run s1_drop01  $S1 --dropout 0.1
run s1_huber   $S1 --loss huber

# --- sweep 2 / 3: lines written by  python analyses/pinn/sweep_next.py  (run it after sweep 1, then after sweep 2)
#     scripts/train_queue.sh s2_      combinations of the sweep-1 winners, 12 epochs
#     scripts/train_queue.sh s3_      best config so far at 60 epochs (+ a second seed), best-val checkpoint from epoch 10
S3="$FULL --year-input --donor-window 15 --epochs 60 --ckpt-from 10 --fig-every 10 --lam 0"
[[ -f models/sweep2.txt ]] && while read -r NAME ARGS; do [[ -z "$NAME" || "$NAME" == \#* ]] || run "$NAME" $S1 $ARGS; done < models/sweep2.txt
[[ -f models/sweep3.txt ]] && while read -r NAME ARGS; do [[ -z "$NAME" || "$NAME" == \#* ]] || run "$NAME" $S3 $ARGS; done < models/sweep3.txt

echo "=== queue finished $(date)" | tee -a $LOG
