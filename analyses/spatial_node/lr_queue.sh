#!/bin/bash
# Sequential learning-rate-schedule queue (conv D-net, t+-4 context, train stride 2, val stride 4), one GPU job at a time.
# lr_cos (already launched separately) -> lr_const -> lr_const_low -> lr_const_high -> lr_const_cont (best constant, +6 ep) -> lr_cos12
P=/mnt/custom-file-systems/s3/shared/ecology/mind_the_chl_gap
LOGS=~/mind_the_chl_gap/logs
COMMON="--dmodel conv --context 4 --ctx-feat --train-stride 2 --val-stride 4"
cd "$P" || exit 1

wait_for() {  # wait until models/spatial/<run>/metrics.json exists; fail if the run's tmux session died without it
  local run=$1
  while [ ! -f "$P/models/spatial/$run/metrics.json" ]; do
    if ! tmux has-session -t "$run" 2>/dev/null && ! pgrep -f -- "--run $run " >/dev/null; then
      echo "$(date -u +%H:%M:%S) $run: no process and no metrics.json -> treating as failed"; return 1
    fi
    sleep 60
  done
  echo "$(date -u +%H:%M:%S) $run finished: $(python -c "import json;m=json.load(open('$P/models/spatial/$run/metrics.json'));print('final %.4f best %.4f min %.1f'%(m['val']['mae'],m['best_val_mae'],m['minutes']))")"
}

run() {  # run <name> <extra args...>; blocks until finished
  local name=$1; shift
  echo "$(date -u +%H:%M:%S) launch $name: $*"
  python -m chl_cnn.train_spatial --run "$name" $COMMON "$@" > "$LOGS/$name.log" 2>&1
  wait_for "$name"
}

wait_for lr_cos
run lr_const --epochs 6 --sched none
run lr_const_low --epochs 6 --sched none --lr 1e-2 --lr-net 1e-3
run lr_const_high --epochs 6 --sched none --lr 3e-2 --lr-net 1e-2

# best constant run by best_val_mae
best=$(python - <<PY
import json
c={"lr_const":("3e-2","3e-3"),"lr_const_low":("1e-2","1e-3"),"lr_const_high":("3e-2","1e-2")}
s=[]
for r,(lr,lrn) in c.items():
    try: s.append((json.load(open("$P/models/spatial/%s/metrics.json"%r))["best_val_mae"],r,lr,lrn))
    except Exception as e: pass
s.sort(); print(*s[0][1:])
PY
)
set -- $best
echo "$(date -u +%H:%M:%S) best constant run: $1 (lr $2, lr-net $3)"
run lr_const_cont --epochs 6 --sched none --lr "$2" --lr-net "$3" --init-from "$1"
run lr_cos12 --epochs 12 --sched cosine
echo "$(date -u +%H:%M:%S) QUEUE DONE"
