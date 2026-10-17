#!/bin/bash
# Year-by-year Copernicus pull for tmux:  scripts/fetch_cmems_all.sh <blk|med> <first year> <last year> [coarsen=2]
# Downloads CHL L3/L4 + physics for one year, writes it to ~/mind_the_chl_gap/data/cmems/<basin>_<coarsen>km/<year>.zarr,
# deletes that year's raw NetCDFs, then moves on. Safe to re-run: finished years are skipped.
set -u
cd "$(dirname "$0")"
BASIN=$1; Y0=$2; Y1=$3; C=${4:-2}
for Y in $(seq $Y0 $Y1); do
  echo "=== $BASIN $Y $(date)"
  python fetch_cmems.py download --basin $BASIN --what chl phys --start $Y-01-01 --end $Y-12-31 --skip-arranged $C 2>&1 | grep -v -E "Fetching|it/s|^\s*$|INFO|WARNING"
  python fetch_cmems.py arrange --basin $BASIN --coarsen $C --years $Y $Y --delete-raw 2>&1 | grep -v -i warning
done
echo "=== done $(date)"; df -h ~ | tail -1
