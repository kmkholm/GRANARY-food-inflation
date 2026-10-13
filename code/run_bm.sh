#!/bin/bash
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for ds in fao wfp; do for m in snaive ar ets arima; do
  [ -f ../runs/bm_${ds}_${m}.npz ] && continue
  $PY benchmarks.py --ds $ds --model $m 2>&1 | grep -vi warn >> ../runs/bm_log.txt
done; done
echo BMDONE >> ../runs/bm_log.txt
