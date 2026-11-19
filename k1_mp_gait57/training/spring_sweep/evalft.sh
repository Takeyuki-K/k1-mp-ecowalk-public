#!/bin/bash
# evaluate fine-tuned policies at their own K (and D variants), incl. push test
cd $(dirname "$0")/../..; OUT=/tmp/sweep_ft; mkdir -p $OUT
for K in ${KS:-10 40 80 160}; do for D in ${DS:-0.6}; do
  export K1_ANKROLL_SPRING=$K K1_ANKROLL_DAMP=$D W_WALK_M=0.11
  W=runs/kft_K${K}_walk/model.pt; R=runs/kft_K${K}_run/model.pt
  python3 spring_diag.py $W 1.0,1.4 8 ft > $OUT/diag_K${K}_D${D}.jsonl 2>/dev/null
  python3 eval_gait56.py $W $R --n 8 2>/dev/null | grep '^{' > $OUT/eval_K${K}_D${D}.jsonl
  python3 eval_gait56.py $W $R --n 8 --push 2>/dev/null | grep '^{' > $OUT/push_K${K}_D${D}.jsonl
  echo "done K=$K D=$D"
done; done; echo ALLDONE
