# zero-shot K x D sweep: straight-walk diag (1.0, 1.4) + full eval (spring on all segments)
cd $(dirname "$0")/../..
W=${1:-runs/final60/walk.pt}; R=${2:-runs/final60/run.pt}; OUT=${3:-/tmp/sweep0}; mkdir -p $OUT
for K in ${KS:-5 10 20 40 80 160}; do for D in ${DS:-0.3 0.6 1.5}; do
  export K1_ANKROLL_SPRING=$K K1_ANKROLL_DAMP=$D W_WALK_M=0.11
  python3 spring_diag.py $W 1.0,1.4 8 z > $OUT/diag_K${K}_D${D}.jsonl 2>/dev/null
  python3 eval_gait56.py $W $R --n 8 2>/dev/null | grep '^{' > $OUT/eval_K${K}_D${D}.jsonl
  echo "done K=$K D=$D"
done; done
echo ALLDONE
