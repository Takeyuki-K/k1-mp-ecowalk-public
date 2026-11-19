#!/bin/bash
# per-K fine-tune: walk (train_env10 recipe) + run (r58a recipe), same budget for every K
cd $(dirname "$0")/../..
for K in ${KS:-10 40 80 160}; do
  ( source training/train_env10.sh; export K1_ANKROLL_SPRING=$K K1_ANKROLL_DAMP=0.6
    python3 ppo_walk4.py --iters 200 --init runs/final60/walk.pt --sym 1.0 --lr 3e-5 --lr_min 2e-5 --kv_walk 60 --w_energy 0.0035 --out runs/kft_K${K}_walk > /tmp/kft_K${K}_walk.log 2>&1 )
  echo "walk done K=$K"
  ( source training/train_run_spring.sh; export K1_ANKROLL_SPRING=$K K1_ANKROLL_DAMP=0.6
    python3 ppo_run4.py --stage 2 --iters 200 --init runs/final60/run.pt --sym 0.3 --lr 4e-5 --lr_min 3e-5 --v_lo 1.8 --v_hi 5.5 --v_max 5.5 --kvel 60 --w_max_start 1.0 --w_max 1.0 --p_brake 0.3 --w_energy 0.0003 --out runs/kft_K${K}_run > /tmp/kft_K${K}_run.log 2>&1 )
  echo "run done K=$K"
done
echo ALLDONE
