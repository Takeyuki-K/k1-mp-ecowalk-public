# v5.5 training history

All stages: `THREADS=1`, 128 robots × 32 steps per iteration, CPU (2-core VM, two trainings in parallel, ~3–6 s per iteration).
Run from `k1_mp_gait55/`. Stage-final checkpoints are in `runs/init/`.

| policy | stage | command | carried over | notes |
|---|---|---|---|---|
| run | r55a | `python3 ppo_run4.py --stage 2 --iters 2000 --init ../k1_mp_gait/runs/final/run.pt --from_v5 --lr 5e-5 --lr_min 3e-5 --v_lo 1.8 --v_hi 5.5 --v_max 5.5 --kvel 60 --w_max_start 0.5 --w_max 1.0 --out runs/r55` | it 400 → `runs/init/r55a.pt` | stopped: only ~35 % of the commanded yaw rate (S5) |
| run | r55b | `python3 ppo_run4.py --stage 2 --iters 2000 --init runs/init/r55a.pt --lr 7e-5 --lr_min 3e-5 --v_lo 1.8 --v_hi 5.5 --v_max 5.5 --kvel 60 --w_max_start 1.0 --w_max 1.0 --out runs/r55b` | it 1999 → `runs/init/r55b_final.pt` | + hip-yaw feed-forward (S5) |
| run | r55c | `P_STAND=0.4 python3 ppo_run4.py ... --init runs/init/r55b_final.pt --iters 1000 --lr 5e-5 --p_brake 0.45 --out runs/r55c` | rejected (S12) | turning at 1 rad/s lost |
| run | r55d | `python3 ppo_run4.py --stage 2 --iters 700 --init runs/init/r55b_final.pt --lr 3e-5 --lr_min 2e-5 --v_lo 1.8 --v_hi 5.5 --v_max 5.5 --kvel 60 --w_max_start 1.0 --w_max 1.0 --p_brake 0.3 --out runs/r55d` | it 699 → `runs/final/run.pt` | final |
| walk | w55a | `P_RUN=0.3 P_STOP=0.3 python3 ppo_walk3.py --iters 1500 --init ../k1_mp_gait/runs/final/walk.pt --from_v5 --lr 5e-5 --lr_min 5e-5 --kv_walk 40 --out runs/w55` | it 700 → `runs/init/w55a.pt` | stopped early: in-place turns fell after ~7 s |
| walk | w55b | `TURN_MIX=inplace P_RUN=0.3 P_STOP=0.3 python3 ppo_walk3.py --iters 2000 --init runs/init/w55a.pt --lr 1e-4 --lr_min 5e-5 --kv_walk 40 --out runs/w55b` | it 1999 → `runs/init/w55b_final.pt` | in-place curriculum (v3 T6) |
| walk | w55c | `WALK_BRAKE=1 P_BRAKE_WALK=0.3 P_RUN=0.3 P_STOP=0.3 python3 ppo_walk3.py --iters 1000 --init runs/init/w55b_final.pt --add_brake --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/w55c` | it 999 → `runs/init/w55c_final.pt` | + brake input / hard stop |
| walk | w55d | as w55c + `TURN_MIX=inplace`, `--init runs/init/w55c_final.pt --iters 800 --out runs/w55d` | rejected | interrupted at it 610 by a machine restart; in-place turns fell after 6–9 s |
| walk | w55e | `TURN_MIX=inplace LONG_INPLACE=1 EP_LEN=1000 WALK_BRAKE=1 P_BRAKE_WALK=0.3 P_RUN=0.3 P_STOP=0.3 python3 ppo_walk3.py --iters 1000 --init runs/init/w55c_final.pt --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/w55e` | **it 500** → `runs/final/walk.pt` (S11) | in-place turns of up to 12 s, 20 s episodes |

Evaluation: `python3 eval_gait55.py runs/final/walk.pt runs/final/run.pt [--push --n 16]`,
`WALK_BRAKE=1 python3 eval55.py walk runs/final/walk.pt`, `python3 eval55.py run runs/final/run.pt`,
`SEED=1 python3 push_compare.py v55 runs/final/walk.pt runs/final/run.pt`, figures: `python3 plot_gait55.py runs/final/walk.pt runs/final/run.pt`.
Note: `eval55.py run` starts running robots mid-stride from random phases (as the training resets); its survival is
lower than through the gait manager for v5 as well (v5 at 4.5 m/s: 75 %) — use `eval_gait55.py` for end-to-end numbers.
