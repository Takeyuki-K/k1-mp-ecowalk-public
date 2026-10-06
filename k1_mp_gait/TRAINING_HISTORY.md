# v5 walk / run policies: training history and reproducibility

- **Evaluation**: `python3 eval_gait.py runs/final/walk.pt runs/final/run.pt`, `python3 push_test.py runs/final/walk.pt runs/final/run.pt 8 out/push.json accel,standrun 11,12,13,14`,
  `python3 eval_run2.py runs/final/run.pt --entry walk --speeds 2,3,4,5,5.5 --n 16 --no_entry`, `python3 eval_walk2.py runs/final/walk.pt out/sweep.json` — reproducible.
- **Training**: staged, with changes to the motor model, the hand-over banks and the episode-start mix between stages
  (G3, G8–G10 in `REPORT_GAIT.md`). Released code = final. Logs: `runs/logs/`. Intermediate checkpoints: release asset `k1-mp-ecowalk_intermediate_checkpoints_2_sprint_gait.zip` (GitHub release).
  `runs/final/init_run_v4_motorlimit.pt` = `../k1_mp_sprint/runs/final/model_motorlimit.pt`,
  `runs/final/init_walk_v4.pt` = `../k1_mp_fastwalk/runs/final/model.pt`.

Hand-over banks: `walk_bank.npz` = `python3 ../k1_mp_fastwalk/make_walk_bank.py ../k1_mp_fastwalk/runs/final/model.pt walk_bank.npz`
(fast-walk policy); `run_bank.npz` regenerated three times with `make_run_bank.py` (from rn1 it 900 at 1.9–2.3 m/s,
from rn2 final at 1.9–2.3 m/s, from rn4 at 1.75–1.9 m/s = the included file).

| policy | stage | command (from `k1_mp_gait/`, `THREADS=1`) | carried over | code / data at that time vs. final |
|---|---|---|---|---|
| run | rn1 | `python3 ppo_run2.py --stage 1 --iters 6000 --init runs/final/init_run_v4_motorlimit.pt --std_reset 0.3 --lr 1e-4 --lr_min 3e-5 --assist 0.3 --v_lo 2.0 --v_hi 3.5 --v_max 5.5 --kvel 60 --out runs/rn1` (interrupted by a machine restart) | it 900 | motor model without back-EMF braking; running speeds ≥ 2.0 m/s; P_STAND 0.25 |
| run | rn2 | `python3 ppo_run2.py --stage 1 --iters 5000 --init runs/rn1/model_it900.pt --lr 7e-5 --lr_min 3e-5 --v_lo 2.0 --v_hi 5.25 --v_max 5.5 --kvel 60 --out runs/rn2` | it 4999 | as rn1 |
| run | rn3 | `python3 ppo_run2.py --stage 1 --iters 2500 --init runs/rn2/model_now.pt --add_cad --cad ...` (cadence action, G7) | rejected | – |
| run | rn4 | `python3 ppo_run2.py --stage 2 --iters 2000 --init runs/rn2/model_final.pt --lr 5e-5 --lr_min 3e-5 --v_lo 2.0 --v_hi 5.5 --v_max 5.5 --kvel 60 --out runs/rn4` | it 1999 | + back-EMF braking (G3); speeds ≥ 2.0 |
| run | rn6 | `P_STAND=0.4 python3 ppo_run2.py --stage 2 --iters 1000 --init runs/rn4/model.pt --lr 5e-5 --lr_min 3e-5 --v_lo 1.8 --v_hi 5.5 --v_max 5.5 --kvel 60 --out runs/rn6` | it 999 → `runs/final/run.pt` | final |
| walk | wk1 | `python3 ppo_walk2.py --iters 1500 --init runs/final/init_walk_v4.pt --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/wk1` (restarted once after the machine restart) | it 1499 | motor model without back-EMF; run bank from rn1 |
| walk | wk2 | `python3 ppo_walk2.py --iters 800 --init runs/wk1/model_final.pt --lr 5e-5 --lr_min 5e-5 --kv_walk 40 --out runs/wk2` | it 799 | back-EMF; run bank from rn2 |
| walk | wk3 | `P_STOP=0.3 python3 ppo_walk2.py --iters 1000 --init runs/wk2/model_final.pt --lr 5e-5 --lr_min 5e-5 --kv_walk 40 --out runs/wk3` | it 999 | run bank from rn2 |
| walk | wk4 | `P_RUN=0.5 P_STOP=0.3 python3 ppo_walk2.py --iters 1000 --init runs/wk3/model_final.pt --lr 5e-5 --lr_min 5e-5 --kv_walk 40 --out runs/wk4` | it 999 → `runs/final/walk.pt` | final (run bank from rn4) |

(rn5 = rn4 + 40 % standing starts was evaluated and not selected; wk3 was briefly used as `final` before wk4.)
