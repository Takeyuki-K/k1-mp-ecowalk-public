# Jog policy (v3): training history and reproducibility

- **Evaluation** of `runs/final/model.pt`: reproducible (`python3 eval_run.py runs/final/model.pt [--push] [--dr] [--hard]`).
- **Training**: staged, with code / reference changes between stages (R7, R8, R5 in `REPORT_RUN.md`). The released code
  is the final one; `ref_run.npz` is the final (5° lean) reference. Logs: `runs/logs/run1.txt` … `run4.txt`.
  Intermediate checkpoints: release asset `k1-mp-ecowalk_intermediate_checkpoints_1_turn_run_fastwalk.zip` (GitHub release).

| stage | command (from `k1_mp_run/`, `THREADS=1`) | checkpoint carried over | code at that time vs. final |
|---|---|---|---|
| run1 | `python3 ppo_run.py --stage 1 --iters 4000 --init ../k1_mp_eco/runs/eco1/model.pt --std_reset 0.3 --lr 1e-4 --lr_min 3e-5 --assist 2.0 --out runs/run1` (stopped) | it 300 | reference without trunk lean (`ref_run_nolean`, not included), no flight / vertical-velocity reward (flags did not exist) |
| run2 | `python3 ppo_run.py --stage 1 --iters 3000 --init runs/run1/model_final.pt --lr 1e-4 --lr_min 3e-5 --assist 0.5 --kv 4 --w_flight 1.0 --out runs/run2` (stopped) | it 1800 | reference without trunk lean |
| run3 | `python3 ppo_run.py --stage 1 --iters 2500 --init runs/run2/model_final.pt --lr 1e-4 --lr_min 3e-5 --assist 0.3 --kv 4 --w_flight 1.0 --out runs/run3` (stopped) | it 1500 | final code and reference |
| run4 | `python3 ppo_run.py --stage 2 --iters 1500 --init runs/run3/model_final.pt --lr 7e-5 --lr_min 3e-5 --kv 4 --w_flight 1.0 --out runs/run4` | **it 1000** (`model_1000.pt`, R12) → `runs/final/model.pt` | final |

Note: one early run1 attempt with two trainings sharing 2 cores (4 threads) was stopped after 50 iterations because of
CPU over-subscription; it was restarted as above.
