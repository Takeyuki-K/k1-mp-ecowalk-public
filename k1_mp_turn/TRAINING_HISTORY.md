# Turning policy (v3): training history and reproducibility

- **Evaluation** of the released policy `runs/final/model.pt`: reproducible (`python3 eval_turn.py runs/final/model.pt`,
  `python3 grid_turn.py runs/final/model.pt out/grid.json`, `python3 reg_v2.py`).
- **Training** was done in stages while the environment code was being changed (decisions T4–T6 in `REPORT_TURN.md`).
  The released `k1env_turn.py` contains the final code. Earlier stages ran with code that no longer exists
  (listed per stage). They are documented exactly but cannot be replayed identically.
- Stage logs: `runs/logs/turn1.txt` … `turn6.txt`. Intermediate checkpoints: release asset
  `k1-mp-ecowalk_intermediate_checkpoints_1_turn_run_fastwalk.zip` (GitHub release) (not in the git tree, to keep it small).

All commands from `k1_mp_turn/`, 1 CPU thread per training (`THREADS=1`; two trainings ran in parallel).

| stage | command | checkpoint carried over | code at that time vs. final |
|---|---|---|---|
| turn1 | `python3 ppo_turn.py --iters 5000 --init runs/v2/model.pt --add_turn_obs --lr 1e-4 --lr_min 5e-5 --out runs/turn1` (stopped at it 370; `runs/v2/model.pt` = `../k1_mp_speed/runs/final/model.pt`) | it 300 | no in-place turning pattern (T4), no heading leash (T5), yaw reward exp(−4e²) weight 0.20 / heading exp(−4e²) weight 0.15, motion mix 30/35/25/10 % |
| turn2 | `THREADS=1 python3 ppo_turn.py --iters 4600 --init runs/turn1/model_it370.pt --lr 1e-4 --lr_min 5e-5 --out runs/turn2` (stopped) | it 800 | as turn1 |
| turn3 | `THREADS=1 python3 ppo_turn.py --iters 4000 --init runs/turn2/model_final.pt --lr 1e-4 --lr_min 5e-5 --out runs/turn3` (stopped) | it 1200 | + in-place pattern (T4), mix 25/30/35/10 %; no leash yet |
| turn4 | `THREADS=1 python3 ppo_turn.py --iters 4000 --init runs/turn3/model_final.pt --lr 1e-4 --lr_min 5e-5 --out runs/turn4` (stopped) | it 700 | + leash and sharper yaw / heading rewards (T5) = final reward |
| turn5 | `TURN_MIX=inplace THREADS=1 python3 ppo_turn.py --iters 2000 --init runs/turn4/model_final.pt --lr 1e-4 --lr_min 5e-5 --out runs/turn5` | it 1999 | final code, in-place curriculum (T6) |
| turn6 | `THREADS=1 python3 ppo_turn.py --iters 1000 --init runs/turn5/model_final.pt --lr 7e-5 --lr_min 5e-5 --out runs/turn6` | it 999 → `runs/final/model.pt` | final code |

`model_final.pt` / `model_it370.pt` are copies of the stage's `model.pt` (saved every 100 iterations) at the moment the
stage was stopped; the iteration stored in the file is given above. Stages turn5 and turn6 can be re-run exactly with
the released code from the turn4 checkpoint (in the release asset).
