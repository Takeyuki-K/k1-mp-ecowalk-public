# Fast-walk policy (v4): training history and reproducibility

- **Evaluation**: `python3 eval_fw.py runs/final/model.pt out/sweep.json` (and `--v2` for the baseline) — reproducible.
- **Training** (released code, `THREADS=1`, from `k1_mp_fastwalk/`; `runs/v2/model.pt` = `../k1_mp_speed/runs/final/model.pt`):

| stage | command | checkpoint carried over |
|---|---|---|
| fw1 | `python3 ppo_speed.py --iters 3000 --init runs/v2/model.pt --lr 1e-4 --lr_min 5e-5 --out runs/fw1` | it 2999 |
| fw2 | `python3 ppo_speed.py --iters 1500 --init runs/fw1/model_final.pt --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/fw2` | it 1499 → `runs/final/model.pt` |

Both stages used the released code (`retarget_speed.py` library 0.30–1.95 m/s, `--kv_walk` added before fw2), so this
chain can be re-run as written (results vary with RL randomness). Logs: `runs/logs/fw1.txt`, `fw2.txt`.

Intermediate checkpoint fw1: release asset `k1-mp-ecowalk_intermediate_checkpoints_1_turn_run_fastwalk.zip` (GitHub release).
