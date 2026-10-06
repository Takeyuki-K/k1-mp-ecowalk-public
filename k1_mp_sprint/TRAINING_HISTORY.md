# Fast-running policies (v4): training history and reproducibility

- **Evaluation**: `python3 eval_sprint.py runs/final/model.pt --speeds 3,4,5,5.5 --n 24` (and `model_motorlimit.pt`) — reproducible.
- **Training** (from `k1_mp_sprint/`, `THREADS=1`; `runs/final/init_jog_v3.pt` = `../k1_mp_run/runs/final/model.pt`):

| stage | command | checkpoint carried over | code at that time vs. final |
|---|---|---|---|
| sp1 | `python3 ppo_sprint.py --stage 1 --iters 10000 --init runs/final/init_jog_v3.pt --from_run --std_reset 0.3 --lr 1e-4 --lr_min 3e-5 --assist 1.0 --v_lo 1.6 --v_hi 2.5 --v_max 5.0 --out runs/sp1` (stopped) | it 5000 | velocity kernel fixed at 25 (`--kvel` added later; default 25 = same) |
| sp2 | `python3 ppo_sprint.py --stage 2 --iters 2000 --init runs/sp1/model_5000.pt --lr 7e-5 --lr_min 3e-5 --v_lo 1.6 --v_hi 5.0 --v_max 5.0 --out runs/sp2` | it 1999 | same |
| sp4 | `python3 ppo_sprint.py --stage 2 --iters 2500 --init runs/sp2/model_final.pt --lr 7e-5 --lr_min 3e-5 --v_lo 1.6 --v_hi 5.5 --v_max 5.5 --kvel 60 --out runs/sp4` | it 2499 → `runs/final/model.pt` | final |
| sp3 | `python3 ppo_sprint.py --stage 2 --iters 2000 --init runs/sp1/model_5000.pt --lr 7e-5 --lr_min 3e-5 --v_lo 1.6 --v_hi 5.0 --v_max 5.0 --w_qd 0.5 --out runs/sp3` | it 1999 | same as sp2 |
| sp5 | `python3 ppo_sprint.py --stage 2 --iters 2500 --init runs/sp3/model_final.pt --lr 7e-5 --lr_min 3e-5 --v_lo 1.6 --v_hi 5.5 --v_max 5.5 --kvel 60 --w_qd 0.5 --out runs/sp5` | it 2499 → `runs/final/model_motorlimit.pt` | final |

The speed curriculum (`--v_hi` → `--v_max`) advances automatically; the logs (`runs/logs/sp*.txt`) contain `v_hi`.
The reference library is the v4 one (`ref_sprint_lib.npz`, heel-first; `retarget_sprint.py` with `STRIKE='heel'`
semantics of that version). Note: in v5 (`k1_mp_gait/`) the same script name builds the forefoot library.

Intermediate checkpoints of sp1–sp3: release asset `k1-mp-ecowalk_intermediate_checkpoints_2_sprint_gait.zip` (GitHub release).
