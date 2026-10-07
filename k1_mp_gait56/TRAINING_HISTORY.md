# v5.6 training history (walking policy only; running policy = v5.5 `runs/init/run_v55.pt`)

Run from `k1_mp_gait56/`, `THREADS=1–2`, 128 robots × 32 steps per iteration, CPU. Reference: `python3 make_ref56.py`.
Common environment variables: `LONG_INPLACE=1 EP_LEN=1000 WALK_BRAKE=1 P_BRAKE_WALK=0.3 P_RUN=0.3`.

| stage | extra env | command | carried over | notes |
|---|---|---|---|---|
| w56a | `P_STOP=0.3` | `python3 ppo_walk4.py --iters 1500 --init runs/init/walk_v55.pt --from_v55 --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/w56a` | it 900 → `runs/init/w56a_level.pt` | strictly level pelvis; stopped by a machine restart at it ~980; showed the energy cost (S3) |
| w56b | `ROLL_DEADBAND=0.045 IP_WIDE=1 P_STOP=0.3` | `python3 ppo_walk4.py --iters 1200 --init runs/init/w56a_level.pt --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/w56b` | it 1000 → `runs/init/w56b_model_1000.pt` | human-level sway allowed, 21 cm stance when stepping in place |
| w56c | `ALIGN_CHECK=1 ROLL_DEADBAND=0.045 IP_WIDE=1 P_STOP=0.4` | `python3 ppo_walk4.py --iters 1200 --init runs/init/w56b_model_1000.pt --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/w56c` | it 1199 → **`runs/final/walk.pt`** | + alignment reward while stepping to stop |
| w56d | `PLACE_FF=1 ALIGN_CHECK=1 ROLL_DEADBAND=0.045 IP_WIDE=1 P_STOP=0.4` | `python3 ppo_walk4.py --iters 1000 --init runs/final/walk.pt --lr 5e-5 --lr_min 4e-5 --kv_walk 40 --out runs/w56d` | rejected → `runs/init/w56d_model.pt` | stop from walking fell 8/8 (S11) |

Deployment settings (set in `gait56.py`): `IP_WIDE`, `ROLL_DEADBAND=0.045` (reward only), `PLACE_FF=1`, `PLACE_GAIN=1.5`,
`ALIGN_CHECK=0`. For the single-env evaluation set the same variables:
`ALIGN_CHECK=0 PLACE_FF=1 PLACE_GAIN=1.5 ROLL_DEADBAND=0.045 IP_WIDE=1 python3 eval56.py runs/final/walk.pt`.
End-to-end: `python3 eval_gait56.py runs/final/walk.pt runs/final/run.pt [--push --n 16]`.
Videos: `MUJOCO_GL=osmesa python3 video_gait56.py runs/final/walk.pt runs/final/run.pt straight|profile|inplace_stop|brake out/X.mp4`.
