# v5.6 training history (walking policy only; running policy = v5.5 `runs/init/run_v55.pt`)

Run from `k1_mp_gait56/`, `THREADS=1–2`, 128 robots × 32 steps per iteration, CPU. Reference: `python3 make_ref56.py`.
Common environment variables: `LONG_INPLACE=1 EP_LEN=1000 WALK_BRAKE=1 P_BRAKE_WALK=0.3 P_RUN=0.3`.

| stage | extra env | command | carried over | notes |
|---|---|---|---|---|
| w56a | `P_STOP=0.3` | `python3 ppo_walk4.py --iters 1500 --init runs/init/walk_v55.pt --from_v55 --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/w56a` | it 900 → `runs/init/w56a_level.pt` | strictly level pelvis; stopped by a machine restart at it ~980; showed the energy cost (S3) |
| w56b | `ROLL_DEADBAND=0.045 IP_WIDE=1 P_STOP=0.3` | `python3 ppo_walk4.py --iters 1200 --init runs/init/w56a_level.pt --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/w56b` | it 1000 → `runs/init/w56b_model_1000.pt` | human-level sway allowed, 21 cm stance when stepping in place |
| w56c | `ALIGN_CHECK=1 ROLL_DEADBAND=0.045 IP_WIDE=1 P_STOP=0.4` | `python3 ppo_walk4.py --iters 1200 --init runs/init/w56b_model_1000.pt --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/w56c` | it 1199 → **`runs/final/walk.pt`** | + alignment reward while stepping to stop (its foot-offset measurement had a frame bug, REPORT S9; not retrained) |
| w56d | `PLACE_FF=1 ALIGN_CHECK=1 ROLL_DEADBAND=0.045 IP_WIDE=1 P_STOP=0.4` | `python3 ppo_walk4.py --iters 1000 --init runs/final/walk.pt --lr 5e-5 --lr_min 4e-5 --kv_walk 40 --out runs/w56d` | rejected → `runs/init/w56d_model.pt` | stop from walking fell 8/8 (S11) |
| w56e | `ROLL_DEADBAND=0.045 IP_WIDE=1 PLACE_FF=1 PLACE_GAIN=1.5 RESTANCE_ENV=1 IP_RHYTHM=1 P_STAGGER=0.5 W_LIFT=0.3 W_PIVOT=0.05 P_STOP=0.4` | `python3 ppo_walk4.py --iters 1500 --init runs/final/walk.pt(w56c) --lr 7e-5 --lr_min 5e-5 --kv_walk 40 --out runs/w56e` | → `runs/init/w56e_model.pt` | quick stop + re-stance in training, human rhythm; still shuffled, −0.6 rad/s fell 8/8 (REPORT S15) |
| w56f | `ROLL_DEADBAND=0.045 IP_WIDE=1 PLACE_FF=1 PLACE_GAIN=1.5 RESTANCE_ENV=1 IP_RHYTHM=1 P_STAGGER=0.5 W_LIFT=1.0 LIFT_PEN=1.0 W_PIVOT=0.15 TURN_MIX=inplace P_STOP=0.4` | `python3 ppo_walk4.py --iters 1500 --init runs/init/w56e_model.pt ... --out runs/w56f` | → `runs/init/w56f_model.pt` | both feet step; +1.0 rad/s fell 8/8 |
| w56g | w56f + `TURN_POS_P=0.7 P_FAST_IP=0.5` | `python3 ppo_walk4.py --iters 1000 --init runs/init/w56f_model.pt ... --out runs/w56g` | → `runs/init/w56g_model.pt` | all turn rates OK, weak under pushes |
| w56h | `ROLL_DEADBAND=0.045 IP_WIDE=1 PLACE_FF=1 PLACE_GAIN=1.5 RESTANCE_ENV=1 IP_RHYTHM=1 P_STAGGER=0.5 W_LIFT=1.0 LIFT_PEN=1.0 W_PIVOT=0.15 TURN_POS_P=0.6 P_FAST_IP=0.3 PUSH_P=0.01 P_STOP=0.4` | `python3 ppo_walk4.py --iters 1500 --init runs/init/w56g_model.pt --lr 7e-5 --lr_min 4e-5 --kv_walk 40 --out runs/w56h` | it 1499 → `runs/init/w56h_model.pt` (v5.6.1) | REPORT §4 |
| w56i | w56h + `RS_OBS=1 W_RSPLACE=3 RS_MAX=1 ALIGN_MAX=4 RS_DX=0.03 RS_DYAW_DEG=6 RS_DSEP=0.04 PLACE_LAT=1 P_STAGGER=0.6 PUSH_P=0.008` | `python3 ppo_walk4.py --iters 1500 --init runs/init/w56h_model.pt --add_rs_obs ... --out runs/w56i` | rejected → `runs/init/w56i_model.pt` | REPORT S20 |
| w56j (v5.6.2) | `ROLL_DEADBAND=0.045 IP_WIDE=1 PLACE_FF=1 PLACE_GAIN=1.5 RESTANCE_ENV=1 IP_RHYTHM=1 P_STAGGER=0.5 W_LIFT=1.0 LIFT_PEN=1.0 W_PIVOT=0.15 TURN_POS_P=0.5 P_FAST_IP=0.3 P_STOP=0.4 PUSH_P=0.01` | `python3 ppo_walk4.py --iters 1500 --init runs/init/w56h_model.pt --sym 1.0 --lr 7e-5 --lr_min 4e-5 --kv_walk 40 --out runs/w56j` | it 1499 → **`runs/final/walk.pt`** (`runs/init/w56j_model.pt`) | mirror-symmetry loss (REPORT §5) |
| w56k | `ROLL_DEADBAND=0.045 IP_WIDE=1 PLACE_FF=1 PLACE_GAIN=1.5 RESTANCE_ENV=1 IP_RHYTHM=1 P_STAGGER=0.5 W_LIFT=1.0 LIFT_PEN=1.0 W_PIVOT=0.15 TURN_POS_P=0.5 P_FAST_IP=0.3 P_STOP=0.4 PUSH_P=0.012` | `python3 ppo_walk4.py --iters 1500 --init runs/init/w56j_model.pt --sym 1.0 --lr 6e-5 --lr_min 4e-5 --kv_walk 40 --out runs/w56k` | rejected → `runs/init/w56k_model.pt` | stop from 4.5 m/s fell 6/8 (S23) |

Deployment settings (set in `gait56.py`, w56h / w56j): `IP_WIDE`, `ROLL_DEADBAND=0.045` (reward only), `IP_RHYTHM=1`, quick stop +
re-stance (`RESTANCE=1`, 1 s, thresholds 3 cm / 6° / 4 cm, ≤ 4 steps, once), `PLACE_FF=1`, `PLACE_GAIN=1.5`, `PLACE_GAIN_YAW=1.0`, `PLACE_LAT=1`.
(Stage w56c settings: `ALIGN_CHECK=0 PLACE_FF=1 PLACE_GAIN=1.5`, `RESTANCE=0 IP_RHYTHM=0`.) For the single-env evaluation set the same variables:
`ALIGN_CHECK=0 PLACE_FF=1 PLACE_GAIN=1.5 ROLL_DEADBAND=0.045 IP_WIDE=1 python3 eval56.py runs/final/walk.pt`.
Power: `python3 eval_power56.py runs/final/walk.pt runs/final/run.pt` (v5.5: `cd ../k1_mp_gait55 && PYTHONPATH=. python3 <copy of eval_power55.py outside k1_mp_gait56> runs/final/walk.pt runs/final/run.pt`); arms: `python3 eval_arms56.py runs/final/walk.pt`; mirror check: `python3 mirror.py runs/final/walk.pt`.
In-place turning: `python3 eval_inplace56.py runs/final/walk.pt runs/final/run.pt [--push --n 16]`; turning rhythm from the capture: `python3 mocap_turn_rhythm.py <motion_and_confidence.npz>`.
Final stance (ground truth): `python3 eval_stance56.py runs/final/walk.pt runs/final/run.pt`; stopping under pushes: `python3 eval_stop_push56.py runs/final/walk.pt runs/final/run.pt`.
End-to-end: `python3 eval_gait56.py runs/final/walk.pt runs/final/run.pt [--push --n 16]`.
Videos: `MUJOCO_GL=osmesa python3 video_gait56.py runs/final/walk.pt runs/final/run.pt straight|profile|inplace_stop|brake out/X.mp4`.
