# K1 + MP toes — ECO walking with variable impedance (脱力・ダンパー的な柔らかさ)

This folder is a **copy** of `k1_mp/` plus new files. The previous package (`K1_MP_walk.zip`, `k1_mp/`) is untouched;
the fixed-gain policy is kept here as `runs/final/model.pt` so both can be compared.

## New files
| file | role |
|---|---|
| `k1env_eco.py` | env: policy sets position **and** Kp, Kd of every leg joint each 20 ms (36 actions) |
| `ppo_eco.py` | PPO + `transfer_fixed_to_eco()` (starts from the fixed-gain policy with identical behaviour) |
| `test_eco_equiv.py` | proves eco env with gain scale 1 == old env (qpos diff 3.5e-8 after 5 s) |
| `energy_analysis.py` | per-joint electrical / positive / negative work |
| `eval_eco.py`, `gait_check_eco.py` | fixed vs eco comparison, survival, contact sequence, plots |
| `final_video_eco.py` | video with live stiffness bars |
| `export_onnx_eco.py`, `deploy_eco/` | ONNX + gain law for the real K1 |
| `runs/eco1/model.pt` | trained eco policy |

## How the softness is implemented
`tau = Kp (q_target − q) − Kd·qd`, exact at every 200 Hz physics step, clamped to the motor torque limit.
This is exactly the K1 motor "MIT mode" (`Goal Position`, `MIT P Gain`, `MIT D Gain` command interfaces in
`ai_sapiens_description/ros2_control/k1_rev1/*`, used by `ai_sapiens_joint_group_impedance_controller`).
- Kp = nominal × [0 … 1.5]  → 0 = fully relaxed (脱力)
- Kd = nominal × [0.05 … 3] → Kp≈0 with Kd>0 = pure damper
- Energy reward: P_elec = Σ τ²/Km² (copper loss) + Σ max(τ·qd, 0) (no regeneration). **Km is an assumed value
  (4.0 / 2.2 Nm/√W)** — replace with measured motor data, the absolute watts depend on it.

## Results (same electrical model for both, steady walking ≈0.88 m/s)
| | fixed gain (previous) | eco |
|---|---|---|
| leg electrical power | 153 W | **108 W (−29 %)** |
| CoT (electrical) | 0.49 | **0.34** |
| survival stand / walk / stand-walk-stop / +DR (64 envs, 8 s) | 100 % | 100 % |
| survival standing with random pushes | 73 % | **88 %** |
| survival stand-walk-stop with DR + pushes | 86 % | **98 %** |
| heel-first touchdowns | 19/20 | 17/18 |
| MP bending at push-off | 43° | 55° |

What the policy learned (`out/eco_gain_profile.png`):
- **hip pitch relaxes to ~0.45× at the end of stance** → the leg is released into swing like a pendulum
- **knee relaxes to ~0.3× at toe-off** → passive knee flexion during push-off (as in human gait)
- **ankle stiffens to 1.1–1.25×** around heel strike / in stance → stiff lever for rollover + push-off;
  ankle power rises slightly (29→33 W) while hip power halves (43→20 W): work moved from hip to ankle push-off
- hip roll relaxed only in swing; in stance it must hold the pelvis → still the largest consumer (34 W)

## Open issues / next steps
1. **Hip roll holding torque** (static copper loss) is not fixable by softness: try a parallel spring on hip roll,
   narrower step width / lateral CoM sway in the reward, or a gear-ratio change.
2. **Knee negative work** (26 W absorbed, was 30 W) → candidate for a physical damper / clutch-spring (series elastic knee).
3. Gains switch every 20 ms: check the K1 motor driver latency for gain updates on the real robot.
4. Km, gearbox friction and back-drivability are not modelled; a relaxed joint on a high-friction gearbox does not swing freely.
