# K1 MP Eco-Walk

**Passive spring toe (MP) joints × human-gait imitation × learned relaxation → energy-efficient humanoid walking (MuJoCo simulation)**

受動バネのMP関節（つま先関節） × 人の歩行の模倣 × 学習による脱力で、ヒューマノイドの省エネ歩行を実現する（MuJoCoシミュレーション）

Idea & direction: **Takeyuki-K** · Implementation: Claude (Anthropic) · License: Apache-2.0 (see [NOTICE](NOTICE)) · Simulation only

![3-way comparison: ROBOTIS walk_default vs MP + imitation vs MP + imitation + relaxation, live power](media/comparison_3way.gif)

> **Simulation only (MuJoCo).** Independent personal research, **not affiliated with or endorsed by ROBOTIS.**
> MuJoCoによるシミュレーションのみ（実機ではありません）。個人の独自研究であり、ROBOTIS社とは無関係で、同社の承認・推奨を受けたものではありません。

---

## Idea / アイディア

Instead of making a humanoid walk with stiff, always-on motors, let **natural physics** do part of the work:

1. **Passive MP (toe) joint** – a motor-less torsion spring that bends when the body weight rolls onto the toes and returns the toe flat in swing. The spring is sized so that it can never lift the foot by itself.
2. **Imitation of human walking data** – only the hip→ankle vector is imitated (scaled by the leg-length ratio), the rest is solved by IK, so the robot learns a human heel-strike → foot-flat → toe-off rollover.
3. **Learned relaxation (脱力)** – the policy chooses joint stiffness Kp and damping Kd every 20 ms (the K1 motors' "MIT mode"), and is rewarded for low electrical power. It learned by itself to relax the hip and knee at push-off / swing (pendulum-like swing, passive knee flexion) and to stiffen the ankle for push-off.

モーターで固く制御し続けるのではなく、自然の力（つま先の受動バネ・人の歩き方・脚の振り子運動）を使って歩かせる、というアイディアです。

## Result / 結果

Three controllers on the same K1 model family, same MuJoCo physics (dt 2 ms, 50 Hz control), same start, straight walking at the same speed (~0.92 m/s), same electrical power model.
同一物理条件・同一速度・同一電力モデルで3つの制御を比較:

| | controller | speed | avg. electrical power | CoT (electrical) | vs ① |
|---|---|---|---|---|---|
| ① | ROBOTIS public `walk_default` policy, original flat foot | 0.92 m/s | 182 W | 0.57 | – |
| ② | MP joint + human-gait imitation, fixed gains | 0.90 m/s | 158 W | 0.50 | −13 % |
| ③ | **MP joint + imitation + learned relaxation** | 0.91 m/s | **114 W** | **0.36** | **−37 %** |

| ② heel strike → flat → toe-off, passive MP bending (slow ×0.25) | ③ learned stiffness within one stride (slow ×0.25) |
|---|---|
| ![heel-to-toe close-up](media/heel_toe_slowmo.gif) | ![stiffness bars](media/eco_stiffness_slowmo.gif) |

Full videos / 動画（MP4）: [`media/K1_3way_energy_comparison.mp4`](media/K1_3way_energy_comparison.mp4) (3-up comparison with live per-joint power),
[`media/K1_MP_heel_toe_walk.mp4`](media/K1_MP_heel_toe_walk.mp4) (②, heel-to-toe close-up),
[`media/K1_MP_eco_walk.mp4`](media/K1_MP_eco_walk.mp4) (③, live stiffness bars).

Other findings (simulation):
- ③ keeps the heel-strike → flat → toe-off sequence (17/18 heel-first touchdowns); MP bends up to ~55° at push-off.
- Robustness improved with relaxation: standing with random pushes 73 % → 88 % survival, stand→walk→stop with domain randomisation + pushes 86 % → 98 % (② vs ③, 64 envs, 8 s).
- The ranking ① > ② > ③ also holds for positive mechanical work (independent of the assumed motor constant): 53 W > 49 W > 44 W.

### Honest limitations / 注意点
- **① is a general-purpose policy** (omnidirectional velocity tracking, made for the real robot). ②③ are specialised for straight walking at one speed. The −37 % is valid for this condition only.
- ②③ have no heading control yet (lateral drift 0.8–1.8 m over 10 m).
- The electrical model uses an **assumed** motor constant Km (4.0 / 2.2 Nm/√W); gearbox friction, driver losses and electronics are not modelled. Compare the numbers relatively, not with other robots.
- No real-robot experiment yet. MP spring, toe mass and contact parameters are estimates.

## Speed command / 速度指令への追従（v2）

One policy now follows a changing forward speed command **0.30 – 1.35 m/s** while keeping the MP toes, the human-gait
imitation and the learned relaxation. Slower = shorter steps and lower cadence, faster = longer steps and higher cadence
(human walk-ratio law). Details and every design decision: [k1_mp_speed/REPORT_SPEED.md](k1_mp_speed/REPORT_SPEED.md).
1つのポリシーで、0.30〜1.35 m/sの速度指令に追従します（遅いほど小股・低ピッチ、速いほど大股・高ピッチ）。

![speed command: ROBOTIS walk_default vs MP + imitation + relaxation on the same speed profile](media/speed_command_comparison.gif)

| command | 0.30 | 0.45 | 0.60 | 0.75 | 0.90 | 1.00 | 1.20 | 1.35 m/s |
|---|---|---|---|---|---|---|---|---|
| measured speed | 0.31 | 0.45 | 0.61 | 0.76 | 0.91 | 1.02 | 1.19 | 1.30 |
| step length (m) | 0.24 | 0.29 | 0.34 | 0.39 | 0.43 | 0.46 | 0.50 | 0.55 |
| leg power vs ROBOTIS walk_default | −8 % | −32 % | −33 % | −36 % | −37 % | −36 % | −36 % | −38 % |

No falls, heel contact at every touchdown, heading drift ≤ 4.3°, 87.5 % survival with random commands + pushes + domain
randomisation. Above the training range the robot saturates at ~1.45 m/s (step length ≈ 0.55 m) without falling — faster
needs a running gait (future work). Full video: [`media/K1_speed_command_comparison.mp4`](media/K1_speed_command_comparison.mp4).

## Turning & safe stop / 旋回・その場旋回・安全停止（v3）

The speed-command walker now also follows a **yaw-rate command**: walking turns, **in-place turns** and a **safe stop**
(decelerate first, then bring the feet together). Details and every design decision:
[k1_mp_turn/REPORT_TURN.md](k1_mp_turn/REPORT_TURN.md).
歩行中の旋回・その場旋回・停止指令での安全停止に対応しました。

![turning demo: walking turns, stop command, in-place turns](media/K1_turning.gif)

| test (8 robots, from standing) | result |
|---|---|
| walking turn 0.6 m/s, ±0.5 rad/s / 1.0 m/s, 1.0 rad/s | 100 % survival, yaw-rate error ≤ 0.3 % |
| in-place turn ±0.3 / ±0.6 / 0.8 rad/s | 100 % survival, drift ≤ 2.8 cm/s |
| in-place turn 1.0 rad/s | 87.5 % (limit) |
| stop command from 1.2 m/s / from a walking turn / from an in-place turn | 8/8 standing after 3.3 / 2.3 / 1.7 s |
| straight walking 0.3–1.35 m/s (regression) | no falls, speed error ≤ 7.5 %, but **+16–27 % power at 0.9–1.35 m/s vs v2** |

Full video: [`media/K1_turning.mp4`](media/K1_turning.mp4).

## Running (jog) / 走行（v3）

Running imitated from **CMU motion-capture data** (subject 16, "run/jog"), with the landing changed to **heel first**
as requested; stability and impact absorption weighted over power. Details: [k1_mp_run/REPORT_RUN.md](k1_mp_run/REPORT_RUN.md).
CMUの走行モーションを模倣し、かかと着地・安定性・衝撃吸収を重視して学習しました。

![running: CMU-derived reference vs learned policy, slow motion](media/K1_running.gif)

| | learned running |
|---|---|
| speed / cadence | 1.54 m/s, 174 steps/min |
| flight phase | 29 % of the time (human reference 35 %) |
| touchdown | heel only, 100 % |
| peak foot force | 2.3 body weights (human running ≈ 2.5) |
| trunk | 1.8° forward lean, ±0.36° |
| robustness | 100 % survival with pushes (±0.6 m/s) + friction/mass randomisation (24 robots, 16 s) |
| relaxation | only in swing (Kp ≈ 0.6 of nominal), stiff again before touchdown |

Full video: [`media/K1_running.mp4`](media/K1_running.mp4).
Mocap: *The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.*

## Fast walking & fast running / 速歩きと高速走行（v4）

**Fast walking** (`k1_mp_fastwalk/`): 1.35–1.65 m/s is covered by walking with longer steps (energy reward kept);
**fast running** (`k1_mp_sprint/`): CMU running as a style prior, automatic speed curriculum to **5 m/s**, stability
and speed first, power only measured. Details: [REPORT_FASTWALK.md](k1_mp_fastwalk/REPORT_FASTWALK.md),
[REPORT_SPRINT.md](k1_mp_sprint/REPORT_SPRINT.md).
1.35〜1.65 m/sは歩幅を伸ばした速歩き（省エネ報酬あり）、それ以上は走行で最高5.2 m/sに到達しました。

![fast running 2 → 5.5 m/s command, slow motion at top speed](media/K1_sprint.gif)

| | result |
|---|---|
| fast walk at 1.63 m/s | 216 W, CoT 0.38 (jog at 1.54 m/s: 342 W → walking ≈ 40 % less) |
| fast running, top speed | **5.21 m/s**, 100 % survival (24 robots; also with pushes + randomisation) |
| fast running gait at top speed | 239 steps/min, step 1.31 m, 45 % flight, heel-first 99 %, trunk +4.5° ± 1.0° |
| **motor-speed limit (URDF 11.5 rad/s)** | exceeded 28 % of the time at 5.2 m/s; a limit-respecting policy reaches **4.6 m/s** |

Full video: [`media/K1_sprint.mp4`](media/K1_sprint.mp4).

## Walk ⇄ run switching, forefoot running within motor limits / 歩行⇄走行の切替（v5）

Running now lands on the **mid/forefoot** (heel strike penalised: 0 % heel-first), the leg motors follow the
**K1 URDF torque–speed limits** (96.9 Nm, 11.5 rad/s, physical model with back-EMF braking), the robot can start
running **from standing**, and a gait manager switches **walk → run** when the command exceeds 1.8 m/s
(reference jumps 1.6 → 2.0 m/s) and **run → walk** when slowing down. Details: [REPORT_GAIT.md](k1_mp_gait/REPORT_GAIT.md).
走行はミッドフット〜前足着地、モーター仕様の速度限界内、立位からの走り出し、歩行⇄走行の自動切替に対応しました。

![stand → walk (0.15 m/s steps) → run → walk → stop](media/K1_walk_to_run.gif)

| | result (MuJoCo) |
|---|---|
| stand → walk 0.3→1.65 (+0.15 m/s) → run → 5.5 cmd → walk → stop | 100 % (8 robots), 93.8 % with random pushes (32 robots) |
| stand → run 3 m/s directly → 4.5 → walk → stop | 100 %, 93.8 % with pushes |
| top running speed within the motor limit | **4.9 m/s** (v4 heel strike: 4.6 m/s) |
| touchdowns while running | 84–99 % forefoot first, rest midfoot, **0 % heel first** |
| at the switching speed | walking 219 W (1.6 m/s, CoT 0.39) vs running 650 W (1.9 m/s, CoT 1.04) |

Videos: [`media/K1_walk_to_run.mp4`](media/K1_walk_to_run.mp4), [`media/K1_stand_to_run.mp4`](media/K1_stand_to_run.mp4).

## Power model / 消費電力の計算
At every 2 ms physics step for every joint: `τ = clip(Kp(q* − q) − Kd·q̇, motor limit)`

`P = Σ τ²/Km² + Σ max(τ·q̇, 0)` — copper (Joule) loss + positive mechanical work, no regeneration.
CoT = P / (m g v). The Joule + mechanical decomposition follows the common practice in legged-robot energetics (e.g. Seok et al., MIT Cheetah, IEEE/ASME T-Mech 2015).

## Repository layout
| path | content |
|---|---|
| `ai_sapiens/ai_sapiens_description/` | K1 model (ROBOTIS, Apache-2.0) + **K1 with MP joints** (`k1_mp.xml`, `k1_mp.urdf`, split foot meshes) |
| `k1_mp/` | MP joint generator, human-gait retargeting, imitation (stage 1) + RL (stage 2), policy `runs/final/model.pt`, ONNX in `deploy/` |
| `k1_mp_eco/` | variable-impedance env & PPO, eco policy `runs/eco1/model.pt`, energy analysis, ONNX + gain law in `deploy_eco/` (see [README_ECO.md](k1_mp_eco/README_ECO.md)) |
| `k1_mp_speed/` | **v2 speed command**: speed-scaled human-gait library, env, PPO, evaluation, report, policy `runs/final/model.pt`, ONNX in `deploy_speed/` |
| `k1_mp_turn/` | **v3 turning**: yaw-rate command, walking / in-place turning, safe stop; policy `runs/final/model.pt`, report |
| `k1_mp_run/` | **v3 running**: CMU mocap (BVH) retargeting with heel-first landing, running env with assist curriculum, policy `runs/final/model.pt`, report |
| `k1_mp_fastwalk/` | **v4 fast walking** to 1.65 m/s (walking library to 1.95 m/s), policy `runs/final/model.pt` |
| `k1_mp_sprint/` | **v4 fast running**: CMU 09_04 speed library 1.6–5.5 m/s, speed curriculum, policies `runs/final/model.pt` (speed priority) and `model_motorlimit.pt` |
| `k1_mp_gait/` | **v5**: forefoot running with motor torque–speed model, stand → run, walk ⇄ run gait manager (`gait.py`), policies `runs/final/walk.pt`, `runs/final/run.pt` |
| `k1_compare/` | 3-way comparison: recording, power evaluation, figures, video composition; results in `results/` |
| `media/` | videos and key figure |

## Reproduce / 再現
```bash
pip install -r requirements.txt          # CPU is enough (2 cores were used)
export MUJOCO_GL=osmesa                   # or egl, for headless rendering
./scripts/fetch_external.sh               # gait data (CC BY 4.0) + ROBOTIS upstream (for ①)

cd k1_mp
python3 gen_model.py && python3 retarget.py && python3 test_mp.py   # model, reference, spring test
python3 ppo.py --stage 1 --iters 1100 --out runs/s1                     # imitation   (~1 h)
python3 ppo.py --stage 2 --iters 2500 --init runs/s1/model.pt --lr 1e-4 --out runs/s2   # RL (~1 h)

cd ../k1_mp_eco
python3 test_eco_equiv.py                                                # eco env == fixed env when gains = 1
python3 ppo_eco.py --stage 2 --iters 2500 --init runs/final/model.pt --from_fixed --lr 1e-4 --out runs/eco1

cd ../k1_compare
python3 record3.py robotis && python3 record3.py mp_fixed && python3 record3.py eco
for k in robotis mp_fixed eco; do python3 render_raw.py $k; done
python3 compose3.py                                                      # -> K1_3way_energy_comparison.mp4
cd ../k1_mp_speed                                                       # v2: speed command
python3 retarget_speed.py                                               # speed library 0.30-1.65 m/s
python3 ppo_speed.py --iters 3500 --init ../k1_mp_eco/runs/eco1/model.pt --from_eco ...   # see REPORT_SPEED.md (stages A-G)
python3 eval_speed.py runs/final/model.pt
cd ../k1_mp_turn                                                        # v3: turning (needs ../k1_mp/data from fetch_external.sh)
python3 retarget_speed.py                                               # library incl. stepping in place (v = 0)
python3 ppo_turn.py --iters 5000 --init ../k1_mp_speed/runs/final/model.pt --add_turn_obs ...   # stages: REPORT_TURN.md
python3 eval_turn.py runs/final/model.pt && python3 video_turn.py runs/final/model.pt out/K1_turning.mp4
cd ../k1_mp_run                                                         # v3: running
python3 retarget_run.py                                                 # CMU 16_35 -> ref_run.npz
python3 ppo_run.py --stage 1 --iters 330 --init ../k1_mp_eco/runs/eco1/model.pt --std_reset 0.3 --assist 2.0 --out runs/run1
#   then --kv 4 --w_flight 1 (run2, run3) and --stage 2 (run4); see REPORT_RUN.md
python3 eval_run.py runs/final/model.pt --push --dr && python3 video_run.py out/rec.npz out/K1_running.mp4
cd ../k1_mp_fastwalk                                                    # v4: fast walking
python3 retarget_speed.py && python3 ppo_speed.py --iters 3000 --init ../k1_mp_speed/runs/final/model.pt --lr 1e-4 --lr_min 5e-5 --out runs/fw1
python3 ppo_speed.py --iters 1500 --init runs/fw1/model.pt --kv_walk 40 --lr 7e-5 --lr_min 5e-5 --out runs/fw2 && python3 eval_fw.py runs/fw2/model.pt out/sweep.json
cd ../k1_mp_sprint                                                      # v4: fast running
python3 retarget_sprint.py
python3 ppo_sprint.py --stage 1 --iters 5000 --init runs/final/init_jog_v3.pt --from_run --std_reset 0.3 --assist 1.0 --out runs/sp1
#   then sp2 (--stage 2), sp4 (--kvel 60 --v_hi 5.5 --v_max 5.5); motor-limit branch: --w_qd 0.5 (see REPORT_SPRINT.md)
python3 eval_sprint.py runs/final/model.pt --speeds 3,4,5,5.5 --n 24 && python3 video_sprint.py runs/final/model.pt out/K1_sprint.mp4
cd ../k1_mp_fastwalk && python3 make_walk_bank.py runs/final/model.pt ../k1_mp_gait/walk_bank.npz   # v5
cd ../k1_mp_gait && python3 retarget_sprint.py                               # forefoot running library
python3 ppo_run2.py --stage 1 --iters 6000 --init runs/final/init_run_v4_motorlimit.pt --std_reset 0.3 --assist 0.3 --v_hi 3.5 --v_max 5.5 --kvel 60 --out runs/rn1
python3 make_run_bank.py runs/rn1/model.pt run_bank.npz
python3 ppo_walk2.py --iters 1500 --init runs/final/init_walk_v4.pt --kv_walk 40 --out runs/wk1    # then robustness stages, see REPORT_GAIT.md
python3 eval_gait.py runs/final/walk.pt runs/final/run.pt && python3 video_gait.py runs/final/walk.pt runs/final/run.pt accel out/K1_walk_to_run.mp4
```
The ROBOTIS `walk_default` policy is **not redistributed**; it is loaded from the upstream clone in `external/`.
Its observation (390 = 78 × 5 history) and action pipeline were reproduced from the upstream C++ sim2real code; it tracks 0.5/0.7/0.9 m/s commands at 0.494/0.703/0.901 m/s in our setup.

## Authorship / 役割分担

**Idea and direction — Takeyuki-K (アイディア・方向付け)**
- The concept: passive spring MP (toe) joints × imitation of human walking × relaxation of the actuators for an
  energy-efficient humanoid (MP関節 × 人の模倣 × 脱力による省エネ歩行という構想)
- Key specifications: motor-less spring MP joint that bends when the weight moves forward but cannot lift the foot;
  heel → foot-flat → toe contact sequence; imitate only the hip-to-ankle motion and solve the rest by IK; separate RL
  for the stand-to-walk transition; slower walking with shorter steps; running treated as a separate gait;
  turning both while walking and on the spot, safe stop on a stop command, relaxation used as shock absorption in
  turns; running: human imitation and stability first, heel landing from the flight phase with the heel as a pivot
  so the body rotates forward without the upper body collapsing, power secondary ("eco is a result");
  1.35–1.6 m/s handled by fast walking with longer steps and the energy reward; fast running aiming at 5 m/s with
  stability and speed first and power evaluated only as a result; running on the mid/forefoot (heel strike was a
  mistake), motor speed limits respected for sim2real, stand → run, and walk → run switching above a threshold
  with an allowed speed jump
- Evaluation policy: comparison with the public ROBOTIS policy at the same speed and conditions; staged development
  with branches so the walking result is preserved

**Implementation — Claude (Anthropic) (実装)**
- All code, the modified robot model (MJCF/URDF, split meshes), gait retargeting, reinforcement learning, evaluation,
  videos, figures and documentation in this repository were produced by Claude under the direction above.
- Design decisions taken by Claude within that direction are recorded with reasons in
  [k1_mp_speed/REPORT_SPEED.md](k1_mp_speed/REPORT_SPEED.md) (D1–D12), [k1_mp_turn/REPORT_TURN.md](k1_mp_turn/REPORT_TURN.md) (T1–T9)
  [k1_mp_run/REPORT_RUN.md](k1_mp_run/REPORT_RUN.md) (R1–R12), [k1_mp_fastwalk/REPORT_FASTWALK.md](k1_mp_fastwalk/REPORT_FASTWALK.md) (F1–F4)
  [k1_mp_sprint/REPORT_SPRINT.md](k1_mp_sprint/REPORT_SPRINT.md) (S1–S12)
  and [k1_mp_gait/REPORT_GAIT.md](k1_mp_gait/REPORT_GAIT.md) (G1–G10).

本リポジトリのアイディアと方向付けは Takeyuki-K、コード・モデル改変・学習・評価・動画・文書などの実装はすべて Claude（Anthropic）によるものです。

## Use this idea / このアイディアの利用について
You are free to use, modify and build on this work (Apache-2.0).
**If this work or its idea inspired yours, please credit `Takeyuki-K` and link this repository** (GitHub "Cite this repository" uses [CITATION.cff](CITATION.cff)).
自由に使ってください。参考にした場合は、ユーザー名 **Takeyuki-K** とこのリポジトリへのリンクの記載をお願いします。
Redistributions must keep the [NOTICE](NOTICE) file (Apache-2.0 §4(d)).

## Credits / クレジット
- **Robot model**: ROBOTIS AI Sapiens K1, [ROBOTIS-GIT/ai_sapiens](https://github.com/ROBOTIS-GIT/ai_sapiens) (Apache-2.0) — modified (passive spring MP toe joints added), see [ai_sapiens/NOTICE_MODIFICATIONS.md](ai_sapiens/NOTICE_MODIFICATIONS.md).
- **Human gait data**: M. Duarte, "Notes on Scientific Computing for Biomechanics and Motor Control" (BMC), [duartexyz/BMC](https://github.com/duartexyz/BMC), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — retargeted to the robot (`ref_gait.npz` stays under CC BY 4.0).
- **Running motion data**: CMU Graphics Lab Motion Capture Database, [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu) (subject 16 trial 35, subject 9 trial 4), BVH conversion by B. Hahne — free for research and commercial use. *The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.*
- **Software used** (not redistributed): MuJoCo (Apache-2.0), PyTorch (BSD-style/Apache-2.0), NumPy/SciPy/NetworkX/Shapely (BSD), trimesh/Rtree/PyYAML/ONNX Runtime (MIT), ONNX/OpenCV (Apache-2.0), imageio (BSD-2), matplotlib (PSF-based), Pillow (MIT-CMU), Noto Sans CJK font (SIL OFL 1.1), FFmpeg (used as an encoding tool only).
- Idea and direction: Takeyuki-K. All implementation: Claude (Anthropic) — see [Authorship](#authorship--役割分担).

"ROBOTIS" and "AI Sapiens" may be trademarks of their respective owner; they are used here only to identify the robot model.
