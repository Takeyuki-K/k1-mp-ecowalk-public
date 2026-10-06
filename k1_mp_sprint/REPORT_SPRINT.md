# Fast running to 5 m/s (CMU style prior): report and decision log

**K1 + passive MP toe joints + CMU running data as style prior + speed curriculum.**
Idea & direction: Takeyuki-K · Implementation generated with Claude (Anthropic) under Takeyuki-K's direction · MuJoCo simulation only

User instruction: with the current CMU data, aim for **5.0 m/s**; running = **stability and speed first**, power is only
evaluated as a result (no energy reward). The 1.35–1.65 m/s range is covered by fast walking instead
(`k1_mp_fastwalk/REPORT_FASTWALK.md`).

## 1. Result

Two policies were trained (S8):

| | `runs/final/model.pt` (**speed priority**) | `runs/final/model_motorlimit.pt` (**respects 11.5 rad/s**) |
|---|---|---|
| top speed (command 5.5 m/s) | **5.21 m/s** | 4.60 m/s |
| speed at command 5.0 | 4.73 m/s | 4.43 m/s |
| survival, 24 robots, 12 s, at every command 3 / 4 / 5 / 5.5 m/s | **100 %** | **100 %** |
| survival with pushes + friction/mass/gain randomisation (8 robots, 5.5 m/s) | 100 % (5.14 m/s) | – |
| time with any leg joint above 11.5 rad/s (at top speed) | 28 % (hip pitch, knee, ankle pitch) | **0.4 %** |
| peak joint speed (p99) | 20 rad/s | 16.6 rad/s (impact spikes, ankle roll limit is 20.9) |

Speed-priority policy in detail (24 robots, start running at 2 m/s, accelerate at 1.5 m/s², measured after 5 s):

| command | speed | cadence | step length | flight | heel-only touchdowns | peak foot force (p95) | trunk lean | leg power | CoT |
|---|---|---|---|---|---|---|---|---|---|
| 3.0 | 2.85 m/s | 194 /min | 0.88 m | 33 % | 100 % | 2.8 BW | +3.7° ± 0.8 | 0.93 kW | 0.93 |
| 4.0 | 3.94 | 214 | 1.10 | 42 % | 100 % | 3.3 BW | +4.5° ± 0.6 | 1.32 kW | 0.96 |
| 5.0 | 4.73 | 231 | 1.23 | 44 % | 99.8 % | 3.6 BW | +4.8° ± 0.7 | 1.72 kW | 1.04 |
| 5.5 | **5.21** | 239 | 1.31 | 45 % | 98.8 % | 3.7 BW | +4.5° ± 1.0 | 1.94 kW | 1.06 |

- **5 m/s is reached** (5.21 m/s at a 5.5 command; the policy runs 5 % below the command, S9). For K1's leg length
  (0.62 m) this is Froude ≈ 4.5, i.e. like a human running ≈ 6.3 m/s (100 m in ~16 s).
- Speed is raised mainly by **step length** (0.88 → 1.31 m = 2.1 leg lengths) and flight (33 → 45 %), cadence rises
  less (194 → 239 /min), as in the human data the reference was built from (S2).
- Landing stays heel first at all speeds (user specification), trunk lean +4–5° with ≤ 1° fluctuation.
- **Power (result only):** ≈ 1.9 kW electrical at 5.2 m/s, CoT ≈ 1.0 (fast walking: 0.38). Legs are stiff
  (mean Kp scale 1.1, no relaxation at speed). With the assumed motor constant the copper loss dominates; whether
  K1's motors / battery can deliver this is not checked (Km is an assumption).

Video: `out/K1_sprint.mp4` (2 → 3 → 4 → 5.5 m/s command, then 0.25× slow motion at top speed; overlays include joint
speed relative to the 11.5 rad/s limit) and `out/K1_sprint.gif`.

## 2. The real limit: motor speed (honest finding)
The K1 URDF lists **11.5 rad/s** for hip, knee and ankle-pitch joints. The human-like running pattern needs more:
at 5.2 m/s the speed-priority policy exceeds it 28 % of the time (hip pitch 10 %, ankle pitch 9 %, knee 6 % of the
time, peaks 14–16 rad/s). When joint speed above the limit is penalised strongly, the same training reaches
**4.6 m/s** with practically no violation. So, **for the real K1, ≈ 4.5 m/s is the realistic top speed with this
gait**; 5 m/s needs either faster motors (≥ ~16 rad/s at the knee / hip) or a gait with smaller joint excursions
(e.g. less heel recovery, shorter swing), which a human-imitation prior does not give.
MuJoCo cannot vary the torque limit per robot inside a batched rollout, so the torque–speed curve itself is not
modelled; the limit is enforced by a penalty and measured.

## 3. Decisions and reasons
| # | decision | reason |
|---|---|---|
| S1 | **CMU 09_04** (3.6 m/s, the fastest clean running cycle in CMU) as the base | No sprint mocap usable under a permissive licence (see the data survey in the conversation). |
| S2 | **Time-warp scaling** of the hip-relative foot path: cadence ∝ v^0.35, contact time ∝ 1/v (constant contact length) | Humans increase speed mainly by stride length up to ~7 m/s while contact time shortens roughly as 1/v; the hip-relative foot path stays similar, so only timing changes. Gives a library 1.6–5.5 m/s from one trial. |
| S3 | Map the human's longest hip–ankle distance to 95 % of the straight robot leg | Found during the first build: with 100 % the IK hit the straight-knee singularity in late swing (errors up to 0.3 m, joint-speed spikes > 100 rad/s). |
| S4 | **No energy reward** | User instruction (power evaluated as a result). |
| S5 | Imitation reduced to a style prior (0.50 total), velocity 0.40, upright / trunk rate 0.25, heel-first, flight match | Speed and stability first. |
| S6 | **Automatic speed curriculum** 2.5 → 5.0 (→ 5.5) m/s: +0.25 m/s whenever falls < 0.05 /s, error near the top speed < 12 % and no assist | Reached 5.0 m/s after ~1900 iterations without manual tuning; every level is mastered without help before the next. |
| S7 | Heel-first landing kept; impact threshold grows with speed (2.5 + 0.2·(v−2) BW) | User specification for running. Note: athletes at these speeds usually land mid/forefoot; heel-first still worked (99 %). |
| S8 | **Two final policies**: speed priority (weak joint-speed penalty) and motor-limit (strong penalty) | The URDF speed limit decides what is realistic; both results are useful (what the gait can do vs. what K1 can do). |
| S9 | Command range trained to 5.5 m/s | The policy runs ~5 % slower than commanded (trade-off with stability); a 5.5 command gives > 5 m/s. |
| S10 | Evaluation starts running at 2 m/s and accelerates | Starting directly at high speed from a kinematic pose caused occasional falls in the first 0.3 s (initialisation artefact, not steady running). |
| S11 | Robustness stage (pushes ± 0.3 m/s, friction / mass / motor-gain randomisation) | Stability requirement. |
| S12 | Start from the jog policy (v3) with two new zero-weight inputs (v_cmd, v_ref) | Keeps the heel-first running already learned. |

## 4. Training stages
| stage | change | iterations (× 4096 steps) |
|---|---|---|
| sp1 | speed library + curriculum (2.5 → 5.0), assist curriculum, init jog | 5000 |
| sp2 | robustness (pushes + randomisation) | 2000 |
| sp4 | sharper speed reward, commands to 5.5 m/s → **final (speed priority)** | 2500 |
| sp3 → sp5 | from sp1: strong joint-speed penalty, then sharper speed reward → **final (motor limit)** | 2000 + 2500 |
Total ≈ 57 M environment steps.

## 5. Limitations / next steps
- Simulation only; no torque–speed curve, assumed Km; power numbers are relative.
- Starts inside the running cycle (2 m/s); no walk → run transition or running stop yet.
- Arms follow a fixed counter-swing (not optimised); at top speed they swing wide.
- A sprint-specific gait (forefoot strike, smaller knee excursion) would be needed to go beyond ~4.6 m/s within
  the motor-speed limit.

## 6. Files
`retarget_sprint.py` (speed library from CMU 09_04 → `ref_sprint_lib.npz`), `k1env_sprint.py` (env: speed command,
joint-speed penalty), `ppo_sprint.py` (curriculum), `eval_sprint.py`, `video_sprint.py`; base files from `k1_mp_run`.
Mocap: The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.

## Reproducibility
Evaluation of the released checkpoints is reproducible with the included code. The training was staged; the exact
commands, the checkpoints carried over and the code changes between stages are listed in
[TRAINING_HISTORY.md](TRAINING_HISTORY.md). Stages that ran with earlier code cannot be replayed identically.
