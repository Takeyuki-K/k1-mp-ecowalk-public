# Running (jog) with human imitation: report and decision log

**K1 + passive MP toe joints + human-running imitation (CMU mocap) + learned relaxation.**
Idea & direction: Takeyuki-K · Implementation generated with Claude (Anthropic) under Takeyuki-K's direction · MuJoCo simulation only

Goal (user instruction): first imitate human running and make it **stable**; give impact absorption and stability
more weight than power ("eco is a result"); land **heel first** from the flight phase, let the heel act as a pivot so
that the body rotates forward over it without the upper body collapsing; speed tracking is not required.

## 1. Result (final policy `runs/final/model.pt`)

10 s, 8 robots, start inside the running cycle, physics dt 5 ms, numbers measured after the first 2 s.

| item | learned running | human reference (retargeted) |
|---|---|---|
| survival (nominal / pushes ±0.3 m/s + friction & mass randomisation, 24 robots 16 s / pushes ±0.6 m/s) | **100 % / 100 % / 100 %** | – |
| speed | 1.54 m/s (Froude 0.39) | 1.65 m/s |
| cadence / cycle | 174 steps/min, 0.69 s | 174 steps/min, 0.69 s |
| flight phase (no foot contact) | **29 %** of time | 35 % |
| stance time | 244–254 ms | 223 ms |
| touchdown | **heel only, 100 %** of 184 touchdowns | heel first (imposed, see R2) |
| heel → forefoot contact (heel rocker) | 33–40 ms | 36 ms |
| peak foot force per stance | **2.30 BW** mean, 2.50 BW p95 | human running ≈ 2.5 BW (literature) |
| trunk forward lean | +1.8° mean, **±0.36°** std | +5° (R5) |
| pelvis vertical bob | 6.4 cm | 4.8 cm |
| leg electrical power / cost of transport | 342 W / 0.64 | – |

Leg stiffness over the right-leg cycle (Kp scale, 1 = nominal servo gain; phase 0 = right touchdown):

| phase | 0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| | 0.85 | 0.82 | 0.96 | 0.97 | 0.92 | 0.82 | 0.68 | 0.60 | **0.59** | 0.76 |

- As the user predicted, **relaxation only happens during swing** (Kp ≈ 0.6 in mid/late swing, 40 % softer) and the leg
  stiffens again before touchdown; it is never fully relaxed while running (walking eco policy went much lower).
  Kd is raised (×1.21 on average): the landing is absorbed by damping rather than by a stiff spring.
- The touchdown is soft compared with a stiff landing: peak foot force stays below the 2.5 BW penalty threshold.
- Power is about 2× the walking policy at 1.35 m/s (CoT 0.64 vs 0.34). Running near the walk–run transition is also
  more expensive than walking for humans; power was deliberately a small term here (R6).

Video: `out/K1_running.mp4` (left: CMU-derived reference, right: learned policy, synchronised by gait phase,
then 0.25× slow motion) and `out/K1_running.gif`.

## 2. Decisions and reasons

| # | decision | reason |
|---|---|---|
| R1 | **CMU subject 16, trial 35 "run/jog"** (BVH conversion by B. Hahne, cgspeed) | Free for research and commercial use; full skeleton incl. hips (needed for hip→ankle imitation). Subject 09 runs are 3.4–3.5 m/s (too fast for K1); 16_35 is a jog (2.83 m/s over the chosen cycle, leg 0.84 m). One clean right-to-right cycle (0.80 s) is used. |
| R2 | **Heel-first landing imposed** on the reference foot pitch (toe up 10° at touchdown, rolled back to the human pitch within the first 25 % of stance; contact flags heel 0–35 % of stance, forefoot from 15 %) | All CMU touchdowns in 16_35 are fore/mid-foot (toe-down). The user asked for heel landing with the heel as a pivot. Leg kinematics and timing stay human; only the foot angle around touchdown is changed. |
| R3 | **Speed 1.65 m/s reference**: stride scaled ×0.68 at the **same cadence** | Froude scaling of the human jog gives 2.44 m/s for K1 (Fr ≈ 1.0), too fast for a first stable running gait and the motors. 1.65 m/s (Fr ≈ 0.45) is just above the walking ceiling of the v2 policy (≈ 1.45 m/s). Keeping cadence keeps the ballistic flight timing; vertical hip→ankle motion is unchanged. |
| R4 | Pelvis height: stance foot exactly on the ground in stance, **ballistic parabola (g)** in flight | Same lesson as walking D8 (per-frame contact); in flight there is no ground to match, so physics (g) decides the shape. |
| R5 | **Trunk lean 5° forward** in the reference and upright reward | Found during training: the first running policy leaned **back** 2.8° (braking against the heel pivot). The user's idea needs the body to rotate forward over the heel without collapsing; human joggers lean ≈ 5–10°. Result: +1.8° forward, very small fluctuation (±0.36°). |
| R6 | Reward priorities: imitation 0.75, stability (upright 0.15, trunk angular rate 0.10), heel-first 0.3 per touchdown, impact penalty above 2.5 BW, **energy 0.0003** (1/5 of the eco walker) | User instruction: stability and impact absorption over power. |
| R7 | **Assist curriculum ("harness")**: pelvis support forces/torques, automatically reduced when falls become rare | Without help every robot fell within 0.4 s (one stride). Assist reached 0 after ~270 iterations; all later training and **all evaluations/videos use assist = 0**. |
| R8 | **Flight reward** (reward flight when the human is airborne) + vertical pelvis velocity imitation | Found during training: the first stage converged to a fast walk with **0 % flight** (1.1 m/s), a local optimum. With R8 flight rose to 29 %. |
| R9 | Start from the eco walking policy (same inputs/outputs) | Keeps the learned relaxation behaviour; it survived ~0.5 s on the running reference vs 0.35 s for zero actions. |
| R10 | MP toe spring unchanged (walking-tuned) | One change at a time. A stiffer MP may help push-off at running speed (future work). |
| R11 | Robustness stage: pushes + friction/mass/motor-gain randomisation | Required for stability; hard pushes (±0.6 m/s) survival 95.8 % → 100 %. |
| R12 | Checkpoint 1000 of the robustness stage | All candidates survived everything; this one had the smallest trunk fluctuation (±0.36° vs ±0.67°) and 100 % under hard pushes. Slightly slower (1.54 vs 1.58 m/s), accepted because speed tracking was not required. |

## 3. Training stages
| stage | change | iterations (× 4096 steps) |
|---|---|---|
| run1 | imitation + stability, assist curriculum, init from eco walker | ~330 |
| run2 | + flight reward, sharper speed reward (R8) | ~1880 |
| run3 | + 5° trunk lean reference (R5) | ~1590 |
| run4 | robustness: pushes + domain randomisation (R11) | 1000 (selected) of 1500 |
Total ≈ 20 M environment steps, 1 CPU core (two trainings ran in parallel).

## 4. Limitations / next steps
- Simulation only; Km assumed; no speed command (single running speed). A speed-command runner can reuse the
  walking speed-library approach (stride scaling).
- Trunk lean reached +1.8°, not the 5° of the reference (trade-off with the upright/rate rewards).
- No walk↔run transition and no running stop yet (episodes start in the running cycle).
- Running from one human trial only; the heel-first foot angle is an imposed modification of the data.

## 5. Files
`bvh.py` (BVH reader), `retarget_run.py` (running reference → `ref_run.npz`), `k1env_run.py` (env, rewards, assist),
`ppo_run.py` (training), `eval_run.py` (metrics), `video_run.py` (video). `k1env.py`/`k1env_eco.py` are copies of the
eco versions with two changes: reference file name (`ref_run.npz`) and pelvis-frame ankle vectors (needed for the
leaned reference; identical for upright walking references).

Mocap acknowledgement: The data used in this project was obtained from mocap.cs.cmu.edu.
The database was created with funding from NSF EIA-0196217.

## Reproducibility
Evaluation of the released checkpoints is reproducible with the included code. The training was staged; the exact
commands, the checkpoints carried over and the code changes between stages are listed in
[TRAINING_HISTORY.md](TRAINING_HISTORY.md). Stages that ran with earlier code cannot be replayed identically.
