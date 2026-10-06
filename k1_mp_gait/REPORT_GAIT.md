# v5: mid/forefoot running within the motor limits, stand → run, and walk ⇄ run switching

**K1 + passive MP toes. Eco walking policy (heel strike) ⇄ running policy (mid/forefoot strike), gait manager.**
Idea & direction: Takeyuki-K · Implementation generated with Claude (Anthropic) under Takeyuki-K's direction · MuJoCo simulation only

> **Simulation only — not tested on a real robot.** The motor limits are a model (K1 URDF values + an assumed torque–speed
> curve); power uses an assumed motor constant. Whether a real K1 can run like this is untested.
> シミュレーションのみ・実機未検証。モーター制限は仮定を含むモデル、消費電力も仮定したモーター定数による推定値です。

**How v5 came about.** v3 (jog, heel first) showed that walking is cheaper than running near 1.5 m/s (342 W vs 216 W);
v4 reached 5.2 m/s only by running the leg joints above the URDF speed limit 28 % of the time, still with a heel landing.
v5 therefore (i) lands on the forefoot like the human runner in the CMU data — the idea being that a heel landing in front
of the body brakes the forward momentum (a hypothesis; the braking impulse was not measured), (ii) models the motor
torque–speed limit physically, and (iii) walks as long as walking can go and switches to running only above that.

User instructions (v5):
1. Running: heel landing was wrong → penalise heel strike, land on the **mid/forefoot**.
2. **Respect the ROBOTIS motor speed** (sim2real); if forefoot landing reduces losses, aim for 5 m/s.
3. Learn **stand → run**.
4. **Walk → run when speeding up** past a threshold (speed may jump, e.g. 1.6 → 2.0 m/s); walking keeps
   0.1–0.2 m/s speed steps.

## 1. Results

### Combined gait (gait manager, `gait.py`, final policies `runs/final/walk.pt` + `runs/final/run.pt`)
| test (8 robots each) | no pushes | with random pushes (±0.3 m/s every ~4 s, 32 robots) |
|---|---|---|
| **A** stand → walk 0.30 m/s, +0.15 m/s every 1.5 s → 1.65 → run 2 → 3 → 4 → 5.5 → 3 → walk 1.5 → 0.6 → stop (43 s) | **100 %**, all standing at the end | 93.8 % |
| **B** stand → **run 3 m/s directly** → 4.5 → walk 1.2 → stop (18 s) | **100 %**, all standing at the end | 93.8 % |

- Walk → run switch at the right heel strike after the walk reached ~1.6 m/s (t = 15.3 s in A); running reference
  starts at 2.0 m/s. Run → walk: running slows to 1.8 m/s, switch at the right touchdown, walking continues at 1.6 m/s.
- Walking speed steps of 0.15 m/s are followed (measured 0.45 / 0.60 / 0.75 / 0.89 / 1.04 / 1.21 / 1.36 / 1.49 m/s).

### Running policy (motor limit on, start from recorded walking states at 1.6 m/s, accelerate, 16 robots, 12 s)
| command | survival | speed | forefoot-first touchdowns | heel-first | trunk lean | torque saturated | leg power |
|---|---|---|---|---|---|---|---|
| 2.0 | 100 % | 1.87 m/s | 99 % | **0 %** | 4–5° ± 1.2 | 6 % | 0.65 kW |
| 3.0 | 100 % | 2.97 | 84 % (rest midfoot) | **0 %** | ± 0.6 | 15 % | 0.99 kW |
| 4.0 | 100 % | 3.94 | 84 % | **0 %** | ± 0.6 | 22 % | 1.33 kW |
| 5.0 | 100 % | 4.67 | 92 % | **0 %** | ± 1.1 | 40 % | 1.68 kW |
| 5.5 | 100 % | **4.92** | 84 % | **0 %** | ± 1.2 | 59 % | 1.84 kW |

- **Heel-first landing eliminated** (0 of all touchdowns); 84–99 % forefoot-first, the rest midfoot (heel and forefoot in
  the same 20 ms).
- **Top speed within the motor limit: 4.9 m/s** (v4 heel-strike with only a speed penalty: 4.6 m/s). At the top speed the
  motors are on their torque–speed limit 59 % of the time; this, not stability, sets the limit.
- Evaluation start: robots start from recorded fast-walking states (the real use case through the gait manager).
  With an artificial kinematic start directly inside the running cycle at 2 m/s (reference-state initialisation),
  1 of 16 robots fell at the 2 / 4 / 5 / 5.5 m/s commands and one robot stalled at low speed
  (`out/final_run.json`); this start is not used by the gait manager.
- Stand → run: 100 %, 90 % of 3 m/s reached in 2.6 s. Robustness (pushes + friction/mass/gain randomisation): 100 % at
  3 / 4 / 5.5 m/s commands (8 robots, top 4.8 m/s).
- Joint speeds above the limit only happen when the joint is **driven by the environment** (landing impact, swing
  inertia) against the motor's braking: in every over-limit sample the motor torque was braking (0 % motoring;
  `audit_joint_speed.py`, `out/joint_speed_audit.json`, added for the public release). This is **not** the same as staying
  within the specification: time with any leg joint above its limit is 0 / 1.3 / 6.2 / 16.0 / 19.1 % at the
  2 / 3 / 4 / 5 / 5.5 m/s commands; median excess 0.4–0.6 rad/s, p99 knee 12.6 rad/s, short peaks at top speed
  13.3 rad/s (knee), 17.8 rad/s (ankle pitch, 1.5× limit) and 40.6 rad/s (ankle roll, 1.9× its 20.9 rad/s limit).
  Whether real gearboxes, motors and drivers tolerate this back-driving is untested.

### Walking policy (eco reward kept, motor limit on; same protocol as v2/v4)
0.30 → 0.32, 0.60 → 0.61, 0.90 → 0.91, 1.20 → 1.22, 1.35 → 1.35, 1.50 → 1.48, 1.65 → 1.62 m/s; no falls; heel
strike 100 %; leg power 69 / 92 / 114 / 146 / 168 / 191 / 219 W (CoT 0.34–0.39 from 0.9 m/s), same as v4.
At the switching speed walking costs 219 W at 1.62 m/s, running 650 W at 1.87 m/s (CoT 0.39 vs 0.99; `out/ew_rn6.json`;
an earlier version of this report quoted CoT 1.04, which came from the reference-state-start run in `out/final_run.json`):
switching to running only when walking cannot go faster is also the efficient choice.

### Energy compared with v4 (estimated)
Walking: no further reduction in v5 (same power as v4). Running (no energy reward in v5) vs the v4 policy that respects
the joint-speed limit (heel first): command 3 / 4 / 5 / 5.5 m/s → v4 2.93 / 3.79 / 4.43 / 4.60 m/s, CoT 0.93 / 0.97 /
1.04 / 1.11; v5 2.97 / 3.94 / 4.67 / 4.92 m/s, CoT 0.95 / 0.96 / 1.03 / 1.07 (`../k1_mp_sprint/out/final_eval_motorlimit.json`,
`out/ew_rn6.json`). Similar at 3 m/s, slightly lower at higher commands while running faster; start conditions differ
(v4: running start at 2 m/s, v5: from fast walking), so this is not a controlled comparison.

Videos: `out/K1_walk_to_run.mp4` (test A), `out/K1_stand_to_run.mp4` (test B), GIFs.

## 2. Decisions and reasons
| # | decision | reason |
|---|---|---|
| G1 | Running reference = CMU 09_04 **with its own landing** (touchdown foot pitch 0° / 13.5° heel-up) and contact flags forefoot from touchdown, heel 30–70 % of stance | User decision ①. The CMU runner lands on the mid/forefoot; v4 had overwritten this. |
| G2 | Touchdown reward: forefoot first +1, midfoot +0.7, **heel first −1.5** | ①. |
| G3 | **Physical motor model** instead of a penalty: URDF 96.9 Nm / 11.5 rad/s (ankle roll 47.3 Nm / 20.9 rad/s); full torque up to 60 % of the speed limit, linear to 0 at the limit, **negative above (back-EMF braking)**; braking against the motion always available. Applied every 5 ms by scaling the joint's PD (implicit damping kept) | ② sim2real. The knee of the curve (60 %) is an assumption (no torque–speed curve published for K1; the 11.5 rad/s ≈ 110 rpm matches ROBOTIS' QC080 class actuator). A first version without back-EMF let passively driven joints exceed the limit 15 % of the time; with braking the motor no longer motors above the limit, but impact/inertia overshoots remain (19 % of the time at top speed, §1). |
| G4 | **Two policies + a gait manager** instead of one network | Walking keeps its eco reward and heel strike, running has no energy reward and forefoot strike: different reward functions and gaits. Humans also switch gait as a discrete event. The hand-overs are trained explicitly (G5). |
| G5 | Hand-over training with **recorded physics states**: running policy starts 25 % of episodes from fast-walking states (1.45–1.65 m/s, right heel strike) and 25–40 % from standing; walking policy starts 20–50 % from running states (right touchdown) | ③④. The policies learn exactly the states they receive from the other policy. |
| G6 | Walk → run at the right heel strike when walking ≥ 1.55 m/s and command ≥ 1.8; reference jumps to 2.0 m/s. Run → walk when command < 1.7: run slows to 1.8, switch at the right touchdown, walking reference 1.6 | ④ (jump allowed); hysteresis avoids chattering; both cycles start at the right foot contact, so the phase is continuous. |
| G7 | Cadence-choice action for running (tested, **rejected**) | Tried to raise the motor-limited top speed by letting the policy trade step frequency for length; it did not use it (factor 0.99) and the top speed dropped to 4.4 m/s. |
| G8 | Running speed range extended down to 1.8 m/s | Hand-over speed; the earlier policy (trained ≥ 2.0) fell 5 % of the time at 1.8 m/s. |
| G9 | Hand-over speed run → walk 1.8 m/s (not 2.0) | At 2.0 the running policy actually ran ~2.3 m/s; the large momentum step to walking caused falls under pushes. |
| G10 | Robustness round: more standing starts (40 %), more stops (30 %), more run → walk entries (50 %) | Push test of the combined gait was 56–88 %; falls clustered 1–2 s after run → walk and in the first second of stand → run. After this round 93.8 % / 93.8 %. |

## 3. Training
| policy | stage | change | iterations |
|---|---|---|---|
| run | rn1, rn2 | forefoot reference, motor model, entries, curriculum 3.5 → 5.5 m/s, init v4 motor-limit policy | 900 + 5000 |
| run | rn4 | robustness (pushes + randomisation), back-EMF motor model | 2000 |
| run | rn6 | 1.8 m/s, 40 % standing starts → **final** | 1000 |
| walk | wk1–wk4 | motor model, run → walk entries, more stops / hand-overs → **final** | 1500 + 800 + 1000 + 1000 |

## 4. Limitations / next steps
- **Simulation only, no real-robot test.** The torque–speed curve shape is assumed; power uses the assumed Km.
- Joints are back-driven above the speed limit by impacts and inertia (see §1); the tolerance of the real actuators is unknown.
- The "forefoot landing brakes less" hypothesis was not measured directly (no braking-impulse analysis).
- Running has no heading control (drift is corrected by the walking policy after the switch); no turning while running.
- Under continuous pushes 2 of 32 robots still fall per 18–43 s test (stand → run start, or just after run → walk).
- 5.0 m/s is not reached within the motor limit (4.9 m/s); faster motors or a gait with smaller joint excursions are needed.

## 5. Files
`retarget_sprint.py` (STRIKE='fore' library), `motor.py` (torque–speed model), `k1env_run2.py` (running env),
`k1env_walk2.py` (walking env), `gait.py` (gait manager), `ppo_run2.py`, `ppo_walk2.py`, `make_walk_bank.py`
(in `k1_mp_fastwalk`) / `make_run_bank.py`, `eval_run2.py`, `eval_walk2.py`, `eval_gait.py`, `push_test.py`,
`audit_joint_speed.py` (joint-speed audit), `video_gait.py`; banks `walk_bank.npz`, `run_bank.npz`; libraries `ref_sprint_lib.npz` (running), `ref_lib.npz` (walking).
Mocap: The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.

## Reproducibility
Evaluation of the released checkpoints is reproducible with the included code. The training was staged; the exact
commands, the checkpoints carried over and the code changes between stages are listed in
[TRAINING_HISTORY.md](TRAINING_HISTORY.md). Stages that ran with earlier code cannot be replayed identically.
