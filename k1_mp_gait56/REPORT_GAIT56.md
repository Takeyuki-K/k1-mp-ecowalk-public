# v5.6: less side-to-side rocking, higher swing foot, arm residuals, feet re-placed before standing

**Walking policy of v5.5 retrained; running policy unchanged.** Idea & direction: Takeyuki-K · Implementation generated
with Claude (Anthropic) under Takeyuki-K's direction · **MuJoCo simulation only, not tested on a real robot.**

User observations and instructions:
- straight walking rocked strongly from side to side, which looked wasteful;
- 1) keep the pelvis as level as possible while walking; 2) a higher swing foot is allowed (a little higher than a
  human); 3) arm swing might help the balance; 4) after an in-place turn the robot stopped in a half step and could
  fall later — re-place the feet to the standing position after stopping;
- decision after the first stage (§2, S3): **human-level sway is allowed**; lateral pelvis travel is fine.

## 1. Result (`runs/final/walk.pt`, `runs/final/run.pt`, `gait56.py`)

8 robots from standing, no pushes (`eval56.py`, `out/eval56_final.json`; v5.5: `out/eval56_baseline_v55.json`):

| | v5.5 | **v5.6** |
|---|---|---|
| pelvis roll peak-to-peak, straight 0.6 / 1.0 / 1.4 m/s | 12.6 / 12.0 / 10.8° | **5.7 / 5.5 / 5.7°** |
| leg power 0.6 / 1.0 / 1.4 m/s | 87 / 125 / 173 W | 96 / 137 / 184 W (**+6 … +11 %**) |
| arm power (newly counted, part of the energy reward) | not counted | 3 / 6 / 12 W |
| in-place turns ±0.3 / ±0.6 / ±1.0 rad/s, 12 s | 100 % | 100 % |
| final stance after stopping from an in-place turn: front-back stagger / relative foot yaw | 3–19 cm / 5–20° | **2–7 cm / 2–7°** |
| stop from 1.2 m/s walking | 100 %, stagger 3.6 cm | 100 %, stagger 0.3 cm |

End-to-end through the gait manager (`eval_gait56.py`; v5.5 numbers from `../k1_mp_gait55/out/`):

| | v5.5 | **v5.6** |
|---|---|---|
| full 54 s profile (walk, turns, run, governor, braking), 8 robots | 8/8 | 8/8 |
| in-place +0.6 then −0.6 rad/s (2 × 12 s) and stop, 8 robots | 8/8 | 7/8 |
| **with random pushes** (±0.3 m/s, independent, ~every 3 s), 16 robots: full profile | 12/16 | **15/16** |
| with pushes: in-place turning profile | 6/16 | **10/16** |
| with pushes: normal stop / hard braking from 4.5 m/s | 15/16 / 15/16 | 15/16 / 16/16 |
| time to standing after a stop / hard braking from 4.5 m/s | 5.9 s / 4.4 s | 7.3 s / 5.2 s (two re-placement steps) |

Video: `media/K1_v55_v56_straight.mp4` (same command, front view, v5.5 left / v5.6 right).

## 2. Decisions and reasons

| # | decision | reason |
|---|---|---|
| S1 | **Diagnosis**: the rocking is a lean of the whole body over the stance leg (left single support +4.7°, right −5.8°), not a pelvis drop; hip-roll motors 27 W of 126 W | Measured before changing anything. The human reference had a level pelvis and only ±1.85 cm lateral pelvis travel (human proportions); K1's hips are 25 cm apart, so the policy shifted the COM by tilting instead. Present since v1 (v3 10.1°, v5 11.5°). |
| S2 | New reference `ref_lib_56.npz` (`make_ref56.py`): stance 20.9 → 16 cm, lateral pelvis travel from the linear inverted pendulum (A = w/2 (1 − 1/cosh(Ts/2Tc))), pelvis kept level by the legs, +2.5 cm swing clearance, pelvis height re-solved | Instructions 1 and 2. With a level pelvis the swing foot no longer gets clearance from the pelvis tilting up on the swing side, so the higher swing (allowed by the user) is needed. Stance-foot slip unchanged (16.6 vs 17.1 mm). |
| S3 | **Physics found in stage A**: strictly level (roll 3.8° p2p) raised hip-roll copper losses 22 → 39 W (+15 % leg power) | In single support the stance hip holds the body weight at the hip half-width: 27.3 kg × g × 0.125 m = 33.5 Nm; leaning 6° moves the COM over the hip (lever 9.4 cm, 25 Nm), and copper loss grows with torque². The v5.5 rocking was the energy optimum of the eco reward (like a compensated gait in humans with weak hip abductors). → user decision: **allow human-level sway** — no penalty within ±2.6° (`ROLL_DEADBAND`), steep outside. |
| S4 | Arm residuals (shoulder pitch / roll, both sides, ±0.3 rad per unit) as 4 extra actions; arm motor power added to the energy term | Instruction 3. No arm-specific reward ("reward design is hard"): the arms are used only if they help the existing rewards. Counting the arm power keeps them from being "free". Mean residual ~0.07 rad. The effect of the arms alone was not isolated (no arms-off ablation). |
| S5 | Stop rule: after stopping, step in place with zero yaw rate for two steps before closing the legs; reward for feet at the standing position | Instruction 4. Measured problem: after in-place turns v5.5 stopped with up to 19 cm stagger / 20° foot yaw. |
| S6 | Stepping in place uses the original 21 cm stance (fades out by 0.4 m/s) | Found in stage A: on the 16 cm stance in-place turns at −1 rad/s fell (0 %); restored 100 %. |
| S7 | Stage C adds a reward for re-aligned feet while stepping to stop | Stage B still stopped with 8–15 cm stagger. |
| S8 | **Foot re-placement feed-forward** at deployment (`PLACE_FF`, gain 1.5): the foot that swings next is placed beside the stance foot (offset measured at the last double support, in the frame of the feet) | User decision (option 2: a fixed procedure instead of more RL). Run → walk → stop stagger 10 → 6.5 cm; no falls. Policy unchanged. |
| S9 | Feet offsets are measured in the frame of the feet's mean heading, not the pelvis frame | Found during evaluation: a pelvis twisted relative to two parallel feet looked like a 7 cm stagger (0.25 m × sin 16°). |
| S10 | "Step until aligned" (`ALIGN_CHECK`) not used at deployment | It kept stepping up to the 8-step limit after running (11 s to stand) for little gain. |
| S11 | Stage D (feed-forward used in training) **rejected** | Leg power fell to +5 %, but stopping from 1.2 m/s walking fell 8/8. |

## 3. Limitations / next steps
- **+6 … +11 % leg power** for walking compared with v5.5 (mostly hip-roll copper loss, S3). Idea under consideration (user): a passive hip-abduction spring, like the passive MP toe spring.
- Stopping takes longer (two re-placement steps); a front-back stagger of 2–7 cm can remain.
- In-place turning under pushes (10/16) is still the weakest case.
- Simulation only.
