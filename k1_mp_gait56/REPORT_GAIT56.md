# v5.6: less side-to-side rocking, higher swing foot, arm residuals, feet re-placed before standing (joint-angle FK + IK)

**Walking policy of v5.5 retrained; running policy unchanged.** Idea & direction: Takeyuki-K · Implementation generated
with Claude (Anthropic) under Takeyuki-K's direction · **MuJoCo simulation only, not tested on a real robot.**

User observations and instructions:
- straight walking rocked strongly from side to side, which looked wasteful;
- 1) keep the pelvis as level as possible while walking; 2) a higher swing foot is allowed (a little higher than a
  human); 3) arm swing might help the balance; 4) after an in-place turn the robot stopped in a half step and could
  fall later — re-place the feet to the standing position after stopping;
- decision after the first stage (§2, S3): **human-level sway is allowed**; lateral pelvis travel is fine.

> **v5.6.1 (§4): `runs/final/walk.pt` is now stage w56h** — quick stop, feet re-placed 1 s after stopping, in-place
> turning with real alternating steps in the human rhythm. §1–§3 describe the first v5.6 result (stage w56c,
> `runs/init/w56c_model.pt`).

## 1. Result (stage w56c, `gait56.py`)

8 robots from standing, no pushes (`eval56.py`, `out/eval56_final.json`; v5.5: `out/eval56_baseline_v55.json`):

| | v5.5 | **v5.6** |
|---|---|---|
| pelvis roll peak-to-peak, straight 0.6 / 1.0 / 1.4 m/s | 12.6 / 12.0 / 10.8° | **5.7 / 5.5 / 5.7°** |
| leg power 0.6 / 1.0 / 1.4 m/s | 87 / 125 / 173 W | 96 / 137 / 184 W (**+6 … +11 %**) |
| arm power (newly counted, part of the energy reward) | not counted | 3 / 6 / 12 W |
| in-place turns ±0.3 / ±0.6 / ±1.0 rad/s, 12 s | 100 % | 100 % |
| stop from 1.2 m/s walking | 100 % | 100 % |

End-to-end through the gait manager (`eval_gait56.py`; v5.5 numbers from `../k1_mp_gait55/out/`):

| | v5.5 | **v5.6** |
|---|---|---|
| full 54 s profile (walk, turns, run, governor, braking), 8 robots | 8/8 | 8/8 |
| in-place +0.6 then −0.6 rad/s (2 × 12 s) and stop, 8 robots | 8/8 | 7/8 (fall while turning) |
| **with random pushes** (±0.3 m/s, independent, ~every 3 s), 16 robots: full profile | 12/16 | **15/16** |
| with pushes: in-place turning profile | 6/16 | **10/16** |
| with pushes: normal stop / hard braking from 4.5 m/s | 15/16 / 15/16 | 15/16 / 16/16 |
| time to standing after a stop / hard braking from 4.5 m/s | 5.9 s / 4.4 s | 7.3 s / 5.2 s (two re-placement steps) |

With pushes the in-place result varies with the random state carried over from the previous profile (8/16 to
10/16 in reruns; most falls happen while turning, before the stop logic acts).

**Final stance after stopping** (`eval_stance56.py`, 8 robots, no pushes). Measured with simulator ground truth
(world poses of both feet) when the robot reaches standing; front-back offset of the ankles along the feet's heading
[cm] / relative foot yaw [°], mean over robots. v5.5: `eval_stance55.py`. JSON: `out/stance*.json`.

| stop after | v5.5 | v5.6, no re-placement | v5.6, first re-placement (bug, S9) | **v5.6, FK + IK re-placement** | same + step until aligned (S10) |
|---|---|---|---|---|---|
| in-place +0.6 rad/s | 6.7 / 14 | 1.4 / 15 | 6.8 / 4 | **1.2 / 5** | 2.3 / 3 |
| in-place −1.0 rad/s | 1.6 / 27 | 2.3 / 7 | 5.3 / 13 | **2.8 / 10** | 3.8 / 14 |
| in-place +0.6 then −0.6 | 0.6 / 6 | 7.7 / 5 | 7.9 / 3 | **1.6 / 3** | 2.7 / 3 |
| walking 1.2 m/s | 10.7 / 19 | 4.4 / 8 | 3.8 / 5 | **3.2 / 11** | 1.7 / 7 |
| walking 0.8 m/s + 0.5 rad/s | 12.3 / 21 | 12.8 / 6 | 11.5 / 5 | **10.4 / 7** | 3.2 / 3 |
| running 4.5 m/s, normal stop | 8.9 / 7 | 11.8 / 3 | 7.2 / 3 | **6.8 / 2** | 3.5 / 2 |
| running 4.5 m/s, hard braking | 3.7 / 15 | 1.9 / 8 | 1.9 / 7 | **0.6 / 7** | 0.6 / 4 |
| time to standing (walk 1.2 / run stop / brake) | 3.3 / 5.9 / 4.4 s | 4.3 / 7.3 / 5.2 s | same | **4.3 / 7.3 / 5.2 s** | 8.4 / 10.0 / 7.1 s |

Stopping under random pushes (`eval_stop_push56.py`, 32 robots × 4 stop cases, falls from the stop command on):
no re-placement 21 / 122, FK + IK re-placement 27 / 122 (gain 1.0: 24 / 122). The difference is within the
statistical spread (about ±4), so the re-placement neither clearly helps nor clearly hurts robustness.

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
| S8 | **Foot re-placement feed-forward** at deployment (`PLACE_FF`, gain 1.5): while stepping in place to stop, the swing foot is placed beside the stance foot. Inputs: **joint angles only** (forward kinematics, `legkin.py`) and the foot contact sensors. The swing foot target seen from the stance foot (FK of the measured stance leg and of the swing leg's joint target) gives the front-back offset and relative yaw; **inverse kinematics** of the swing leg (6×6 damped least squares) removes them, keeping foot height, sideways distance, pitch and roll | User decision (option 2: a fixed procedure instead of more RL). Revised after the user asked whether this works only because it is a simulation: the first version read simulator sensors (world-frame foot headings) and used a linearised gain (0.63 m per rad hip pitch at the standing pose) instead of IK. FK reproduces MuJoCo's kinematics exactly (1e-12 m on the same joint angles) and matches the ground-truth stance to 0.2 cm on average at double support. Policy unchanged. |
| S9 | **Bug found and fixed**: the first version combined pelvis-frame ankle positions with world-frame foot headings | It was wrong whenever the robot did not face the world x axis: 17 cm mean error (58 cm max) against ground truth. Consequences: the first re-placement made in-place stops worse (6.8 vs 1.4 cm, table above); the stance numbers reported for that version (2–7 cm) were measured in the pelvis frame and are withdrawn, and so is the explanation "a twisted pelvis looked like a stagger". Stage C (w56c) was trained with the same wrong measurement in its alignment reward and stop rule; it was **not retrained** (the policy's own inputs never contained it). |
| S10 | "Step until aligned" (`ALIGN_CHECK`) **not used by default** | Re-checked with the correct measurement: it brings the front-back offset to ≤ 4 cm on average in every case, but standing takes 1–4 s longer (walk 4.3 → 8.4 s, run stop 7.3 → 10.0 s). Left to the user to decide; `ALIGN_CHECK=1` enables it. (The first rejection was based on the wrong measurement.) |
| S11 | Stage D (feed-forward used in training) **rejected** | Leg power fell to +5 %, but stopping from 1.2 m/s walking fell 8/8. |
| S12 | **Double support is detected with foot contact sensors** (heel / ball / toe on each foot) | User decision: a real K1 will need foot contact sensors for this; not replaced by a kinematic contact estimate. |

## 3. Limitations / next steps
- **+6 … +11 % leg power** for walking compared with v5.5 (mostly hip-roll copper loss, S3). Idea under consideration (user): a passive hip-abduction spring, like the passive MP toe spring.
- Stopping takes longer (two re-placement steps). After a turning walk the feet still end 10 cm apart front-back
  (the re-placement starts only when the speed command has reached zero, two steps are not enough); relative foot
  yaw of 5–10° remains in several cases. `ALIGN_CHECK=1` fixes most of it at the cost of time.
- The feet creep while standing (e.g. 1.6 → 5.9 cm and 3 → 7.5° within 7 s after an in-place stop; v5.5 shows the same,
  6 → 15°): the standing posture pushes against a staggered stance and MuJoCo's soft contact lets the feet slide
  slowly. Not investigated yet; on a real floor static friction may hide it, which should not be relied on.
- Stopping under pushes falls in about 20 % of the cases (with or without re-placement).
- Stage C was trained with the wrong alignment measurement (S9); not retrained.
- In-place turning under pushes (10/16) is still the weakest case.
- Simulation only.

## 4. v5.6.1: quick stop, re-stance after 1 s, in-place turning in the human rhythm (stages w56e–w56i)

User instructions:
- stopping does not need aligned feet — **stopping quickly is safer**; but about 1 s after stopping, without a speed
  command, return to the initial standing posture ("a normal robot can do this; can a policy?");
- in-place turning: check the attached motion capture (`IMG_2938`, monocular video → MediaPipe → 21-joint skeleton);
  if usable, retarget to K1 and train imitation + reinforcement learning that does not fall;
- decision after the data check: **use only the turning rhythm** of the capture (S13).

### Result (`runs/final/walk.pt` = w56h, deployment settings in `gait56.py`)

Stop → standing (gait manager, 8 robots): from in-place turning **1.7 s** (before 2.6 s), from 1.2 m/s walking
**3.3 s** (4.3 s), normal stop from 4.5 m/s **6.3 s** (7.3 s), hard braking **4.2 s** (5.2 s) — as fast as v5.5.

In-place turning, 8 robots, 12 s (`eval_inplace56.py`; simulator ground truth):

| ω [rad/s] | steps/s w56c → **w56h** | yaw per step w56c → **w56h** (human) | foot lift L / R [cm] w56c → **w56h** | loaded-foot spin rms [rad/s] | leg power [W] |
|---|---|---|---|---|---|
| +0.3 | 0.56 → **1.07** | 31° → **16°** (17°) | 1.4 / 7.7 → **16.7 / 11.1** | 0.65 → **0.10** | 41 → 82 |
| +0.6 | 0.86 → **1.17** | 40° → **30°** (32°) | 4.3 / 11.0 → **16.7 / 12.1** | 1.00 → **0.16** | 67 → 90 |
| +1.0 | 1.17 → **1.25** | 50° → **46°** (37–58°) | 4.8 / 11.7 → **17.3 / 12.4** | 1.41 → **0.34** | 109 → 107 |
| −0.6 | 0.75 → **1.17** | 46° → **30°** | 9.2 / 0.9 → **13.3 / 10.5** | 0.72 → **0.27** | 45 → 84 |
| −1.0 | 0.76 → **1.25** | 76° → **46°** | 12.8 / 1.0 → **12.2 / 8.4** | 1.04 → **0.50** | 69 → 116 |
| with random pushes, 16 robots: survival | 63–75 % → **69–88 %** | | | | |

The w56c policy turned by lifting only the outer foot and spinning the inner foot on the floor (1 cm lift);
w56h steps with both feet. Stepping costs more energy at low turn rates (+40 W at 0.3–0.6 rad/s): the spinning foot
was cheap in the simulation, but its grip depends on the real floor (S15).

End-to-end (`eval_gait56.py`): full profile 8/8, in-place profile 8/8, normal stop from 4.5 m/s **7/8**, braking 8/8;
with pushes (16 robots): full profile 15/16, in-place profile **12/16** (w56c 8–10/16), stop 16/16, braking 15/16.
Stopping under pushes (`eval_stop_push56.py`, 4 × 32 robots): 27 falls (first v5.6: 21–27); the stop after a
−1.0 rad/s in-place turn is the weakest case (14/31).
Walking (`eval56.py`): pelvis roll 6.8–6.9° peak-to-peak (w56c 5.5–5.7°), leg power 96 / 141 / 190 W at
0.6 / 1.0 / 1.4 m/s (w56c 96 / 137 / 184 W).

Re-stance (`eval_stance56.py`, ground truth; front-back offset [cm] / relative foot yaw [°], mean of 8 robots,
at the quick stop → at the end, several seconds later):

| stop after | quick stop | after re-stance | time to final stance |
|---|---|---|---|
| in-place −1.0 rad/s | 12.3 / 20 | **3.0 / 3** | 8.0 s |
| in-place +0.6 rad/s | 3.0 / 9 | 5.0 / 8 | 6.4 s |
| in-place +0.6 then −0.6 | 5.5 / 2 | 5.4 / 2 | 8.2 s |
| walking 1.2 m/s | 1.1 / 8 | 5.5 / 6 | 7.9 s |
| walking 0.8 m/s + 0.5 rad/s | 2.6 / 4 | 2.5 / 6 (no re-stance) | 2.6 s |
| running 4.5 m/s, normal stop | 6.6 / 2 | 3.7 / 6 | 12.3 s |
| running 4.5 m/s, hard braking | 3.3 / 7 | 6.9 / 1 | 10.7 s |

The re-stance removes large misalignments (the risky ones), but its precision is about ±5 cm / ±6°: stepping in
place itself scatters the feet by that much. Video: `media/K1_v56_inplace_restance.mp4`.

### Decisions and reasons

| # | decision | reason |
|---|---|---|
| S13 | **Only the turning rhythm of the capture is used** (user decision after the check); leg joint angles are not imitated | Data check (`mocap_turn_rhythm.py`): both foot heights move up and down together by ~10 cm with the facing direction (floor-height heuristic), both feet move ±20 cm front-back together relative to the pelvis (monocular depth), left/right thigh and shank lengths differ by 6 cm, 64 % of the frames have a low-visibility joint. The pelvis heading is robust: turning comes in bursts (one per step, ~0.5 s) separated by pauses in double support. Extracted (`mocap/turn_rhythm.json`, 36 steps): yaw per step 17° / 32° / 37° / 58° and step period 1.17 / 0.97 / 0.83 / 0.92 s at about 0.2 / 0.55 / 0.85 / >1 rad/s. The raw capture (video-derived personal data) is not in the repository. |
| S14 | Rhythm on K1: step period from the human data scaled by Froude similarity √(0.70 / 0.90); the swing keeps its duration, the **double support is stretched** (pause between steps); yaw per step = ω × step period (`IP_RHYTHM`) | Matches the "turn – pause – turn" pattern of the capture without making single support longer. |
| S15 | Rewards for real stepping: **+ for an unloaded, lifted (> 3 cm) swing foot, − for a swing foot still carrying load** (`W_LIFT`, `LIFT_PEN`), penalty for a loaded foot spinning about the vertical (`W_PIVOT`) | Found while testing the re-stance: the w56c policy did not lift the inner foot at all; it spun it on the floor (MuJoCo floor, condim 3). The first stage with only a bonus (w56e) kept shuffling and fell 8/8 at −0.6 rad/s → rejected. |
| S16 | Quick stop: the legs close as soon as the speed is zero (no re-placement steps) | User instruction (stopping quickly is safer). |
| S17 | **Re-stance**: after 1.0 s of standing without a command, if the feet are off the standing position (front-back > 3 cm, relative yaw > 6° or width off by > 4 cm; joint-angle FK), step in place with the FK + IK re-placement (front-back, yaw, **and standing width**) until aligned or 4 steps, then close the legs; once per standing period. Trained in the env (`RESTANCE_ENV`, staggered standing starts `P_STAGGER`) | The recommended way for a learned controller: a fixed, checkable procedure (like a footstep planner) around the learned stepping; the policy learns to step while it is used. Width added because closing the legs from a narrow stance pushed the feet apart. Re-placement gain for yaw 1.0 (1.5 overshot). |
| S18 | Trigger thresholds kept low (3 cm / 6°) | With 5 cm / 10° a staggered stance after an in-place stop was left alone, the feet crept, and 5/8 robots fell later while standing — the original problem 4. |
| S19 | w56g: in-place turns biased to the weak (left) side and to fast turns; w56h: twice as many pushes | w56f fell 8/8 at +1.0 rad/s (left/right asymmetry); w56g was weak under pushes (in-place 44–81 %, stops 46 falls / 120). |
| S20 | Stage w56i (re-stance state and stance error as policy inputs, reward per re-stance touchdown) **rejected** | Front-back alignment slightly better, but in-place turning under pushes fell much more (+0.3 rad/s: 3/16 survived). |

### Limitations / next steps
- Re-stance precision ±5 cm / ±6° and 5–9 s until the final stance; a learned footstep-target input (S20) did not
  work yet without losing robustness.
- **Feet creep while standing** in the simulation (a few cm / degrees in several seconds). The foot contacts are two
  capsules and a box with condim 3 (no torsional friction). Before trusting standing behaviour, the contact model
  should be checked (condim 4/6, contact softness) — this may change all stages.
- In-place stepping costs +40 W at slow turns; lower swing height for in-place steps would save energy (not tried).
- One normal stop from 4.5 m/s fell (7/8) without pushes; stopping after −1.0 rad/s under pushes is weak.
- Walking rocks slightly more than w56c (6.8° vs 5.5°).
