# v5.5: turning while walking and running, speed governor, lean into the turn, hard braking

**K1 + passive MP toes, v5 walk ⇄ run, now with a yaw-rate command at every speed.**
Idea & direction: Takeyuki-K · Implementation generated with Claude (Anthropic) under Takeyuki-K's direction ·
**MuJoCo simulation only, not tested on a real robot.** Motor limits are the v5 model (K1 URDF + assumed torque–speed curve).

User instructions for this version:
1. add turning to the running policy (and keep it in walking);
2. a large yaw-rate command should automatically lower the forward speed so the robot can turn;
3. the faster it runs, the more the whole body should lean into the turn (easier to turn, harder to fall);
4. hard braking: the landing leg flexes the knee to absorb the momentum while the trunk extends upward, straight up
   or slightly behind the foot.

## 1. Result (`runs/final/walk.pt`, `runs/final/run.pt`, gait manager `gait55.py`)

End-to-end through the gait manager, every robot starts standing, 8 robots (1° rms start-pose difference), no pushes
(`eval_gait55.py`, `out/eval_gait55.json`):

| segment (6 s each, measured after 1.5 s) | mode | yaw rate cmd → measured | speed reference → measured |
|---|---|---|---|
| walk 1.0 m/s | walk | 0 → −0.01 | 1.00 → 1.01 |
| walk 1.0 m/s, turn +0.6 rad/s | walk | 0.6 → 0.59 | 1.00 → 1.00 |
| run 3.0 m/s | run | 0 → −0.01 | 3.00 → 2.82 |
| run 3.0 m/s, turn ±0.5 | run | ±0.5 → +0.51 / −0.50 | 3.00 → 3.03 / 2.85 |
| run 4.5 m/s | run | 0 → 0.02 | 4.50 → 4.32 |
| **command 4.5 m/s + 1.0 rad/s (governor)** | run | 1.0 → 1.03 | **3.00 (governed)** → 2.60 |
| hard braking from 4.5 m/s | run → walk → stand | – | – |
| **survival over the whole 54 s profile** | | | **8 / 8** |
| in-place turn +0.6 then −0.6 rad/s, 12 s each | walk | ±0.6 → +0.61 / −0.61 | drift 6 cm/s |

In-place grid (`eval55.py walk`, 12 s each, 8 robots): survival 100 % for ω = ±0.3, ±0.6, ±1.0 rad/s.

**Lean into the turn** (running, measured trunk roll, 0.5 s mean, Fig. 1): 3 m/s × 0.5 rad/s → 7.7° (atan(vω/g) = 8.7°),
4.5 m/s × 0.5 → 10.4° (12.9°), governed 2.7 m/s × 1.0 → 13.5° (15.4°). The robot leans in the direction and by roughly the
amount that steady turning physically requires.

**Hard braking vs normal stop from 4.5 m/s** (8 robots, Fig. 2):

| | time to < 2 m/s | distance to < 2 m/s | time to standing | distance to standing | survival |
|---|---|---|---|---|---|
| normal stop (v5 rules) | 1.92 s | 6.6 m | 5.87 s | 8.7 m | 8/8 |
| **hard braking (v5.5)** | **0.88 s** | **3.1 m** | **4.36 s** | **6.2 m** | 8/8 |

During braking the pelvis first drops (knee flexion absorbing the impact, lowest 0.65 m) and then rises above the
running height (0.83 m) before standing — qualitatively the user's description. The rise comes mostly during the
following walking steps, not in the same stance.

**Random pushes** (±0.3 m/s horizontal, independently per robot, on average every 3 s):

| test | v5 | v5.5 |
|---|---|---|
| stand → walk 1.0 → run 4.5 → stop (straight), 2 seeds × 16 robots, same pushes | 28/32 (15/16, 13/16) | **30/32** (16/16, 14/16) |
| full v5.5 profile above (16 robots) | – | 12/16 (falls during the governed 1 rad/s turn at 17° lean) |
| hard braking / normal stop from 4.5 m/s (16 robots) | – | 15/16 / 15/16 |
| in-place turning ±0.6 rad/s 2 × 12 s (16 robots) | – (no in-place turn) | **6/16 — weak** |

**Straight walking power** (eco reward kept): 1.0 m/s 126 W vs v5 119 W (+6 %); like v3, turning needs slightly
stiffer legs. Running power is not optimised (as v5).

## 2. Decisions and reasons

| # | decision | reason |
|---|---|---|
| S1 | **Speed governor** v·\|ω\| ≤ a_lat (walk 1.0, run 3.0 m/s²): yaw-rate reference limited by the *current* speed, speed target lowered to a_lat/\|ω\|; references still ramped | User instruction 2. Centripetal acceleration v·ω must come from foot friction and lean; 3.0 m/s² ≈ 17° lean, well inside friction 1.0. Limiting ω by the current speed means the robot first slows, then turns harder, never an instant jump. |
| S2 | **Lean target** φ = atan(v·ω/g) in the trunk-orientation reward + **half of it as feed-forward** on hip/ankle roll (feet to the outside, soles flat) | User instruction 3; the physics of steady turning. Sign checked with forward kinematics before training. Measured lean follows φ (§1). |
| S3 | Turning inputs added with zero weights (v3 method): run +[heading error, ω_cmd, ω_ref, brake], walk +[ω_cmd, ω_ref, (brake)] | Iteration 0 behaves exactly like v5 (verified: identical actions and physics at ω = 0). |
| S4 | Running velocity rewarded in the **target-heading frame**, yaw-rate penalty replaced by yaw-rate tracking | v5 rewarded world-x speed and penalised every yaw rate, i.e. it was trained *not* to turn (baseline: 0.03–0.17 rad/s for 0.5–1.0 commands). |
| S5 | **Stride-wise hip-yaw feed-forward** for running turns (v3 T4 pattern, pelvis yaws ω·T/2 per stance) | Found during training: with rewards alone the running policy reached ~35 % of the commanded yaw rate after 400 iterations. Sign/magnitude tested (+1 better than 0, −1 worse) before continuing. |
| S6 | Walking: v = 0 reference + v3 in-place pattern + v3 safe stop, merged library 0 … 1.95 m/s | v5 walking had no in-place turn (baseline: 0 % survival at v 0 / ω 0.6). |
| S7 | **Hard braking**: running reference speed falls at 4 m/s² to 1.8 m/s; shaping = landing foot ahead of the pelvis, upright trunk, impact threshold +0.5 BW (knee may absorb), downward pelvis speed penalised but rising allowed, imitation weight ×0.3; walking part 1.5 m/s² (2.5× the comfortable 0.6) | User instruction 4. **Correction of the idea, as discussed**: the forward momentum can only be removed by a backward ground force, i.e. by landing with the foot ahead of the centre of mass; "sending it upward" converts part of the kinetic energy into height (at most v²/2g), so from 4.5 m/s several braking steps are still needed. |
| S8 | Gait manager: hand-over decisions use the running governor; turn rate and heading target are carried across walk ⇄ run; **one steady running cycle after braking before the hand-over to walking** | Found during evaluation: handing over right after braking fell 5/8 — the braking posture is not in the walking policy's hand-over training states. With the extra cycle 8/8. |
| S9 | Normal stop after run → walk: keep walking while slowing to 0.35 m/s, then close the legs (as v5) | Found during evaluation: stopping right at the hand-over fell 4/4 (not a trained situation); after the change 8/8. |
| S10 | Push tests with **independent** per-robot pushes | A first protocol pushed all robots at the same instant; survival then depended on which gait phase the push happened to hit (v5 56 %, v5.5 13 % for one seed) and did not compare policies. |
| S11 | Final walking checkpoint = stage E at iteration 500 (not the last one) | Fine-tuning kept trading skills: stage D lost long in-place turns, stage E's last checkpoint lost in-place −1.0 rad/s; it 500 passed all in-place cells and the full profile. |
| S12 | Running stage C (brake share 0.45, 40 % standing starts) **rejected** | Turning at 1 rad/s collapsed (survival 0 %). Stage D (brake 0.3, normal starts, lower lr) kept turning and improved braking. |

## 3. Limitations / next steps
- **Push robustness while turning on the spot (6/16) and during the governed high-lean running turn (12/16)** is the weakest point; next: push curriculum in those situations.
- The running speed drops more than the governor asks when a large turn starts (4.5 → 1.7 m/s at the entry of the 1 rad/s turn, Fig. 1).
- Speed under turn is 5–15 % below the reference.
- Braking: the walking part (1.8 m/s → stand) still takes ~3.5 s; a braking state bank for the walking policy (instead of the extra running cycle, S8) would shorten it.
- Yaw rate is taken from the pelvis angular velocity in the body frame (≈ cos(lean) error, < 5 % at 17°).
- Simulation only; whether real K1 motors can follow the lean and braking torques has not been checked.
