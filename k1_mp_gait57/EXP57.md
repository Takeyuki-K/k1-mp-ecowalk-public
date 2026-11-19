# v5.7 experiment: level torso by foot placement, not by hip-roll stiffness

Direction: Takeyuki-K (2026-11-12). Execution: metapen environment (CPU, 128 robots).

## Problem (measured on v5.6.3, direct walking env, n=8, 12 s, meas. after 4 s)
At 1.0 m/s straight: pelvis roll 7.2 deg peak-to-peak (RMS 3.1 deg), leg power 122.6 W,
of which hip-roll copper loss alone 31.2 W (25 %). Step width at touchdown 13.1 cm,
touchdown foot offset from pelvis 6.8 cm, single-support pelvis-to-foot lever 5.6 cm.
REPORT_GAIT56 S1/S3 physics: the rocking is a whole-body lean over the stance leg;
forcing a level pelvis by hip-roll torque raised hip-roll copper 22 -> 39 W in w56a.

## Hypothesis
If the support foot is placed further inwards (closer to under the pelvis/COM), the
stance hip-roll lever arm shrinks, so the pelvis can stay level with *less* hip-roll
torque: level torso and low power at the same time, with the gains left free to relax.

## Changes (this folder only; k1_mp_gait56 untouched)
1. Reference stance 16 -> 11 cm: `REF_W_TARGET=0.11 python3 make_ref56.py` +
   `make_sym_ref.py` (LIPM lateral pelvis travel is recomputed for the narrower base).
   In-place widening in `k1env_walk4._extra_targets` now uses `W_WALK_M` so stepping
   in place still re-widens towards 20.9 cm.
2. New foot-placement reward in `K1Walk4Batch._extra_reward` (`W_PLACE`, default 0):
   (a) at each touchdown, reward exp(-((offset - target)/sigma)^2) with target
       |lateral foot offset from pelvis| = `PLACE_TD_M` = 5.5 cm, sigma 3 cm;
   (b) dense term: in single support, same reward for the support foot.
   Active only while walking forwards (cmd on, alpha > 0.5, v_ref > 0.25 m/s).
3. Level-pelvis reward strengthened, but paired with an anti-stiffness guard:
   `ROLL_DEADBAND` 0.045 -> 0.02, `W_LEVEL` 0.35 -> 0.6, `LEVEL_K` 60 -> 100, and
   `W_KPROLL=0.15` penalises mean hip-roll kp_scale above 0.75 while walking
   (baseline policy mean is ~0.69), so buying level with stiffness is not rewarded.
4. Energy reward unchanged (`--w_energy` default 0.0015): relaxation still pays.

## Training recipe (base: w56l, the v5.6.3 stage)
Env: see exp57/train_env.sh (LONG_INPLACE=1 EP_LEN=1000 WALK_BRAKE=1 P_BRAKE_WALK=0.3
P_RUN=0.3 P_STOP=0.4 PUSH_P=0.01 SYM_REF=1 KP_STAND_MIN=0.3 KD_STAND_MIN=0.5
IP_WIDE=1 PLACE_FF=1 PLACE_GAIN=1.5 PLACE_GAIN_YAW=1.0 PLACE_LAT=1 RESTANCE_ENV=1
IP_RHYTHM=1 P_STAGGER=0.5 W_LIFT=1.0 LIFT_PEN=1.0 W_PIVOT=0.15 TURN_POS_P=0.5
P_FAST_IP=0.3, plus the v5.7 variables above), THREADS=2.
Init: runs/final/walk.pt (v5.6.3). `--sym 1.0 --kv_walk 40`, staged chunks of
150 iterations with a diagnostic evaluation after each chunk.

## Reference-swap control (old policy, new 11 cm reference, no training)
At 1.0 m/s: roll p2p 7.12 deg (was 7.20), step width 9.1 cm (13.1), lever 3.9 cm (5.6),
but leg power 126.9 W (122.6) and hip-roll copper 33.4 W (31.2): narrowing alone
does not fix the tilt; the learned lean-over-stance-leg strategy has to be unlearned.

## Results (2026-11-12, metapen CPU; diag = direct walking env, n=8, 12 s, meas. after 4 s)
Straight walking at 1.0 m/s (0.6 / 1.4 m/s follow the same pattern):

| policy | iters | roll p2p | roll RMS | leg power | hip-roll copper | hip-roll kp | step width | support lever |
|---|---|---|---|---|---|---|---|---|
| v5.6.3 baseline (own env) | - | 7.20 deg | 3.10 deg | 122.6 W | 31.2 W | 0.688 | 13.1 cm | 5.6 cm |
| old policy + 11 cm ref (control) | 0 | 7.12 | 2.99 | 126.9 | 33.4 | 0.695 | 9.1 | 3.9 |
| exp57a (HIPCU_W=0) | 50 | 6.00 | 2.56 | 125.3 | 35.9 | 0.716 | 11.6 | 5.1 |
| exp57b (+HIPCU_W=0.002) | 150 | 5.31 | 2.10 | 130.9 | 36.9 | 0.675 | 11.0 | 4.6 |
| exp57c (from b, HIPCU_W=0.004) | 300 | 6.23 | 2.60 | 131.4 | **27.9** | 0.686 | 11.9 | 5.1 |
| exp57d (from b, HIPCU_W=0.003) | 250 | **4.65** | **1.68** | 137.5 | 38.4 | 0.667 | 11.9 | 4.8 |

Reading: the tilt and the hip-roll copper are a Pareto trade in this model. exp57d
halves the roll (RMS -46 %) at +12 % leg power, bought *without* stiffness
(kp fell); exp57c is the only point with hip-roll copper below baseline (-11 %)
at a moderate roll improvement (-16 % RMS) and +7 % leg power. No run of a few
hundred iterations reached "level AND cheaper" — but the exchange rate is much
better than w56a's (+15 % power), and it improves with training, which suggests
the placement path is being learned and longer training (the original stages
used 1200-1500 iterations) is the main missing ingredient.

Full-course check (eval_gait56.py, n=4) for exp57c and exp57d: survival 1.0 on the
54 s course, in-place turns and stops; walk 1.0 m/s -> 0.96 / 0.98 m/s.
Minor regression vs v5.6.3: brake from 4.5 m/s to stand 4.5 s (was 3.84 s).

Checkpoints: runs/exp57a|b|c|d/model.pt (init chain: final -> a -> b -> {c, d}).
Deployment note: gait56.py in this folder defaults W_WALK_M=0.11.

## Posture-first stage (2026-11-12 evening, user: ignore power for now)
Goal: about half the sway of exp57d again, and arms should not swing sideways.
Settings (train_env2.sh): init exp57d, --w_energy 0, HIPCU_W=0, W_KPROLL=0,
ROLL_DEADBAND=0, W_LEVEL=1.2, LEVEL_K=150, ROLLRATE_W=0.02, W_PLACE unchanged.
New terms in _extra_reward: ARM_LAT_W penalises shoulder-roll deviation from the
default pose; ARM_ACT_W penalises shoulder-roll residuals (chunk E only).
Straight walking at 1.0 m/s (same diag):
| policy | roll RMS | roll p2p | arm roll p2p | arm dev RMS | leg power | full course |
|---|---|---|---|---|---|---|
| exp57d | 1.68 deg | 4.65 deg | 22.8 deg | 13.1 deg | 137.5 W | 1.00 |
| exp57e (150 it, ARM_LAT 2.0 + ACT 0.05) | 0.82 | 3.03 | 21.1 | 10.4 | 159.3 | 1.00 |
| exp57f (from e, ARM_LAT 4.0, ACT 0) | 0.65 | 2.21 | 10.2 | 4.7 | 161.9 | 0.75 (1/4 fell in the final brake of the 54 s course; isolated stop/brake tests 1.00) |
Note: chunk E's residual penalty fought the cancellation of the reference arm
swing, so the arm p2p only moved in chunk F once ARM_ACT_W was removed.

## Brake-robustness repair (exp57g, from exp57f, 200 it)
Diagnosis: in the 54 s course, falls happened at t=50.26 s = 2.26 s into the final
brake segment (run -> walk hand-over while braking from 4.5 m/s); n=8 reproduced
2 falls with identical fall time (same deterministic start). Isolated brake test
had passed, so it was a specific hand-over state, not general fragility.
Recipe (train_env3.sh): P_BRAKE_WALK 0.3 -> 0.55, ARM_LAT_W 4.0 -> 3.0, rest as
the posture stage, lr 3e-5 -> 2e-5.
Result: full course n=8 survival 1.00 (course, in-place, stop 5.74 s, brake
4.41 s). Straight 1.0 m/s: roll RMS 0.68 deg, roll p2p 2.16 deg, arm roll p2p
7.8 deg (better than exp57f's 10.2), leg power 155.1 W. Note the policy re-widened
its stance (13.0 cm) and raised hip kp (0.77) to buy the robustness - acceptable
while power is ignored, and the first thing to revisit when power matters again.

## Passive ankle-roll spring (v5.8 probe, 2026-11-14)
User request: replace the ankle-roll actuators with a passive spring that conforms to the
ground at touchdown and whose restoring force does not disturb posture.
Implementation (k1env_eco.py): env vars K1_ANKROLL_SPRING [Nm/rad] / K1_ANKROLL_DAMP zero the
three actuation channels (motor/kp/kd) of both ankle-roll joints in the compiled model and
set jnt_stiffness / dof_damping / qpos_spring=0 (same mechanism as the MP toe: 2.63 / 0.04).
Actuator count and ctrl layout unchanged -> checkpoints load as-is.
Zero-shot with the exp57g walk policy (+ stock run policy, spring applied to both envs):
- Straight 1.0 m/s: survival 1.00 for K = 2.63 / 5 / 10 / 20; pelvis roll RMS 0.62-0.67 deg
  (actuated exp57g: 0.68), roll p2p 1.72-1.82 (actuated: 2.16 - slightly BETTER); passive ankle
  articulates RMS ~5 deg, max ~8 deg; hip-roll copper rises 49 -> 58-60 W (hips take over
  lateral balance). Reported P_leg is inflated: tau_sub still books phantom PD torque/power
  for the dead ankle channels - power accounting must be fixed before any power claim.
- Full course n=8: survival 1.00 at K=10 and K=5, stop 1.00, brake 1.00 (4.44 s).
- BUT in-place turning fails zero-shot: survival 0.625 at K=10 (falls 9.7-13.4 s, first turn
  segment), 0.0 at K=5. In-place pivoting is where the active ankle roll was doing real work.
Conclusion: feasible; straight/run/brake transfer zero-shot, in-place turning (and final
polish / honest power accounting) need spring-aware retraining.

## Spring-aware retraining (exp58a, from exp57g, K=10 / D=0.6)
Recipe: train_env4.sh = brake-repair recipe + K1_ANKROLL_SPRING=10, P_FAST_IP 0.3 -> 0.5,
lr 3e-5 -> 2e-5, w_energy 0. 300 iterations run; the FINAL (300 it) checkpoint regressed
(course 0.625 - brake hand-over falls at 50.3 s again; in-place 0.75), and the 100 it
checkpoint had in-place 0.25 (falls at the turn reversal ~16.1 s). The 200 it checkpoint
(runs/exp58a/model_best.pt) passes everything at n=8: course 1.00, in-place 1.00,
stop 1.00 (5.83 s), brake 1.00 (4.40 s).
Straight diag (spring on): 1.0 m/s roll RMS 0.49 deg, p2p 1.51 deg, arm roll p2p 7.8 deg,
ankle roll RMS 4.7 deg (max 7.5), hip-roll copper 60.7 W (still above the actuated 49 W -
hips keep compensating; hip kp_scale rose to 0.88). P_leg readings (246.8 W) remain
phantom-inflated by the dead ankle channels and are NOT valid power numbers.
Lesson: under the spring, the brake and in-place basins are both fragile transients;
checkpoint selection along the run mattered more than total iterations.

## Running with the spring + push-robustness control (2026-11-16)
Running was never separately retrained: the stock v5.6.3 run policy, with the spring applied
to the run env as well, already passes the full course (3.0 / 4.5 m/s running, running turns,
brake from 4.5) at n=8 survival 1.00 together with exp58a/model_best.
Push test (--push, n=8) is harsh for every variant: course survival v5.6.3 stock 0.875,
exp57g actuated 0.875, exp58a spring 0.625; brake 0.75 / 0.75 / 0.625; in-place 0.25 / 0.625 /
0.25. The spring costs some disturbance margin (mainly stop/brake transients) but is the
same order as the actuated baselines; push seeds/timing differ per run, so treat as noisy.
A spring-native run-policy retrain is the obvious next step if running margin matters.

## Corrected power accounting (2026-11-16)
The spring's tau_sub entries for ankle roll (idx 5,11) were fictitious PD torques. Fix: when
K1_ANKROLL_SPRING > 0, tau_sub[:, :, [5,11]] is zeroed in k1env_walk3 / k1env_eco / k1env_run4
(passive joint = no electrical power). Diag independently decomposes P into total / ankle /
rest; decomposition is exact (parts sum to the env total) and after the fix the env P_leg
equals the corrected value (165.2 W), while actuated numbers are unchanged.
Straight walking, n=8, same conditions (P in W; ankle = ankle-roll motors' share):
  v=1.0: v5.6.3 126.8 (ankle 9.2, hip copper 33.4, roll RMS 3.00)
         exp57g 155.1 (ankle 8.5, hip copper 49.1, roll RMS 0.68)
         exp58a spring 165.2 corrected (display was 246.8; phantom part 81.6; hip copper 60.7, roll RMS 0.49)
  v=0.6: 96.1 / 123.8 / 124.7 corrected     v=1.4: 164.3 / 209.4 / 228.9 corrected
Reading: the ankle motors only ever cost ~9 W (~7%); removing them saves that, but the spring
policy currently pays ~+19 W extra elsewhere vs exp57g (hip-roll copper 60.7 vs 49.1, hip
kp_scale 0.88) because it was trained with w_energy = 0. The stability gain is real
(roll RMS 0.49, p2p 1.51 - best of the three) but is bought with hip torque. A power-aware
retrain under the spring (now with honest accounting) is the open opportunity.

## Power-aware spring retraining (exp58b, 2026-11-17)

From exp58a/model_best.pt, 300 it, train_env5.sh = train_env4 + --w_energy 0.0015,
HIPCU_W=0.003, W_KPROLL=0.15 (KPROLL_SOFT=0.75); posture terms unchanged
(W_LEVEL=1.2, deadband 0, ARM_LAT_W=3.0). Energy now uses the corrected
accounting (ankle-roll tau_sub zeroed), so the reward prices real watts.
Full suite n=8 (spring K=10): model_100 in-place 0.75 (dip), model_200 and
model (300) all 1.00 (brake 4.40/4.44 s). Straight diag n=8, corrected:

| ckpt | 0.6 m/s | 1.0 m/s | hipcu | rollRMS | arm p2p | kp_hip | 1.4 m/s |
|---|---|---|---|---|---|---|---|
| exp58a/model_best | 124.7 | 165.2 | 60.7 | 0.49 | 7.8 | 0.88 | 228.9 |
| exp58b/model_200 | 121.3 | 159.5 | 60.1 | 0.64 | 7.1 | 0.84 | 217.0 |
| exp58b/model (300) | 116.4 | 151.9 | 56.1 | 0.73 | 7.0 | 0.81 | 204.6 |

Power fell monotonically with training while roll RMS rose modestly
(0.49 -> 0.73, still ~4x better than v5.6.3's 3.00 and ~equal to actuated
exp57g's 0.68). At 1.0 m/s the spring policy (151.9 W) now undercuts the
actuated exp57g (155.1 W) at matched posture. Kept: runs/exp58b/model.pt.
Hip copper (56.1) is still above the actuated 49.1 W; the power curve had
not flattened by it300, so longer training likely buys more.

## Longer power-aware spring training (exp58c, 2026-11-17)

From exp58b/model.pt, +700 it, same train_env5.sh (cumulative power-aware
1000 it). Straight diag @1.0 (spring K=10, n=8, corrected): power fell from
exp58b's 151.9 W into 137.7-147.9 W across checkpoints; hipcu ~50-55.
Full suite n=8 all 1.00 for model_300, model_600 and model (700):
brake 4.54/4.45/4.46 s. Two keepers:

| ckpt | 0.6 | 1.0 | rollRMS@1.0 | arm | hipcu | 1.4 |
|---|---|---|---|---|---|---|
| exp58c/model_600 | 108.1 | 142.8 | 0.44 | 6.0 | 53.6 | 199.3 |
| exp58c/model (700) | 108.8 | 137.7 | 0.63 | 6.6 | 55.0 | 184.2 |

model_600 dominates exp58a/model_best on both power (142.8 < 165.2) and
roll (0.44 < 0.49) at 1.0 m/s. model (final) is the cheapest: 137.7 W at
1.0 m/s = -27.5 W (-17%) vs the power-blind spring exp58a, and 17.4 W
below the actuated exp57g (155.1) at comparable roll (0.63 vs 0.68).
Primary kept: runs/exp58c/model.pt; posture-priority alt: model_600.pt.
Remaining gap to v5.6.3 (126.8 W) is the price of near-zero sway itself;
hipcu plateaued ~53-55 W (actuated exp57g: 49.1).
