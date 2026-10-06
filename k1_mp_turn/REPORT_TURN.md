# Turning, in-place turning and safe stop: report and decision log

**K1 + passive MP toe joints + human-gait imitation + learned relaxation, now with a yaw-rate command.**
Idea & direction: Takeyuki-K · Implementation generated with Claude (Anthropic) under Takeyuki-K's direction · MuJoCo simulation only

Goal (user instruction): turn while walking **and** on the spot; on a stop command stop safely; use relaxation for
impact absorption during turning; if one of the two turning modes cannot be learned, leave it as future work.
**Both modes were learned.** Remaining weak spot: very fast turning (1.0 rad/s) at almost zero speed (§4).

## 1. Result (final policy `runs/final/model.pt`)

8 robots per test, start from standing, 12 s, measured after 4 s, no pushes, physics dt 5 ms,
each robot starts with a slightly different joint pose (1° rms) so that survival is a rate, not all-or-nothing.

### Walking turn and in-place turn
| command v [m/s] / ω [rad/s] | survival | measured ω | measured v | leg power | foot force p95 |
|---|---|---|---|---|---|
| 0.6 / +0.5 | 100 % | +0.499 | 0.61 | 94 W | 1.29 BW |
| 0.6 / −0.5 | 100 % | −0.499 | 0.61 | 94 W | 1.25 BW |
| 1.0 / +1.0 | 100 % | +0.999 | 0.85 | 133 W | 1.31 BW |
| 0.3 / +0.3 | 100 % | +0.305 | 0.27 | 72 W | 1.21 BW |
| **0 / ±0.3 (in place)** | 100 % | ±0.30 | drift 2.5–2.8 cm/s | 32 W | 0.94 BW |
| **0 / ±0.6 (in place)** | 100 % | +0.59 / −0.60 | drift 2.0–2.1 cm/s | 45–52 W | 0.99 BW |
| 0 / +0.8 (in place) | 100 % | +0.80 | – | 62 W | – |
| 0 / +1.0 (in place) | 87.5 % | +1.00 | drift 1.6 cm/s | 73 W | 1.19 BW |

Survival map (8 robots, 10 s): 100 % for all v ∈ {0, 0.15, 0.3, 0.6, 1.0} × ω ∈ {0.3, 0.6, 0.8, 1.0}
**except v 0.15 / ω 1.0 (12.5 %)**; yaw-rate error ≤ 2 % in every surviving cell (`out/grid_final.json`).

### Safe stop (stop command at t = 6 s)
| before the stop | survival | time to standing | distance after the command | standing at the end |
|---|---|---|---|---|
| straight 1.2 m/s | 100 % | 3.3 s | 1.10 m | 8/8 |
| walking turn 0.6 m/s / 0.5 rad/s | 100 % | 2.3 s | 0.29 m | 8/8 |
| in-place turn 0.6 rad/s | 100 % | 1.7 s | 0.02 m | 8/8 |

Stop sequence: speed and yaw rate are first ramped down (0.6 m/s², 1.5 rad/s²) to stepping in place, then the feet are
brought together (the reference blends to the standing pose over one gait cycle) — the robot never stops mid-stride.

### Straight walking regression vs v2 (same protocol, `reg_v2.py`)
| command | v2 speed error | **v3** speed error | v2 heading drift | **v3** heading drift | v2 leg power | **v3** leg power |
|---|---|---|---|---|---|---|
| 0.30 | 0.8 % | 4.9 % | 4.0° | 0.5° | 79 W | **66 W (−16 %)** |
| 0.60 | 1.1 % | 7.5 % (faster) | 1.5° | 2.8° | 85 W | 91 W (+8 %) |
| 0.90 | 0.7 % | 4.6 % (faster) | 1.4° | 3.6° | 104 W | 120 W (+16 %) |
| 1.20 | 1.2 % | 1.3 % | 2.2° | 1.2° | 135 W | 166 W (+23 %) |
| 1.35 | 4.3 % | 1.6 % | 3.2° | 2.4° | 153 W | 195 W (+27 %) |

- No falls; speed and heading still follow the command (worst 7.5 %, 3.6°).
- **Honest regression: straight walking at 0.9–1.35 m/s costs 16–27 % more power than v2.** Turning needs the
  legs to stay stiffer (mean Kp scale during turns 0.65–0.72). v2 remains the best policy for long straight walks;
  a power fine-tuning stage for v3 is listed as future work.

### Impact absorption
Peak foot force (95th percentile) stays at 1.2–1.3 body weights in walking turns and ≈ 1.0 in in-place turns
(penalty starts at 1.3 BW). Legs stay partly relaxed while turning (Kp scale 0.65–0.72 of nominal on average,
lowest values ≈ 0.45 during in-place turning, visible in the video).

Video: `out/K1_turning.mp4` (stand → walk → walking turns L/R → straight 1.2 m/s → **stop command** → in-place turns
L/R → stop; overlays: command vs measured v/ω, leg Kp, foot force, top-view path) and `out/K1_turning.gif` (2× speed).

## 2. Decisions and reasons

| # | decision | reason |
|---|---|---|
| T1 | **v = 0 reference = stepping in place** (stride 0, cadence 0.655 × human, 87 steps/min, cycle 1.38 s) added to the speed library | An in-place turn needs a stepping pattern without forward stride; the walk-ratio law alone would give a cadence of 0. |
| T2 | Yaw command: inputs ω_cmd and ω_ref (ramped at 1.5 rad/s²), target heading integrates ω_ref; rewards on yaw rate and heading; v2 policy reused with zero-weighted new inputs | Same approach as the heading hold of v2 (D11); iteration 0 behaves exactly like v2. |
| T3 | **Safe stop** = decelerate first, then legs together | Stopping the legs mid-stride at 1.2 m/s would need a large braking step; ramping down to stepping in place keeps the human gait until the speed is gone. |
| T4 | **In-place turning pattern** added to the reference: stance hip yaw unwinds while the pelvis turns, swing leg re-opens; stance 0.05 rad wider; faded out above 0.4 m/s | Found during evaluation: after the first run walking turns already worked (< 1 % error) but **in-place turning failed 100 %** — the legs crossed and the robot fell (training logs in `runs/logs/`). The reference had no turning motion at v = 0, so the policy had to invent it. |
| T5 | Heading target may lead the robot by at most 0.5 rad ("leash"); sharper yaw-rate and heading rewards | Found during evaluation: the policy learned to **stand still** under an in-place command (0.05 rad/s for a 0.3 command) because the yaw reward was too flat and the heading error wrapped around / saturated. |
| T6 | **In-place curriculum**: 60 % in-place commands for 2000 iterations, then the normal mix for 1000 iterations | With the normal mix (25–35 % in-place) in-place turning was not learned in 1100 iterations; with the curriculum it was learned within ~900 iterations; the consolidation stage restored straight-walking speed tracking (0.3 m/s error 15.6 % → 4.9 %). |
| T7 | Impact penalty on foot force above 1.3 body weights (weight 0.15) | User instruction: relaxation as shock absorption during turns. |
| T8 | Bug fix found before training: the gait-choice clip had a lower bound of 0.25 m/s | At v = 0 the "in-place" reference was walking forward (ratio 1.83). Lower bound set to 0. |
| T9 | Final = end of consolidation stage | All turning / stop tests pass; better straight-speed tracking than the curriculum checkpoint (which had 15.6 % error at 0.3 m/s) at the price of higher power at speed (§1). |

## 3. Training stages
| stage | change | iterations (× 4096 steps) |
|---|---|---|
| turn1 | yaw command, stepping-in-place reference, safe stop, init from v2 | 370 |
| turn2 | (continued, one core per training) | 840 |
| turn3 | in-place turning pattern (T4), more in-place samples | 1210 |
| turn4 | heading leash, sharper yaw rewards (T5) | 700 |
| turn5 | in-place curriculum (T6) | 2000 |
| turn6 | consolidation with the normal mix (final) | 1000 |
Total ≈ 25 M environment steps.

## 4. Unsolved / future work
- **Fast turn at almost zero speed**: v 0.15 m/s with ω 1.0 rad/s survives only 12.5 %; in place at 1.0 rad/s 87.5 %.
  Recommended command limit for now: |ω| ≤ 0.8 rad/s when v < 0.3 m/s. Likely cause: the in-place pattern (T4) is
  faded out between 0 and 0.4 m/s, so at 0.15 m/s neither the in-place pattern nor the walking-turn strategy is
  fully active.
- Power regression on straight walking at 0.9–1.35 m/s (+16–27 % vs v2).
- No lateral (sideways) walking, no turning while running.
- Simulation only; Km assumed.

## 5. Files
`retarget_speed.py` (library incl. v = 0), `k1env_turn.py` (env), `ppo_turn.py` (training), `eval_turn.py`,
`grid_turn.py`, `reg_v2.py` (v2 baseline under the same protocol), `video_turn.py`.

## Reproducibility
Evaluation of the released checkpoints is reproducible with the included code. The training was staged; the exact
commands, the checkpoints carried over and the code changes between stages are listed in
[TRAINING_HISTORY.md](TRAINING_HISTORY.md). Stages that ran with earlier code cannot be replayed identically.
