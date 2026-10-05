# Fast walking (eco) up to 1.65 m/s: report

**K1 + passive MP toes + human-gait imitation + learned relaxation; speed command extended 1.35 → 1.65 m/s.**
Idea & direction: Takeyuki-K · Implementation: Claude (Anthropic) · MuJoCo simulation only

User instruction: speeds around 1.35–1.6 m/s should be covered by **walking with longer steps** (not by running),
and the power-saving reward stays.

## Result (final policy `runs/final/model.pt`)
Same protocol as the v2 table (dt 2 ms, 3 robots per speed, 12 s from standing, measured after 4 s).

| command | v2 speed | **fast-walk** speed | v2 step / cadence | **fast-walk** step / cadence | v2 leg power | **fast-walk** leg power | CoT v2 → fast-walk |
|---|---|---|---|---|---|---|---|
| 0.30 | 0.31 | 0.33 | 0.24 m / 78 | 0.25 m / 78 | 86 W | 72 W | 0.79 → 0.62 |
| 0.60 | 0.61 | 0.63 | 0.34 / 107 | 0.34 / 110 | 87 W | 87 W | 0.41 → 0.40 |
| 0.90 | 0.91 | 0.92 | 0.43 / 128 | 0.42 / 130 | 108 W | 114 W | 0.34 → 0.35 |
| 1.20 | 1.19 | 1.21 | 0.51 / 142 | 0.50 / 147 | 144 W | 146 W | 0.35 → 0.34 |
| 1.35 | 1.30 | 1.36 | 0.55 / 148 | 0.53 / 153 | 163 W | 169 W | 0.36 → 0.36 |
| **1.50** | 1.40 (saturated) | **1.51** | 0.55 / 154 | **0.57 / 160** | 183 W | 190 W | 0.37 → 0.36 |
| **1.65** | 1.49 (saturated) | **1.63** | 0.56 / 159 | **0.59 / 166** | 199 W | 216 W | 0.38 → 0.38 |

- No falls, heel contact at every touchdown, heading drift ≤ 3.4°. Speed error ≤ 2 % at 0.6–1.65 m/s
  (+10 % at 0.30 m/s, slightly fast).
- Cost of transport is unchanged (0.34–0.38) from 0.9 to 1.65 m/s: the extra speed is bought with a longer step
  (0.59 m ≈ 0.95 leg length) and a slightly higher cadence, without leaving the walking gait.
- **Walking vs running at the same speed:** the jog policy (v3, 1.54 m/s) used 342 W; fast walking at 1.51–1.63 m/s
  uses 190–216 W, i.e. **about 40 % less**. This confirms the user's judgement that this speed range belongs to
  walking (as in humans, the walk–run transition is near Froude 0.5 ≈ 1.9 m/s for K1).

## Decisions
| # | decision | reason |
|---|---|---|
| F1 | Start from the v2 speed policy (not from the turning policy) | v2 is the most efficient straight walker; the turning policy costs +16–27 % at speed. |
| F2 | Walking library extended with 1.80 / 1.95 m/s (walk-ratio law) and command range to 1.65 m/s | The gait-choice action needs references up to ~1.3 × the command. |
| F3 | Energy reward unchanged (0.0015 · P_elec) | User instruction. |
| F4 | Stage 2: sharper speed reward (kv 15 → 40) | Stage 1 still walked 7 % slow at 1.65 m/s (1.54); stage 2 reaches 1.63 at a CoT of 0.38. |

## Training
| stage | change | iterations |
|---|---|---|
| fw1 | command range 0.30–1.65, library to 1.95, init v2 | 3000 |
| fw2 | kv 40 | 1500 |

Files: `retarget_speed.py` (library 0.30–1.95), `k1env_speed.py` (range), `ppo_speed.py` (`--kv_walk`), `eval_fw.py`,
`out/sweep_v2.json`, `out/sweep_fw2.json`. All other code identical to `k1_mp_speed` (v2, unchanged).
