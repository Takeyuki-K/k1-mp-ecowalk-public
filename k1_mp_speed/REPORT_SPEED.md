# Speed-command walking: report and decision log

**K1 + passive MP toe joints + human-gait imitation + learned relaxation, now following a speed command.**
Idea & direction: Takeyuki-K · Implementation: Claude (Anthropic) · MuJoCo simulation only

Goal: keep the eco walking concept (MP toes × human-gait imitation × relaxation) and make one policy follow a changing
forward speed command. Running is out of scope (separate future work).

## 1. Result (final policy `runs/final/model.pt`)

Straight walking, 12 s per speed, 3 robots per speed, physics dt 2 ms (same as the ROBOTIS comparison),
same electrical model (Σ τ²/Km² + Σ max(τ·ω, 0), Km assumed 4.0 / 2.2 Nm/√W). Leg power compared with leg power.

| command | measured | error | step length | cadence | heading drift | leg power | ROBOTIS leg power | difference |
|---|---|---|---|---|---|---|---|---|
| 0.30 m/s | 0.31 | 3.8 % | 0.24 m | 78 /min | 4.3° | 87 W | 95 W | −8 % |
| 0.45 | 0.45 | 0.9 % | 0.29 | 94 | 3.0° | 77 W | 113 W | −32 % |
| 0.60 | 0.61 | 1.8 % | 0.34 | 107 | 0.3° | 87 W | 130 W | −33 % |
| 0.75 | 0.76 | 1.9 % | 0.39 | 118 | 0.4° | 96 W | 150 W | −36 % |
| 0.90 | 0.91 | 1.5 % | 0.43 | 128 | 2.7° | 108 W | 172 W | −37 % |
| 1.00 | 1.02 | 1.5 % | 0.46 | 133 | 0.4° | 120 W | 189 W | −36 % |
| 1.20 | 1.19 | 1.0 % | 0.50 | 142 | 1.8° | 144 W | 227 W | −36 % |
| 1.35 | 1.30 | 3.4 % | 0.55 | 148 | 4.0° | 163 W | 261 W | −38 % |

- No falls at any speed; heel contact at every touchdown (heel-strike or heel+forefoot); step length **and** cadence both increase with speed (Fig. 2).
- Speed-profile test (stand → 0.4 → 1.2 → 0.6 → 1.35 → 0.3 → stop, step changes): no fall, mean |v − v_ref| 0.06 m/s
  including the acceleration transients.
- Random speed commands + random pushes (±0.35 m/s) + domain randomisation: 87.5 % survival over 16 s (24 robots).
- **Above the training range** (commands 1.50 / 1.65 m/s) the robot does not fall but saturates at 1.38 / 1.45 m/s:
  step length stops growing at ≈ 0.55 m (≈ 0.88 × leg length). This is consistent with the user's expectation that
  higher speeds need a running gait.

Figures: `out/fig_speed_tracking.png`, `out/fig_step_cadence.png`, `out/fig_power_vs_speed.png`;
video: `out/K1_speed_command_comparison.mp4` (ROBOTIS walk_default vs this policy on the same profile).

## 2. Decisions and reasons

| # | decision | reason |
|---|---|---|
| D1 | **Synthesise speed-specific references from the existing human trial** instead of new mocap | Multi-speed open datasets (e.g. figshare "multimodal dataset of human gait at different walking speeds") could not be downloaded from this environment. A slow-walk file found on GitHub (GaitPhase subset, 0.6 m/s) has only foot markers (no hip → cannot do hip→ankle imitation) and no stated licence. |
| D2 | Scale with the **walk-ratio law**: step length ∝ √v, cadence ∝ √v | In healthy adults the walk ratio (step length / cadence) is roughly constant over normal speeds (e.g. Sekiya & Nagasaki 1998). It also matches the instruction "slower = shorter steps, faster = longer steps". Hip→ankle *distance* is kept per frame (knee pattern preserved); foot pitch (heel-strike / toe-off angle) scaled with step length. |
| D3 | Command range **0.30 – 1.35 m/s**; reference library 0.30 – 1.65 m/s | Walk–run transition happens near Froude number 0.5 (≈ 1.85 m/s for the 0.70 m hip height). 1.35 m/s (Fr ≈ 0.27) keeps a clear margin and stays within motor limits; the library above 1.35 is only used by the gait-choice action (D7). 0.30 m/s lower bound: below this, single support becomes very long (cycle 1.7 s) and walking becomes quasi-static. |
| D4 | Reference speed follows the command with **0.6 m/s² acceleration limit**; references of neighbouring speeds blended at the same phase | Human-like acceleration; all cycles start at right heel strike so blending is phase-consistent and the phase rate changes smoothly. |
| D5 | Start from the eco single-speed policy (new inputs zero-weighted) | Keeps the learned heel-to-toe gait and relaxation; behaviour at iteration 0 identical to the eco policy. |
| D6 | Velocity reward sharpened (stage B) | Stage A walked ~15 % slower than commanded at every speed. |
| D7 | **Gait-choice action**: policy picks which human-derived gait (step length + cadence) to imitate, ±30 % around the command | Lets the robot adapt the human gait to its own body instead of fighting the imitation reward joint by joint. Final policy uses ≈ 1.05 × on average (slightly shorter steps, higher cadence than the human-scaled reference, Fig. 2). |
| D8 | **Pelvis height per frame** so the stance foot touches the ground exactly | Found during debugging: with constant pelvis height the scaled reference foot penetrated / floated up to 30 mm at long strides, which cut steps short. Now within ±2 mm at all speeds. |
| D9 | Plain power penalty and **relative** speed error | Power-per-distance made standing still "cheap" at 0.3 m/s (robot stalled at 0.17 m/s); a relative error gives the same pressure for 10 % error at 0.3 and 1.35 m/s. |
| D10 | Imitation weight 0.70 → 0.45, velocity 0.35 → 0.50 | Imitation dominated and kept ~10 % under-speed; with lower weight the human gait acts as a style prior (similar to AMP-style approaches) while speed is the main objective. Heel-first ratio stayed 100 %. |
| D11 | **Heading hold** (heading-error input, heading reward, velocity measured in the target-heading frame) | Found during evaluation: the robot drifted up to 87° in 10 s at slow speed, so world-x speed looked 25 % too low although path speed was correct. The policy could not see its yaw. Now drift ≤ 4.3°. |
| D12 | Checkpoint at 3500 iterations of stage G | Best push/DR survival (87.5 %) and lowest high-speed power among 1770/2000/2500/3000/3500, relative error 1.9 %, max heading drift 6° (2500 had 1.8° but lower push survival). |

## 3. Training stages
| stage | change | iterations (× 4096 steps) |
|---|---|---|
| A | speed library + speed command, init from eco | 1160 |
| B | sharper velocity reward | 600 |
| C | + gait-choice action | 610 |
| D | per-frame pelvis height in the reference | 610 |
| E | relative speed error, plain power penalty | 1720 |
| F | imitation / velocity rebalance | 1240 |
| G | heading hold | 3500 |
Total ≈ 39 M environment steps on 2 CPU cores (MuJoCo 3.14, PPO).

## 4. Limitations (honest list)
- Simulation only; Km assumed; gearbox friction not modelled.
- The slow and fast human references are **synthesised** from one human trial by a general law, not measured.
  Replace with measured multi-speed mocap when available (pipeline unchanged: `retarget_speed.py`).
- At 0.30 m/s power is higher than at 0.45 m/s (long single support needs holding torque, mostly hip roll); the gain over
  ROBOTIS is small there (−8 %).
- Forward speed only (no lateral / turning / backward commands). The heading target input can be extended to turning.
- ROBOTIS walk_default is a general omnidirectional policy designed for the real robot; ours is specialised.
- Above ~1.4 m/s a running gait is needed (not trained, as instructed).

## 5. Files
`retarget_speed.py` (speed library), `k1env_speed.py` (env), `ppo_speed.py` (training + weight transfers),
`eval_speed.py`, `select_ckpt.py`, `robotis_sweep.py`, `plot_speed.py`, `record_profile.py`, `video_profile.py`.

## 6. Reproducibility note
Stages A–F were run while the reward settings were being corrected (D6–D10); the code in this folder contains the
**final** settings (stage G). The released policy is `runs/final/model.pt`; stage logs are in `runs/logs/`.
Re-running all stages from scratch with the final code (init from `../k1_mp_eco/runs/eco1/model.pt` with `--from_eco`,
then `--from_speed36`, then `--add_heading_obs`) follows the same path but has not been re-executed end-to-end.
