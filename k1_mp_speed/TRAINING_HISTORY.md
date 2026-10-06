# Speed policy (v2): training history and reproducibility

## What is reproducible
| | status |
|---|---|
| **Evaluation of the released policy** (`runs/final/model.pt`) | reproducible with the included code: `python3 eval_speed.py runs/final/model.pt` (results in `out/`) |
| **Historical training path A → G** | documented below with the exact checkpoint chain; every stage-final checkpoint is included in `runs/stages/`, every stage log in `runs/logs/` |
| **Re-running stages A–F exactly** | **not possible with the released code**: the reward settings were changed in the code between stages (D6–D10 in `REPORT_SPEED.md`); the released code contains only the final settings (stage G) |
| **End-to-end re-training with the released (final) code** | see "Re-training with the final code" below |

## Historical chain (reconstructed from the saved checkpoints, their stored iteration numbers and the first line of
each stage's stdout, `loaded <file>`; 2 CPU threads, 128 envs × 32 steps per iteration, default seed of the code at the time — 207 in the released code)

| stage | init checkpoint | transfer flag | code settings at that time (differences to the final code) | checkpoint taken | file in `runs/stages/` |
|---|---|---|---|---|---|
| A | `../k1_mp_eco/runs/eco1/model.pt` | `--from_eco` (70 → 72 inputs) | kv_walk 6.0; w_im / w_vel 0.70 / 0.35; energy term = power / speed; absolute speed error; reference with constant pelvis height; 36 actions; no heading input | it 1100 (log runs to 1160) | `stage_A.pt` |
| B | stage A | – | as A but kv_walk 15.0 (D6) | it 600 | `stage_B.pt` |
| C | stage B | `--from_speed36` (+ gait-speed action, 37 actions) | as B (D7) | it 600 | `stage_C.pt` |
| D | stage C | – | reference library rebuilt with per-frame pelvis height (D8) | it 600 | `stage_D.pt` |
| E | stage D | – | relative speed error, plain power penalty (D9) | it 1700 | `stage_E.pt` |
| F | stage E | – | w_im / w_vel 0.45 / 0.50 (D10) | it 1200 | `stage_F.pt` |
| G | stage F | `--add_heading_obs` (+ heading-error input, 74 inputs) | heading hold (D11) = **final code** | it 3500 → `runs/final/model.pt` (identical tensors) | – (`stage_G_candidate_it1700.pt` = the rejected candidate of D12) |

Common arguments of all stages: `--lr 1e-4`; `--lr_min` 1e-5 (default) in A–B, 5e-5 in C–G (inferred from the logged
learning rate, which never went below these values). Example of the form used (stage G):

```bash
python3 ppo_speed.py --iters 3500 --init runs/stages/stage_F.pt --add_heading_obs --lr 1e-4 --lr_min 5e-5 --out runs/speed7
```
Running this command with the released code reproduces stage G itself (stage G used the final code); its starting point
is the included `stage_F.pt`. Stages A–F cannot be re-run identically because their environment settings no longer
exist in the code (column "code settings").

## Re-training with the final code (end-to-end check)
To check that the released code alone produces a comparable policy, the final architecture was trained directly from
the eco policy with the final settings, for the same total number of iterations as A–G (9 300):

```bash
python3 ppo_speed.py --iters 9300 --init ../k1_mp_eco/runs/eco1/model.pt --from_eco_full --lr 1e-4 --lr_min 5e-5 --out runs/repro_final_code
python3 eval_speed.py runs/repro_final_code/model.pt
```
`--from_eco_full` composes the three historical transfers (A, C, G) in one step. This is a single-stage run, i.e.
**not** an exact replay of A–G; RL results also vary with the random seed.

**Result (run on 2026-10-06, 2.6 h on 2 CPU threads):** no falls at 0.30–1.35 m/s, heel-first 100 %, speeds within 3.1 %
and leg power within −18 … 0 % of the released policy (`out/repro_final_code.json`, table in `REPORT_SPEED.md` §7).
The checkpoint is included as `runs/repro_final_code/model.pt`.
