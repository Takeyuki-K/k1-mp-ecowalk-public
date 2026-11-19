# K1 ankle roll as a passive spring (k1_mp_gait56_ankle_spring)

v5.6.3 (k1_mp_gait56) variant in which the ankle **roll** joints have no motor and act as a
passive spring-damper (K = 10 Nm/rad, D = 0.6 Nms/rad). Policies were PPO fine-tuned from v5.6.3.

## Policies (runs/final_spring/)
| file | use | notes |
|---|---|---|
| `walk_Qup075.pt` | walking | level pelvis + relaxed arms (stage Q_up075 it100) |
| `run_K10_D0.6.pt` | running | frozen (it1400) |

`runs/final/` and `runs/init/` are the unchanged v5.6.3 policies used as the baseline / warm start.

## Required settings for `walk_Qup075.pt`
```
ANKLE_SPRING=1 ANK_K=10 ANK_D=0.6 \
ARM_KPM=0.25 REF_LIB=ref_lib_56_w11_sym.npz REF_W=0.11 \
python3 eval_spring.py runs/final_spring/walk_Qup075.pt runs/final_spring/run_K10_D0.6.pt --n 32 --meas 10 --straight
```
`ARM_KPM=0.25` (arm servo gain x0.25) is **not stored in the checkpoint** and must be set.
The 11 cm stance reference is `ref_lib_56_w11_sym.npz` (`make_ref56.py`, `make_sym_ref.py`).

## Results vs v5.6.3 (32 robots, 10 s, straight; power interpolated to the same actual speed)
| command | 0.6 m/s | 1.0 m/s | 1.4 m/s |
|---|---|---|---|
| actual speed | 0.635 | 1.047 | 1.388 |
| power vs v5.6.3 | +12.9 % | +7.9 % | +9.5 % |
| pelvis roll RMS (v5.6.3 ~3.0 deg) | 0.80 deg | 0.94 deg | 0.92 deg |
| torso tilt RMS (v5.6.3 3.16/3.21/3.24) | 1.18 deg | 1.20 deg | 1.57 deg |

- Pushed walking: 1 / 0 / 0 falls in 128 (v5.6.3: 3 / 2 / 1).
- Pushed hard brake survival: 98.4 / 96.9 / 100 / 98.4 %.
- Arm swing fore-aft 28 / 40 / 47 deg, min arm-body clearance 3.9 cm (no contact).
- Power is higher than v5.6.3 because the pelvis is held level (v5.6.3 leans ~3 deg over the stance leg,
  which unloads hip roll). Relaxed arms recovered ~5-7 points vs the level-pelvis policy without arm change.

## Caveats
- Arm gearbox back-drive friction is not modelled; relaxed arms may swing less on hardware.
- Details and the full training history: `NOTES_ANKLE_SPRING.md`.
