# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Compare the released speed policy with the end-to-end re-trained one (same protocol as the README table:
dt 2 ms, 3 robots per speed, 12 s from standing, measured after 4 s; plus the speed-profile test).
python3 compare_repro.py runs/final/model.pt runs/repro_final_code/model.pt out/repro_final_code.json"""
import sys, json
import numpy as np, torch
from eval_speed import sweep, profile
SPEEDS = [0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.35]
out = {}
for name, path in (('released', sys.argv[1]), ('retrained_final_code', sys.argv[2])):
    res = sweep(path, SPEEDS, per=3, dt=0.002)
    pr = profile(path); L = pr['log']; on = L['alpha'] > 0.99
    out[name] = dict(policy=path, sweep=res, profile_fell=bool(pr['fell']),
                     profile_err=float(np.abs(L['vx'] - L['v_ref'])[on].mean()))
    print(name, 'profile fell', pr['fell'])
    for r in res:
        print(f"  {r['v_cmd']:.2f}: surv {r['survival']:.2f} v {r['v_meas']:.3f} step {r['step_len']:.3f} cad {r['cadence']:.0f} "
              f"heel {r['heel_first']:.2f} P {r['P_elec']:.1f} W CoT {r['CoT']:.3f}")
json.dump(out, open(sys.argv[3], 'w'), indent=1)
