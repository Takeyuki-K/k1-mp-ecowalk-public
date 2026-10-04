# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""ROBOTIS public walk_default policy: power vs commanded speed (same electrical model, dt 2 ms).
Its trained forward range is up to 1.0 m/s; 1.2 / 1.35 are tested as extrapolation."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'k1_compare'))
import compare3 as C

out = {}
for v in [0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.2, 1.35]:
    r = C.run_robotis(v)
    out[str(v)] = dict(v_cmd=v, v_meas=r['speed'], P_total=r['P_total'], P_legs=r['P_legs'], fell=r['fell'])
    print(v, {k: round(x, 3) if isinstance(x, float) else x for k, x in out[str(v)].items()}, flush=True)
json.dump(out, open(os.path.join(HERE, 'out', 'robotis_sweep.json'), 'w'), indent=1)
