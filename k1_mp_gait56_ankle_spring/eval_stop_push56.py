# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Stopping under random pushes (+-0.3 m/s, independent per robot, ~every 3 s): falls counted only from the stop
command on (robots that fell before are excluded). Used to check that the foot re-placement does not cost stability.
python3 eval_stop_push56.py runs/final/walk.pt runs/final/run.pt [--n 32]    (PLACE_FF=0 to compare)
"""
import argparse, json
import numpy as np
from eval_gait56 import Gait55, run_profile

CASES = {'inplace+0.6_stop': [(2, 'cmd', 0.0, 0.0), (6, 'cmd', 0.0, 0.6), (10, 'stop', 0, 0)],
         'inplace-1.0_stop': [(2, 'cmd', 0.0, 0.0), (4.1, 'cmd', 0.0, -1.0), (10, 'stop', 0, 0)],
         'turnwalk_stop': [(2, 'cmd', 0.0, 0.0), (8, 'cmd', 0.8, 0.5), (10, 'stop', 0, 0)],
         'run4.5_stop': [(2, 'cmd', 0.0, 0.0), (5, 'cmd', 1.65, 0.0), (6, 'cmd', 4.5, 0.0), (12, 'stop', 0, 0)]}

ap = argparse.ArgumentParser(); ap.add_argument('walk'); ap.add_argument('run'); ap.add_argument('--n', type=int, default=32)
ap.add_argument('--json', default='')
a = ap.parse_args()
g = Gait55(a.walk, a.run, n=a.n, nthread=2)
res = {}
for name, prof in CASES.items():
    L = run_profile(g, prof, push=True, seed=11)
    t_stop = sum(p[0] for p in prof[:-1])
    before = (L['fall_t'] < t_stop)
    after = (L['fall_t'] >= t_stop)
    res[name] = dict(n=a.n, fell_before_stop=int(before.sum()), fell_after_stop=int(after.sum()),
                     survival_after_stop=round(float(1 - after.sum() / max(1, (~before).sum())), 3))
    print(name, json.dumps(res[name]), flush=True)
if a.json:
    json.dump(res, open(a.json, 'w'), indent=1)
