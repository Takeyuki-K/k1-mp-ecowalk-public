# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Combined gait under random pushes: both profiles, 2 seeds x N robots; prints survival and where falls happen."""
import sys, json
import numpy as np
from gait import Gait
from eval_gait import PROFILES, FPS
walk, runp = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SEEDS = [int(x) for x in sys.argv[6].split(',')] if len(sys.argv) > 6 else (5, 6)
res = {}
for prof in (sys.argv[5].split(',') if len(sys.argv) > 5 else ('accel', 'standrun')):
    falls = []; tot = 0
    for seed in SEEDS:
        g = Gait(N, walk, runp, push=True, seed=seed)
        P = PROFILES[prof]; prev = g.alive.copy()
        for k in range(int(P[-1][0] * FPS)):
            t = k / FPS; u = [v for ts, v in P if ts <= t and v is not None][-1]
            g.step(u, t)
            for i in np.where(prev & ~g.alive)[0]:
                falls.append(dict(t=t, mode='run' if g.mode[i] else 'walk', cmd=u))
            prev = g.alive.copy()
        tot += N
    res[prof] = dict(survival=1 - len(falls) / tot, falls=falls)
    print(prof, round(1 - len(falls) / tot, 3), falls, flush=True)
if len(sys.argv) > 4:
    json.dump(res, open(sys.argv[4], 'w'), indent=1)
