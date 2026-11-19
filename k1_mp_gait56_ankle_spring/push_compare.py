# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Same straight profile and same independent pushes for v5 (gait.py) and v5.5 (gait55.py).
SEED=1 python3 push_compare.py v5
SEED=1 python3 push_compare.py v55 runs/final/walk.pt runs/final/run.pt"""
import sys, os, json, numpy as np
which = sys.argv[1]
N = 16
if which == 'v5':
    sys.path.insert(0, '/root/k1-mp-ecowalk-public/k1_mp_gait'); os.chdir('/root/k1-mp-ecowalk-public/k1_mp_gait')
    from gait import Gait
    g = Gait(N, 'runs/final/walk.pt', 'runs/final/run.pt')
    def step(v, t): g.step(np.full(N, v), t); return g.alive.copy()
    envs = (g.W, g.R)
else:
    sys.path.insert(0, '/root/k1-mp-ecowalk-public/k1_mp_gait55'); os.chdir('/root/k1-mp-ecowalk-public/k1_mp_gait55')
    import eval_gait55 as E
    from gait55 import Gait55
    g = Gait55(sys.argv[2], sys.argv[3], n=N, nthread=2); E.stand_noise(g)
    g._oaw, _ = g.W.obs(); g._oar, _ = g.R.obs()
    alive = np.ones(N, bool)
    def step(v, t):
        global alive
        if v > 0: g.command(v, 0.0)
        else: g.stop()
        term, _ = g.step(); alive &= ~term; return alive.copy()
    envs = (g.W, g.R)
rng = np.random.default_rng(int(os.environ.get('SEED', '0')))
prof = [(2, 0.0 if which != 'v5' else 0.0), (5, 1.0), (6, 4.5), (8, 0.0)]
k = 0; out = []; prev = np.ones(N, bool); ft = {}
for T, v in prof:
    for _ in range(int(T * 50)):
        hit = rng.random(N) < 1 / 150                      # independent per robot (no synchronised pushes)
        dv = rng.uniform(-0.3, 0.3, (N, 2)) * hit[:, None]
        for e in envs:
            e.state[:, 1 + e.m.nq:1 + e.m.nq + 2] += dv
        a = step(v, k); k += 1
        for i in np.where(prev & ~a)[0]: ft[int(i)] = round(k/50, 2)
        prev = a
    out.append(float(a.mean()))
print(which, 'survival after each segment', out, 'falls', sorted(ft.values()))
