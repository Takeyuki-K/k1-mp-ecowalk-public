# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""(v5.5 baseline: run from ../k1_mp_gait55 with PYTHONPATH=.)
Electrical power of walking, turning and running through the gait manager (8 robots from standing, no pushes).

Each case: stand 1 s, command (v, w) for the case duration, measure the last 5 s:
  P_leg  leg motors only (comparable with v5.5, which did not count the arms)
  P      electrical power of the active policy's motors [W] (legs: copper loss + positive mechanical power,
         motor model of v5; walking also the arm motors)
  v, w   measured forward speed [m/s] and yaw rate [rad/s]
  CoT    cost of transport P / (m g v) (only for v >= 0.3 m/s; m = 35.7 kg)
  mode   fraction of the time in the running policy
python3 eval_power56.py runs/final/walk.pt runs/final/run.pt [--json out.json]
"""
import argparse, json
import numpy as np
from eval_gait55 import Gait55, run_profile, yaw_of, HZ

MG = 35.7 * 9.81
CASES = [  # (group, v, w, duration)
    ('walk', 0.6, 0.0, 9), ('walk', 1.0, 0.0, 9), ('walk', 1.4, 0.0, 9),
    ('walk turn', 0.6, 0.5, 9), ('walk turn', 0.6, -0.5, 9), ('walk turn', 1.0, 0.6, 9), ('walk turn', 1.0, -0.6, 9),
    ('in-place', 0.0, 0.3, 10), ('in-place', 0.0, -0.3, 10), ('in-place', 0.0, 0.6, 10), ('in-place', 0.0, -0.6, 10),
    ('in-place', 0.0, 1.0, 10), ('in-place', 0.0, -1.0, 10),
    ('run', 2.5, 0.0, 12), ('run', 3.5, 0.0, 12), ('run', 4.5, 0.0, 12),
    ('run turn', 3.0, 0.5, 12), ('run turn', 3.0, -0.5, 12), ('run turn', 4.5, 1.0, 12), ('run turn', 4.5, -1.0, 12),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('walk'); ap.add_argument('run')
    ap.add_argument('--json', default=''); ap.add_argument('--n', type=int, default=8)
    a = ap.parse_args()
    g = Gait55(a.walk, a.run, n=a.n, nthread=2)
    res = []
    for grp, v, w, T in CASES:
        L = run_profile(g, [(1, 'cmd', 0.0, 0.0), (T - 5, 'cmd', v, w)], seed=3)
        alive = L['alive'].copy()
        P, PL, V, Y, M = [], [], [], [], []
        for k in range(5 * HZ):
            term, _ = g.step(); alive &= ~term
            q = g.qpos(); qv = g.active('state')[:, 1 + g.W.m.nq:1 + g.W.m.nq + g.W.m.nv]
            yw = yaw_of(q)
            P.append(np.where(g.mode == 1, g.R.P_elec, g.W.P_elec)); PL.append(np.where(g.mode == 1, g.R.P_elec, getattr(g.W, 'P_leg', g.W.P_elec))); V.append(np.cos(yw) * qv[:, 0] + np.sin(yw) * qv[:, 1])
            Y.append(yw); M.append(g.mode.copy())
        al = alive
        r = dict(group=grp, v_cmd=v, w_cmd=w, survival=float(al.mean()))
        if al.any():
            p = float(np.array(P)[:, al].mean()); vm = float(np.array(V)[:, al].mean())
            yaw = np.unwrap(np.array(Y), axis=0)
            r.update(P_W=round(p, 1), P_leg_W=round(float(np.array(PL)[:, al].mean()), 1), v=round(vm, 2), w=round(float(((yaw[-1] - yaw[0]) / 5.0)[al].mean()), 2),
                     CoT=round(p / (MG * vm), 3) if vm >= 0.3 else None, run_frac=round(float(np.array(M)[:, al].mean()), 2))
        res.append(r)
        print(json.dumps(r), flush=True)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
