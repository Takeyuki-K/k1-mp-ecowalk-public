# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Running turns through the gait manager (8 robots): does the running policy still lean into the turn like v5.5?
For each case (v, w), last 4 s of 10 s: measured speed, yaw rate, lean (pelvis roll toward the inside of the turn,
mean) vs the physical target atan(v w / g), leg power, survival. Left and right turns are listed separately so that
the effect of the mirror-symmetry loss can be seen.
python3 eval_runturn56.py runs/final/walk.pt runs/final/run.pt [--json out.json]
"""
import argparse, json
import numpy as np
from eval_gait56 import Gait55, run_profile, yaw_of, HZ

CASES = [(3.0, 0.5), (3.0, -0.5), (4.0, 0.7), (4.0, -0.7), (4.5, 0.0)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('walk'); ap.add_argument('run')
    ap.add_argument('--json', default=''); ap.add_argument('--n', type=int, default=8)
    a = ap.parse_args()
    g = Gait55(a.walk, a.run, n=a.n, nthread=2)
    res = []
    for v, w in CASES:
        L = run_profile(g, [(1, 'cmd', 0.0, 0.0), (4, 'cmd', 1.65, 0.0), (3, 'cmd', v, 0.0), (6, 'cmd', v, w)], seed=3)   # turn once running
        alive = L['alive'].copy()
        V, Y, R, P, M = [], [], [], [], []
        for k in range(4 * HZ):
            term, _ = g.step(); alive &= ~term
            q = g.qpos(); qv = g.active('state')[:, 1 + g.W.m.nq:1 + g.W.m.nq + g.W.m.nv]
            yw = yaw_of(q)
            V.append(np.cos(yw) * qv[:, 0] + np.sin(yw) * qv[:, 1]); Y.append(yw)
            _, gr = g.R.base_frame(); _, gw = g.W.base_frame()
            gv = np.where(g.mode[:, None] == 1, gr, gw)
            R.append(np.degrees(np.arcsin(np.clip(gv[:, 1], -1, 1)))); P.append(np.where(g.mode == 1, g.R.P_elec, g.W.P_elec))
            M.append(g.mode.copy())
        al = alive
        r = dict(v_cmd=v, w_cmd=w, survival=float(al.mean()))
        if al.any():
            vm = float(np.array(V)[:, al].mean()); yaw = np.unwrap(np.array(Y), axis=0)
            wm = float(((yaw[-1] - yaw[0]) / 4.0)[al].mean())
            r.update(v=round(vm, 2), w=round(wm, 3), lean_deg=round(float(np.array(R)[:, al].mean()), 1),
                     lean_target_deg=round(float(np.degrees(np.arctan(vm * wm / 9.81))), 1),
                     P_W=round(float(np.array(P)[:, al].mean()), 0), run_frac=round(float(np.array(M)[:, al].mean()), 2))
        res.append(r)
        print(json.dumps(r), flush=True)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
