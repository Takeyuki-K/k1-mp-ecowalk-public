# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""In-place turning quality through the gait manager (8 robots from standing, 12 s per case).

Measured per case (simulator ground truth, after 3 s settling):
  w_meas       mean yaw rate [rad/s]
  steps_s      touchdowns per second (a foot loaded again after >= 0.1 s unloaded), both feet
  yaw_step     yaw per step [deg] = |w_meas| / steps_s
  lift_L/R     max ankle lift of each foot during the case [cm] (asymmetry = one foot shuffling)
  unload_L/R   fraction of time each foot is unloaded
  pivot        rms yaw rate of a loaded foot [rad/s] (foot spinning on the floor)
  P_leg        leg electrical power [W]
python3 eval_inplace56.py runs/final/walk.pt runs/final/run.pt [--push] [--n 8] [--json out.json]
"""
import argparse, json
import numpy as np
from eval_gait56 import Gait55, run_profile, HZ

CASES = [0.3, 0.6, 1.0, -0.6, -1.0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('walk'); ap.add_argument('run')
    ap.add_argument('--push', action='store_true'); ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    g = Gait55(a.walk, a.run, n=a.n, nthread=2)
    W = g.W
    res = []
    for w in CASES:
        L = run_profile(g, [(1, 'cmd', 0.0, 0.0), (2, 'cmd', 0.0, w)], push=a.push, seed=3)
        alive = L['alive'].copy()
        rng = np.random.default_rng(5)
        yaw0 = W.yaw().copy(); yaws = []
        prev_load = None; air = np.zeros((a.n, 2)); td = np.zeros(a.n)
        zmin = np.full((a.n, 2), 9.0); zmax = np.full((a.n, 2), -9.0); unl = np.zeros((a.n, 2))
        piv = []; P = []
        fy_prev = None
        T = 12 * HZ
        for k in range(T):
            if a.push:
                hit = rng.random(a.n) < 1 / 150
                W.state[:, 1 + W.m.nq:1 + W.m.nq + 2] += rng.uniform(-0.3, 0.3, (a.n, 2)) * hit[:, None]
            term, _ = g.step(); alive &= ~term
            yaws.append(W.yaw().copy())
            load = np.stack([np.concatenate([W.s(f'{s}_{p}_touch') for p in ('heel', 'meta', 'toe')], 1).sum(1)
                             for s in ('left', 'right')], 1) > 5
            fz = np.stack([W.s('left_foot_pos')[:, 2], W.s('right_foot_pos')[:, 2]], 1)
            fx = np.stack([W.s('left_foot_x'), W.s('right_foot_x')], 1); fy = np.arctan2(fx[:, :, 1], fx[:, :, 0])
            if k >= 1:
                td += ((air >= 0.1 * HZ) & load).sum(1)
                wz = np.arctan2(np.sin(fy - fy_prev), np.cos(fy - fy_prev)) * HZ
                piv.append(np.where(load, wz, np.nan))
            air = np.where(load, 0, air + 1)
            fy_prev = fy
            zmin = np.minimum(zmin, fz); zmax = np.maximum(zmax, fz); unl += ~load
            P.append(W.P_elec.copy())
        yaw = np.unwrap(np.array(yaws), axis=0)
        al = alive
        r = dict(w_cmd=w, survival=float(al.mean()))
        if al.any():
            wm = (yaw[-1] - yaw[0]) / 12.0
            steps = td / 12.0
            pv = np.array(piv)[:, al]
            r.update(w_meas=round(float(wm[al].mean()), 3), steps_s=round(float(steps[al].mean()), 2),
                     yaw_step_deg=round(float(np.degrees(np.abs(wm[al]) / np.maximum(steps[al], 1e-3)).mean()), 1),
                     lift_L_cm=round(float(100 * (zmax - zmin)[al, 0].mean()), 1), lift_R_cm=round(float(100 * (zmax - zmin)[al, 1].mean()), 1),
                     unload_L=round(float(unl[al, 0].mean() / T), 2), unload_R=round(float(unl[al, 1].mean() / T), 2),
                     pivot_rms=round(float(np.sqrt(np.nanmean(pv ** 2))), 3),
                     P_leg=round(float(np.array(P)[:, al].mean()), 1))
        res.append(r)
        print(json.dumps(r), flush=True)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
