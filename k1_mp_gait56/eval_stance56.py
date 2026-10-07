# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Final stance after stopping, through the gait manager (8 robots, no pushes).

The stance is MEASURED with simulator ground truth (world poses of both feet), independent of what the controller
uses (the controller only uses joint-angle kinematics and foot contact sensors):
  dx   front-back offset of the ankles along the feet's mean heading [cm]
  dyaw relative yaw of the feet [deg]
measured when the robot reaches standing (alpha = 0) and at the end of the case ("_end", several seconds later);
values are [mean, max] over the robots.
python3 eval_stance56.py runs/final/walk.pt runs/final/run.pt [--json out.json]
(PLACE_FF=0 switches the re-placement feed-forward off, PLACE_GAIN sets its gain)
"""
import argparse, json
import numpy as np
from eval_gait56 import Gait55, run_profile, HZ

CASES = {
    'inplace+0.6_stop': [(2, 'cmd', 0.0, 0.0), (12, 'cmd', 0.0, 0.6), (10, 'stop', 0, 0)],
    'inplace-1.0_stop': [(2, 'cmd', 0.0, 0.0), (4.1, 'cmd', 0.0, -1.0), (10, 'stop', 0, 0)],
    'inplace+-0.6_stop': [(2, 'cmd', 0.0, 0.0), (12, 'cmd', 0.0, 0.6), (12, 'cmd', 0.0, -0.6), (10, 'stop', 0, 0)],
    'walk1.2_stop': [(2, 'cmd', 0.0, 0.0), (8, 'cmd', 1.2, 0.0), (10, 'stop', 0, 0)],
    'turnwalk_stop': [(2, 'cmd', 0.0, 0.0), (8, 'cmd', 0.8, 0.5), (10, 'stop', 0, 0)],
    'run4.5_stop': [(2, 'cmd', 0.0, 0.0), (5, 'cmd', 1.65, 0.0), (6, 'cmd', 4.5, 0.0), (12, 'stop', 0, 0)],
    'run4.5_brake': [(2, 'cmd', 0.0, 0.0), (5, 'cmd', 1.65, 0.0), (6, 'cmd', 4.5, 0.0), (12, 'brake', 0, 0)],
}


def stance_truth(W):
    fl, fr = W.s('left_foot_x'), W.s('right_foot_x')
    yl, yr = np.arctan2(fl[:, 1], fl[:, 0]), np.arctan2(fr[:, 1], fr[:, 0])
    dyaw = np.arctan2(np.sin(yl - yr), np.cos(yl - yr)); ym = yr + dyaw / 2
    d = W.s('left_foot_pos') - W.s('right_foot_pos')
    return np.cos(ym) * d[:, 0] + np.sin(ym) * d[:, 1], -np.sin(ym) * d[:, 0] + np.cos(ym) * d[:, 1], dyaw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('walk'); ap.add_argument('run')
    ap.add_argument('--json', default=''); ap.add_argument('--n', type=int, default=8)
    a = ap.parse_args()
    g = Gait55(a.walk, a.run, n=a.n, nthread=2)
    res = {}
    for name, prof in CASES.items():
        L = run_profile(g, prof[:-1])
        alive = L['alive'].copy()
        T, kind = prof[-1][0], prof[-1][1]
        g.stop() if kind == 'stop' else g.brake()
        t_st = np.full(g.n, np.nan); at_stand = np.full((3, g.n), np.nan)
        for k in range(int(T * HZ)):
            term, _ = g.step(); alive &= ~term
            st = alive & (g.mode == 0) & (g.W.alpha == 0) & np.isnan(t_st)
            if st.any():
                t_st[st] = k / HZ
                at_stand[:, st] = np.array(stance_truth(g.W))[:, st]
        end = np.array(stance_truth(g.W))
        s_ = alive & ~np.isnan(t_st)
        r = dict(survival=float(alive.mean()), standing=float(s_.mean()),
                 t_to_stand=round(float(np.nanmean(t_st[s_])), 2) if s_.any() else None)
        if s_.any():
            for tag, v in (('', at_stand), ('_end', end)):
                r.update({f'dx_cm{tag}': [round(float(np.abs(100 * v[0, s_]).mean()), 1), round(float(np.abs(100 * v[0, s_]).max()), 1)],
                          f'dyaw_deg{tag}': [round(float(np.degrees(np.abs(v[2, s_])).mean()), 1), round(float(np.degrees(np.abs(v[2, s_])).max()), 1)]})
            r['sep_cm'] = round(float(100 * at_stand[1, s_].mean()), 1)
        res[name] = r
        print(name, json.dumps(r), flush=True)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
