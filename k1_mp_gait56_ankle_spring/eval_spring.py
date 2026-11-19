# Local experiment (ankle-roll spring variant). Same measurement as eval_power56.py (gait manager, 8 robots from
# standing, stand 1 s, command for T-5 s, measure the last 5 s) plus torso tilt and ankle-roll angle statistics.
# ANKLE_SPRING=0 -> v5.6.3 model; ANKLE_SPRING=1 -> passive ankle roll (k1env.py).
# python3 eval_spring.py walk.pt run.pt [--json out.json] [--n 8] [--quick]
import argparse, json, os
import numpy as np
from eval_gait56 import Gait55, run_profile, yaw_of, HZ

MG = 35.7 * 9.81
CASES = [('walk', 0.6, 0.0, 9), ('walk', 1.0, 0.0, 9), ('walk', 1.4, 0.0, 9),
         ('walk turn', 1.0, 0.6, 9), ('walk turn', 1.0, -0.6, 9), ('in-place', 0.0, 0.6, 10),
         ('run', 2.5, 0.0, 12), ('run', 3.0, 0.0, 12), ('run', 3.5, 0.0, 12), ('run', 4.5, 0.0, 12),
         ('run turn', 3.0, 0.5, 12), ('run turn', 3.0, -0.5, 12)]
QUICK = [('walk', 1.0, 0.0, 9), ('run', 3.0, 0.0, 12)]


def tilt(q):
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.degrees(pitch), np.degrees(roll)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('walk'); ap.add_argument('run')
    ap.add_argument('--json', default=''); ap.add_argument('--n', type=int, default=8); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', default='', help='walk|run: quick cases of one gait only')
    ap.add_argument('--profile', action='store_true', help='also the stand-walk-run-stop profile')
    ap.add_argument('--seed', type=int, default=3); ap.add_argument('--meas', type=float, default=5.0, help='measurement window [s]')
    ap.add_argument('--straight', action='store_true', help='straight walk 0.6/1.0/1.4 only (precise mode)')
    a = ap.parse_args()
    g = Gait55(a.walk, a.run, n=a.n, nthread=2)
    qa = g.W.qadr
    ar = [qa[5], qa[11]]
    res = []
    for grp, v, w, T in [c for c in (QUICK if a.quick else CASES) if (not a.straight or c[0] == 'walk') and (not a.only or c[0].startswith(a.only))]:
        L = run_profile(g, [(1, 'cmd', 0.0, 0.0), (T - 5, 'cmd', v, w)], seed=a.seed)
        alive = L['alive'].copy()
        P, V, Y, M, PI, RO, AR = [], [], [], [], [], [], []; SH, KH, FY = [], [], []
        for k in range(int(a.meas * HZ)):
            term, _ = g.step(); alive &= ~term
            q = g.qpos(); qv = g.active('state')[:, 1 + g.W.m.nq:1 + g.W.m.nq + g.W.m.nv]
            yw = yaw_of(q)
            P.append(np.where(g.mode == 1, g.R.P_elec, g.W.P_elec)); V.append(np.cos(yw) * qv[:, 0] + np.sin(yw) * qv[:, 1])
            Y.append(yw); M.append(g.mode.copy())
            p, r = tilt(q); PI.append(p); RO.append(r); AR.append(np.degrees(q[:, ar]))
            SH.append(np.degrees(q[:, [qa[14], qa[19]]])); KH.append(g.W.kp_scale[:, [1, 7]].mean(1))
            ay = np.stack([g.W.s('left_ankle_rel')[:, 1], g.W.s('right_ankle_rel')[:, 1]], 1)
            tl = g.W.touch_now() if hasattr(g.W, 'touch_now') else None
            FY.append(np.abs(ay).min(1))
        al = alive
        r = dict(group=grp, v_cmd=v, w_cmd=w, survival=float(al.mean()))
        if al.any(): r['P_per_robot'] = [round(float(x), 1) for x in np.array(P)[:, al].mean(0)]; r['v_per_robot'] = [round(float(x), 3) for x in np.array(V)[:, al].mean(0)]
        if al.any():
            p = float(np.array(P)[:, al].mean()); vm = float(np.array(V)[:, al].mean())
            yaw = np.unwrap(np.array(Y), axis=0)
            PI_, RO_, AR_ = np.array(PI)[:, al], np.array(RO)[:, al], np.array(AR)[:, al]
            r.update(P_W=round(p, 1), v=round(vm, 2), w=round(float(((yaw[-1] - yaw[0]) / a.meas)[al].mean()), 2),
                     CoT=round(p / (MG * vm), 3) if vm >= 0.3 else None, run_frac=round(float(np.array(M)[:, al].mean()), 2),
                     pitch_mean=round(float(PI_.mean()), 2), pitch_absmax=round(float(np.abs(PI_).max()), 2),
                     roll_rms=round(float(np.sqrt((RO_ ** 2).mean())), 2), roll_absmax=round(float(np.abs(RO_).max()), 2),
                     tilt_rms=round(float(np.sqrt((PI_ ** 2 + RO_ ** 2).mean())), 2),
                     ankroll_rms=round(float(np.sqrt((AR_ ** 2).mean())), 2), ankroll_absmax=round(float(np.abs(AR_).max()), 2),
                     ankroll_p2p=round(float((AR_.max(0) - AR_.min(0)).mean()), 2),
                     roll_p2p=round(float((RO_.max(0) - RO_.min(0)).mean()), 2),
                     arm_roll_p2p=round(float((np.array(SH)[:, al].max(0) - np.array(SH)[:, al].min(0)).mean()), 2),
                     hiproll_kp=round(float(np.array(KH)[:, al].mean()), 3),
                     foot_y_inner_cm=round(float(100 * np.array(FY)[:, al].mean()), 2))
        res.append(r); print(json.dumps(r), flush=True)
    if a.profile:
        prof = [(1, 'cmd', 0, 0), (6, 'cmd', 1.0, 0), (8, 'cmd', 3.0, 0), (6, 'cmd', 1.0, 0), (6, 'stop', 0, 0)]
        L = run_profile(g, prof, seed=3)
        r = dict(group='profile stand-walk1.0-run3.0-walk1.0-stop', survival=float(L['alive'].mean()))
        res.append(r); print(json.dumps(r), flush=True)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
