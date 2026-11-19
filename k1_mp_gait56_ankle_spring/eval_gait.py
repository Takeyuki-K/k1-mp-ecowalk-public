# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Combined walk <-> run evaluation with the gait manager (gait.py).
 A 'accel'   : stand -> walk 0.3 m/s, +0.15 m/s every 1.5 s up to 1.65 -> command 2.0 (switch to run) -> 3 -> 4 -> 5.5
               -> 3 -> 1.5 (switch to walk) -> 0.6 -> stop
 B 'standrun': stand -> command 3.0 (direct stand -> run) -> 4.5 -> 1.2 (switch to walk) -> stop
python3 eval_gait.py WALK.pt RUN.pt [--n 8] [--push] [--rec A:out/rec_A.npz] [--json out.json]
"""
import sys, json
import numpy as np
from gait import Gait

FPS = 50
PROFILES = {
    'accel': [(0, 0.0), (1, 0.3)] + [(1 + 1.5 * k, round(0.3 + 0.15 * k, 2)) for k in range(1, 10)]
             + [(15, 2.0), (18, 3.0), (21, 4.0), (24, 5.5), (30, 3.0), (33, 1.5), (36, 0.6), (39, 0.0), (43, None)],
    'standrun': [(0, 0.0), (1, 3.0), (6, 4.5), (11, 1.2), (14, 0.0), (18, None)],
}


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def run(walk, runp, prof, N, push=False):
    g = Gait(N, walk, runp, push=push)
    P = PROFILES[prof]
    T = P[-1][0]
    L = {k: [] for k in ('qpos', 'mode', 'v_user', 'P', 'kp', 'heelL', 'heelR', 'foreL', 'foreR', 'alive')}
    for k in range(int(T * FPS)):
        t = k / FPS
        u = [v for ts, v in P if ts <= t and v is not None][-1]
        g.step(u, t)
        q = g.qpos()
        L['qpos'].append(q.copy()); L['mode'].append(g.mode.copy()); L['v_user'].append(u)
        L['P'].append(g.active('P_elec').copy()); L['kp'].append(g.active('kp_scale').mean(1))
        sdW = g.W.sd_out; sdR = g.R.sd_out_all
        for s, S in (('left', 'L'), ('right', 'R')):
            h = np.where(g.mode == 0, sdW[:, :, g.W.sens[f'{s}_heel_touch'][0]].max(1), sdR[:, :, g.R.sens[f'{s}_heel_touch'][0]].max(1))
            f = np.where(g.mode == 0, np.maximum(sdW[:, :, g.W.sens[f'{s}_meta_touch'][0]], sdW[:, :, g.W.sens[f'{s}_toe_touch'][0]]).max(1),
                         np.maximum(sdR[:, :, g.R.sens[f'{s}_meta_touch'][0]], sdR[:, :, g.R.sens[f'{s}_toe_touch'][0]]).max(1))
            L['heel' + S].append(h); L['fore' + S].append(f)
        L['alive'].append(g.alive.copy())
    L = {k: np.array(v) for k, v in L.items()}
    standing_end = (g.mode == 0) & (g.W.alpha == 0) & (g.W.cmd < 0.5)
    x = L['qpos'][:, :, 0]
    v = np.gradient(x, axis=0) * FPS
    # mean speed in each command segment (last 60 % of the segment), active robots
    seg = []
    for (ts, u), (te, _) in zip(P[:-1], P[1:]):
        a, b = int((ts + 0.4 * (te - ts)) * FPS), int(te * FPS)
        seg.append(dict(t=ts, cmd=u, speed=float(v[a:b, g.alive].mean()) if g.alive.any() else None))
    R = dict(profile=prof, survival=float(g.alive.mean()), standing_at_end=float(standing_end[g.alive].mean()) if g.alive.any() else 0,
             switches=[(round(t, 2), i, s) for t, i, s in g.switches if i == 0],
             n_switch_events=len(g.switches), segments=seg,
             max_speed=float(np.max([s['speed'] for s in seg if s['speed'] is not None])))
    return R, L


def main():
    walk, runp = sys.argv[1], sys.argv[2]
    N = int(arg('--n', 8))
    res = []
    rec = arg('--rec')
    for prof in arg('--profiles', 'accel,standrun').split(','):
        R, L = run(walk, runp, prof, N, push='--push' in sys.argv)
        print(json.dumps(R), flush=True); res.append(R)
        if rec and rec.split(':')[0] == prof:
            np.savez(rec.split(':')[1], **{k: (v[:, 0] if v.ndim > 1 else v) for k, v in L.items() if k != 'qpos'},
                     qpos=L['qpos'][:, 0], profile=prof)
    if arg('--json'):
        json.dump(res, open(arg('--json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
