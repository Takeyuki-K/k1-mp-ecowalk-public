# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5.5 end-to-end evaluation through the gait manager (walk <-> run hand-over, turning, governor, braking).
Every robot starts standing; 8 robots with a slightly different start pose (1 deg rms); optional random pushes.

python3 eval_gait55.py runs/final/walk.pt runs/final/run.pt [--json out.json] [--push]
"""
import argparse, json
import numpy as np, mujoco
from k1env import DEFAULT_POSE
from gait55 import Gait55

HZ = 50


def stand_noise(g, seed=7):
    W = g.W; m = W.m; d = mujoco.MjData(m); rng = np.random.default_rng(seed)
    for i in range(g.n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.key_qpos[0]; d.qpos[W.qadr] = DEFAULT_POSE; d.qpos[2] = W.ref.z_stand + 0.002
        d.qpos[W.qadr[:12]] += rng.normal(0, 0.017, 12)
        mujoco.mj_forward(m, d)
        st = np.zeros(W.nstate); mujoco.mj_getState(m, d, st, W.spec_state)
        W.state[i] = st; W.sdata[i] = d.sensordata


def yaw_of(q):
    return np.arctan2(2 * (q[:, 3] * q[:, 6] + q[:, 4] * q[:, 5]), 1 - 2 * (q[:, 5] ** 2 + q[:, 6] ** 2))


def run_profile(g, profile, push=False, seed=0):
    """profile: list of (duration_s, action, v, w); action in {'cmd', 'stop', 'brake'}"""
    g.reset(); stand_noise(g, seed + 7)
    g._oaw, _ = g.W.obs(); g._oar, _ = g.R.obs()
    rng = np.random.default_rng(seed)
    alive = np.ones(g.n, bool); fall_t = np.full(g.n, np.nan)
    log = dict(xy=[], yaw=[], vx=[], mode=[], vref=[], wref=[], alpha=[], seg=[])
    k = 0
    for si, (T, act, v, w) in enumerate(profile):
        if act == 'cmd':
            g.command(v, w)
        elif act == 'stop':
            g.stop()
        else:
            g.brake()
        for _ in range(int(T * HZ)):
            if push:      # independent per robot: about one +-0.3 m/s horizontal push every 3 s
                hit = rng.random(g.n) < 1 / 150
                dv = rng.uniform(-0.3, 0.3, (g.n, 2)) * hit[:, None]
                for e in (g.W, g.R):
                    e.state[:, 1 + e.m.nq:1 + e.m.nq + 2] += dv
            term, _ = g.step()
            new = term & alive
            fall_t[new] = k / HZ
            alive &= ~term
            q = g.qpos(); qv = g.active('state')[:, 1 + g.W.m.nq:1 + g.W.m.nq + g.W.m.nv]
            yw = yaw_of(q)
            log['xy'].append(q[:, :2].copy()); log['yaw'].append(yw)
            log['vx'].append(np.cos(yw) * qv[:, 0] + np.sin(yw) * qv[:, 1])
            log['mode'].append(g.mode.copy()); log['vref'].append(g.active('v_ref').copy())
            log['wref'].append(g.active('w_ref').copy()); log['alpha'].append(g.active('alpha').copy())
            log['seg'].append(si)
            k += 1
    L = {kk: np.array(vv) for kk, vv in log.items()}
    L['alive'] = alive; L['fall_t'] = fall_t
    return L


def seg_stats(L, profile, si, settle=1.5):
    idx = np.where(L['seg'] == si)[0]
    idx = idx[int(settle * HZ):] if len(idx) > settle * HZ + 10 else idx
    al = L['alive']
    if not al.any() or len(idx) < 10:
        return {}
    yaw = np.unwrap(L['yaw'][idx], axis=0)
    w = (yaw[-1] - yaw[0]) / (len(idx) / HZ)
    T, act, v, wc = profile[si]
    return dict(seg=si, cmd=f'{act} v={v} w={wc}', w_meas=round(float(w[al].mean()), 3),
                v_meas=round(float(L['vx'][idx][:, al].mean()), 3), v_ref=round(float(L['vref'][idx][-1, al].mean()), 3),
                run_frac=round(float(L['mode'][idx][:, al].mean()), 2))


PROFILE = [(2, 'cmd', 0.0, 0.0), (5, 'cmd', 1.0, 0.0), (6, 'cmd', 1.0, 0.6), (6, 'cmd', 3.0, 0.0), (6, 'cmd', 3.0, 0.5),
           (6, 'cmd', 3.0, -0.5), (6, 'cmd', 4.5, 0.0), (6, 'cmd', 4.5, 1.0), (5, 'cmd', 4.5, 0.0),
           (6, 'brake', 0, 0)]
INPLACE = [(2, 'cmd', 0.0, 0.0), (12, 'cmd', 0.0, 0.6), (12, 'cmd', 0.0, -0.6), (5, 'stop', 0, 0)]


def stop_test(g, kind, v=4.5, push=False):
    prof = [(2, 'cmd', 0.0, 0.0), (5, 'cmd', 1.65, 0.0), (6, 'cmd', v, 0.0), (8, kind, 0, 0)]
    L = run_profile(g, prof, push=push)
    k0 = np.where(L['seg'] == 3)[0][0]
    al = L['alive']
    sp = np.linalg.norm(np.diff(L['xy'], axis=0), axis=2) * HZ
    out = dict(kind=kind, v=v, survival=float(al.mean()))
    t2, d2, ts, ds = [], [], [], []
    for i in np.where(al)[0]:
        s = sp[k0:, i]
        a = np.where(s < 2.0)[0]
        if len(a):
            t2.append(a[0] / HZ); d2.append(float(s[:a[0]].sum() / HZ))
        st = np.where(L['alpha'][k0:, i] == 0)[0]
        if len(st):
            ts.append(st[0] / HZ); ds.append(float(np.linalg.norm(L['xy'][k0 + st[0], i] - L['xy'][k0, i])))
    out.update(t_to_2mps=round(float(np.mean(t2)), 2) if t2 else None, dist_to_2mps=round(float(np.mean(d2)), 2) if d2 else None,
               t_to_stand=round(float(np.mean(ts)), 2) if ts else None, dist_to_stand=round(float(np.mean(ds)), 2) if ds else None,
               standing_end=float(len(ts) / max(1, al.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('walk'); ap.add_argument('run')
    ap.add_argument('--json', default=''); ap.add_argument('--push', action='store_true')
    ap.add_argument('--n', type=int, default=8)
    a = ap.parse_args()
    g = Gait55(a.walk, a.run, n=a.n, nthread=2)
    res = {}
    L = run_profile(g, PROFILE, push=a.push)
    res['profile'] = dict(survival=float(L['alive'].mean()), fall_t=[None if np.isnan(x) else x for x in L['fall_t']],
                          segments=[seg_stats(L, PROFILE, i) for i in range(len(PROFILE))])
    print(json.dumps(res['profile']), flush=True)
    L = run_profile(g, INPLACE, push=a.push)
    res['inplace'] = dict(survival=float(L['alive'].mean()), fall_t=[None if np.isnan(x) else x for x in L['fall_t']],
                          segments=[seg_stats(L, INPLACE, i) for i in (1, 2)])
    print(json.dumps(res['inplace']), flush=True)
    for kind in ('stop', 'brake'):
        res[kind] = stop_test(g, kind, push=a.push); print(json.dumps(res[kind]), flush=True)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
