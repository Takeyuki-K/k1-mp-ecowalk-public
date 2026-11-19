# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5.6 walking evaluation (8 robots, from standing, no pushes):
  straight 0.6 / 1.0 / 1.4 m/s : pelvis roll peak-to-peak, leg / arm power, speed
  turning 0.6 / +-0.5, in-place +-0.3 / +-0.6 / +-1.0 (12 s)
  stop after in-place turns: final stance (foot offset, separation, relative foot yaw), standing 15 s later
python3 eval56.py runs/final/walk.pt [--v55]      (--v55: evaluate a v5.5 walking policy in the v5.5 env)
"""
import sys, os, json, argparse
import numpy as np, torch, mujoco
from k1env import DEFAULT_POSE
from ppo_walk4 import ACEco

N = 8


def make(v55):
    os.environ['WALK_BRAKE'] = '1'
    if v55:
        from k1env_walk3 import K1Walk3Batch as C
    else:
        from k1env_walk4 import K1Walk4Batch as C
    C.P_RUN = 0.0
    env = C(N, stage=2, randomize=False, seed=31, ep_len=10 ** 9, nthread=2)
    env.pushes = False; env.scripted = True
    return env


def load(path, env):
    oa, oc = env.obs()
    net = ACEco(oa.shape[1], oc.shape[1], env.nact)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    return net


def act(net, oa):
    with torch.no_grad():
        return net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)


def stand(env):
    m = env.m; d = mujoco.MjData(m); rng = np.random.default_rng(3)
    for i in range(env.n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.key_qpos[0]; d.qpos[env.qadr] = DEFAULT_POSE; d.qpos[2] = env.ref.z_stand + 0.002
        d.qpos[env.qadr[:12]] += rng.normal(0, 0.01, 12)
        mujoco.mj_forward(m, d)
        st = np.zeros(env.nstate); mujoco.mj_getState(m, d, st, env.spec_state)
        env.state[i] = st; env.sdata[i] = d.sensordata
    for k in ('ph', 'alpha', 'cmd', 'v_cmd', 'v_ref', 'w_cmd', 'w_ref', 'yaw_t', 'brake'):
        getattr(env, k)[:] = 0
    if hasattr(env, 'home_cnt'):
        env.home_cnt[:] = 0; env._prev_ph[:] = 0
    env.t[:] = 0; env.last_a[:] = 0; env.last_a2[:] = 0; env.next_evt[:] = 10 ** 9


def run(net, env, v, w, T, stop_at=None, t0=4.0):
    stand(env); env.cmd[:] = 1; env.v_cmd[:] = v; env.w_cmd[:] = w
    oa, _ = env.obs(); alive = np.ones(env.n, bool)
    L = dict(roll=[], yaw=[], vx=[], Pl=[], Pa=[])
    for k in range(int(T * 50)):
        if stop_at is not None and k == int(stop_at * 50):
            env.cmd[:] = 0
        _, term, _, _ = env.step(act(net, oa)); oa, _ = env.obs()
        alive &= ~term
        if k >= t0 * 50 and (stop_at is None or k < stop_at * 50):
            _, g = env.base_frame(); yw = env.yaw(); qv = env.qvel()
            L['roll'].append(np.degrees(np.arcsin(np.clip(g[:, 1], -1, 1)))); L['yaw'].append(yw)
            L['vx'].append(np.cos(yw) * qv[:, 0] + np.sin(yw) * qv[:, 1])
            L['Pl'].append(getattr(env, 'P_leg', env.P_elec).copy()); L['Pa'].append(np.array(getattr(env, 'P_arm', np.zeros(env.n))).copy())
    L = {k_: np.array(v_) for k_, v_ in L.items()}
    out = dict(v=v, w=w, survival=float(alive.mean()))
    if alive.any() and len(L['roll']):
        al = alive
        yaw = np.unwrap(L['yaw'], axis=0)
        out.update(roll_p2p=float(np.mean(np.percentile(L['roll'][:, al], 95, 0) - np.percentile(L['roll'][:, al], 5, 0))),
                   w_meas=float(np.mean((yaw[-1, al] - yaw[0, al]) / (len(yaw) / 50))),
                   v_meas=float(L['vx'][:, al].mean()), P_leg=float(L['Pl'][:, al].mean()), P_arm=float(L['Pa'][:, al].mean()))
    if stop_at is not None:
        # ground truth (world poses of the feet), in the frame of the feet's mean heading
        fl, fr = env.s('left_foot_x'), env.s('right_foot_x')
        yl, yr = np.arctan2(fl[:, 1], fl[:, 0]), np.arctan2(fr[:, 1], fr[:, 0])
        fyaw = np.arctan2(np.sin(yl - yr), np.cos(yl - yr)); ym = yr + fyaw / 2
        d = env.s('left_foot_pos') - env.s('right_foot_pos')
        a = np.stack([np.cos(ym) * d[:, 0] + np.sin(ym) * d[:, 1], -np.sin(ym) * d[:, 0] + np.cos(ym) * d[:, 1]], 1)[:, None, :]
        a = np.concatenate([a, np.zeros_like(a)], 1)
        fyaw = np.degrees(fyaw)
        out.update(standing=float(((env.alpha == 0) & alive).mean()),
                   foot_dx_cm=float(np.mean(np.abs(100 * (a[alive, 0, 0] - a[alive, 1, 0])))) if alive.any() else None,
                   foot_sep_cm=float(np.mean(100 * (a[alive, 0, 1] - a[alive, 1, 1]))) if alive.any() else None,
                   foot_yaw_deg=float(np.mean(np.abs(fyaw[alive]))) if alive.any() else None)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('--v55', action='store_true')
    ap.add_argument('--json', default=''); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    env = make(a.v55); net = load(a.model, env)
    res = []
    for v in ((1.0,) if a.quick else (0.6, 1.0, 1.4)):
        res.append(run(net, env, v, 0.0, 12.0))
    for v, w in ([] if a.quick else [(0.6, 0.5), (0.6, -0.5)]) + [(0.0, 0.6), (0.0, -1.0)] + ([] if a.quick else [(0.0, 0.3), (0.0, -0.6), (0.0, 1.0)]):
        res.append(run(net, env, v, w, 12.0))
    for w, Ts in ((0.6, 6.0), (-1.0, 4.1), (0.3, 5.2)) if not a.quick else ((0.6, 6.0),):
        r = run(net, env, 0.0, w, Ts + 15.0, stop_at=Ts, t0=2.0); r['case'] = 'stop_after_inplace'; res.append(r)
    r = run(net, env, 1.2, 0.0, 22.0, stop_at=8.0); r['case'] = 'stop_from_walk'; res.append(r)
    for r in res:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
