# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Evaluate the turning policy (no pushes, no randomisation, start from standing).
  walkturn : v 0.6 m/s, yaw rate +-0.5 rad/s, and v 1.0 / w 1.0 (hardest), 12 s
  inplace  : v 0, yaw rate +-0.3 / +-0.6 / +-1.0 rad/s, 12 s -> yaw rate, horizontal drift
  stop     : walk 1.2 m/s (and walking turn 0.6/0.5), stop command at 6 s -> stop distance/time, survival
  straight : v 0.3 .. 1.35 straight -> regression vs v2 (speed error, power, heading drift)
python3 eval_turn.py runs/turn1/model.pt [--json out.json] [--rec out/rec.npz]
"""
import sys, json, os
import numpy as np, torch, mujoco
from k1env import DEFAULT_POSE
from k1env_turn import K1TurnBatch
from ppo_turn import ACEco

N = 8
NOISE_RNG = np.random.default_rng(123)


def make_env(n=N):
    env = K1TurnBatch(n, stage=2, randomize=False, seed=11)
    env.pushes = False
    env.scripted = True
    return env


def load(path, env):
    oa, oc = env.obs()
    net = ACEco(oa.shape[1], oc.shape[1], env.nact)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    return net


def stand_all(env):
    m = env.m; d = mujoco.MjData(m)
    for i in range(env.n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.key_qpos[0]; d.qpos[env.qadr] = DEFAULT_POSE; d.qpos[2] = env.ref.z_stand + 0.002
        d.qpos[env.qadr[:12]] += NOISE_RNG.normal(0, 0.01, 12)     # small per-robot difference (1 deg rms)
        mujoco.mj_forward(m, d)
        st = np.zeros(env.nstate); mujoco.mj_getState(m, d, st, env.spec_state)
        env.state[i] = st; env.sdata[i] = d.sensordata
    for k in ('ph', 'alpha', 'cmd', 'v_cmd', 'v_ref', 'w_cmd', 'w_ref', 'yaw_t'):
        getattr(env, k)[:] = 0
    env.t[:] = 0; env.last_a[:] = 0; env.last_a2[:] = 0
    env.next_evt[:] = 10 ** 9


def run(net, env, v, w, T=12.0, stop_at=None, rec=False):
    stand_all(env)
    n = env.n
    env.cmd[:] = 1.0; env.v_cmd[:] = v; env.w_cmd[:] = w
    alive = np.ones(n, bool)
    log = dict(xy=[], yaw=[], wz=[], vx=[], P=[], impact=[], kp=[], alpha=[], cmd=[], vref=[], wref=[])
    qs = []
    oa, _ = env.obs()
    for k in range(int(T * 50)):
        if stop_at is not None and k == int(stop_at * 50):
            env.cmd[:] = 0.0
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        rew, term, trunc, info = env.step(a)
        alive &= ~term
        q = env.qpos(); qv = env.qvel()
        yaw = env.yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        log['xy'].append(q[:, :2].copy()); log['yaw'].append(yaw)
        log['wz'].append(qv[:, 5].copy())
        log['vx'].append(c * qv[:, 0] + s * qv[:, 1])
        log['P'].append(env.P_elec.copy()); log['impact'].append(env.impact / (35.706 * 9.81))
        log['kp'].append(env.kp_scale.copy()); log['alpha'].append(env.alpha.copy())
        log['cmd'].append(env.cmd.copy()); log['vref'].append(env.v_ref.copy()); log['wref'].append(env.w_ref.copy())
        if rec:
            qs.append(q[0].copy())
        oa, _ = env.obs()
    L = {k: np.array(v_) for k, v_ in log.items()}
    L['alive'] = alive
    if rec:
        L['qpos'] = np.array(qs)
    return L


def summarize_turn(L, v, w, t0=4.0):
    k0 = int(t0 * 50)
    al = L['alive']
    yaw = np.unwrap(L['yaw'], axis=0)
    wz = (yaw[-1] - yaw[k0]) / ((len(yaw) - 1 - k0) / 50)
    vx = L['vx'][k0:].mean(0)
    drift = np.linalg.norm(L['xy'][-1] - L['xy'][k0], axis=1)
    # mean speed of the path (for in-place turning = drift speed)
    path_v = drift / ((len(yaw) - 1 - k0) / 50)
    return dict(v_cmd=v, w_cmd=w, survival=float(al.mean()),
                w_meas=float(np.mean(wz[al])) if al.any() else None,
                w_err_pct=float(100 * np.mean(np.abs(wz[al] - w)) / abs(w)) if al.any() and w else None,
                v_meas=float(np.mean(vx[al])) if al.any() else None,
                drift_m_per_s=float(np.mean(path_v[al])) if al.any() else None,
                P_leg=float(L['P'][k0:, al].mean()) if al.any() else None,
                impact_p95=float(np.percentile(L['impact'][k0:, al], 95)) if al.any() else None)


def summarize_stop(L, t_stop):
    k0 = int(t_stop * 50)
    al = L['alive']
    xy = L['xy']
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=2) * 50
    res = []
    for i in np.where(al)[0]:
        settled = np.where((L['alpha'][k0:, i] == 0))[0]
        t_al = settled[0] / 50 if len(settled) else None
        still = np.where(sp[k0:, i] < 0.05)[0]
        dist = np.linalg.norm(xy[-1, i] - xy[k0, i])
        res.append((t_al, dist, sp[-50:, i].mean()))
    ok = [r for r in res if r[0] is not None]
    return dict(survival=float(al.mean()),
                t_to_stand=float(np.mean([r[0] for r in ok])) if ok else None,
                stop_distance=float(np.mean([r[1] for r in res])) if res else None,
                residual_speed=float(np.mean([r[2] for r in res])) if res else None,
                stood=f'{len(ok)}/{len(res)}')


def main():
    path = sys.argv[1]
    out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    env = make_env()
    net = load(path, env)
    R = dict(policy=path, walkturn=[], inplace=[], stop=[], straight=[])
    for v, w in ((0.6, 0.5), (0.6, -0.5), (1.0, 1.0), (0.3, 0.3)):
        L = run(net, env, v, w)
        R['walkturn'].append(summarize_turn(L, v, w)); print('walkturn', R['walkturn'][-1], flush=True)
    for w in (0.3, -0.3, 0.6, -0.6, 1.0):
        L = run(net, env, 0.0, w)
        R['inplace'].append(summarize_turn(L, 0.0, w)); print('inplace', R['inplace'][-1], flush=True)
    for v, w in ((1.2, 0.0), (0.6, 0.5), (0.0, 0.6)):
        L = run(net, env, v, w, T=11.0, stop_at=6.0)
        s = summarize_stop(L, 6.0); s.update(v=v, w=w); R['stop'].append(s); print('stop', s, flush=True)
    for v in (0.3, 0.6, 0.9, 1.2, 1.35):
        L = run(net, env, v, 0.0)
        s = summarize_turn(L, v, 0.0)
        yaw = np.unwrap(L['yaw'], axis=0)
        s['heading_drift_deg'] = float(np.degrees(np.abs(yaw[-1] - yaw[0]))[L['alive']].max()) if L['alive'].any() else None
        s['v_err_pct'] = float(100 * abs(s['v_meas'] - v) / v) if s['v_meas'] is not None else None
        R['straight'].append(s); print('straight', s, flush=True)
    if out:
        json.dump(R, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
