# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5.5 evaluation: turning while walking / running, speed governor, hard braking (no pushes, no randomisation).

python3 eval55.py run  runs/r1/model.pt  [--v5]   # --v5: evaluate the original v5 policy (no turn inputs)
python3 eval55.py walk runs/w1/model.pt  [--v5]
"""
import sys, json, argparse
import numpy as np, torch, mujoco
from k1env import DEFAULT_POSE
from ppo_eco import AC
from ppo_run4 import ACEco

N = 8


def policy(path, env, nact, v5_layout_at=None, add=0):
    oa, oc = env.obs()
    sd = torch.load(path, map_location='cpu')['model']
    net = ACEco(oa.shape[1], oc.shape[1], nact)
    if v5_layout_at is not None:
        from ppo_run4 import transfer_insert_obs
        transfer_insert_obs(sd, net, at=v5_layout_at, add=add)
    else:
        net.load_state_dict(sd)
    net.eval()
    return net


def act(net, oa):
    with torch.no_grad():
        return net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)


def unwrap_rate(yaws, k0, hz=50):
    y = np.unwrap(np.array(yaws), axis=0)
    return (y[-1] - y[k0]) / ((len(y) - 1 - k0) / hz)


# ------------------------------------------------------------------ running
def run_env():
    from k1env_run4 import K1Run4Batch
    K1Run4Batch.P_STAND = 0.0; K1Run4Batch.P_WALK = 0.0
    env = K1Run4Batch(N, v_lo=2.0, v_hi=2.0, stage=2, randomize=False, seed=21, ep_len=10 ** 9, nthread=2)
    env.pushes = False; env.scripted = True; env.assist = 0.0
    return env


def run_start(env, v0):
    """steady running start at v0 (same as the running RSI start), heading 0"""
    env.v_lo = env.v_hi = v0
    env.reset(np.arange(env.n))
    env.v_cmd[:] = v0; env.v_ref[:] = v0; env.w_cmd[:] = 0; env.w_ref[:] = 0; env.brake[:] = 0
    env.yaw_t[:] = env.yaw(); env.next_evt[:] = 10 ** 9


def run_case(net, env, v, w, T=8.0, t0=3.0, brake_at=None):
    run_start(env, v)
    alive = np.ones(env.n, bool)
    oa, _ = env.obs()
    L = dict(yaw=[], vx=[], vref=[], wz=[], tilt_y=[], xy=[], impact=[], qdv=[], sat=[])
    for k in range(int(T * 50)):
        if k == 25:
            env.w_cmd[:] = w
        if brake_at is not None and k == int(brake_at * 50):
            env.brake[:] = 1.0; env.v_cmd[:] = 1.8; env.w_cmd[:] = 0.0
        _, term, _, info = env.step(act(net, oa))
        alive &= ~term
        oa, _ = env.obs()
        qv = env.qvel(); yaw = env.yaw()
        quat, g = env.base_frame()
        L['yaw'].append(yaw); L['vx'].append(np.cos(yaw) * qv[:, 0] + np.sin(yaw) * qv[:, 1])
        L['vref'].append(env.v_ref.copy()); L['wz'].append(qv[:, 5].copy()); L['tilt_y'].append(g[:, 1].copy())
        L['xy'].append(env.qpos()[:, :2].copy()); L['impact'].append(info['impact']); L['qdv'].append(env.qd_viol)
        L['sat'].append(env.tau_sat)
    L = {k: np.array(v_) for k, v_ in L.items()}
    k0 = int(t0 * 50)
    out = dict(v_cmd=v, w_cmd=w, survival=float(alive.mean()))
    if alive.any() and brake_at is None:
        wz = unwrap_rate(L['yaw'], k0)
        out.update(w_meas=float(np.mean(wz[alive])), v_meas=float(L['vx'][k0:, alive].mean()),
                   v_ref_end=float(L['vref'][-1, alive].mean()),
                   lean_deg=float(np.degrees(np.arcsin(np.clip(L['tilt_y'][k0:, alive].mean(), -1, 1)))),
                   impact_p95=float(np.percentile(L['impact'][k0:, alive], 95)),
                   qd_over=float(L['qdv'][k0:, alive].mean()))
    if alive.any() and brake_at is not None:
        kb = int(brake_at * 50)
        sp = np.linalg.norm(np.diff(L['xy'], axis=0), axis=2) * 50
        t18 = [np.argmax(sp[kb:, i] < 2.0) / 50 if (sp[kb:, i] < 2.0).any() else None for i in np.where(alive)[0]]
        dist = [float(np.sum(sp[kb:kb + int(t * 50) if t else None, i]) / 50) if t else None
                for t, i in zip(t18, np.where(alive)[0])]
        out.update(t_to_2mps=t18, dist_to_2mps=dist, impact_p95=float(np.percentile(L['impact'][kb:, alive], 95)))
    return out


# ------------------------------------------------------------------ walking
def walk_env():
    from k1env_walk3 import K1Walk3Batch
    K1Walk3Batch.P_RUN = 0.0
    env = K1Walk3Batch(N, stage=2, randomize=False, seed=31, ep_len=10 ** 9, nthread=2)
    env.pushes = False; env.scripted = True
    return env


def walk_stand(env):
    m = env.m; d = mujoco.MjData(m); rng = np.random.default_rng(3)
    for i in range(env.n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.key_qpos[0]; d.qpos[env.qadr] = DEFAULT_POSE; d.qpos[2] = env.ref.z_stand + 0.002
        d.qpos[env.qadr[:12]] += rng.normal(0, 0.01, 12)
        mujoco.mj_forward(m, d)
        st = np.zeros(env.nstate); mujoco.mj_getState(m, d, st, env.spec_state)
        env.state[i] = st; env.sdata[i] = d.sensordata
    for k in ('ph', 'alpha', 'cmd', 'v_cmd', 'v_ref', 'w_cmd', 'w_ref', 'yaw_t'):
        getattr(env, k)[:] = 0
    env.t[:] = 0; env.last_a[:] = 0; env.last_a2[:] = 0; env.next_evt[:] = 10 ** 9


def walk_case(net, env, v, w, T=12.0, t0=5.0, stop_at=None):
    walk_stand(env)
    env.cmd[:] = 1; env.v_cmd[:] = v; env.w_cmd[:] = w
    alive = np.ones(env.n, bool)
    oa, _ = env.obs()
    L = dict(yaw=[], vx=[], xy=[], P=[], alpha=[])
    for k in range(int(T * 50)):
        if stop_at is not None and k == int(stop_at * 50):
            env.cmd[:] = 0
        _, term, _, info = env.step(act(net, oa))
        alive &= ~term
        oa, _ = env.obs()
        qv = env.qvel(); yaw = env.yaw()
        L['yaw'].append(yaw); L['vx'].append(np.cos(yaw) * qv[:, 0] + np.sin(yaw) * qv[:, 1])
        L['xy'].append(env.qpos()[:, :2].copy()); L['P'].append(env.P_elec.copy()); L['alpha'].append(env.alpha.copy())
    L = {k: np.array(v_) for k, v_ in L.items()}
    k0 = int(t0 * 50)
    out = dict(v_cmd=v, w_cmd=w, survival=float(alive.mean()))
    if alive.any():
        wz = unwrap_rate(L['yaw'], k0)
        out.update(w_meas=float(np.mean(wz[alive])), v_meas=float(L['vx'][k0:, alive].mean()),
                   P_leg=float(L['P'][k0:, alive].mean()))
        if stop_at is not None:
            out['standing_end'] = float((L['alpha'][-1, alive] == 0).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['run', 'walk'])
    ap.add_argument('model')
    ap.add_argument('--v5', action='store_true')
    ap.add_argument('--json', default='')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    res = []
    if a.kind == 'run':
        env = run_env()
        net = policy(a.model, env, 36, *( (10, 4) if a.v5 else (None, 0)))
        cases = [(3.0, 0.0), (3.0, 0.5), (3.0, -0.5), (4.5, 0.0), (4.5, 0.5), (4.5, 1.0), (2.5, 1.0)]
        if a.quick:
            cases = [(3.0, 0.0), (3.0, 0.5), (4.5, 1.0)]
        for v, w in cases:
            r = run_case(net, env, v, w); res.append(r); print(json.dumps(r), flush=True)
        for v in ([4.5] if a.quick else [3.0, 4.5]):
            r = run_case(net, env, v, 0.0, T=6.0, brake_at=2.0); r['case'] = 'brake'; res.append(r); print(json.dumps(r), flush=True)
    else:
        env = walk_env()
        net = policy(a.model, env, 37, *((11, 2) if a.v5 else (None, 0)))
        cases = [(1.0, 0.0), (0.6, 0.5), (0.6, -0.5), (1.5, 1.0), (0.0, 0.6), (0.0, -1.0)]
        if a.quick:
            cases = [(1.0, 0.0), (0.6, 0.5), (0.0, 0.6)]
        for v, w in cases:
            r = walk_case(net, env, v, w); res.append(r); print(json.dumps(r), flush=True)
        r = walk_case(net, env, 1.2, 0.0, T=12.0, stop_at=6.0); r['case'] = 'stop'; res.append(r); print(json.dumps(r), flush=True)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
