# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Evaluate the speed-command policy.
 sweep   : constant commands 0.30 ... 1.35 m/s (+ 1.50 / 1.65 extrapolation), N envs each, 12 s
           -> survival, tracking error, step length, cadence, electrical power, CoT, heel-first ratio
 profile : one robot, stand -> 0.4 -> 1.2 -> 0.6 -> 1.35 -> 0.3 -> stop (step changes) -> time series
python3 eval_speed.py runs/speed1/model.pt
"""
import sys, json, os
import numpy as np, torch, mujoco
from k1env import DEFAULT_POSE
from k1env_speed import K1SpeedBatch
from ppo_speed import ACEco

MG = 35.706 * 9.81


def load(path, env):
    oa, oc = env.obs()
    net = ACEco(oa.shape[1], oc.shape[1], env.nact)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    return net


def act(net, oa):
    with torch.no_grad():
        return net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)


def stand_all(env):
    m = env.m; d = mujoco.MjData(m)
    for i in range(env.n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = m.key_qpos[0]; d.qpos[env.qadr] = DEFAULT_POSE; d.qpos[2] = env.ref.z_stand + 0.002
        mujoco.mj_forward(m, d)
        st = np.zeros(env.nstate); mujoco.mj_getState(m, d, st, env.spec_state)
        env.state[i] = st; env.sdata[i] = d.sensordata
    env.ph[:] = 0; env.alpha[:] = 0; env.cmd[:] = 0; env.t[:] = 0
    if hasattr(env, 'yaw_t'):
        env.yaw_t[:] = 0.0
    env.last_a[:] = 0; env.last_a2[:] = 0


class Gait:
    """touchdown detection -> step length / cadence / heel-first"""

    def __init__(self, n):
        self.air = np.zeros((n, 2), bool); self.at = np.zeros((n, 2)); self.td = [[[], []] for _ in range(n)]

    def update(self, env, t):
        for k, s in enumerate(('left', 'right')):
            h = env.s(f'{s}_heel_touch')[:, 0] > 5
            f = (env.s(f'{s}_meta_touch')[:, 0] > 5) | (env.s(f'{s}_toe_touch')[:, 0] > 5)
            c = h | f
            down = self.air[:, k] & c
            x = env.s(f'{s}_foot_pos')[:, 0]
            for i in np.where(down)[0]:
                self.td[i][k].append((t, x[i], bool(h[i]) and not bool(f[i]) or bool(h[i])))
            self.at[:, k] = np.where(c, 0, self.at[:, k] + 0.02)
            self.air[:, k] = (~c) & (self.at[:, k] > 0.06) | (self.air[:, k] & ~c)

    def stats(self, i, t0, t1):
        ev = sorted([(t, x, hf, k) for k in range(2) for (t, x, hf) in self.td[i][k] if t0 <= t <= t1])
        if len(ev) < 4:
            return dict(step_len=np.nan, cadence=np.nan, heel_first=np.nan)
        steps = [abs(ev[j + 1][1] - ev[j][1]) for j in range(len(ev) - 1) if ev[j + 1][3] != ev[j][3]]
        dur = ev[-1][0] - ev[0][0]
        return dict(step_len=float(np.median(steps)) if steps else np.nan,
                    cadence=60.0 * (len(ev) - 1) / dur, heel_first=float(np.mean([e[2] for e in ev])))


def sweep(path, speeds, per=6, T=12.0, warm=4.0, pushes=False, dt=0.005):
    n = per * len(speeds)
    env = K1SpeedBatch(n, stage=2, randomize=False, seed=3, ep_len=10 ** 9, v_range=(0.3, 1.35), dt=dt)
    env.pushes = pushes; env.scripted = True
    stand_all(env)
    vc = np.repeat(speeds, per)
    net = load(path, env)
    g = Gait(n)
    alive = np.ones(n, bool); P = np.zeros(n); cnt = 0
    x0 = None
    oa, _ = env.obs()
    for k in range(int(T * 50)):
        t = k * 0.02
        if k == 25:
            env.cmd[:] = 1; env.v_cmd[:] = vc; env.v_ref[:] = vc
        a = act(net, oa)
        _, term, _, info = env.step(a)
        alive &= ~term
        g.update(env, t)
        if t >= warm:
            if x0 is None:
                x0 = env.qpos()[:, 0].copy(); tw0 = t
            P += info['P_elec']; cnt += 1
        oa, _ = env.obs()
    v_meas = (env.qpos()[:, 0] - x0) / (T - tw0)
    yaw_end = np.degrees(env.heading_err()) if hasattr(env, 'heading_err') else np.zeros(n)
    P /= cnt
    res = []
    for si, v in enumerate(speeds):
        ids = np.arange(si * per, (si + 1) * per)
        ok = ids[alive[ids]]
        st = [g.stats(i, warm, T) for i in ok]
        vm = v_meas[ok].mean() if len(ok) else np.nan
        res.append(dict(v_cmd=float(v), survival=float(alive[ids].mean()), v_meas=float(vm),
                        err=float(abs(vm - v)) if len(ok) else np.nan,
                        step_len=float(np.nanmean([s['step_len'] for s in st])) if st else np.nan,
                        cadence=float(np.nanmean([s['cadence'] for s in st])) if st else np.nan,
                        heel_first=float(np.nanmean([s['heel_first'] for s in st])) if st else np.nan,
                        P_elec=float(P[ok].mean()) if len(ok) else np.nan,
                        yaw_end_deg=float(np.abs(yaw_end[ok]).max()) if len(ok) else np.nan,
                        CoT=float(P[ok].mean() / (MG * vm)) if len(ok) else np.nan))
    return res


PROFILE = [(0.0, 0, 0.0), (2.0, 1, 0.4), (7.0, 1, 1.2), (12.0, 1, 0.6), (17.0, 1, 1.35), (22.0, 1, 0.3), (27.0, 0, 0.3)]


def profile(path, T=30.0, record=False):
    env = K1SpeedBatch(1, stage=2, randomize=False, seed=5, ep_len=10 ** 9)
    env.pushes = False; env.scripted = True
    stand_all(env)
    net = load(path, env)
    g = Gait(1)
    oa, _ = env.obs()
    log = dict(t=[], v_cmd=[], v_ref=[], vx=[], P=[], kp=[], cmd=[], alpha=[])
    states = []
    fell = False
    for k in range(int(T * 50)):
        t = k * 0.02
        for (ts, c, v) in PROFILE:
            if abs(t - ts) < 1e-6:
                env.cmd[:] = c; env.v_cmd[:] = v
        a = act(net, oa)
        _, term, _, info = env.step(a)
        g.update(env, t)
        log['t'].append(t); log['v_cmd'].append(env.v_cmd[0] * env.cmd[0]); log['v_ref'].append(env.v_ref[0] * env.alpha[0])
        log['vx'].append(env.qvel()[0, 0]); log['P'].append(info['P_elec'][0]); log['kp'].append(env.kp_scale[0].copy())
        log['cmd'].append(env.cmd[0]); log['alpha'].append(env.alpha[0])
        if record:
            states.append(env.st_out[0].copy())
        if term[0]:
            fell = True; print('profile: FELL at', t); break
        oa, _ = env.obs()
    td = g.td[0]
    return dict(fell=fell, log={k: np.array(v) for k, v in log.items()}, td=td,
                states=np.concatenate(states, 0) if record else None, env=env)


if __name__ == '__main__':
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'out/eval_speed.json'
    speeds = [0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.35, 1.50, 1.65]
    res = sweep(path, speeds)
    print(f"{'v_cmd':>6} {'surv':>5} {'v_meas':>6} {'err':>5} {'step':>5} {'cad':>5} {'heel1':>5} {'P[W]':>6} {'CoT':>5}")
    for r in res:
        print(f"{r['v_cmd']:6.2f} {r['survival']:5.2f} {r['v_meas']:6.3f} {r['err']:5.3f} {r['step_len']:5.3f} "
              f"{r['cadence']:5.0f} {r['heel_first']:5.2f} {r['P_elec']:6.1f} {r['CoT']:5.2f}")
    pr = profile(path)
    L = pr['log']
    on = L['alpha'] > 0.99
    print('profile fell:', pr['fell'], ' mean |v - v_ref| while walking: %.3f m/s' % np.abs(L['vx'] - L['v_ref'])[on].mean())
    json.dump(dict(sweep=res, profile_fell=pr['fell'],
                   profile_err=float(np.abs(L['vx'] - L['v_ref'])[on].mean())), open(out, 'w'), indent=1)
