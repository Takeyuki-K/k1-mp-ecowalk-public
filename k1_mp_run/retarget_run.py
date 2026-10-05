# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Running reference for K1 + MP toes from CMU mocap (subject 16, trial 16_35 'run/jog', 2.74 m/s).

Same rule as walking: imitate only the hip->ankle motion (scaled by leg-length ratio, Froude time scaling),
solve the leg joints by IK.
Running-specific decisions (see REPORT_RUN.md):
  * stride shortened (kL) at constant cadence -> slow jog ~1.65 m/s for the robot (Froude ~0.45);
    the vertical hip->ankle profile is kept (ballistic flight timing unchanged), so the knee bends more.
  * CMU runners land on the forefoot. The foot pitch is replaced around touchdown by a heel-first landing
    (toe up ~10 deg at touchdown, rolling to flat within the first part of stance), as requested.
  * pelvis height = scaled human hip height (keeps the flight phase), offset so that the stance foot is on
    the ground on average.
Output: ref_run.npz (same layout as ref_gait.npz, usable by k1env.Ref)
"""
import os
import numpy as np
import mujoco
import bvh

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, '..', 'ai_sapiens', 'ai_sapiens_description', 'mujoco', 'k1', 'scene_mp.xml')
TRIAL = os.path.join(HERE, 'data', 'cmu', 'data', '016', '16_35.bvh')
V_TARGET = 1.65
LEAN = np.radians(5.0)      # trunk forward lean (human jogging ~5-10 deg); pelvis pitched forward in the reference
HEEL_PITCH = -0.17          # rad, toe up at touchdown (walking convention: + = heel up)
REST_PITCH = np.arctan2(-0.46332, 2.12791)   # BVH rest-pose ankle->toebase direction (toe slightly down)


def smooth(x, w=5):
    k = np.ones(w) / w
    pad = np.concatenate([x[-w:], x, x[:w]], 0)
    return np.stack([np.convolve(pad[:, j], k, 'same') for j in range(x.shape[1])], 1)[w:-w]


def periodic_fix(x):
    n = len(x) - 1
    ramp = np.linspace(0, 1, n + 1)[:, None]
    return (x - ramp * (x[-1] - x[0]))[:-1]


def to_robot(P, fwd):
    """BVH (Y up) -> robot world (x forward along fwd, y left, z up)"""
    up = np.array([0, 1.0, 0])
    f = fwd / np.linalg.norm(fwd)
    left = np.cross(up, f)
    return np.stack([P @ f, P @ left, P @ up], -1)


def build(v_target=V_TARGET, verbose=True):
    B = bvh.load(TRIAL)
    n = B['names']; rate = 1.0 / B['dt']
    g = lambda k: B['P'][:, n.index(k)]
    hipm = 0.5 * (g('LeftUpLeg') + g('RightUpLeg'))
    fwd = (hipm[-1] - hipm[0]) * np.array([1, 0, 1])
    J = {}
    for s, S in (('l', 'Left'), ('r', 'Right')):
        J[s] = dict(hip=to_robot(g(S + 'UpLeg'), fwd), ankle=to_robot(g(S + 'Foot'), fwd),
                    mt=to_robot(g(S + 'ToeBase'), fwd))
    ground = min(J[s][k][:, 2].min() for s in 'rl' for k in ('ankle', 'mt'))
    # stance detection per foot (low and slow)
    st = {}
    for s in 'rl':
        low = np.minimum(J[s]['ankle'][:, 2], J[s]['mt'][:, 2]) - ground
        hv = np.linalg.norm(np.gradient(J[s]['mt'][:, :2], axis=0), axis=1) * rate
        st[s] = (low < 0.06) & (hv < 1.5)
    td = [i for i in range(1, len(st['r'])) if st['r'][i] and not st['r'][i - 1]]
    a, b = td[0], td[1]
    if verbose:
        print('right touchdowns', td, f'cycle {(b - a) / rate:.3f} s')
    sl = slice(a, b + 1)
    Lh = max(np.linalg.norm(J[s]['hip'] - J[s]['ankle'], axis=1).max() for s in 'rl')

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    QL = np.array([np.cos(LEAN / 2), 0, np.sin(LEAN / 2), 0])
    data.qpos[3:7] = QL
    mujoco.mj_forward(model, data)
    pel = model.body('pelvis').id
    hip_o, Lr = {}, {}
    for s, side in (('l', 'left'), ('r', 'right')):
        hip_o[s] = data.xpos[model.body(f'{side}_hip_roll_link').id] - data.xpos[pel]
        Lr[s] = np.linalg.norm(data.xpos[model.body(f'{side}_ankle_roll_link').id] - data.xpos[pel] - hip_o[s])
    sc = np.mean(list(Lr.values())) / Lh
    tsc = np.sqrt(sc)

    t_h = np.arange(b - a + 1) / rate
    midhip = 0.5 * (J['r']['hip'] + J['l']['hip'])[sl]
    speed = (midhip[-1, 0] - midhip[0, 0]) / t_h[-1]
    pel_h = midhip.copy(); pel_h[:, 0] -= speed * t_h
    pel_h = periodic_fix(pel_h)
    N = len(pel_h)
    T_cycle = N / rate * tsc
    v_base = speed * sc / tsc
    kL = v_target / v_base
    vec, fpitch, contact = {}, {}, {}
    for s in 'rl':
        v = smooth(periodic_fix((J[s]['ankle'] - J[s]['hip'])[sl]), 5)
        xm = v[:, 0].mean()
        v[:, 0] = xm + kL * (v[:, 0] - xm)            # shorter stride, same vertical profile
        vec[s] = v
        f = (J[s]['mt'] - J[s]['ankle'])[sl]
        p = -(np.arctan2(f[:, 2], np.hypot(f[:, 0], f[:, 1])) - REST_PITCH)    # + = heel up
        p = smooth(periodic_fix(p[:, None]), 5)[:, 0]
        stn = st[s][sl][:-1]
        # heel-first landing: blend pitch to HEEL_PITCH at touchdown, roll to the human pitch within the
        # first 25 % of stance (heel rocker); start the dorsiflexion during late swing
        tds = [i for i in range(N) if stn[i] and not stn[i - 1]]
        stance_len = int(stn.sum() / max(len(tds), 1))
        w = np.zeros(N)
        pre, post = int(0.12 * N), max(int(0.25 * stance_len), 2)
        for i0 in tds:
            for k in range(-pre, post + 1):
                j = (i0 + k) % N
                w[j] = max(w[j], (1 + k / pre) if k < 0 else (1 - k / post))
        w = 0.5 - 0.5 * np.cos(np.pi * np.clip(w, 0, 1))
        fpitch[s] = (1 - w) * p * kL + w * HEEL_PITCH
        heel_c = np.zeros(N, bool); fore_c = np.zeros(N, bool)
        for i0 in tds:
            for k in range(stance_len):
                j = (i0 + k) % N
                if k <= 0.35 * stance_len:
                    heel_c[j] = True
                if k >= 0.15 * stance_len:
                    fore_c[j] = True
        contact[s] = np.stack([heel_c, fore_c], 1)
    v_robot = v_base * kL
    if verbose:
        print(f'human {speed:.2f} m/s, leg {Lh:.3f} m, s={sc:.3f}; base robot speed {v_base:.2f} m/s -> '
              f'kL={kL:.3f} -> {v_robot:.2f} m/s, cycle {T_cycle:.3f} s, cadence {120 / T_cycle:.0f} steps/min')

    # ------------- IK (same as walking) -------------
    names = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
    qadr = {sd: [model.jnt_qposadr[model.joint(f'{sd}_{nm}_joint').id] for nm in names] for sd in ('left', 'right')}
    dadr = {sd: [model.jnt_dofadr[model.joint(f'{sd}_{nm}_joint').id] for nm in names] for sd in qadr}
    lo = {sd: np.array([model.jnt_range[model.joint(f'{sd}_{nm}_joint').id][0] for nm in names]) for sd in qadr}
    hi = {sd: np.array([model.jnt_range[model.joint(f'{sd}_{nm}_joint').id][1] for nm in names]) for sd in qadr}
    q = model.key_qpos[0].copy()
    for sd in qadr:
        q[qadr[sd]] = [-0.4, 0, 0, 0.9, -0.4, 0]
    Q = np.zeros((N, model.nq)); err = []
    for i in range(N):
        t = i * T_cycle / N
        q[0] = v_robot * t + sc * kL * pel_h[i, 0]; q[1] = sc * pel_h[i, 1]; q[3:7] = QL
        for s, sd in (('l', 'left'), ('r', 'right')):
            tgt = hip_o[s] + sc * vec[s][i]
            fp = fpitch[s][i]
            Rt = np.array([[np.cos(fp), 0, np.sin(fp)], [0, 1, 0], [-np.sin(fp), 0, np.cos(fp)]])
            bid = model.body(f'{sd}_ankle_roll_link').id
            for it in range(80):
                q[2] = 0.0
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
                ep = tgt - (data.xpos[bid] - data.xpos[pel])
                Re = Rt @ data.xmat[bid].reshape(3, 3).T
                eo = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
                e = np.concatenate([ep, 0.3 * eo])
                if np.linalg.norm(e) < 1e-5:
                    break
                jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jp, jr, bid)
                Jm = np.vstack([jp[:, dadr[sd]], 0.3 * jr[:, dadr[sd]]])
                q[qadr[sd]] = np.clip(q[qadr[sd]] + Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(6), e), lo[sd], hi[sd])
            err.append(np.linalg.norm(ep))
        Q[i] = q
    # pelvis height: human hip height (scaled, keeps the flight), offset so stance feet touch the ground
    hz = sc * pel_h[:, 2]
    sites = [f'{sd}_{p}_site' for sd in ('left', 'right') for p in ('heel', 'meta')]
    zs = []
    for i in range(N):
        data.qpos[:] = Q[i]; data.qpos[2] = 0
        mujoco.mj_kinematics(model, data)
        zs.append([data.site_xpos[model.site(sn).id][2] for sn in sites])
    zs = np.array(zs)
    in_stance = contact['l'].any(1) | contact['r'].any(1)
    # stance: pelvis height so that the lowest sole point of the stance foot touches the ground (as walking D8)
    # flight: ballistic parabola between take-off and landing heights (g, given flight time)
    z = np.full(N, np.nan)
    for i in range(N):
        if in_stance[i]:
            cs = [k for k, (sd, c) in enumerate((('l', contact['l']), ('r', contact['r']))) if c[i].any()]
            z[i] = -min(zs[i, 2 * k:2 * k + 2].min() for k in cs)
    dtf = T_cycle / N
    i = 0
    while i < N:
        if not in_stance[i]:
            j = i
            while not in_stance[j % N]:
                j += 1
            z0, z1 = z[(i - 1) % N], z[j % N]
            Tf = (j - i + 1) * dtf
            v0 = (z1 - z0) / Tf + 9.81 * Tf / 2
            for k in range(i, j):
                tt = (k - i + 1) * dtf
                z[k % N] = z0 + v0 * tt - 9.81 * tt * tt / 2
            i = j
        else:
            i += 1
    Q[:, 2] = smooth(z[:, None], 3)[:, 0]
    ground_gap = Q[:, 2] + zs.min(1)
    flight = ~in_stance
    # arms: counter-phase swing, elbows bent more for running
    jq = lambda nm: model.jnt_qposadr[model.joint(nm).id]
    hpL = Q[:, qadr['left'][0]]; hpR = Q[:, qadr['right'][0]]
    Q[:, jq('left_shoulder_pitch_joint')] = 0.9 * (hpR - hpR.mean())
    Q[:, jq('right_shoulder_pitch_joint')] = 0.9 * (hpL - hpL.mean())
    Q[:, jq('left_shoulder_roll_joint')] = 0.15; Q[:, jq('right_shoulder_roll_joint')] = -0.15
    Q[:, jq('left_elbow_joint')] = 1.5; Q[:, jq('right_elbow_joint')] = 1.5
    if verbose:
        print(f'IK err max {max(err) * 1000:.2f} mm; pelvis z {Q[:, 2].min():.3f}..{Q[:, 2].max():.3f}; '
              f'flight fraction {flight.mean():.2f}; sole height in stance {1000 * ground_gap[in_stance].min():.0f}..'
              f'{1000 * ground_gap[in_stance].max():.0f} mm, in flight min {1000 * ground_gap[flight].min() if flight.any() else 0:.0f} mm')
    return dict(qpos=Q, T=T_cycle, speed=v_robot, contact_l=contact['l'], contact_r=contact['r'],
                fpitch_l=fpitch['l'], fpitch_r=fpitch['r'], dt=T_cycle / N, scale=sc, lean=LEAN)


if __name__ == '__main__':
    r = build()
    np.savez(os.path.join(HERE, 'ref_run.npz'), **r)
    print('saved ref_run.npz')
