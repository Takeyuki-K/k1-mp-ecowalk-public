# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Running speed library 1.6 ... 5.5 m/s for K1 + MP toes from the fastest CMU runner (subject 09, trial 09_04,
3.54 m/s, one right-to-right cycle).

Same imitation rule as before: only the hip->ankle vector is imitated (scaled by the leg-length ratio, Froude time
scaling), the leg joints are solved by IK.
Speed scaling (see REPORT_SPRINT.md, S2): the hip-relative foot path is kept and the cycle is TIME-WARPED:
  * cadence  ~ v^0.35  (humans raise speed mostly by stride length up to ~7 m/s)
  * contact time ~ 1/v (contact length roughly constant), the rest of the cycle is swing / flight
  => stride and flight time grow with speed, stance stays a "heel-rocker" over the same ground distance.
Heel-first landing imposed as in the jog reference (user specification), 5 deg trunk lean.
Output: ref_sprint_lib.npz (speeds, T, qpos, cl, cr, lean) - same layout as the walking speed library.
"""
import os
import numpy as np
import mujoco
import bvh

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, '..', 'ai_sapiens', 'ai_sapiens_description', 'mujoco', 'k1', 'scene_mp.xml')
TRIAL = os.path.join(HERE, 'data', 'cmu', 'data', '009', '09_04.bvh')
SPEEDS = [1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
STRIKE = 'fore'     # v5: CMU's own mid/forefoot landing (user decision); 'heel' = v4 heel-first override
N = 96
LEAN = np.radians(5.0)
HEEL_PITCH = -0.17
REST_PITCH = np.arctan2(-0.46332, 2.12791)
CAD_EXP = 0.35


def smooth(x, w=5):
    k = np.ones(w) / w
    pad = np.concatenate([x[-w:], x, x[:w]], 0)
    return np.stack([np.convolve(pad[:, j], k, 'same') for j in range(x.shape[1])], 1)[w:-w]


def periodic_fix(x):
    n = len(x) - 1
    ramp = np.linspace(0, 1, n + 1)[:, None]
    return (x - ramp * (x[-1] - x[0]))[:-1]


def to_robot(P, fwd):
    up = np.array([0, 1.0, 0]); f = fwd / np.linalg.norm(fwd); left = np.cross(up, f)
    return np.stack([P @ f, P @ left, P @ up], -1)


def resample(x, n):
    """periodic linear resampling of (M, ...) to n samples"""
    M = len(x); t = np.arange(n) * M / n
    i0 = np.floor(t).astype(int) % M; i1 = (i0 + 1) % M; w = (t - np.floor(t)).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - w) + x[i1] * w


def base_cycle(verbose=True):
    B = bvh.load(TRIAL)
    n = B['names']; rate = 1.0 / B['dt']
    g = lambda k: B['P'][:, n.index(k)]
    hipm = 0.5 * (g('LeftUpLeg') + g('RightUpLeg'))
    fwd = (hipm[-1] - hipm[0]) * np.array([1, 0, 1])
    J = {s: dict(hip=to_robot(g(S + 'UpLeg'), fwd), ankle=to_robot(g(S + 'Foot'), fwd), mt=to_robot(g(S + 'ToeBase'), fwd))
         for s, S in (('l', 'Left'), ('r', 'Right'))}
    ground = min(J[s][k][:, 2].min() for s in 'rl' for k in ('ankle', 'mt'))
    st = {}
    for s in 'rl':
        low = np.minimum(J[s]['ankle'][:, 2], J[s]['mt'][:, 2]) - ground
        hv = np.linalg.norm(np.gradient(J[s]['mt'][:, :2], axis=0), axis=1) * rate
        st[s] = (low < 0.06) & (hv < 2.0)
    td = [i for i in range(1, len(st['r'])) if st['r'][i] and not st['r'][i - 1]]
    a, b = td[0], td[1]
    sl = slice(a, b + 1)
    Lh = max(np.linalg.norm(J[s]['hip'] - J[s]['ankle'], axis=1).max() for s in 'rl')
    t_h = np.arange(b - a + 1) / rate
    midhip = 0.5 * (J['r']['hip'] + J['l']['hip'])[sl]
    speed = (midhip[-1, 0] - midhip[0, 0]) / t_h[-1]
    pel = midhip.copy(); pel[:, 0] -= speed * t_h
    pel = periodic_fix(pel)
    M = len(pel)
    vec, fp, stn = {}, {}, {}
    for s in 'rl':
        vec[s] = smooth(periodic_fix((J[s]['ankle'] - J[s]['hip'])[sl]), 5)
        f = (J[s]['mt'] - J[s]['ankle'])[sl]
        p = -(np.arctan2(f[:, 2], np.hypot(f[:, 0], f[:, 1])) - REST_PITCH)
        fp[s] = smooth(periodic_fix(p[:, None]), 5)[:, 0]
        stn[s] = st[s][sl][:-1]
    if verbose:
        print(f'CMU 09_04: {speed:.2f} m/s, cycle {M / rate:.3f} s, leg {Lh:.3f} m, '
              f'stance R {stn["r"].mean():.2f} L {stn["l"].mean():.2f}, touchdowns R {td}')
    return dict(pel=pel, vec=vec, fp=fp, stn=stn, speed=speed, T=M / rate, Lh=Lh, M=M)


def build_lib(verbose=True):
    bc = base_cycle(verbose)
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    QL = np.array([np.cos(LEAN / 2), 0, np.sin(LEAN / 2), 0])
    data.qpos[3:7] = QL
    mujoco.mj_forward(model, data)
    pel_id = model.body('pelvis').id
    hip_o, Lr = {}, {}
    for s, side in (('l', 'left'), ('r', 'right')):
        hip_o[s] = data.xpos[model.body(f'{side}_hip_roll_link').id] - data.xpos[pel_id]
        Lr[s] = np.linalg.norm(data.xpos[model.body(f'{side}_ankle_roll_link').id] - data.xpos[pel_id] - hip_o[s])
    # 0.95: the human's longest hip-ankle distance maps to 95 % of the straight robot leg (avoid the knee
    # singularity; found during the first build: IK failed in late swing with the full-length mapping)
    Lmax = np.mean(list(Lr.values()))
    sc = 0.95 * Lmax / bc['Lh']; tsc = np.sqrt(sc)
    M = bc['M']
    T_b = bc['T'] * tsc; v_b = bc['speed'] * sc / tsc
    # base quantities on N samples (phase 0 = right touchdown)
    vec = {s: resample(bc['vec'][s], N) for s in 'rl'}
    fpb = {s: resample(bc['fp'][s][:, None], N)[:, 0] for s in 'rl'}
    stn = {s: resample(bc['stn'][s].astype(float)[:, None], N)[:, 0] > 0.5 for s in 'rl'}
    pel = resample(bc['pel'], N)
    td = {s: [i for i in range(N) if stn[s][i] and not stn[s][i - 1]][0] for s in 'rl'}
    d_b = {s: stn[s].mean() for s in 'rl'}
    if verbose:
        print('natural foot pitch at touchdown (deg, + = heel up):', {s: round(float(np.degrees(fpb[s][td[s]])), 1) for s in 'rl'})
    # heel-first override on the base cycle (v4 only)
    for s in ('rl' if STRIKE == 'heel' else ''):
        w = np.zeros(N); L = int(round(d_b[s] * N))
        pre, post = int(0.12 * N), max(int(0.25 * L), 2)
        for k in range(-pre, post + 1):
            j = (td[s] + k) % N
            w[j] = max(w[j], (1 + k / pre) if k < 0 else (1 - k / post))
        w = 0.5 - 0.5 * np.cos(np.pi * np.clip(w, 0, 1))
        fpb[s] = (1 - w) * fpb[s] + w * HEEL_PITCH
    if verbose:
        print(f'scale {sc:.3f}; base robot speed {v_b:.2f} m/s, cycle {T_b:.3f} s, duty R {d_b["r"]:.2f}')
    names = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
    qadr = {sd: [model.jnt_qposadr[model.joint(f'{sd}_{nm}_joint').id] for nm in names] for sd in ('left', 'right')}
    dadr = {sd: [model.jnt_dofadr[model.joint(f'{sd}_{nm}_joint').id] for nm in names] for sd in qadr}
    lo = {sd: np.array([model.jnt_range[model.joint(f'{sd}_{nm}_joint').id][0] for nm in names]) for sd in qadr}
    hi = {sd: np.array([model.jnt_range[model.joint(f'{sd}_{nm}_joint').id][1] for nm in names]) for sd in qadr}
    sites = [f'{sd}_{p}_site' for sd in ('left', 'right') for p in ('heel', 'meta')]
    jq = lambda nm: model.jnt_qposadr[model.joint(nm).id]
    out = dict(speeds=np.array(SPEEDS), T=[], qpos=[], cl=[], cr=[], duty=[], qd_max=[])
    for v in SPEEDS:
        kf = (v / v_b) ** CAD_EXP
        T = T_b / kf
        warped = {}
        for s in 'rl':
            tc = d_b[s] * T_b * v_b / v
            d = float(np.clip(tc / T, 0.12, 0.65))
            phl = (np.arange(N) / N - td[s] / N) % 1.0                 # local phase from this foot's touchdown
            psi = np.where(phl < d, phl * d_b[s] / d, d_b[s] + (phl - d) * (1 - d_b[s]) / (1 - d))
            bph = (psi + td[s] / N) % 1.0
            idx = bph * N
            i0 = np.floor(idx).astype(int) % N; i1 = (i0 + 1) % N; w = idx - np.floor(idx)
            warped[s] = dict(vec=vec[s][i0] * (1 - w[:, None]) + vec[s][i1] * w[:, None],
                             fp=fpb[s][i0] * (1 - w) + fpb[s][i1] * w,
                             stance=phl < d,
                             heel=(phl < 0.35 * d) if STRIKE == 'heel' else ((phl >= 0.30 * d) & (phl < 0.70 * d)),
                             fore=((phl >= 0.15 * d) & (phl < d)) if STRIKE == 'heel' else (phl < d), d=d)
        # pelvis lateral oscillation: warped with the right-foot map; vertical from contact / ballistic flight
        q = model.key_qpos[0].copy()
        for sd in qadr:
            q[qadr[sd]] = [-0.4, 0, 0, 0.9, -0.4, 0]
        Q = np.zeros((N, model.nq)); err = np.zeros(N)
        for i in range(N):
            t = i * T / N
            q[0] = v * t; q[1] = sc * pel[i, 1]; q[3:7] = QL
            for s, sd in (('l', 'left'), ('r', 'right')):
                vv = sc * warped[s]['vec'][i]
                nv = np.linalg.norm(vv)
                if nv > 0.97 * Lmax:
                    vv = vv * 0.97 * Lmax / nv
                tgt = hip_o[s] + vv
                fpi = warped[s]['fp'][i]
                Rt = np.array([[np.cos(fpi), 0, np.sin(fpi)], [0, 1, 0], [-np.sin(fpi), 0, np.cos(fpi)]])
                bid = model.body(f'{sd}_ankle_roll_link').id
                for it in range(80):
                    q[2] = 0.0
                    data.qpos[:] = q
                    mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
                    ep = tgt - (data.xpos[bid] - data.xpos[pel_id])
                    Re = Rt @ data.xmat[bid].reshape(3, 3).T
                    eo = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
                    e = np.concatenate([ep, 0.3 * eo])
                    if np.linalg.norm(e) < 1e-5:
                        break
                    jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
                    mujoco.mj_jacBody(model, data, jp, jr, bid)
                    Jm = np.vstack([jp[:, dadr[sd]], 0.3 * jr[:, dadr[sd]]])
                    q[qadr[sd]] = np.clip(q[qadr[sd]] + Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(6), e), lo[sd], hi[sd])
                err[i] = max(err[i], np.linalg.norm(ep))
            Q[i] = q
        zs = []
        for i in range(N):
            data.qpos[:] = Q[i]; data.qpos[2] = 0
            mujoco.mj_kinematics(model, data)
            zs.append([data.site_xpos[model.site(sn).id][2] for sn in sites])
        zs = np.array(zs)
        cl = np.stack([warped['l']['heel'], warped['l']['fore']], 1); cr = np.stack([warped['r']['heel'], warped['r']['fore']], 1)
        stl, str_ = warped['l']['stance'], warped['r']['stance']
        z = np.full(N, np.nan)
        for i in range(N):
            ks = [k for k, st_ in enumerate((stl, str_)) if st_[i]]
            if ks:
                z[i] = -min(zs[i, 2 * k:2 * k + 2].min() for k in ks)
        dtf = T / N
        i = 0
        while i < N:
            if np.isnan(z[i]):
                j = i
                while np.isnan(z[j % N]):
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
        hpL = Q[:, qadr['left'][0]]; hpR = Q[:, qadr['right'][0]]
        Q[:, jq('left_shoulder_pitch_joint')] = 0.9 * (hpR - hpR.mean())
        Q[:, jq('right_shoulder_pitch_joint')] = 0.9 * (hpL - hpL.mean())
        Q[:, jq('left_shoulder_roll_joint')] = 0.15; Q[:, jq('right_shoulder_roll_joint')] = -0.15
        Q[:, jq('left_elbow_joint')] = 1.5; Q[:, jq('right_elbow_joint')] = 1.5
        legq = Q[:, qadr['left'] + qadr['right']]
        qd = (np.roll(legq, -1, 0) - legq) / dtf
        qdm = np.abs(qd).max(0)
        flight = (~(stl | str_)).mean()
        if verbose:
            print(f'v {v:.2f}: cycle {T:.3f} s, cadence {120 / T:.0f}/min, step {v * T / 2:.2f} m, duty {warped["r"]["d"]:.2f}, '
                  f'flight {flight:.2f}, IKerr max {1000 * err.max():.0f} mm, pelvis z {Q[:, 2].min():.3f}..{Q[:, 2].max():.3f}, '
                  f'max joint speed hip_pitch {qdm[0]:.1f} knee {qdm[3]:.1f} ankle {qdm[4]:.1f} rad/s')
        out.setdefault('dbg_err', []).append(err); out.setdefault('dbg_knee', []).append(Q[:, qadr['right'][3]])
        out['T'].append(T); out['qpos'].append(Q); out['cl'].append(cl); out['cr'].append(cr)
        out['duty'].append(warped['r']['d']); out['qd_max'].append(qdm)
    for k in ('T', 'qpos', 'cl', 'cr', 'duty', 'qd_max', 'dbg_err', 'dbg_knee'):
        out[k] = np.array(out[k])
    out['lean'] = LEAN
    return out


if __name__ == '__main__':
    L = build_lib()
    np.savez(os.path.join(HERE, 'ref_sprint_lib.npz'), **L)
    print('saved ref_sprint_lib.npz (strike = %s)' % STRIKE)
