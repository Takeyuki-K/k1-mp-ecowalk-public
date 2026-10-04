# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Retarget human gait (BMC walk2.trc) to K1+MP.

Rules (as requested):
  * Only the hip->ankle vector is imitated: v_robot = s * v_human, s = L_robot/L_human (leg length ratio).
  * Foot pitch (heel->MT axis vs ground) is copied so heel-strike / foot-flat / heel-off are kept.
  * All leg joints are solved by numeric IK (6-DOF per leg: ankle position + foot orientation).
  * Time scaled by sqrt(s) (Froude number kept = dynamically similar gait, natural pendulum timing).
  * MP joints are passive: reference value only for visualisation (toe kept flat on ground in stance).
  * Arms: synthetic counter-phase swing (no arm markers in dataset).
Output: ref_gait.npz
"""
import os, numpy as np, mujoco
from gait_data import load_trc, joint_centres

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, '..', 'ai_sapiens', 'ai_sapiens_description', 'mujoco', 'k1', 'scene_mp.xml')
SOLE = 0.065  # ankle pitch axis height above sole (robot)


def smooth(x, w=5):
    k = np.ones(w) / w
    pad = np.concatenate([x[-w:], x, x[:w]], 0)
    out = np.stack([np.convolve(pad[:, j], k, 'same') for j in range(x.shape[1])], 1)
    return out[w:-w]


def periodic_fix(x):
    """remove end-to-start mismatch with linear ramp so x[0]==x[N] (N = len)"""
    n = len(x) - 1
    d = x[-1] - x[0]
    ramp = np.linspace(0, 1, n + 1)[:, None]
    return (x - ramp * d)[:-1]


def heel_strikes(J, s, rate):
    hz = J[s]['heel'][:, 2]
    hv = np.gradient(J[s]['heel'][:, 0]) * rate
    st = hv < 0.25
    return [i for i in range(1, len(hz)) if st[i] and not st[i - 1]]


def main():
    mk, rate = load_trc(os.path.join(HERE, 'data', 'bmc', 'data', 'walk2.trc'))
    J = joint_centres(mk)
    hs = heel_strikes(J, 'r', rate)
    hs = [h for i, h in enumerate(hs) if i == 0 or h - hs[i - 1] > 30]
    a, b = hs[0], hs[1]
    print('right heel strikes', hs, 'cycle', a, b, (b - a) / rate, 's')
    sl = slice(a, b + 1)
    Lh = np.nanmax([np.linalg.norm(J[s]['hip'] - J[s]['ankle'], axis=1).max() for s in 'rl'])

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pel = model.body('pelvis').id
    hip_o, Lr = {}, {}
    for s, side in (('l', 'left'), ('r', 'right')):
        hr = data.xpos[model.body(f'{side}_hip_roll_link').id] - data.xpos[pel]
        an = data.xpos[model.body(f'{side}_ankle_roll_link').id] - data.xpos[pel]
        hip_o[s] = hr
        Lr[s] = np.linalg.norm(an - hr)
    L_robot = np.mean(list(Lr.values()))
    sc = L_robot / Lh
    tsc = np.sqrt(sc)
    print(f'human leg {Lh:.3f} m, robot leg {L_robot:.3f} m, s={sc:.3f}, time scale={tsc:.3f}')

    # ------------- human signals over one cycle (periodic) ---------------
    t_h = (np.arange(b - a + 1)) / rate
    midhip = 0.5 * (J['r']['hip'] + J['l']['hip'])[sl]
    speed = (midhip[-1, 0] - midhip[0, 0]) / t_h[-1]
    pel_h = midhip.copy(); pel_h[:, 0] -= speed * t_h
    pel_h = periodic_fix(pel_h)
    ank_min = {s: np.percentile(J[s]['ankle'][sl, 2], 5) for s in 'rl'}
    ankle_ref = np.mean(list(ank_min.values()))
    vec, fpitch, contact = {}, {}, {}
    for s in 'rl':
        v = periodic_fix((J[s]['ankle'] - J[s]['hip'])[sl])
        vec[s] = smooth(v, 5)
        f = (J[s]['mt'] - J[s]['heel'])[sl]
        p = -np.arctan2(f[:, 2], np.hypot(f[:, 0], f[:, 1]))   # + = heel up
        heel_z = J[s]['heel'][sl, 2]; mt_z = J[s]['mt'][sl, 2]
        both = ((heel_z - heel_z.min()) < 0.012) & ((mt_z - mt_z.min()) < 0.012)
        flat = np.median(p[both])
        print(s, 'foot-flat frames', both.sum(), 'flat offset deg %.1f' % np.degrees(flat))
        fpitch[s] = smooth(periodic_fix((p - flat)[:, None]), 5)[:, 0]
        hc = (J[s]['heel'][sl, 2] - J[s]['heel'][sl, 2].min()) < 0.015
        mc = (J[s]['mt'][sl, 2] - J[s]['mt'][sl, 2].min()) < 0.015
        contact[s] = np.stack([hc, mc], 1)[:-1]
    N = len(pel_h)
    T_cycle = N / rate * tsc
    v_robot = speed * sc / tsc
    print(f'human speed {speed:.2f} m/s -> robot {v_robot:.2f} m/s, cycle {T_cycle:.3f} s')

    # ------------- IK -------------
    qadr = {}
    names = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
    for side in ('left', 'right'):
        qadr[side] = [model.jnt_qposadr[model.joint(f'{side}_{n}_joint').id] for n in names]
    dadr = {side: [model.jnt_dofadr[model.joint(f'{side}_{n}_joint').id] for n in names] for side in qadr}
    lo = {side: np.array([model.jnt_range[model.joint(f'{side}_{n}_joint').id][0] for n in names]) for side in qadr}
    hi = {side: np.array([model.jnt_range[model.joint(f'{side}_{n}_joint').id][1] for n in names]) for side in qadr}

    q = model.key_qpos[0].copy()
    for side in qadr:
        q[qadr[side]] = [-0.25, 0, 0, 0.5, -0.25, 0]
    Q = np.zeros((N, model.nq))
    err_log = []
    for i in range(N):
        t = i / rate * tsc
        q[0] = v_robot * t + sc * pel_h[i, 0]
        q[1] = sc * pel_h[i, 1]
        q[3:7] = [1, 0, 0, 0]
        for s, side in (('l', 'left'), ('r', 'right')):
            tgt_rel = hip_o[s] + sc * vec[s][i]
            fp = fpitch[s][i]
            Rt = np.array([[np.cos(fp), 0, np.sin(fp)], [0, 1, 0], [-np.sin(fp), 0, np.cos(fp)]])
            bid = model.body(f'{side}_ankle_roll_link').id
            for it in range(60):
                q[2] = 0.0
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                pos = data.xpos[bid] - data.xpos[pel]
                ep = tgt_rel - pos
                R = data.xmat[bid].reshape(3, 3)
                Re = Rt @ R.T
                eo = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
                e = np.concatenate([ep, 0.3 * eo])
                if np.linalg.norm(e) < 1e-5:
                    break
                jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jp, jr, bid)
                Jm = np.vstack([jp[:, dadr[side]], 0.3 * jr[:, dadr[side]]])
                dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(6), e)
                q[qadr[side]] = np.clip(q[qadr[side]] + dq, lo[side], hi[side])
            err_log.append(np.linalg.norm(ep))
        Q[i] = q
    print('IK pos error max %.4f m mean %.4f m' % (max(err_log), np.mean(err_log)))

    # pelvis height so that the stance foot sole touches ground (ankle axis = SOLE above ground)
    sites = [f'{sd}_{p}_site' for sd in ('left', 'right') for p in ('heel', 'meta')]
    zs = []
    for i in range(N):
        data.qpos[:] = Q[i]; data.qpos[2] = 0
        mujoco.mj_kinematics(model, data)
        zs.append([data.site_xpos[model.site(sn).id][2] for sn in sites])
    zs = np.array(zs)  # rear-foot sole points with pelvis z=0
    # human pelvis vertical oscillation is already inside hip->ankle vectors; constant offset
    # so that the lowest rear-foot sole point touches the ground (median over cycle)
    z_off = -np.median(np.min(zs, axis=1))
    Q[:, 2] = z_off
    # foot-flat check: heel/toe height
    print('pelvis z %.3f..%.3f' % (Q[:, 2].min(), Q[:, 2].max()))

    # MP reference (visual only) - keep toe flat when foot pitched heel-up and in contact
    for s, side in (('l', 'left'), ('r', 'right')):
        ia = model.jnt_qposadr[model.joint(f'{side}_mp_joint').id]
        Q[:, ia] = np.where(contact[s][:, 1], -np.clip(fpitch[s], 0, 1.0), 0.0)

    # arms: counter-phase swing driven by opposite hip pitch
    def jq(n): return model.jnt_qposadr[model.joint(n).id]
    hpL = Q[:, qadr['left'][0]]; hpR = Q[:, qadr['right'][0]]
    Q[:, jq('left_shoulder_pitch_joint')] = 0.7 * (hpR - hpR.mean())
    Q[:, jq('right_shoulder_pitch_joint')] = 0.7 * (hpL - hpL.mean())
    Q[:, jq('left_shoulder_roll_joint')] = 0.12
    Q[:, jq('right_shoulder_roll_joint')] = -0.12
    Q[:, jq('left_elbow_joint')] = 1.1
    Q[:, jq('right_elbow_joint')] = 1.1

    np.savez(os.path.join(HERE, 'ref_gait.npz'), qpos=Q, dt=1.0 / rate * tsc, T=T_cycle,
             speed=v_robot, contact_l=contact['l'], contact_r=contact['r'],
             fpitch_l=fpitch['l'], fpitch_r=fpitch['r'], scale=sc)
    print('saved ref_gait.npz frames', N)


if __name__ == '__main__':
    main()
