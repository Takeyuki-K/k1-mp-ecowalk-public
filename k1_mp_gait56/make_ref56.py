# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5.6 reference library: level pelvis walking (idea & direction: Takeyuki-K).

Starts from ref_lib_turn.npz (v5.5: human-derived cycles 0 ... 1.95 m/s incl. stepping in place) and changes, per frame:
  1. narrower stance: ankle separation 20.9 -> 16 cm (hip-roll adduction, ankle roll keeps the soles flat).
     Less side-to-side travel is needed over a narrower base (the trained v5.5 policy already walked at ~15 cm).
  2. lateral pelvis travel from the linear inverted pendulum: for half-width w/2, step time Ts and Tc = sqrt(z/g)
     the periodic lateral COM excursion is A = w/2 * (1 - 1/cosh(Ts / (2 Tc))). The human reference only had
     +-1.85 cm (human proportions); the existing oscillation is scaled up to A, keeping its phase. The pelvis is moved
     by the legs (both hip rolls, ankle rolls compensate) so the feet stay where they were -> pelvis stays level.
     Before v5.6 the policy got the missing COM shift by tilting the whole body over the stance leg (pelvis roll
     12 deg peak-to-peak, hip-roll motors ~21 % of the leg power).
  3. more swing-foot clearance (user instruction: a little higher than a human is fine): during swing, hip flexion a,
     knee flexion 2a, ankle a (foot stays parallel), a = 0.12 rad * sin(pi * swing progress) -> about +2.5 cm.
     With a level pelvis the swing foot can no longer get clearance from the pelvis tilting up on the swing side.
  4. pelvis height re-solved so the stance foot touches the ground exactly as before (same site heights).
python3 make_ref56.py   -> ref_lib_56.npz
"""
import os
import numpy as np
import mujoco
from k1env import load_model, ACT_JOINTS

HERE = os.path.dirname(os.path.abspath(__file__))
W_TARGET = 0.16
CLEAR = 0.12
G, ZC = 9.81, 0.70
L_LEG = 0.70
J = {n: i for i, n in enumerate(ACT_JOINTS)}


def swing_profile(contact):
    """contact (M,) bool -> sin(pi * progress) during each (cyclic) swing interval, 0 in stance"""
    M = len(contact); prof = np.zeros(M)
    if contact.all() or (~contact).all():
        return prof
    start = int(np.argmax(contact))                    # rotate so that index 0 is in stance
    c = np.roll(contact, -start)
    i = 0
    while i < M:
        if not c[i]:
            j = i
            while j < M and not c[j]:
                j += 1
            n = j - i
            prof_idx = (np.arange(i, j) + start) % M
            prof[prof_idx] = np.sin(np.pi * (np.arange(n) + 0.5) / n)
            i = j
        else:
            i += 1
    return prof


def main():
    L = dict(np.load(os.path.join(HERE, 'ref_lib_turn.npz')))
    m = load_model(0.005); d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[m.joint(j).id] for j in ACT_JOINTS])
    sites = {s: [m.site(f'{s}_{p}_site').id for p in ('heel', 'meta')] for s in ('left', 'right')}
    Q = L['qpos'].copy(); S, M, _ = Q.shape
    Tc = np.sqrt(ZC / G)
    report = []
    for si in range(S):
        T = L['T'][si]
        cl = L['cl'][si].astype(bool).any(-1) if L['cl'].ndim == 3 else L['cl'][si].astype(bool)
        cr = L['cr'][si].astype(bool).any(-1) if L['cr'].ndim == 3 else L['cr'][si].astype(bool)
        swl, swr = swing_profile(cl), swing_profile(cr)
        y0 = Q[si, :, 1].copy(); yc = y0 - y0.mean(); a0 = 0.5 * (yc.max() - yc.min())
        A = W_TARGET / 2 * (1 - 1 / np.cosh((T / 2) / (2 * Tc)))
        gain = A / max(a0, 1e-3)
        # original stance-foot site heights (to keep the ground contact identical)
        orig_min = np.zeros(M)
        for i in range(M):
            d.qpos[:] = Q[si, i]; mujoco.mj_kinematics(m, d)
            zs = [d.site_xpos[k][2] for s, c in (('left', cl[i]), ('right', cr[i])) if c for k in sites[s]]
            orig_min[i] = min(zs) if zs else np.nan
        sep0 = []
        for i in range(M):
            q = Q[si, i].copy()
            # 1. narrower stance
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)
            sep = d.xpos[m.body('left_ankle_roll_link').id][1] - d.xpos[m.body('right_ankle_roll_link').id][1]
            sep0.append(sep)
            dl = np.arctan2((0.209 - W_TARGET) / 2, L_LEG)
            q[qa[J['left_hip_roll_joint']]] -= dl; q[qa[J['left_ankle_roll_joint']]] += dl
            q[qa[J['right_hip_roll_joint']]] += dl; q[qa[J['right_ankle_roll_joint']]] -= dl
            # 2. lateral pelvis travel (feet fixed, pelvis level)
            dy = (gain - 1.0) * yc[i]
            th = np.arctan2(dy, L_LEG)
            q[1] += dy
            for s in ('left', 'right'):
                q[qa[J[f'{s}_hip_roll_joint']]] -= th; q[qa[J[f'{s}_ankle_roll_joint']]] += th
            # 3. swing clearance
            for s, b in (('left', swl[i]), ('right', swr[i])):
                a = CLEAR * b
                q[qa[J[f'{s}_hip_pitch_joint']]] -= a; q[qa[J[f'{s}_knee_joint']]] += 2 * a
                q[qa[J[f'{s}_ankle_pitch_joint']]] -= a
            # 4. pelvis height: stance sites at the original height
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)
            zs = [d.site_xpos[k][2] for s, c in (('left', cl[i]), ('right', cr[i])) if c for k in sites[s]]
            if zs and not np.isnan(orig_min[i]):
                q[2] += orig_min[i] - min(zs)
            Q[si, i] = q
        report.append((L['speeds'][si], T, 100 * a0, 100 * A, 100 * np.mean(sep0)))
    L['qpos'] = Q
    L['v56'] = np.array([W_TARGET, CLEAR])
    np.savez(os.path.join(HERE, 'ref_lib_56.npz'), **L)
    for v, T, a0, A, sep in report:
        print(f'v {v:4.2f}  cycle {T:4.2f} s  pelvis lateral amplitude {a0:4.2f} -> {A:4.2f} cm  (old ankle sep {sep:4.1f} cm)')


if __name__ == '__main__':
    main()
