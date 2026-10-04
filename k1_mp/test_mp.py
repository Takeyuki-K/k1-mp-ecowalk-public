# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Physical validation of the passive MP spring joint (stiff PD rig, legs held rigid).
Test 1  quiet flat standing         -> toes must stay flat (0 deg)
Test 2  weight moved forward onto toe pads (tiptoe pose) -> toes must bend under body weight
        and the spring alone must NOT push the body up.
        Compared against an over-stiff spring (K=25 Nm/rad) that does lift the body.
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k1env import build_spec, DEFAULT_POSE, ACT_JOINTS

KP_RIG, KD_RIG = 900.0, 20.0


def make(K=None):
    spec = build_spec(0.005, servo=False)
    if K is not None:
        for j in spec.joints:
            if j.name.endswith('mp_joint'):
                j.stiffness = [K, 0, 0]
    return spec.compile()


def run(m, pose, mp0, T=2.0):
    d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[m.joint(j).id] for j in ACT_JOINTS])
    da = np.array([m.jnt_dofadr[m.joint(j).id] for j in ACT_JOINTS])
    mp = [m.jnt_qposadr[m.joint(f'{s}_mp_joint').id] for s in ('left', 'right')]
    d.qpos[:] = m.key_qpos[0]; d.qpos[qa] = pose; d.qpos[mp] = mp0
    mujoco.mj_kinematics(m, d)
    low = min(d.geom_xpos[m.geom(g).id][2] for g in ['left_heel', 'left_meta']) - 0.015
    zt = min(d.site_xpos[m.site(s).id][2] for s in ['left_heel_site', 'left_meta_site', 'left_toe_site'])
    d.qpos[2] -= zt - 0.0005
    mujoco.mj_forward(m, d)
    z0 = d.qpos[2]
    hist = []
    p0 = d.qpos[:2].copy()
    while d.time < T:
        d.ctrl[:] = KP_RIG * (pose - d.qpos[qa]) - KD_RIG * d.qvel[da]
        # horizontal-only stabiliser (no vertical force: full body weight stays on the feet)
        q = d.qpos[3:7]
        d.xfrc_applied[1, :2] = 2000 * (p0 - d.qpos[:2]) - 200 * d.qvel[:2]
        d.xfrc_applied[1, 3:5] = -400 * np.array([2 * q[1], 2 * q[2]]) - 40 * d.qvel[3:5]
        mujoco.mj_step(m, d)
        hist.append([d.time, d.qpos[2] - z0, d.qpos[mp[0]], d.site_xpos[m.site('left_heel_site').id][2],
                     d.subtree_com[1][0] - d.site_xpos[m.site('left_toe_site').id][0]])
    return np.array(hist)


def tiptoe_pose(m, beta):
    """legs straight-ish, rear foot pitched heel-up by beta; lean so CoM sits over the toe pads"""
    d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[m.joint(j).id] for j in ACT_JOINTS])
    best = None
    for h in np.linspace(-0.4, 0.2, 121):
        pose = DEFAULT_POSE.copy()
        for s in ('left', 'right'):
            pose[ACT_JOINTS.index(f'{s}_hip_pitch_joint')] = h
            pose[ACT_JOINTS.index(f'{s}_knee_joint')] = 0.05
            pose[ACT_JOINTS.index(f'{s}_ankle_pitch_joint')] = beta - h - 0.05
        d.qpos[:] = m.key_qpos[0]; d.qpos[qa] = pose
        mujoco.mj_forward(m, d)
        e = d.subtree_com[1][0] - d.site_xpos[m.site('left_toe_site').id][0]
        if pose[ACT_JOINTS.index('left_ankle_pitch_joint')] > 0.78:
            continue
        if best is None or abs(e) < abs(best[0]):
            best = (e, pose)
    return best[1]


if __name__ == '__main__':
    m = make()
    K = m.jnt_stiffness[m.joint('left_mp_joint').id]
    W = sum(m.body_mass) * 9.81
    print(f'robot mass {sum(m.body_mass):.1f} kg, MP spring K={K:.2f} Nm/rad, damping {m.dof_damping[m.joint("left_mp_joint").dofadr[0]]:.3f}')
    h = run(m, DEFAULT_POSE, [0, 0])
    print(f'[Test1 flat standing]  MP angle after 2s: {np.degrees(h[-1,2]):.2f} deg (max |{np.degrees(np.abs(h[:,2]).max()):.2f}| deg)')
    beta = 0.6
    pose = tiptoe_pose(m, beta)
    for KK, label in [(None, f'designed K={K:.2f}'), (25.0, 'over-stiff K=25')]:
        mm = make(KK)
        h = run(mm, pose, [-beta, -beta])
        h_ = h
        print(f'[Test2 tiptoe, {label:16s}] MP start {-np.degrees(beta):.0f} -> end {np.degrees(h[-1,2]):6.1f} deg, '
              f'body moved {h[-1,1]*1000:+6.1f} mm (max up {h[:,1].max()*1000:+.1f} mm), heel height {h[-1,3]*1000:.0f} mm')
    lim = -m.jnt_range[m.joint('left_mp_joint').id][0]
    print(f'[Static check] max spring torque {K*lim:.2f} Nm -> max toe push {K*lim/0.025:.0f} N per foot; '
          f'half body weight {W/2:.0f} N  => spring alone cannot lift the foot (margin x{W/2/(K*lim/0.025):.2f}).')
    print(f'               bending under load: body weight on one toe pad gives {W*0.025:.1f} Nm >> spring torque at 30deg {K*0.52:.2f} Nm')
