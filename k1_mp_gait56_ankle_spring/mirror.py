# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Left/right mirror maps for the v5.6 walking policy (v5.6.2, user idea check: "in the mirrored situation, do the
mirrored motion").

Mirror plane = the robot's sagittal plane (y -> -y). For a rotation (axial vector) the components transform as
(wx, wy, wz) -> (-wx, wy, -wz), so with identical joint axes on both sides (checked in build()):
  joints about y (hip/ankle/shoulder pitch, knee, elbow, MP): swap left/right, same sign
  joints about x or z (hip/ankle/shoulder roll, hip/shoulder yaw, wrist roll): swap, negated; waist yaw: negated
The gait phase is shifted by half a cycle (left step <-> right step): sin, cos -> -sin, -cos.

Actor observation of K1Walk4Batch (81, use_brake):
  0-2 base angular velocity (x, y, z)   3-5 gravity in the base frame   6 cmd  7 alpha  8 v_cmd  9 v_ref
  10 heading error  11 w_cmd  12 w_ref  13 brake  14 sin(phase)  15 cos(phase)  16-27 leg q  28-39 leg qd
  40-80 last action (41)
Actions (41): 0-11 leg target residuals, 12-23 Kp scales, 24-35 Kd scales, 36 gait-speed choice,
  37-40 arm residuals (L shoulder pitch, L shoulder roll, R shoulder pitch, R shoulder roll)

Each map is (perm, sign): mirrored[i] = sign[i] * x[perm[i]].  mirror_state() mirrors a full MuJoCo state, used to
check the observation map against the simulator (python3 mirror.py).
"""
import numpy as np
from k1env import ACT_JOINTS, LEG_JOINTS

Y_AXIS = np.array([0.0, 1.0, 0.0])


def _joint_sign(m, j):
    ax = m.jnt_axis[m.joint(j).id]
    mirrored = np.array([-ax[0], ax[1], -ax[2]])           # axial vector under y -> -y
    return float(np.sign(np.dot(mirrored, ax))) if abs(np.dot(mirrored, ax)) > 0.5 else 0.0


def _other(j):
    if j.startswith('left_'):
        return 'right_' + j[5:]
    if j.startswith('right_'):
        return 'left_' + j[6:]
    return j


def joint_map(m, names):
    perm = np.array([names.index(_other(j)) for j in names])
    sign = np.array([_joint_sign(m, j) for j in names])
    for j in names:                                         # same axes on both sides
        assert np.allclose(m.jnt_axis[m.joint(j).id], m.jnt_axis[m.joint(_other(j)).id]), j
    return perm, sign


def build(m, nact=41, use_brake=True):
    """nact 41: walking policy; nact 36: running policy (K1Run4Batch: same observation layout with 36 last actions:
    0-2 gyro, 3-5 gravity, 6 cmd, 7 alpha, 8 v_cmd, 9 v_ref, 10 heading error, 11 w_cmd, 12 w_ref, 13 brake,
    14-15 phase, 16-27 q, 28-39 qd, 40-75 last action; actions = 12 targets, 12 Kp, 12 Kd)"""
    jp, js = joint_map(m, LEG_JOINTS)
    # actions
    ap = list(jp) + list(12 + jp) + list(24 + jp)
    asg = list(js) + [1.0] * 12 + [1.0] * 12
    if nact == 41:
        ap += [36]; asg += [1.0]
        arm = ['left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint']
        ap += [37 + arm.index(_other(j)) for j in arm]
        asg += [_joint_sign(m, j) for j in arm]
    ap = np.array(ap); asg = np.array(asg)
    assert len(ap) == nact
    # observations
    op = list(range(16)); osg = [-1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1]
    if not use_brake:
        raise NotImplementedError
    op += list(16 + jp) + list(28 + jp) + list(40 + ap)
    osg += list(js) + list(js) + list(asg)
    op = np.array(op); osg = np.array(osg, float)
    for p, s in ((ap, asg), (op, osg)):                     # an involution: mirroring twice is the identity
        assert np.all(p[p] == np.arange(len(p))) and np.all(s * s[p] == 1)
    return (op, osg), (ap, asg)


def apply(x, mp):
    perm, sign = mp
    return x[..., perm] * sign


def mirror_state(m, qpos, qvel):
    """mirror a MuJoCo state of the K1 model (free base + all hinge joints) about the robot's world x-z plane"""
    q = qpos.copy(); v = qvel.copy()
    q[1] = -qpos[1]
    w, x, y, z = qpos[3:7]
    q[3:7] = [w, -x, y, -z]                                 # reflected rotation
    v[1] = -qvel[1]
    v[3:6] = [-qvel[3], qvel[4], -qvel[5]]
    for jid in range(1, m.njnt):
        name = m.joint(jid).name
        o = m.joint(_other(name)).id
        s = _joint_sign(m, name)
        q[m.jnt_qposadr[jid]] = s * qpos[m.jnt_qposadr[o]]
        v[m.jnt_dofadr[jid]] = s * qvel[m.jnt_dofadr[o]]
    return q, v


def check_run(path='runs/final/run.pt', nsteps=100):
    """same check for the running policy / env"""
    import os, torch, mujoco
    from k1env_run4 import K1Run4Batch
    from ppo_run4 import ACEco
    K1Run4Batch.P_STAND = 0.0; K1Run4Batch.P_WALK = 0.0
    env = K1Run4Batch(2, v_lo=3.0, v_hi=3.0, stage=2, randomize=False, seed=3, ep_len=10 ** 9)
    env.pushes = False; env.scripted = True
    om, amp = build(env.m, nact=36)
    oa, oc = env.obs()
    net = ACEco(oa.shape[1], oc.shape[1], 36)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    env.v_cmd[:] = 3.0; env.w_cmd[:] = 0.4
    for _ in range(nsteps):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        env.step(a); oa, _ = env.obs()
    m = env.m; d = mujoco.MjData(m)
    q1, v1 = mirror_state(m, env.qpos()[0], env.qvel()[0])
    d.qpos[:] = q1; d.qvel[:] = v1; mujoco.mj_forward(m, d)
    st = np.zeros(env.nstate); mujoco.mj_getState(m, d, st, env.spec_state)
    env.state[1] = st; env.sdata[1] = d.sensordata
    env.ph[1] = (env.ph[0] + 0.5) % 1.0
    for k in ('cmd', 'alpha', 'v_cmd', 'v_ref', 'brake'):
        getattr(env, k)[1] = getattr(env, k)[0]
    for k in ('w_cmd', 'w_ref'):
        getattr(env, k)[1] = -getattr(env, k)[0]
    env.yaw_t[1] = -env.yaw_t[0]
    env.last_a[1] = apply(env.last_a[0], amp)
    oa, _ = env.obs()
    err = np.abs(apply(oa[0], om) - oa[1])
    print('run: max |mirror(obs) - obs(mirrored state)| =', err.max(), 'at index', int(err.argmax()))
    with torch.no_grad():
        mu = net.dist(torch.from_numpy(oa)).mean.numpy()
    print('run policy asymmetry: mean %.3f max %.3f' % (np.abs(apply(mu[0], amp) - mu[1]).mean(), np.abs(apply(mu[0], amp) - mu[1]).max()))


def check(path='runs/final/walk.pt', nsteps=150):
    """roll out a policy, mirror the simulator state + commands, and compare the env observation of the mirrored
    state with the mirror map applied to the original observation"""
    import os, torch, mujoco
    os.environ.setdefault('WALK_BRAKE', '1')
    from k1env_walk4 import K1Walk4Batch
    from ppo_walk4 import ACEco
    K1Walk4Batch.P_RUN = 0.0
    env = K1Walk4Batch(2, stage=2, randomize=False, seed=3, ep_len=10 ** 9)
    env.pushes = False; env.scripted = True
    om, amp = build(env.m)
    oa, _ = env.obs()
    net = ACEco(oa.shape[1], env.obs()[1].shape[1], env.nact)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    env.cmd[:] = 1; env.v_cmd[:] = 0.6; env.w_cmd[:] = 0.4
    for _ in range(nsteps):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        env.step(a); oa, _ = env.obs()
    # robot 1 := mirror of robot 0
    m = env.m; d = mujoco.MjData(m)
    q0 = env.qpos()[0]; v0 = env.qvel()[0]
    q1, v1 = mirror_state(m, q0, v0)
    d.qpos[:] = q1; d.qvel[:] = v1; mujoco.mj_forward(m, d)
    st = np.zeros(env.nstate); mujoco.mj_getState(m, d, st, env.spec_state)
    env.state[1] = st; env.sdata[1] = d.sensordata
    env.ph[1] = (env.ph[0] + 0.5) % 1.0
    for k in ('cmd', 'alpha', 'v_cmd', 'v_ref', 'brake'):
        getattr(env, k)[1] = getattr(env, k)[0]
    for k in ('w_cmd', 'w_ref'):
        getattr(env, k)[1] = -getattr(env, k)[0]
    env.yaw_t[1] = -env.yaw_t[0]
    env.last_a[1] = apply(env.last_a[0], amp)
    oa, _ = env.obs()
    err = np.abs(apply(oa[0], om) - oa[1])
    print('max |mirror(obs) - obs(mirrored state)| =', err.max(), 'at index', int(err.argmax()))
    with torch.no_grad():
        mu = net.dist(torch.from_numpy(oa)).mean.numpy()
    print('policy asymmetry |mirror(a(o)) - a(mirror(o))|: mean %.3f max %.3f (0 for a symmetric policy)'
          % (np.abs(apply(mu[0], amp) - mu[1]).mean(), np.abs(apply(mu[0], amp) - mu[1]).max()))
    return err.max()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == 'run':
        check_run(sys.argv[2])
    else:
        check(*sys.argv[1:2])
