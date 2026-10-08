# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Record the three controllers under identical conditions for video.
Saves rec_<name>.npz: qpos (T,nq) at 50 Hz, P (T,23) per-joint electrical power averaged over each 20 ms,
joint names, x. Same physics (dt 0.002, implicitfast), same start pose, stand 2 s then straight walk at 0.92 m/s.
python3 record3.py robotis|mp_fixed|eco
"""
import os, sys, json
import numpy as np, mujoco, yaml, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.environ.get('AI_SAPIENS_UPSTREAM', os.path.join(ROOT, 'external', 'ai_sapiens'))
DT = 0.002
SUB = int(round(0.02 / DT))
STAND, WALK = 2.0, 12.0
VX = 0.92


def km_of(j):
    return 2.2 if any(k in j for k in ('ankle_roll', 'shoulder', 'elbow', 'wrist')) else 4.0


def tlim_of(j):
    return 47.277 if any(k in j for k in ('ankle_roll', 'shoulder', 'elbow', 'wrist')) else 96.864


def power(tau, qd, names):
    Km = np.array([km_of(j) for j in names])
    return tau ** 2 / Km ** 2 + np.clip(tau * qd, 0, None)


def rec_robotis():
    import onnxruntime as ort
    A = f'{UPSTREAM}/ai_sapiens_sim2real/assets/k1/locomotion/velocity/walk_default'
    cfg = yaml.safe_load(open(f'{A}/params/sim2real.yaml'))
    pj = cfg['policy_joints']; jp = cfg['joint_properties']
    sess = ort.InferenceSession(f'{A}/exported/policy.onnx')
    m = mujoco.MjModel.from_xml_path(f'{ROOT}/ai_sapiens/ai_sapiens_description/mujoco/k1/scene.xml')
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[m.joint(j).id] for j in pj]); da = np.array([m.jnt_dofadr[m.joint(j).id] for j in pj])
    act = np.array([[i for i in range(m.nu) if m.actuator(i).trnid[0] == m.joint(j).id][0] for j in pj])
    q0 = np.array([jp[j]['default_position'] for j in pj])
    kp = np.array([jp[j]['stiffness'] for j in pj]); kd = np.array([jp[j]['damping'] for j in pj])
    sc = np.array(cfg['actions']['joint_pos']['scale']); off = np.array(cfg['actions']['joint_pos']['offset'])
    lim = np.array([tlim_of(j) for j in pj])
    oc = cfg['observations']; order = list(oc.keys())
    d.qpos[:] = m.key_qpos[0]; d.qpos[qa] = q0
    mujoco.mj_forward(m, d)
    zmin = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in range(m.ngeom)
               if m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
    d.qpos[2] -= zmin - 0.0005
    mujoco.mj_forward(m, d)
    last = np.zeros(23); cmd = np.zeros(3)

    def term(n):
        if n == 'base_ang_vel': v = d.qvel[3:6].copy()
        elif n == 'projected_gravity':
            R = np.zeros(9); mujoco.mju_quat2Mat(R, d.qpos[3:7]); v = R.reshape(3, 3).T @ np.array([0, 0, -1.0])
        elif n == 'velocity_commands': v = cmd.copy()
        elif n == 'joint_pos_rel': v = d.qpos[qa] - q0
        elif n == 'joint_vel_rel': v = d.qvel[da].copy()
        else: v = last.copy()
        s = oc[n].get('scale')
        return v * np.array(s) if s is not None else v
    hist = {n: [term(n)] * oc[n]['history_length'] for n in order}
    Q, P, X = [], [], []
    for k in range(int((STAND + WALK) / 0.02)):
        cmd[:] = [VX if k * 0.02 >= STAND else 0.0, 0, 0]
        for n in order:
            hist[n] = hist[n][1:] + [term(n)]
        obs = np.concatenate([np.concatenate(hist[n]) for n in order]).astype(np.float32)[None]
        a = np.clip(sess.run(None, {'obs': obs})[0][0].astype(np.float64), -5, 5)
        last = a.copy(); tgt = a * sc + off
        pw = np.zeros(23)
        for s in range(SUB):
            tau = np.clip(kp * (tgt - d.qpos[qa]) - kd * d.qvel[da], -lim, lim)
            d.ctrl[:] = 0; d.ctrl[act] = tau
            mujoco.mj_step(m, d)
            pw += power(tau, d.qvel[da], pj) / SUB
        Q.append(d.qpos.copy()); P.append(pw); X.append(d.qpos[0])
    return dict(qpos=np.array(Q), P=np.array(P), names=np.array(pj), x=np.array(X),
                xml=f'{ROOT}/ai_sapiens/ai_sapiens_description/mujoco/k1/scene.xml')


def rec_ours(kind):
    if kind == 'mp_fixed':
        sys.path.insert(0, f'{ROOT}/k1_mp'); os.chdir(f'{ROOT}/k1_mp')
        from k1env import K1Batch as Env, DEFAULT_POSE, ACT_JOINTS, KP, KD
        from ppo import AC
        path, nact = 'runs/final/model.pt', 12
    else:
        sys.path.insert(0, f'{ROOT}/k1_mp_eco'); os.chdir(f'{ROOT}/k1_mp_eco')
        from k1env import DEFAULT_POSE, ACT_JOINTS, KP, KD
        from k1env_eco import K1EcoBatch as Env
        from ppo_eco import ACEco as AC
        path, nact = 'runs/eco1/model.pt', 36
    env = Env(1, stage=2, seed=7, dt=DT, ep_len=10 ** 9); env.pushes = False
    m = env.m; d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]; d.qpos[env.qadr] = DEFAULT_POSE; d.qpos[2] = env.ref.z_stand + 0.002
    mujoco.mj_forward(m, d)
    st = np.zeros(env.nstate); mujoco.mj_getState(m, d, st, env.spec_state)
    env.state[0] = st; env.sdata[0] = d.sensordata
    env.ph[:] = 0; env.alpha[:] = 0; env.cmd[:] = 0
    env.switch_t[:] = int(STAND * 50); env.stop_t[:] = 10 ** 9
    oa, oc = env.obs()
    net = AC(oa.shape[1], oc.shape[1], nact)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    lim = np.array([tlim_of(j) for j in ACT_JOINTS])
    Q, P, X = [], [], []
    for k in range(int((STAND + WALK) * 50)):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        _, term, _, _ = env.step(a)
        tgt = env.ctrl[0]
        kp = KP.copy(); kd = KD.copy()
        if kind == 'eco':
            kp[:12] *= env.kp_scale[0]; kd[:12] *= env.kd_scale[0]
        pw = np.zeros(23)
        for s in range(env.nsub):
            x = env.st_out[0, s]
            q = x[1:1 + m.nq][env.qadr]; qd = x[1 + m.nq:1 + m.nq + m.nv][env.dadr]
            tau = np.clip(kp * (tgt - q) - kd * qd, -lim, lim)
            pw += power(tau, qd, ACT_JOINTS) / env.nsub
        Q.append(env.qpos()[0].copy()); P.append(pw); X.append(env.qpos()[0, 0])
        if term[0]:
            print('FELL'); break
        oa, oc = env.obs()
    xml = f'{ROOT}/ai_sapiens/ai_sapiens_description/mujoco/k1/scene_mp.xml'
    return dict(qpos=np.array(Q), P=np.array(P), names=np.array(ACT_JOINTS), x=np.array(X), xml=xml)


def rec_v563():
    """latest walking policy (v5.6.3) through its gait manager, same conditions (added 2026-10-08)"""
    os.environ.setdefault('K1_NOSLIP', '0')            # friction model of the original v1 comparison
    sys.path.insert(0, f'{ROOT}/k1_mp_gait56'); os.chdir(f'{ROOT}/k1_mp_gait56')
    from gait56 import Gait56
    from k1env import ACT_JOINTS, KP, KD
    g = Gait56('runs/final/walk.pt', 'runs/final/run.pt', n=1, dt=DT)
    W = g.W; m = W.m
    lim = np.array([tlim_of(j) for j in ACT_JOINTS])
    Q, P, X = [], [], []
    g.stop()
    for k in range(int((STAND + WALK) * 50)):
        if k == int(STAND * 50):
            g.command(VX, 0.0)
        term, _ = g.step()
        tgt = W.ctrl[0]
        kp = KP.copy(); kd = KD.copy()
        kp[:12] *= W.kp_scale[0] * W.kp_dr[0]; kd[:12] *= W.kd_scale[0] * W.kp_dr[0]
        pw = np.zeros(23)
        for s in range(W.nsub):
            x = W.st_out[0, s]
            q = x[1:1 + m.nq][W.qadr]; qd = x[1 + m.nq:1 + m.nq + m.nv][W.dadr]
            tau = np.clip(kp * (tgt - q) - kd * qd, -lim, lim)
            pw += power(tau, qd, ACT_JOINTS) / W.nsub
        Q.append(W.qpos()[0].copy()); P.append(pw); X.append(W.qpos()[0, 0])
        if term[0]:
            print('FELL'); break
    xml = f'{ROOT}/ai_sapiens/ai_sapiens_description/mujoco/k1/scene_mp.xml'
    return dict(qpos=np.array(Q), P=np.array(P), names=np.array(ACT_JOINTS), x=np.array(X), xml=xml)


if __name__ == '__main__':
    k = sys.argv[1]
    r = rec_robotis() if k == 'robotis' else (rec_v563() if k == 'v563' else rec_ours(k))
    np.savez(f'{ROOT}/k1_compare/rec_{k}.npz', **r)
    t = np.arange(len(r['x'])) * 0.02
    w = (t >= STAND + 4) & (t < STAND + WALK)
    v = (r['x'][w][-1] - r['x'][w][0]) / (w.sum() * 0.02)
    print(k, 'frames', len(r['x']), 'speed %.3f' % v, 'mean P %.1f W' % r['P'][w].sum(1).mean())
