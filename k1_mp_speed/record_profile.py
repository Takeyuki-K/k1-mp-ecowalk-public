# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Record ROBOTIS walk_default and the speed-command eco policy on the SAME speed profile (dt 2 ms, 50 Hz).
Saves out/prof_<name>.npz: qpos (T,nq) 50 Hz, P (T,23) per-joint electrical power, v_cmd (T,), touchdowns.
python3 record_profile.py robotis|speed <model.pt>
"""
import os, sys
import numpy as np, mujoco, yaml, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UPSTREAM = os.environ.get('AI_SAPIENS_UPSTREAM', os.path.join(ROOT, 'external', 'ai_sapiens'))
DT = 0.002
SUB = 10
# (start time [s], walk flag, speed [m/s])
PROFILE = [(0.0, 0, 0.0), (2.0, 1, 0.4), (7.0, 1, 1.2), (12.0, 1, 0.6), (17.0, 1, 1.35), (22.0, 1, 0.3), (27.0, 0, 0.0)]
T_END = 30.0


def cmd_at(t):
    c = [p for p in PROFILE if p[0] <= t + 1e-9][-1]
    return c[1], c[2]


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
    Q, P, V = [], [], []
    for k in range(int(T_END / 0.02)):
        w, v = cmd_at(k * 0.02)
        cmd[:] = [v * w, 0, 0]
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
        Q.append(d.qpos.copy()); P.append(pw); V.append(v * w)
    return dict(qpos=np.array(Q), P=np.array(P), names=np.array(pj), v_cmd=np.array(V), xml='scene.xml')


def rec_speed(path):
    sys.path.insert(0, HERE); os.chdir(HERE)
    from k1env import ACT_JOINTS, KP, KD
    import eval_speed as E
    from k1env_speed import K1SpeedBatch
    env = K1SpeedBatch(1, stage=2, randomize=False, seed=5, dt=DT, ep_len=10 ** 9)
    env.pushes = False; env.scripted = True
    E.stand_all(env)
    net = E.load(path, env)
    lim = np.array([tlim_of(j) for j in ACT_JOINTS])
    oa, _ = env.obs()
    Q, P, V, VL = [], [], [], []
    for k in range(int(T_END * 50)):
        w, v = cmd_at(k * 0.02)
        env.cmd[:] = w
        if w:
            env.v_cmd[:] = v
        a = E.act(net, oa)
        _, term, _, _ = env.step(a)
        tgt = env.ctrl[0]
        kp = KP.copy(); kd = KD.copy(); kp[:12] *= env.kp_scale[0]; kd[:12] *= env.kd_scale[0]
        pw = np.zeros(23)
        for s in range(env.nsub):
            x = env.st_out[0, s]
            q = x[1:1 + env.m.nq][env.qadr]; qd = x[1 + env.m.nq:1 + env.m.nq + env.m.nv][env.dadr]
            pw += power(np.clip(kp * (tgt - q) - kd * qd, -lim, lim), qd, ACT_JOINTS) / env.nsub
        Q.append(env.qpos()[0].copy()); P.append(pw); V.append(v * w); VL.append(env.v_lib[0])
        if term[0]:
            print('FELL at', k * 0.02); break
        oa, _ = env.obs()
    return dict(qpos=np.array(Q), P=np.array(P), names=np.array(ACT_JOINTS), v_cmd=np.array(V), v_lib=np.array(VL),
                kp=None, xml='scene_mp.xml')


if __name__ == '__main__':
    k = sys.argv[1]
    r = rec_robotis() if k == 'robotis' else rec_speed(sys.argv[2])
    np.savez(os.path.join(HERE, 'out', f'prof_{k}.npz'), **{a: b for a, b in r.items() if b is not None})
    x = r['qpos'][:, 0]; y = r['qpos'][:, 1]
    print(k, 'frames', len(x), 'final x %.2f y %.2f' % (x[-1], y[-1]), 'mean P while walking %.1f W' %
          r['P'][r['v_cmd'] > 0].sum(1).mean())
