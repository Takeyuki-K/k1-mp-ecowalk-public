# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Same-condition energy comparison of three walking controllers on K1 (MuJoCo).

 (1) ROBOTIS official walk_default policy (ONNX) on the ORIGINAL K1 model (no MP joint)
 (2) MP-joint K1, human-gait imitation + RL, fixed ROBOTIS PD gains   (k1_mp/runs/final)
 (3) MP-joint K1, eco policy with variable impedance (Kp/Kd per step)   (k1_mp_eco/runs/eco1)

Common conditions
 - MuJoCo 3.14, implicitfast, physics dt = 0.002 s (ROBOTIS sim default), control 50 Hz
 - flat floor, friction 1.0, no pushes, no randomisation
 - start from the ROBOTIS default standing pose, stand 2 s, then walk; energy window = steady walking 4..10 s after start
 - speed: (1) commanded vx = 0.9 m/s (inside its trained range, close to the speed of (2)/(3)); also 0.5 / 0.7 m/s
 - torque law at every physics step: tau = clip(Kp (q* - q) - Kd qd, motor limit)
 - electrical model for every joint: P = sum(tau^2 / Km^2) + sum(max(tau*qd, 0))   (no regeneration)
   Km = 4.0 Nm/sqrt(W) (QC080: hips, knee, ankle pitch, waist), 2.2 (QC060: ankle roll, arms)  <- assumed
"""
import os, sys, json
import numpy as np, mujoco, yaml
import onnxruntime as ort
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.environ.get('AI_SAPIENS_UPSTREAM', os.path.join(ROOT, 'external', 'ai_sapiens'))
DT = 0.002
SUB = int(round(0.02 / DT))
STAND, WALK = 2.0, 10.0
WIN = (STAND + 4.0, STAND + WALK)


def km_of(j):
    return 2.2 if ('ankle_roll' in j or 'shoulder' in j or 'elbow' in j or 'wrist' in j) else 4.0


def tlim_of(j):
    return 47.277 if ('ankle_roll' in j or 'shoulder' in j or 'elbow' in j or 'wrist' in j) else 96.864


def energy(names, Q, QD, TAU, X, t):
    """Q,QD,TAU: (T, nj) at physics rate; t: time stamps; X: base x"""
    m = (t >= WIN[0]) & (t < WIN[1])
    tau, qd = TAU[m], QD[m]
    dur = m.sum() * DT
    Km = np.array([km_of(j) for j in names])
    p = tau * qd
    pos = np.clip(p, 0, None).sum(0) * DT / dur
    neg = -np.clip(p, None, 0).sum(0) * DT / dur
    cu = (tau ** 2 / Km ** 2).sum(0) * DT / dur
    v = (X[m][-1] - X[m][0]) / dur
    leg = np.array([('hip' in j or 'knee' in j or 'ankle' in j) for j in names])
    out = dict(speed=float(v),
               P_legs=float((pos + cu)[leg].sum()), P_arms_waist=float((pos + cu)[~leg].sum()),
               P_total=float((pos + cu).sum()),
               legs_copper=float(cu[leg].sum()), legs_pos_mech=float(pos[leg].sum()), legs_neg_mech=float(neg[leg].sum()))
    for key, sub in [('hip_pitch', 'hip_pitch'), ('hip_roll', 'hip_roll'), ('hip_yaw', 'hip_yaw'), ('knee', 'knee'),
                     ('ankle_pitch', 'ankle_pitch'), ('ankle_roll', 'ankle_roll')]:
        s = np.array([sub in j for j in names])
        out['legs_' + key] = float((pos + cu)[s].sum())
    return out


# ------------------------------------------------------------------ (1) ROBOTIS official
def run_robotis(vx):
    A = f'{UPSTREAM}/ai_sapiens_sim2real/assets/k1/locomotion/velocity/walk_default'
    cfg = yaml.safe_load(open(f'{A}/params/sim2real.yaml'))
    pj = cfg['policy_joints']
    jp = cfg['joint_properties']
    sess = ort.InferenceSession(f'{A}/exported/policy.onnx')
    m = mujoco.MjModel.from_xml_path(f'{ROOT}/ai_sapiens/ai_sapiens_description/mujoco/k1/scene.xml')
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    qa = np.array([m.jnt_qposadr[m.joint(j).id] for j in pj])
    da = np.array([m.jnt_dofadr[m.joint(j).id] for j in pj])
    act = np.array([[i for i in range(m.nu) if m.actuator(i).trnid[0] == m.joint(j).id][0] for j in pj])
    q0 = np.array([jp[j]['default_position'] for j in pj])
    kp = np.array([jp[j]['stiffness'] for j in pj]); kd = np.array([jp[j]['damping'] for j in pj])
    sc = np.array(cfg['actions']['joint_pos']['scale']); off = np.array(cfg['actions']['joint_pos']['offset'])
    lim = np.array([tlim_of(j) for j in pj])
    obs_cfg = cfg['observations']
    order = list(obs_cfg.keys())
    d.qpos[:] = m.key_qpos[0]; d.qpos[qa] = q0
    mujoco.mj_forward(m, d)
    # place the feet on the floor
    zmin = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in range(m.ngeom)
               if m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
    d.qpos[2] -= zmin - 0.0005
    mujoco.mj_forward(m, d)
    last = np.zeros(23)
    cmd = np.zeros(3)

    def term(name):
        quat = d.qpos[3:7]
        if name == 'base_ang_vel':
            v = d.qvel[3:6].copy()
        elif name == 'projected_gravity':
            R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); v = R.reshape(3, 3).T @ np.array([0, 0, -1.0])
        elif name == 'velocity_commands':
            v = cmd.copy()
        elif name == 'joint_pos_rel':
            v = d.qpos[qa] - q0
        elif name == 'joint_vel_rel':
            v = d.qvel[da].copy()
        elif name == 'last_action':
            v = last.copy()
        s = obs_cfg[name].get('scale')
        if s is not None:
            v = v * np.array(s)
        return v
    hist = {n: [term(n)] * obs_cfg[n]['history_length'] for n in order}
    tgt = q0.copy()
    rec = dict(Q=[], QD=[], TAU=[], X=[], t=[])
    T = int((STAND + WALK) / 0.02)
    fell = False
    for k in range(T):
        cmd[:] = [vx if k * 0.02 >= STAND else 0.0, 0, 0]
        for n in order:
            hist[n] = hist[n][1:] + [term(n)]
        obs = np.concatenate([np.concatenate(hist[n]) for n in order]).astype(np.float32)[None]
        a = sess.run(None, {'obs': obs})[0][0].astype(np.float64)
        a = np.clip(a, -5, 5)
        last = a.copy()
        tgt = a * sc + off
        for s in range(SUB):
            tau = np.clip(kp * (tgt - d.qpos[qa]) - kd * d.qvel[da], -lim, lim)
            d.ctrl[:] = 0; d.ctrl[act] = tau
            mujoco.mj_step(m, d)
            rec['Q'].append(d.qpos[qa].copy()); rec['QD'].append(d.qvel[da].copy()); rec['TAU'].append(tau)
            rec['X'].append(d.qpos[0]); rec['t'].append(d.time)
        if d.qpos[2] < 0.45:
            fell = True; break
    R = {k: np.array(v) for k, v in rec.items()}
    res = energy(pj, R['Q'], R['QD'], R['TAU'], R['X'], R['t'])
    res['fell'] = fell
    res['foot_contacts'] = 'flat sole (9 small spheres), no toe joint'
    return res


# ------------------------------------------------------------------ (2) and (3)
def run_ours(kind):
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
    env = Env(1, stage=2, seed=7, dt=DT, ep_len=10 ** 9)
    env.pushes = False
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
    rec = dict(Q=[], QD=[], TAU=[], X=[], t=[])
    fell = False
    for k in range(int((STAND + WALK) * 50)):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        _, term, _, _ = env.step(a)
        tgt = env.ctrl[0]                         # 23 position targets
        kp = KP.copy(); kd = KD.copy()
        if kind == 'eco':
            kp[:12] *= env.kp_scale[0]; kd[:12] *= env.kd_scale[0]
        for s in range(env.nsub):
            x = env.st_out[0, s]
            q = x[1:1 + m.nq][env.qadr]; qd = x[1 + m.nq:1 + m.nq + m.nv][env.dadr]
            rec['Q'].append(q); rec['QD'].append(qd)
            rec['TAU'].append(np.clip(kp * (tgt - q) - kd * qd, -lim, lim))
            rec['X'].append(x[1]); rec['t'].append(x[0])
        if term[0]:
            fell = True; break
        oa, oc = env.obs()
    R = {k: np.array(v) for k, v in rec.items()}
    res = energy(ACT_JOINTS, R['Q'], R['QD'], R['TAU'], R['X'], R['t'])
    res['fell'] = fell
    return res


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'robotis':
        out = {f'robotis_v{v}': run_robotis(v) for v in (0.5, 0.7, 0.9)}
    else:
        out = {which: run_ours(which)}
    json.dump(out, open(f'{ROOT}/k1_compare/res_{which}.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
