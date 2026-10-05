"""Batched MuJoCo environment for K1 + passive MP joints (CPU, mujoco.rollout threads).

Action  : 12 leg joint residuals around the (blended) human reference, PD position targets.
Arms    : follow reference arm swing (not in policy). Waist: 0.
MP joint: passive spring - never actuated, not observed by the actor (no encoder assumed).
Command : c in {0: stand, 1: walk}. alpha ramps toward c over ~1 gait cycle. Phase runs while alpha>0.
"""
import os
import numpy as np
import mujoco
from mujoco import rollout

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, '..', 'ai_sapiens', 'ai_sapiens_description', 'mujoco', 'k1', 'scene_mp.xml')

LEG = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
LEG_JOINTS = [f'{s}_{j}_joint' for s in ('left', 'right') for j in LEG]
ARM_JOINTS = [f'{s}_{j}_joint' for s in ('left', 'right') for j in
              ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll')]
ACT_JOINTS = LEG_JOINTS + ['waist_yaw_joint'] + ARM_JOINTS
# ROBOTIS walk_default sim2real gains / default pose
_D = {'hip_pitch': (-0.205, 100, 2), 'hip_roll': (0.0, 100, 2), 'hip_yaw': (0.0, 100, 2), 'knee': (0.517, 150, 4),
      'ankle_pitch': (-0.307, 40, 2), 'ankle_roll': (0.0, 40, 2), 'waist_yaw': (0.0, 200, 5),
      'shoulder_pitch': (0.218, 40, 1), 'shoulder_roll': (0.315, 40, 1), 'shoulder_yaw': (0.0, 40, 1),
      'elbow': (1.08, 40, 1), 'wrist_roll': (0.0, 40, 1)}


def _jn(j):
    return j.replace('left_', '').replace('right_', '').replace('_joint', '')


DEFAULT_POSE = np.array([_D[_jn(j)][0] * (-1 if (j.startswith('right') and 'shoulder_roll' in j) else 1)
                         for j in ACT_JOINTS])
KP = np.array([_D[_jn(j)][1] for j in ACT_JOINTS], float)
KD = np.array([_D[_jn(j)][2] for j in ACT_JOINTS], float)
FOOT_GEOMS = [f'{s}_{p}' for s in ('left', 'right') for p in ('heel', 'meta', 'toe')]


def build_spec(dt=0.005, friction=1.0, mass_scale=1.0, kp_scale=1.0, servo=True):
    spec = mujoco.MjSpec.from_file(XML)
    spec.option.timestep = dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # cheaper collisions: arms/head do not collide (only legs, torso, feet)
    for g in spec.geoms:
        if g.name and any(k in g.name for k in ('shoulder', 'elbow', 'wrist', 'head')):
            g.contype = 0; g.conaffinity = 0
        if g.name == 'floor':
            g.friction = [friction, 0.02, 0.001]
    for b in spec.bodies:
        if b.name in ('torso_link',):
            b.mass *= mass_scale
    if servo:
        for a in spec.actuators:
            j = a.target
            i = ACT_JOINTS.index(j)
            kp, kd = KP[i] * kp_scale, KD[i] * kp_scale
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = kp
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[0] = 0; a.biasprm[1] = -kp; a.biasprm[2] = -kd
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    # touch sensor sites around foot contact geoms
    for side in ('left', 'right'):
        foot = spec.body(f'{side}_ankle_roll_link')
        toe = spec.body(f'{side}_toe_link')
        foot.add_site(name=f'{side}_heel_touch', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.047, 0, -0.055],
                      size=[0.022, 0.035, 0.016], group=5)
        foot.add_site(name=f'{side}_meta_touch', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.066, 0, -0.057],
                      size=[0.014, 0.045, 0.012], group=5)
        toe.add_site(name=f'{side}_toe_touch', type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.027, 0, -0.011],
                     size=[0.028, 0.04, 0.012], group=5)
    S = mujoco.mjtSensor
    for side in ('left', 'right'):
        for p in ('heel', 'meta', 'toe'):
            spec.add_sensor(name=f'{side}_{p}_touch', type=S.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE,
                            objname=f'{side}_{p}_touch')
    for side in ('left', 'right'):
        spec.add_sensor(name=f'{side}_ankle_rel', type=S.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_XBODY,
                        objname=f'{side}_ankle_roll_link', reftype=mujoco.mjtObj.mjOBJ_XBODY, refname='pelvis')
        spec.add_sensor(name=f'{side}_foot_x', type=S.mjSENS_FRAMEXAXIS, objtype=mujoco.mjtObj.mjOBJ_XBODY,
                        objname=f'{side}_ankle_roll_link')
        spec.add_sensor(name=f'{side}_foot_pos', type=S.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_XBODY,
                        objname=f'{side}_ankle_roll_link')
    spec.add_sensor(name='base_linvel', type=S.mjSENS_VELOCIMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu')
    for side in ('left', 'right'):
        spec.add_sensor(name=f'{side}_foot_linvel', type=S.mjSENS_FRAMELINVEL, objtype=mujoco.mjtObj.mjOBJ_XBODY,
                        objname=f'{side}_ankle_roll_link')
    return spec


def load_model(dt=0.005, **kw):
    return build_spec(dt, **kw).compile()


class Ref:
    """cyclic reference sampled by phase in [0,1)"""

    def __init__(self, model):
        r = np.load(os.path.join(HERE, 'ref_gait.npz'))
        self.Q = r['qpos']; self.N = len(self.Q); self.T = float(r['T']); self.speed = float(r['speed'])
        d = mujoco.MjData(model)
        qa = [model.jnt_qposadr[model.joint(j).id] for j in ACT_JOINTS]
        self.q = self.Q[:, qa]                       # (N, 23)
        self.contact = np.stack([r['contact_l'], r['contact_r']], 1).astype(float)  # (N,2,2) heel, fore
        ank, pitch = [], []
        for q in self.Q:
            d.qpos[:] = q
            mujoco.mj_kinematics(model, d)
            pel = d.xpos[model.body('pelvis').id]
            a, p = [], []
            for s in ('left', 'right'):
                b = model.body(f'{s}_ankle_roll_link').id
                a.append(d.xpos[b] - pel)
                xa = d.xmat[b].reshape(3, 3)[:, 0]
                p.append(np.arcsin(-xa[2]))
            ank.append(a); pitch.append(p)
        self.ank = np.array(ank); self.pitch = np.array(pitch)
        self.z = self.Q[:, 2]; self.y = self.Q[:, 1]
        xs = self.Q[:, 0] - self.speed * np.arange(self.N) * self.T / self.N
        self.xosc = xs - xs.mean()
        dQ = np.roll(self.Q, -1, 0) - self.Q
        dQ[-1, 0] = self.Q[0, 0] + self.speed * self.T - self.Q[-1, 0]
        dt = self.T / self.N
        self.qd = dQ[:, [model.jnt_qposadr[model.joint(j).id] for j in ACT_JOINTS]] / dt
        self.vroot = np.stack([dQ[:, 0] / dt, dQ[:, 1] / dt, dQ[:, 2] / dt], 1)
        # standing pose quantities
        d.qpos[:] = model.key_qpos[0]; d.qpos[qa] = DEFAULT_POSE
        mujoco.mj_kinematics(model, d)
        pel = d.xpos[model.body('pelvis').id]
        self.ank_stand = np.array([d.xpos[model.body(f'{s}_ankle_roll_link').id] - pel for s in ('left', 'right')])
        self.z_stand = -min(d.site_xpos[model.site(f'{s}_{p}_site').id][2] - pel[2]
                            for s in ('left', 'right') for p in ('heel', 'meta'))

    def sample(self, ph):
        f = ph * self.N
        i0 = np.floor(f).astype(int) % self.N
        i1 = (i0 + 1) % self.N
        w = (f - np.floor(f))
        def lerp(A):
            sh = (-1,) + (1,) * (A.ndim - 1)
            return A[i0] * (1 - w.reshape(sh)) + A[i1] * w.reshape(sh)
        return dict(q=lerp(self.q), qd=lerp(self.qd), ank=lerp(self.ank), pitch=lerp(self.pitch),
                    contact=self.contact[i0], z=lerp(self.z), y=lerp(self.y), vroot=lerp(self.vroot),
                    xosc=lerp(self.xosc))


def quat_rot_inv(q, v):
    """rotate world vector v into body frame given quat q (w,x,y,z) batched"""
    w, x, y, z = q[:, 0:1], -q[:, 1:2], -q[:, 2:3], -q[:, 3:4]
    u = np.concatenate([x, y, z], 1)
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class K1Batch:
    CTRL_HZ = 50

    def __init__(self, n, stage=1, nthread=2, seed=0, dt=0.005, randomize=False, ep_len=500):
        self.n, self.stage, self.dt = n, stage, dt
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.pushes = True
        self.ep_len = ep_len
        self.base_model = load_model(dt)
        if randomize:
            var = [load_model(dt, friction=self.rng.uniform(0.5, 1.2), mass_scale=self.rng.uniform(0.9, 1.15),
                              kp_scale=self.rng.uniform(0.9, 1.1)) for _ in range(12)]
            self.models = [var[i % len(var)] for i in range(n)]
        else:
            self.models = [self.base_model] * n
        m = self.base_model
        self.m = m
        self.datas = [mujoco.MjData(m) for _ in range(nthread)]
        self.nsub = int(round(1.0 / self.CTRL_HZ / dt))
        self.ref = Ref(m)
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ACT_JOINTS])
        self.dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ACT_JOINTS])
        self.mpadr = np.array([m.jnt_qposadr[m.joint(f'{s}_mp_joint').id] for s in ('left', 'right')])
        self.spec_state = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = mujoco.mj_stateSize(m, self.spec_state)
        self.sadr = {}
        o = 0
        for nm in ['touch', 'ankle_rel', 'foot_x', 'foot_pos', 'base_linvel', 'foot_linvel']:
            pass
        self.sens = {m.sensor(i).name: (m.sensor_adr[i], m.sensor_dim[i]) for i in range(m.nsensor)}
        self.state = np.zeros((n, self.nstate))
        self.warm = np.zeros((n, m.nv))
        self.sdata = np.zeros((n, m.nsensordata))
        self.ph = np.zeros(n); self.alpha = np.zeros(n); self.cmd = np.zeros(n)
        self.t = np.zeros(n, int); self.last_a = np.zeros((n, 12)); self.last_a2 = np.zeros((n, 12))
        self.air = np.zeros((n, 2), bool); self.air_time = np.zeros((n, 2))
        self.switch_t = np.zeros(n, int); self.stop_t = np.zeros(n, int)
        self.x0 = np.zeros(n); self.ref_x = np.zeros(n)
        self.ctrl = np.zeros((n, m.nu))
        self.roll = rollout.Rollout(nthread=nthread)
        self.st_out = np.zeros((n, self.nsub, self.nstate))
        self.sd_out = np.zeros((n, self.nsub, m.nsensordata))
        self.reset(np.arange(n))

    # ---------------- state helpers ----------------
    def qpos(self, ids=slice(None)):
        return self.state[ids, 1:1 + self.m.nq]

    def qvel(self, ids=slice(None)):
        return self.state[ids, 1 + self.m.nq:1 + self.m.nq + self.m.nv]

    def s(self, name):
        a, dd = self.sens[name]
        return self.sdata[:, a:a + dd]

    def reset(self, ids):
        m = self.m
        k = len(ids)
        if k == 0:
            return
        d = self.datas[0]
        walk_start = np.ones(k, bool) if self.stage == 1 else (self.rng.random(k) < 0.5)
        ph = self.rng.random(k)
        r = self.ref.sample(ph)
        for j, i in enumerate(ids):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = m.key_qpos[0]
            if walk_start[j]:
                d.qpos[self.qadr] = r['q'][j]
                d.qpos[0] = 0.0; d.qpos[1] = r['y'][j]; d.qpos[2] = r['z'][j] + 0.005
                d.qvel[self.dadr] = r['qd'][j]
                d.qvel[0:3] = r['vroot'][j]
                d.qpos[self.mpadr] = self.ref.Q[int(ph[j] * self.ref.N) % self.ref.N][self.mpadr] * 0.5
            else:
                d.qpos[self.qadr] = DEFAULT_POSE
                d.qpos[2] = self.ref.z_stand + 0.003
            if self.stage >= 2:
                d.qpos[self.qadr[:12]] += self.rng.normal(0, 0.02, 12)
            mujoco.mj_forward(m, d)
            st = np.zeros(self.nstate)
            mujoco.mj_getState(m, d, st, self.spec_state)
            self.state[i] = st
            self.warm[i] = 0
            self.sdata[i] = d.sensordata
            self.ctrl[i] = d.qpos[self.qadr]
        self.ph[ids] = np.where(walk_start, ph, 0.0)
        self.alpha[ids] = walk_start.astype(float)
        self.cmd[ids] = walk_start.astype(float)
        self.t[ids] = 0
        self.last_a[ids] = 0; self.last_a2[ids] = 0
        self.air[ids] = False; self.air_time[ids] = 0
        # command schedule for stage 2 (standing starts switch to walk at random time, maybe stop later)
        sw = np.where(self.rng.random(k) < 0.15, 10 ** 9, self.rng.integers(50, 200, k))
        self.switch_t[ids] = np.where(walk_start, -1, sw)
        self.stop_t[ids] = np.where(self.rng.random(k) < (0.0 if self.stage == 1 else 0.35),
                                    self.rng.integers(250, 420, k), 10 ** 9)
        self.x0[ids] = 0.0
        self.ref_x[ids] = 0.0

    # ---------------- observations ----------------
    def base_frame(self):
        q = self.qpos()
        quat = q[:, 3:7]
        g = quat_rot_inv(quat, np.tile([0, 0, -1.0], (self.n, 1)))
        return quat, g

    def obs(self):
        quat, g = self.base_frame()
        qv = self.qvel()
        gyro = qv[:, 3:6]  # free joint angular velocity is in local frame
        q = self.qpos()[:, self.qadr[:12]] - DEFAULT_POSE[:12]
        qd = qv[:, self.dadr[:12]]
        tw = 2 * np.pi * self.ph
        actor = np.concatenate([gyro * 0.25, g, self.cmd[:, None], self.alpha[:, None],
                                np.sin(tw)[:, None], np.cos(tw)[:, None], q, qd * 0.05, self.last_a], 1)
        touch = np.concatenate([self.s(f'{s}_{p}_touch') for s in ('left', 'right') for p in ('heel', 'meta', 'toe')], 1)
        priv = np.concatenate([self.s('base_linvel'), self.qpos()[:, 2:3], np.minimum(touch / 200.0, 2.0),
                               self.qpos()[:, self.mpadr], self.s('left_ankle_rel'), self.s('right_ankle_rel')], 1)
        return actor.astype(np.float32), np.concatenate([actor, priv], 1).astype(np.float32)

    # ---------------- step ----------------
    def step(self, a):
        n, m = self.n, self.m
        a = np.clip(a, -4, 4)
        # command schedule
        self.cmd = np.where(self.t == self.switch_t, 1.0, self.cmd)
        self.cmd = np.where(self.t == self.stop_t, 0.0, self.cmd)
        rate = 1.0 / (self.ref.T * self.CTRL_HZ)
        self.alpha = np.clip(self.alpha + np.where(self.cmd > 0.5, rate, -rate), 0, 1)
        moving = self.alpha > 0
        self.ph = np.where(moving, (self.ph + rate) % 1.0, self.ph)
        r = self.ref.sample(self.ph)
        al = self.alpha[:, None]
        qbase = (1 - al) * DEFAULT_POSE[None] + al * r['q']
        tgt = qbase.copy()
        tgt[:, :12] += 0.25 * a
        self.ctrl = tgt
        ctrl = np.ascontiguousarray(np.repeat(tgt[:, None, :], self.nsub, 1))
        self.roll.rollout(self.models, self.datas, np.ascontiguousarray(self.state), ctrl,
                          nstep=self.nsub, skip_checks=True, state=self.st_out, sensordata=self.sd_out)
        st, sd = self.st_out, self.sd_out
        # substep actuator torque log (legs) = servo law, clipped to joint actuator limits
        qs = st[:, :, 1 + self.m.nq:][:, :, :0]  # placeholder for shape
        q_sub = st[:, :, 1:1 + self.m.nq][:, :, self.qadr[:12]]
        qd_sub = st[:, :, 1 + self.m.nq:1 + self.m.nq + self.m.nv][:, :, self.dadr[:12]]
        lim = np.array([47.277 if 'ankle_roll' in j else 96.864 for j in ACT_JOINTS[:12]])
        self.tau_sub = np.clip(KP[:12] * (tgt[:, None, :12] - q_sub) - KD[:12] * qd_sub, -lim, lim)
        self.state = st[:, -1].copy()
        self.sdata = sd[:, -1].copy()
        if self.stage >= 2 and self.pushes:  # random pushes (velocity kicks) ~ every 4 s
            k = np.where(self.rng.random(n) < 1.0 / 200)[0]
            if len(k):
                vo = 1 + self.m.nq
                self.state[k, vo:vo + 2] += self.rng.uniform(-0.35, 0.35, (len(k), 2))
        touch_max = np.zeros((n, 6))
        names = [f'{s}_{p}_touch' for s in ('left', 'right') for p in ('heel', 'meta', 'toe')]
        for k, nm in enumerate(names):
            adr = self.sens[nm][0]
            touch_max[:, k] = sd[:, :, adr].max(1)
        self.t += 1
        self.ref_x += np.where(moving, self.alpha * self.ref.speed / self.CTRL_HZ, 0)
        rew, info = self.reward(a, r, qbase, touch_max)
        quat, g = self.base_frame()
        z = self.qpos()[:, 2]
        fell = (z < 0.45) | (np.linalg.norm(g[:, :2], axis=1) > 0.7)
        bad = np.isnan(self.state).any(1)
        self.state[bad] = 0
        term = fell | bad
        trunc = self.t >= self.ep_len
        rew = np.where(term, rew - 2.0, rew)
        self.last_a2 = self.last_a.copy(); self.last_a = a.copy()
        info['term'] = term; info['trunc'] = trunc
        return rew.astype(np.float32), term, trunc, info

    def reward(self, a, r, qbase, touch):
        n = self.n
        q = self.qpos(); qv = self.qvel()
        al = self.alpha
        ql = q[:, self.qadr[:12]]
        e_q = ((ql - qbase[:, :12]) ** 2).sum(1)
        r_q = np.exp(-2.0 * e_q)
        # hip->ankle vector imitation (ankle position in pelvis frame)
        ank = np.stack([self.s('left_ankle_rel'), self.s('right_ankle_rel')], 1)
        ank_t = (1 - al)[:, None, None] * self.ref.ank_stand[None] + al[:, None, None] * r['ank']
        e_a = ((ank - ank_t) ** 2).sum((1, 2))
        r_ank = np.exp(-40.0 * e_a)
        fx = np.stack([self.s('left_foot_x'), self.s('right_foot_x')], 1)
        pitch = np.arcsin(np.clip(-fx[:, :, 2], -1, 1))
        e_p = ((pitch - al[:, None] * r['pitch']) ** 2).sum(1)
        r_pitch = np.exp(-4.0 * e_p)
        # base velocity (world) & posture
        quat, g = self.base_frame()
        vlin = qv[:, 0:3]
        v_t = al * self.ref.speed
        kv = np.where(al < 0.05, 20.0, 5.0)
        r_vel = np.exp(-kv * ((vlin[:, 0] - v_t) ** 2 + vlin[:, 1] ** 2))
        r_up = np.exp(-20.0 * (g[:, :2] ** 2).sum(1))
        r_yaw = np.exp(-2.0 * qv[:, 5] ** 2)
        z_t = (1 - al) * self.ref.z_stand + al * r['z']
        r_h = np.exp(-200.0 * (q[:, 2] - z_t) ** 2)
        # contact sequence (heel / forefoot) vs human
        heel = touch[:, [0, 3]] > 5.0
        fore = (touch[:, [1, 4]] > 5.0) | (touch[:, [2, 5]] > 5.0)
        hc = r['contact'][:, :, 0] > 0.5; fc = r['contact'][:, :, 1] > 0.5
        match = 0.5 * ((heel == hc).astype(float) + (fore == fc).astype(float)).mean(1)
        flat = 0.5 * (heel.astype(float) + fore.astype(float)).mean(1)
        r_contact = np.where(al > 0.9, match, np.where(al < 0.05, flat, 0.5))
        # heel-first touchdown bonus
        anyc = heel | fore
        touchdown = self.air & anyc
        hs_good = (touchdown & heel & ~fore).sum(1) + 0.5 * (touchdown & heel & fore).sum(1)
        hs_bad = (touchdown & ~heel & fore).sum(1)
        r_hs = (hs_good - hs_bad) * (al > 0.5)
        self.air_time = np.where(anyc, 0, self.air_time + 1.0 / self.CTRL_HZ)
        self.air = (~anyc) & (self.air_time > 0.06) | (self.air & ~anyc)
        # foot slip when in contact
        fv = np.stack([self.s('left_foot_linvel'), self.s('right_foot_linvel')], 1)
        slip = (np.linalg.norm(fv[:, :, :2], axis=2) ** 2 * anyc).sum(1)
        # energy (eco): mechanical power of leg motors
        qdl = qv[:, self.dadr[:12]]
        tau = KP[:12] * (self.ctrl[:, :12] - ql) - KD[:12] * qdl
        power = np.abs(tau * qdl).sum(1)
        a_rate = ((a - self.last_a) ** 2).sum(1)
        a_acc = ((a - 2 * self.last_a + self.last_a2) ** 2).sum(1)
        w_im = 1.0 if self.stage == 1 else 0.7
        rew = (w_im * (0.20 * r_q + 0.25 * r_ank + 0.10 * r_pitch) + 0.20 * r_vel + 0.10 * r_up + 0.05 * r_yaw
               + 0.05 * r_h + 0.15 * r_contact + 0.3 * r_hs
               - 0.1 * slip - (0.0004 if self.stage == 1 else 0.0008) * power - 0.005 * a_rate - 0.002 * a_acc)
        info = dict(r_q=r_q, r_ank=r_ank, r_pitch=r_pitch, r_vel=r_vel, r_up=r_up, r_contact=r_contact,
                    hs_good=hs_good, hs_bad=hs_bad, power=power, vx=vlin[:, 0])
        return rew, info
