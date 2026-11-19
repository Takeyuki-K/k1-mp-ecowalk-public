# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Running env: K1 + passive MP toes + variable impedance (same actuation as the eco walker),
imitating the CMU-derived running reference (ref_run.npz).

Reward priorities (user instruction): human imitation and stability first, impact absorption next,
energy only a small term ("eco is a result"). Speed tracking is loose.
  imitation : joint angles, hip->ankle vector, foot pitch, pelvis height, contact pattern incl. flight
  stability : torso upright, low torso pitch/roll rate, heel-first touchdown
  impact    : penalty on per-foot peak contact force above 2.5 m g (human running GRF peak ~2.5 BW)
  energy    : 0.0003 * P_elec (eco walker used 0.0015)
"""
import numpy as np
import mujoco
from k1env import K1Batch, DEFAULT_POSE, KP, KD, ACT_JOINTS
from k1env_eco import K1EcoBatch, build_eco_spec, KP_RANGE, KD_RANGE, TLIM, KM

MG = 35.706 * 9.81


ASSIST_GEAR = [[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]]


def build_run_spec(dt=0.005, **kw):
    """eco actuation + 5 'assist' motors on the pelvis (vertical / fore-aft / lateral force, roll / pitch torque).
    They are a training curriculum only (like a harness): scaled by env.assist, which goes to 0; evaluation
    and videos always run with assist = 0."""
    spec = build_eco_spec(dt, **kw)
    spec.body('pelvis').add_site(name='assist', pos=[0, 0, 0], size=[0.01, 0.01, 0.01], group=5)
    for k, gr in enumerate(ASSIST_GEAR):
        a = spec.add_actuator(name=f'assist{k}', target='assist', trntype=mujoco.mjtTrn.mjTRN_SITE)
        g = np.zeros(6); g[:] = gr
        a.gear = g
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec


class K1RunBatch(K1EcoBatch):
    def __init__(self, n, randomize=False, seed=0, w_energy=0.0003, w_impact=0.3, stage=2, nthread=2,
                 dt=0.005, ep_len=500):
        self.w_impact = w_impact
        self.w_energy = w_energy
        self.assist = 0.0
        self.kv = 2.0
        self.push_mag = 0.3
        self.w_flight = 0.0
        K1Batch.__init__(self, n, stage=stage, nthread=nthread, seed=seed, dt=dt, randomize=False, ep_len=ep_len)
        rng = np.random.default_rng(seed + 1)
        self.base_model = build_run_spec(dt).compile()
        self.m = self.base_model
        if randomize:
            var = [build_run_spec(dt, friction=rng.uniform(0.5, 1.2), mass_scale=rng.uniform(0.9, 1.15)).compile()
                   for _ in range(12)]
            self.models = [var[i % 12] for i in range(n)]
            self.kp_dr = rng.uniform(0.9, 1.1, (n, 1))
        else:
            self.models = [self.base_model] * n
            self.kp_dr = np.ones((n, 1))
        self.randomize = randomize
        self.datas = [mujoco.MjData(self.m) for _ in range(nthread)]
        self.ctrl = np.zeros((n, len(ACT_JOINTS)))
        self.sd_out = np.zeros((n, self.nsub, self.m.nsensordata))
        self.kp_scale = np.ones((n, 12)); self.kd_scale = np.ones((n, 12))
        self.P_elec = np.zeros(n); self.P_neg = np.zeros(n)
        self.reset(np.arange(n))

    def assist_ctrl(self, r):
        n = self.n
        if self.assist <= 0:
            return np.zeros((n, 5))
        q = self.qpos(); qv = self.qvel()
        quat, g = self.base_frame()
        gyro = qv[:, 3:6]
        fz = np.clip(1500 * (r['z'] - q[:, 2]) - 100 * qv[:, 2], -0.3 * MG, 0.6 * MG)
        fx = np.clip(60 * (self.alpha * self.ref.speed - qv[:, 0]), -60, 60)
        fy = np.clip(-80 * qv[:, 1], -50, 50)
        tx = np.clip(200 * g[:, 1] - 20 * gyro[:, 0], -50, 50)
        ty = np.clip(-200 * (g[:, 0] - np.sin(self.ref.lean)) - 20 * gyro[:, 1], -50, 50)
        return self.assist * np.stack([fz, fx, fy, tx, ty], 1)

    def step(self, a):
        n, m = self.n, self.m
        a = np.clip(a, -4, 4)
        rate = 1.0 / (self.ref.T * self.CTRL_HZ)
        self.alpha[:] = 1.0
        self.ph = (self.ph + rate) % 1.0
        r = self.ref.sample(self.ph)
        qbase = r['q'].copy()
        tgt = qbase.copy()
        tgt[:, :12] += 0.25 * a[:, :12]
        self.kp_scale = np.clip(1 + 0.5 * a[:, 12:24], *KP_RANGE)
        self.kd_scale = np.clip(1 + 0.5 * a[:, 24:36], *KD_RANGE)
        kp = KP[:12] * self.kp_scale * self.kp_dr
        kd = KD[:12] * self.kd_scale * self.kp_dr
        ctrl = tgt.copy()
        ctrl[:, :12] = kp * tgt[:, :12]
        ctrl = np.concatenate([ctrl, kp, kd, self.assist_ctrl(r)], 1)
        self.ctrl = tgt
        self.roll.rollout(self.models, self.datas, np.ascontiguousarray(self.state),
                          np.ascontiguousarray(np.repeat(ctrl[:, None, :], self.nsub, 1)),
                          nstep=self.nsub, skip_checks=True, state=self.st_out, sensordata=self.sd_out)
        st, sd = self.st_out, self.sd_out
        q_sub = st[:, :, 1:1 + m.nq][:, :, self.qadr[:12]]
        qd_sub = st[:, :, 1 + m.nq:1 + m.nq + m.nv][:, :, self.dadr[:12]]
        self.tau_sub = np.clip(kp[:, None] * (tgt[:, None, :12] - q_sub) - kd[:, None] * qd_sub, -TLIM, TLIM)
        p_mech = self.tau_sub * qd_sub
        self.P_elec = ((self.tau_sub ** 2 / KM ** 2).sum(2) + np.clip(p_mech, 0, None).sum(2)).mean(1)
        self.P_neg = (-np.clip(p_mech, None, 0)).sum(2).mean(1)
        self.state = st[:, -1].copy()
        self.sdata = sd[:, -1].copy()
        if self.stage >= 2 and self.pushes:
            k = np.where(self.rng.random(n) < 1.0 / 200)[0]
            if len(k):
                vo = 1 + m.nq
                self.state[k, vo:vo + 2] += self.rng.uniform(-self.push_mag, self.push_mag, (len(k), 2))
        touch_max = np.zeros((n, 6))
        names = [f'{s}_{p}_touch' for s in ('left', 'right') for p in ('heel', 'meta', 'toe')]
        for k, nm in enumerate(names):
            touch_max[:, k] = sd[:, :, self.sens[nm][0]].max(1)
        self.t += 1
        rew, info = self.reward(a, r, qbase, touch_max)
        quat, g = self.base_frame()
        fell = (self.qpos()[:, 2] < 0.5) | (np.linalg.norm(g[:, :2], axis=1) > 0.6)
        bad = np.isnan(self.state).any(1)
        self.state[bad] = 0
        term = fell | bad
        trunc = self.t >= self.ep_len
        rew = np.where(term, rew - 2.0, rew)
        self.last_a2 = self.last_a.copy(); self.last_a = a.copy()
        info['term'] = term; info['trunc'] = trunc
        return rew.astype(np.float32), term, trunc, info

    def reset(self, ids):
        s = self.stage
        self.stage = 1                       # base reset with stage 1 = always start inside the running cycle
        super().reset(ids)
        self.stage = s
        if len(ids):
            L = self.ref.lean
            self.state[np.ix_(ids, np.arange(4, 8))] = [np.cos(L / 2), 0, np.sin(L / 2), 0]
        if len(ids) and s >= 2:
            m = self.m
            o = 1 + m.nq
            self.state[np.ix_(ids, 1 + self.qadr[:12])] += self.rng.normal(0, 0.02, (len(ids), 12))
            self.state[np.ix_(ids, np.arange(o, o + 2))] += self.rng.normal(0, 0.1, (len(ids), 2))

    def reward(self, a, r, qbase, touch):
        q = self.qpos(); qv = self.qvel()
        al = self.alpha
        ql = q[:, self.qadr[:12]]
        r_q = np.exp(-1.5 * ((ql - qbase[:, :12]) ** 2).sum(1))
        ank = np.stack([self.s('left_ankle_rel'), self.s('right_ankle_rel')], 1)
        ank_t = (1 - al)[:, None, None] * self.ref.ank_stand[None] + al[:, None, None] * r['ank']
        r_ank = np.exp(-20.0 * ((ank - ank_t) ** 2).sum((1, 2)))
        fx = np.stack([self.s('left_foot_x'), self.s('right_foot_x')], 1)
        pitch = np.arcsin(np.clip(-fx[:, :, 2], -1, 1))
        r_pitch = np.exp(-4.0 * ((pitch - al[:, None] * r['pitch']) ** 2).sum(1))
        quat, g = self.base_frame()
        vlin = qv[:, 0:3]
        r_vel = np.exp(-self.kv * ((vlin[:, 0] - al * self.ref.speed) ** 2 + vlin[:, 1] ** 2))
        r_vz = np.exp(-10.0 * (vlin[:, 2] - r['vroot'][:, 2]) ** 2)
        r_up = np.exp(-20.0 * ((g[:, 0] - np.sin(self.ref.lean)) ** 2 + g[:, 1] ** 2))     # target: reference trunk lean
        gyro = qv[:, 3:6]
        r_rate = np.exp(-0.5 * (gyro[:, 0] ** 2 + gyro[:, 1] ** 2) - 1.0 * gyro[:, 2] ** 2)
        r_h = np.exp(-200.0 * (q[:, 2] - r['z']) ** 2)
        heel = touch[:, [0, 3]] > 5.0
        fore = (touch[:, [1, 4]] > 5.0) | (touch[:, [2, 5]] > 5.0)
        hc = r['contact'][:, :, 0] > 0.5; fc = r['contact'][:, :, 1] > 0.5
        r_contact = 0.5 * ((heel == hc).astype(float) + (fore == fc).astype(float)).mean(1)
        anyc = heel | fore
        flight = ~anyc.any(1)
        flight_ref = ~(hc | fc).any(1)
        touchdown = self.air & anyc
        hs_good = (touchdown & heel & ~fore).sum(1) + 0.5 * (touchdown & heel & fore).sum(1)
        hs_bad = (touchdown & ~heel & fore).sum(1)
        r_hs = hs_good - hs_bad
        self.air_time = np.where(anyc, 0, self.air_time + 1.0 / self.CTRL_HZ)
        self.air = (~anyc) & (self.air_time > 0.06) | (self.air & ~anyc)
        fv = np.stack([self.s('left_foot_linvel'), self.s('right_foot_linvel')], 1)
        slip = (np.linalg.norm(fv[:, :, :2], axis=2) ** 2 * anyc).sum(1)
        f_foot = np.stack([touch[:, :3].sum(1), touch[:, 3:].sum(1)], 1)
        impact = f_foot.max(1) / MG
        p_impact = np.clip(impact - 2.5, 0, None)
        a_rate = ((a - self.last_a) ** 2).sum(1)
        a_acc = ((a - 2 * self.last_a + self.last_a2) ** 2).sum(1)
        rew = (0.20 * r_q + 0.25 * r_ank + 0.10 * r_pitch + 0.05 * r_h + 0.15 * r_contact
               + 0.10 * r_vel + 0.15 * r_up + 0.10 * r_rate + 0.3 * r_hs
               + self.w_flight * (0.10 * r_vz + 0.3 * (flight & flight_ref) - 0.1 * (flight & ~flight_ref))
               - self.w_impact * p_impact - 0.1 * slip - self.w_energy * self.P_elec
               - 0.005 * a_rate - 0.002 * a_acc)
        info = dict(r_q=r_q, r_ank=r_ank, r_pitch=r_pitch, r_vel=r_vel, r_up=r_up, r_contact=r_contact,
                    r_rate=r_rate, r_vz=r_vz, hs_good=hs_good, hs_bad=hs_bad, vx=vlin[:, 0], impact=impact,
                    flight=flight.astype(float), flight_ref=flight_ref.astype(float),
                    P_elec=self.P_elec, P_neg=self.P_neg, kp_mean=self.kp_scale.mean(1),
                    kd_mean=self.kd_scale.mean(1), torso_tilt=np.linalg.norm(g[:, :2], axis=1))
        return rew, info
