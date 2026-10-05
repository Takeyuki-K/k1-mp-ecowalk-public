"""Eco variant: variable-impedance legs (policy sets per-joint Kp and Kd every 20 ms).

Actuation (exact PD law at every 200 Hz physics substep, no extra Python stepping):
  per leg joint 3 actuators, summed and clamped by the joint's actuatorfrcrange (= motor torque limit)
    u1 : gain 1                 -> force = Kp*q_target           (ctrl = Kp*q_target)
    u2 : affine gain [0,-1,0]   -> force = -q  * Kp               (ctrl = Kp)
    u3 : affine gain [0,0,-1]   -> force = -qd * Kd               (ctrl = Kd)
  => tau = Kp (q_target - q) - Kd qd   (same as the K1 motor "MIT mode": Goal Position, MIT P Gain, MIT D Gain)
  Arms / waist keep the fixed ROBOTIS servo gains.

Action (36): [12 position residual | 12 Kp scale | 12 Kd scale]
  Kp = KP_nom * clip(1 + 0.5 a, 0.0, 1.5)    -> 0 = fully relaxed (脱力)
  Kd = KD_nom * clip(1 + 0.5 a, 0.05, 3.0)   -> Kp~0 with Kd>0 = pure damper
Energy reward: electrical model (copper loss tau^2/Km^2 + positive mech. work, no regeneration).
With all scales = 1 this env is identical to k1env.K1Batch (verified in test_eco_equiv.py).
"""
import numpy as np
import mujoco
from k1env import (K1Batch, build_spec, ACT_JOINTS, LEG_JOINTS, KP, KD, DEFAULT_POSE)

KM = np.array([2.2 if 'ankle_roll' in j else 4.0 for j in LEG_JOINTS])     # Nm/sqrt(W) at joint output (assumed)
TLIM = np.array([47.277 if 'ankle_roll' in j else 96.864 for j in LEG_JOINTS])
KP_RANGE = (0.0, 1.5)
KD_RANGE = (0.05, 3.0)


def build_eco_spec(dt=0.005, friction=1.0, mass_scale=1.0):
    spec = build_spec(dt, friction=friction, mass_scale=mass_scale, kp_scale=1.0, servo=True)
    for act in spec.actuators:
        if act.target in LEG_JOINTS:          # direct-torque channel
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.gainprm[0] = 1.0
            act.biastype = mujoco.mjtBias.mjBIAS_NONE
            act.biasprm[:3] = 0
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
    for kind, prm in (('kp', [0, -1, 0]), ('kd', [0, 0, -1])):
        for j in LEG_JOINTS:
            a = spec.add_actuator(name=f'{j}_{kind}', target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.gaintype = mujoco.mjtGain.mjGAIN_AFFINE
            g = np.zeros(10); g[:3] = prm
            a.gainprm = g
            a.biastype = mujoco.mjtBias.mjBIAS_NONE
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec


class K1EcoBatch(K1Batch):
    nact = 36

    def __init__(self, n, stage=2, nthread=2, seed=0, dt=0.005, randomize=False, ep_len=500, w_energy=0.0015):
        self.w_energy = w_energy
        self._eco_init = True
        super().__init__(n, stage=stage, nthread=nthread, seed=seed, dt=dt, randomize=False, ep_len=ep_len)
        # replace models with the eco actuation
        rng = np.random.default_rng(seed + 1)
        self.base_model = build_eco_spec(dt).compile()
        self.m = self.base_model
        if randomize:
            var = [build_eco_spec(dt, friction=rng.uniform(0.5, 1.2), mass_scale=rng.uniform(0.9, 1.15)).compile()
                   for _ in range(12)]
            self.models = [var[i % 12] for i in range(n)]
            self.kp_dr = rng.uniform(0.9, 1.1, (n, 1))      # motor gain / torque-constant error
        else:
            self.models = [self.base_model] * n
            self.kp_dr = np.ones((n, 1))
        self.randomize = randomize
        self.datas = [mujoco.MjData(self.m) for _ in range(nthread)]
        self.ctrl = np.zeros((n, len(ACT_JOINTS)))
        self.sd_out = np.zeros((n, self.nsub, self.m.nsensordata))
        self.kp_scale = np.ones((n, 12)); self.kd_scale = np.ones((n, 12))
        self.P_elec = np.zeros(n)
        self.reset(np.arange(n))

    def reset(self, ids):
        super().reset(ids)
        if not hasattr(self, 'kp_scale'):
            self.last_a = np.zeros((self.n, 36)); self.last_a2 = np.zeros((self.n, 36))
            return
        self.last_a[ids] = 0; self.last_a2[ids] = 0
        self.kp_scale[ids] = 1; self.kd_scale[ids] = 1

    def step(self, a):
        n, m = self.n, self.m
        a = np.clip(a, -4, 4)
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
        tgt[:, :12] += 0.25 * a[:, :12]
        self.kp_scale = np.clip(1 + 0.5 * a[:, 12:24], *KP_RANGE)
        self.kd_scale = np.clip(1 + 0.5 * a[:, 24:36], *KD_RANGE)
        kp = KP[:12] * self.kp_scale * self.kp_dr
        kd = KD[:12] * self.kd_scale * self.kp_dr
        ctrl = tgt.copy()                      # arms/waist: servo position targets
        ctrl[:, :12] = kp * tgt[:, :12]
        ctrl = np.concatenate([ctrl, kp, kd], 1)
        self.ctrl = tgt                        # position targets (used by base reward bookkeeping)
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
                self.state[k, vo:vo + 2] += self.rng.uniform(-0.35, 0.35, (len(k), 2))
        touch_max = np.zeros((n, 6))
        names = [f'{s}_{p}_touch' for s in ('left', 'right') for p in ('heel', 'meta', 'toe')]
        for k, nm in enumerate(names):
            touch_max[:, k] = sd[:, :, self.sens[nm][0]].max(1)
        self.t += 1
        rew, info = self.reward(a, r, qbase, touch_max)
        quat, g = self.base_frame()
        fell = (self.qpos()[:, 2] < 0.45) | (np.linalg.norm(g[:, :2], axis=1) > 0.7)
        bad = np.isnan(self.state).any(1)
        self.state[bad] = 0
        term = fell | bad
        trunc = self.t >= self.ep_len
        rew = np.where(term, rew - 2.0, rew)
        self.last_a2 = self.last_a.copy(); self.last_a = a.copy()
        info['term'] = term; info['trunc'] = trunc
        return rew.astype(np.float32), term, trunc, info

    def reward(self, a, r, qbase, touch):
        rew, info = super().reward(a, r, qbase, touch)
        # remove the base mechanical-power term, add the electrical energy term
        rew = rew + (0.0004 if self.stage == 1 else 0.0008) * info['power']
        rew = rew - self.w_energy * self.P_elec
        info['P_elec'] = self.P_elec
        info['P_neg'] = self.P_neg
        info['kp_mean'] = self.kp_scale.mean(1)
        info['kd_mean'] = self.kd_scale.mean(1)
        return rew, info
