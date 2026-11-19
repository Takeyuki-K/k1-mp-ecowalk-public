# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5 running env: mid/forefoot landing, motor torque-speed limit, stand -> run and walk -> run entries.

Changes from k1env_sprint (v4):
  * reference: CMU 09_04 with its own mid/forefoot landing (retarget_sprint.py, STRIKE='fore')
  * touchdown reward: forefoot-first +1, midfoot (fore+heel together) +0.7, heel-first -1.5
  * MOTOR MODEL (physical, not only a penalty): leg joints follow the K1 URDF limits (96.9 Nm, 11.5 rad/s;
    ankle roll 47.3 Nm, 20.9 rad/s). Motoring torque (tau * omega > 0) is available in full up to 60 % of the
    speed limit and falls linearly to 0 at the limit; braking torque stays available. Implemented every 5 ms
    physics step by scaling that joint's PD (Kp, Kd and target term together), so the implicit damping of the
    PD actuators is kept.
  * episode starts: 50 % running at random speed / phase, 25 % standing with a run command (stand -> run),
    25 % from recorded fast-walking states at right heel strike (walk -> run hand-over, speed jumps to 2.0 m/s)
"""
import os
import numpy as np
import mujoco
from k1env import ACT_JOINTS, DEFAULT_POSE, KP, KD
from k1env_eco import KP_RANGE, KD_RANGE, TLIM, KM
from k1env_sprint import K1SprintBatch, MG, QD_LIM
from motor import motor_limit

HERE = os.path.dirname(os.path.abspath(__file__))
KNEE_FRAC = 0.6          # full torque up to 60 % of the speed limit (assumption, see REPORT_GAIT.md)
V_RUN_MIN = 1.8          # lowest speed of the running library


def motor_scale(tau, qd):
    """factor (0..1) applied to the PD torque so that motoring torque respects the torque-speed curve"""
    w = np.abs(qd)
    lim = TLIM * np.clip((QD_LIM - w) / ((1 - KNEE_FRAC) * QD_LIM), 0.0, 1.0)
    motoring = tau * qd > 0
    lim = np.where(motoring, lim, TLIM)
    return np.where(np.abs(tau) > lim, lim / np.maximum(np.abs(tau), 1e-6), 1.0)


def yaw_of(q):
    w, x, y, z = q[3], q[4], q[5], q[6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class K1Run2Batch(K1SprintBatch):
    P_STAND = float(os.environ.get('P_STAND', '0.25'))      # robustness stage: 0.4
    P_WALK = 0.25

    def __init__(self, n, **kw):
        self.motor_limit = True
        self.entry = np.zeros(n, int)                 # 0 running, 1 standing start, 2 from walking
        bank = os.path.join(HERE, 'walk_bank.npz')
        self.bank = np.load(bank) if os.path.exists(bank) else None
        self.tau_sat = np.zeros(n)
        super().__init__(n, **kw)

    # ---------------- episode starts ----------------
    def reset(self, ids):
        if not hasattr(self, 'kp_scale') or not hasattr(self, 'lib'):
            return super().reset(ids)
        k = len(ids)
        if k == 0:
            return
        super().reset(ids)                            # running start (RSI) for all, then overwrite some
        u = self.rng.random(k)
        m = self.m; d = self.datas[0]
        for j, i in enumerate(ids):
            if u[j] < self.P_STAND:                   # standing start with a run command
                mujoco.mj_resetData(m, d)
                d.qpos[:] = m.key_qpos[0]; d.qpos[self.qadr] = DEFAULT_POSE
                d.qpos[2] = self.ref.z_stand + 0.003
                if self.stage >= 2:
                    d.qpos[self.qadr[:12]] += self.rng.normal(0, 0.02, 12)
                self._set(i, d)
                self.entry[i] = 1
                self.alpha[i] = 0.0; self.ph[i] = 0.0
                self.v_ref[i] = 0.0; self.v_cmd[i] = self.rng.uniform(2.0, max(2.2, min(self.v_hi, 3.5)))
            elif u[j] < self.P_STAND + self.P_WALK and self.bank is not None:   # walk -> run hand-over
                b = self.rng.integers(len(self.bank['qpos']))
                q = self.bank['qpos'][b].copy(); v = self.bank['qvel'][b].copy()
                yw = yaw_of(q)                        # rotate to heading 0
                c, s = np.cos(-yw / 2), np.sin(-yw / 2)
                qz = np.array([c, 0, 0, s]); qq = q[3:7]
                q[3:7] = [qz[0] * qq[0] - qz[3] * qq[3], qz[0] * qq[1] - qz[3] * qq[2],
                          qz[0] * qq[2] + qz[3] * qq[1], qz[0] * qq[3] + qz[3] * qq[0]]
                cy, sy = np.cos(-yw), np.sin(-yw)
                v[0:2] = [cy * v[0] - sy * v[1], sy * v[0] + cy * v[1]]
                mujoco.mj_resetData(m, d)
                d.qpos[:] = q; d.qvel[:] = v
                self._set(i, d)
                self.entry[i] = 2
                self.alpha[i] = 1.0; self.ph[i] = 0.0     # running cycle starts at right touchdown
                self.v_ref[i] = 2.0; self.v_cmd[i] = self.rng.uniform(2.0, max(2.2, self.v_hi))
            else:
                self.entry[i] = 0

    def _set(self, i, d):
        mujoco.mj_forward(self.m, d)
        st = np.zeros(self.nstate)
        mujoco.mj_getState(self.m, d, st, self.spec_state)
        self.state[i] = st; self.sdata[i] = d.sensordata
        self.last_a[i] = 0; self.last_a2[i] = 0

    def _sample_v(self, k):
        v = super()._sample_v(k)
        return np.maximum(v, V_RUN_MIN)          # v5: down to 1.8 m/s (hand-over speed to walking)

    # ---------------- step ----------------
    def step(self, a):
        n, m = self.n, self.m
        a = np.clip(a, -4, 4)
        self._schedule()
        dv = np.clip(self.v_cmd - self.v_ref, -self.ACC / self.CTRL_HZ, self.ACC / self.CTRL_HZ)
        self.v_ref = self.v_ref + dv
        v_lib = np.maximum(self.v_ref, V_RUN_MIN)
        rate = 1.0 / (self.lib.T(v_lib) * self.CTRL_HZ)
        if hasattr(self, 'cad'):
            rate = rate * self.cad
        self.alpha = np.clip(self.alpha + rate, 0, 1)            # standing start: blend into running over 1 cycle
        self.ph = (self.ph + rate) % 1.0
        r = self.lib.sample(self.ph, v_lib)
        self._r = r
        al = self.alpha[:, None]
        qbase = (1 - al) * DEFAULT_POSE[None] + al * r['q']
        tgt = qbase.copy()
        tgt[:, :12] += 0.25 * a[:, :12]
        self.kp_scale = np.clip(1 + 0.5 * a[:, 12:24], *KP_RANGE)
        self.kd_scale = np.clip(1 + 0.5 * a[:, 24:36], *KD_RANGE)
        kp = KP[:12] * self.kp_scale * self.kp_dr
        kd = KD[:12] * self.kd_scale * self.kp_dr
        self.ctrl = tgt
        assist = self.assist_ctrl(r)
        st = self.state
        nq, nv = m.nq, m.nv
        if not hasattr(self, 'st1'):
            self.st1 = np.zeros((n, 1, self.nstate)); self.sd1 = np.zeros((n, 1, m.nsensordata))
        sd_all, qd_all, tau_all = [], [], []
        sat = np.zeros(n)
        for sub in range(self.nsub):
            q = st[:, 1:1 + nq][:, self.qadr[:12]]
            qd = st[:, 1 + nq:1 + nq + nv][:, self.dadr[:12]]
            tau = kp * (tgt[:, :12] - q) - kd * qd
            if self.motor_limit:
                f, tdir, direct = motor_limit(tau, qd)
            else:
                f, tdir, direct = np.ones_like(tau), tau, np.zeros(tau.shape, bool)
            sat += ((f < 0.999) | direct).any(1)
            kpe = np.where(direct, 0.0, kp * f); kde = np.where(direct, 0.0, kd * f)
            ctrl = tgt.copy(); ctrl[:, :12] = np.where(direct, tdir, kpe * tgt[:, :12])
            ctrl = np.concatenate([ctrl, kpe, kde, assist], 1)
            self.roll.rollout(self.models, self.datas, np.ascontiguousarray(st),
                              np.ascontiguousarray(ctrl[:, None, :]), nstep=1, skip_checks=True,
                              state=self.st1, sensordata=self.sd1)
            st = self.st1[:, 0].copy()
            sd_all.append(self.sd1[:, 0].copy())
            q1 = st[:, 1:1 + nq][:, self.qadr[:12]]; qd1 = st[:, 1 + nq:1 + nq + nv][:, self.dadr[:12]]
            qd_all.append(qd1)
            tau_all.append(np.clip(np.where(direct, tdir, kpe * (tgt[:, :12] - q1) - kde * qd1), -TLIM, TLIM))
        sd = np.stack(sd_all, 1); qd_sub = np.stack(qd_all, 1); self.tau_sub = np.stack(tau_all, 1)
        self.tau_sat = sat / self.nsub
        p_mech = self.tau_sub * qd_sub
        self.P_elec = ((self.tau_sub ** 2 / KM ** 2).sum(2) + np.clip(p_mech, 0, None).sum(2)).mean(1)
        self.P_neg = (-np.clip(p_mech, None, 0)).sum(2).mean(1)
        over = np.clip(np.abs(qd_sub) - QD_LIM, 0, None)
        self.qd_over = (over ** 2).sum(2).mean(1)
        self.qd_viol = (over > 0).any(2).mean(1)
        self.qd_peak = np.abs(qd_sub).max((1, 2))
        self.state = st; self.sdata = sd[:, -1].copy()
        self.sd_out_all = sd
        if self.stage >= 2 and self.pushes:
            kk = np.where(self.rng.random(n) < 1.0 / 200)[0]
            if len(kk):
                vo = 1 + nq
                self.state[kk, vo:vo + 2] += self.rng.uniform(-self.push_mag, self.push_mag, (len(kk), 2))
        touch = np.zeros((n, 6))
        for k, nm in enumerate([f'{s}_{p}_touch' for s in ('left', 'right') for p in ('heel', 'meta', 'toe')]):
            touch[:, k] = sd[:, :, self.sens[nm][0]].max(1)
        self.t += 1
        rew, info = self.reward(a, r, qbase, touch)
        quat, g = self.base_frame()
        fell = (self.qpos()[:, 2] < 0.5) | (np.linalg.norm(g[:, :2], axis=1) > 0.6)
        bad = np.isnan(self.state).any(1)
        self.state[bad] = 0
        term = fell | bad
        trunc = self.t >= self.ep_len
        rew = np.where(term, rew - 2.0, rew)
        self.last_a2 = self.last_a.copy(); self.last_a = a.copy()
        info['term'] = term; info['trunc'] = trunc
        info['tau_sat'] = self.tau_sat
        info['cad'] = getattr(self, 'cad', np.ones(n))
        return rew.astype(np.float32), term, trunc, info

    def reward(self, a, r, qbase, touch):
        """sprint reward with (i) mid/forefoot touchdown instead of heel-first, (ii) alpha-blended imitation for
        standing starts"""
        q = self.qpos(); qv = self.qvel(); al = self.alpha
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
        rel = (vlin[:, 0] - self.v_ref) / np.maximum(self.v_ref, 1.0)
        r_vel = np.exp(-self.kvel * rel ** 2 - 4.0 * vlin[:, 1] ** 2)
        r_vz = np.exp(-10.0 * (vlin[:, 2] - al * r['vroot'][:, 2]) ** 2)
        lean = al * np.sin(self.lib.lean)
        r_up = np.exp(-20.0 * ((g[:, 0] - lean) ** 2 + g[:, 1] ** 2))
        gyro = qv[:, 3:6]
        r_rate = np.exp(-0.5 * (gyro[:, 0] ** 2 + gyro[:, 1] ** 2) - 1.0 * gyro[:, 2] ** 2)
        z_t = (1 - al) * self.ref.z_stand + al * r['z']
        r_h = np.exp(-200.0 * (q[:, 2] - z_t) ** 2)
        heel = touch[:, [0, 3]] > 5.0
        fore = (touch[:, [1, 4]] > 5.0) | (touch[:, [2, 5]] > 5.0)
        hc = r['contact'][:, :, 0] > 0.5; fc = r['contact'][:, :, 1] > 0.5
        r_contact = np.where(al > 0.9, 0.5 * ((heel == hc).astype(float) + (fore == fc).astype(float)).mean(1), 0.5)
        anyc = heel | fore
        flight = ~anyc.any(1); flight_ref = ~(hc | fc).any(1) & (al > 0.9)
        touchdown = self.air & anyc
        fore_first = (touchdown & fore & ~heel).sum(1)
        mid = (touchdown & fore & heel).sum(1)
        heel_first = (touchdown & heel & ~fore).sum(1)
        r_strike = fore_first + 0.7 * mid - 1.5 * heel_first
        self.air_time = np.where(anyc, 0, self.air_time + 1.0 / self.CTRL_HZ)
        self.air = (~anyc) & (self.air_time > 0.06) | (self.air & ~anyc)
        fv = np.stack([self.s('left_foot_linvel'), self.s('right_foot_linvel')], 1)
        slip = (np.linalg.norm(fv[:, :, :2], axis=2) ** 2 * anyc).sum(1)
        f_foot = np.stack([touch[:, :3].sum(1), touch[:, 3:].sum(1)], 1)
        impact = f_foot.max(1) / MG
        thr = 2.5 + 0.2 * np.clip(self.v_ref - 2.0, 0, None)
        p_impact = np.clip(impact - thr, 0, None)
        a_rate = ((a - self.last_a) ** 2).sum(1)
        a_acc = ((a - 2 * self.last_a + self.last_a2) ** 2).sum(1)
        rew = (0.15 * r_q + 0.15 * r_ank + 0.05 * r_pitch + 0.05 * r_h + 0.10 * r_contact
               + 0.40 * r_vel + 0.15 * r_up + 0.10 * r_rate + 0.3 * r_strike
               + 0.10 * r_vz + 0.3 * (flight & flight_ref) - 0.1 * (flight & ~flight_ref & (al > 0.9))
               - self.w_impact * p_impact - 0.1 * slip - self.w_qd * self.qd_over
               - self.w_energy * self.P_elec - 0.005 * a_rate - 0.002 * a_acc)
        info = dict(r_q=r_q, r_ank=r_ank, r_pitch=r_pitch, r_vel=r_vel, r_up=r_up, r_rate=r_rate, r_vz=r_vz,
                    r_contact=r_contact, hs_good=fore_first + mid, hs_bad=heel_first, vx=vlin[:, 0],
                    v_ref=self.v_ref.copy(), verr_rel=np.abs(rel), impact=impact, flight=flight.astype(float),
                    P_elec=self.P_elec, kp_mean=self.kp_scale.mean(1), kd_mean=self.kd_scale.mean(1),
                    torso_tilt=np.linalg.norm(g[:, :2], axis=1), qd_viol=self.qd_viol, qd_peak=self.qd_peak,
                    fore_first=fore_first, mid=mid, heel_first=heel_first, entry=self.entry.copy())
        return rew, info


class K1Run3Batch(K1Run2Batch):
    """+ cadence-choice action (37th): the policy scales the gait-phase rate by 0.7-1.3 (step frequency vs step
    length trade-off, like the gait-choice action of the walking policy). Added because with the motor speed limit
    the fixed human cadence capped the top speed at ~4.7 m/s (REPORT_GAIT.md G7)."""
    nact = 37

    def __init__(self, n, **kw):
        super().__init__(n, **kw)
        self.last_a = np.zeros((n, 37)); self.last_a2 = np.zeros((n, 37))
        self.cad = np.ones(n)

    def step(self, a):
        a = np.asarray(a, float)
        self.cad = np.clip(1.0 + 0.15 * np.clip(a[:, 36], -2, 2), 0.7, 1.3)
        return super().step(a)
