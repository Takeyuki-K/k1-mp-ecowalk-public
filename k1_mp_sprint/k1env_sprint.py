# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Fast-running env (speed command 1.6 -> 5.0 m/s): K1 + passive MP toes + variable impedance.

Reference: running speed library from CMU 09_04 (retarget_sprint.py), blended at the same gait phase.
User instruction: running = stability and speed first; power is only measured (w_energy = 0).
Imitation is kept as a style prior (lower weight than for the jog), velocity tracking is a main objective.
Motor speed: the K1 URDF lists 11.5 rad/s for the leg joints (20.9 rad/s ankle roll). MuJoCo cannot vary the torque
limit per robot inside a batched rollout, so joint speed above the limit is penalised and reported (see REPORT_SPRINT.md).

Observation: run observation + [v_cmd/5, v_ref/5] inserted after alpha (index 8).
"""
import os
import numpy as np
import mujoco
from k1env import ACT_JOINTS, DEFAULT_POSE, KP, KD
from k1env_eco import KP_RANGE, KD_RANGE, TLIM, KM
from k1env_run import K1RunBatch, MG

HERE = os.path.dirname(os.path.abspath(__file__))
QD_LIM = np.array([20.9 if j.endswith('ankle_roll_joint') else 11.5 for j in ACT_JOINTS[:12]])


class RunLib:
    def __init__(self, model, path=os.path.join(HERE, 'ref_sprint_lib.npz')):
        L = np.load(path)
        self.speeds = L['speeds'].astype(float); self.Tc = L['T'].astype(float); self.lean = float(L['lean'])
        Qs = L['qpos']; S, M, _ = Qs.shape
        self.S, self.M = S, M
        d = mujoco.MjData(model)
        qa = [model.jnt_qposadr[model.joint(j).id] for j in ACT_JOINTS]
        self.Qfull = Qs; self.q = Qs[:, :, qa]
        self.z = Qs[:, :, 2]; self.y = Qs[:, :, 1]
        self.contact = np.stack([L['cl'], L['cr']], 2).astype(float)
        self.mpadr = [model.jnt_qposadr[model.joint(f'{s}_mp_joint').id] for s in ('left', 'right')]
        ank = np.zeros((S, M, 2, 3)); pitch = np.zeros((S, M, 2))
        pid = model.body('pelvis').id
        for si in range(S):
            for i in range(M):
                d.qpos[:] = Qs[si, i]
                mujoco.mj_kinematics(model, d)
                R = d.xmat[pid].reshape(3, 3)
                for k, s in enumerate(('left', 'right')):
                    b = model.body(f'{s}_ankle_roll_link').id
                    ank[si, i, k] = R.T @ (d.xpos[b] - d.xpos[pid])
                    pitch[si, i, k] = np.arcsin(-d.xmat[b].reshape(3, 3)[2, 0])
        self.ank, self.pitch = ank, pitch
        dt = (self.Tc / M)[:, None, None]
        dQ = (np.roll(Qs, -1, 1) - Qs)
        dQ[:, -1, 0] = Qs[:, 0, 0] + self.speeds * self.Tc - Qs[:, -1, 0]
        self.qd = dQ[:, :, qa] / dt
        self.vroot = dQ[:, :, 0:3] / dt
        # standing pose quantities (unused: running env starts inside the cycle)
        self.ank_stand = ank[0, 0]

    def T(self, v):
        return np.interp(v, self.speeds, self.Tc)

    def sample(self, ph, v):
        u = np.interp(v, self.speeds, np.arange(self.S))
        s0 = np.clip(np.floor(u).astype(int), 0, self.S - 2); ws = (u - s0)
        f = ph * self.M
        i0 = np.floor(f).astype(int) % self.M; i1 = (i0 + 1) % self.M; wp = f - np.floor(f)

        def g(A):
            sh = (-1,) + (1,) * (A.ndim - 2)
            a0 = A[s0, i0] * (1 - wp.reshape(sh)) + A[s0, i1] * wp.reshape(sh)
            a1 = A[s0 + 1, i0] * (1 - wp.reshape(sh)) + A[s0 + 1, i1] * wp.reshape(sh)
            return a0 * (1 - ws.reshape(sh)) + a1 * ws.reshape(sh)
        sn = np.where(ws < 0.5, s0, s0 + 1)
        return dict(q=g(self.q), qd=g(self.qd), ank=g(self.ank), pitch=g(self.pitch), z=g(self.z), y=g(self.y),
                    vroot=g(self.vroot), contact=self.contact[sn, i0], mp=g(self.Qfull[:, :, self.mpadr]))


class K1SprintBatch(K1RunBatch):
    ACC = 1.5         # m/s^2 reference acceleration limit

    def __init__(self, n, v_lo=1.6, v_hi=2.5, **kw):
        self.v_lo, self.v_hi = v_lo, v_hi
        self.v_cmd = np.zeros(n); self.v_ref = np.zeros(n); self.next_evt = np.zeros(n, int)
        self.qd_over = np.zeros(n); self.scripted = False
        self.w_qd = 0.02
        self.kvel = 25.0
        self._lib_ready = False
        super().__init__(n, **kw)

    def _ensure_lib(self):
        if not self._lib_ready:
            self.lib = RunLib(self.base_model)
            self.ref.lean = self.lib.lean
            self._lib_ready = True

    def _sample_v(self, k):
        top = self.rng.random(k) < 0.4                       # 40 % near the current top speed
        v = self.rng.uniform(self.v_lo, self.v_hi, k)
        return np.where(top, self.rng.uniform(max(self.v_lo, self.v_hi - 0.6), self.v_hi, k), v)

    def reset(self, ids):
        if not hasattr(self, 'base_model') or not hasattr(self, 'kp_scale'):
            return super().reset(ids)                          # during base construction
        self._ensure_lib()
        m = self.m; k = len(ids)
        if k == 0:
            return
        d = self.datas[0]
        v0 = self._sample_v(k)
        ph = self.rng.random(k)
        r = self.lib.sample(ph, v0)
        L = self.lib.lean
        for j, i in enumerate(ids):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = m.key_qpos[0]
            d.qpos[self.qadr] = r['q'][j]
            d.qpos[0] = 0.0; d.qpos[1] = r['y'][j]; d.qpos[2] = r['z'][j] + 0.005
            d.qpos[3:7] = [np.cos(L / 2), 0, np.sin(L / 2), 0]
            d.qvel[self.dadr] = r['qd'][j]; d.qvel[0:3] = r['vroot'][j]
            d.qpos[self.mpadr] = r['mp'][j] * 0.5
            if self.stage >= 2:
                d.qpos[self.qadr[:12]] += self.rng.normal(0, 0.02, 12)
                d.qvel[0:2] += self.rng.normal(0, 0.1, 2)
            mujoco.mj_forward(m, d)
            st = np.zeros(self.nstate)
            mujoco.mj_getState(m, d, st, self.spec_state)
            self.state[i] = st; self.sdata[i] = d.sensordata
        self.ph[ids] = ph; self.alpha[ids] = 1.0; self.cmd[ids] = 1.0
        self.v_cmd[ids] = v0; self.v_ref[ids] = v0
        self.t[ids] = 0
        self.last_a[ids] = 0; self.last_a2[ids] = 0
        self.air[ids] = False; self.air_time[ids] = 0
        self.kp_scale[ids] = 1; self.kd_scale[ids] = 1
        self.next_evt[ids] = self.rng.integers(100, 300, k)

    def _schedule(self):
        if self.scripted:
            return
        ev = np.where(self.t == self.next_evt)[0]
        if len(ev):
            self.v_cmd[ev] = self._sample_v(len(ev))
            self.next_evt[ev] = self.t[ev] + self.rng.integers(100, 300, len(ev))

    def obs(self):
        oa, oc = super().obs()
        na = oa.shape[1]
        ins = np.stack([self.v_cmd / 5.0, self.v_ref / 5.0], 1).astype(np.float32)
        oa2 = np.concatenate([oa[:, :8], ins, oa[:, 8:]], 1)
        oc2 = np.concatenate([oc[:, :8], ins, oc[:, 8:na], oc[:, na:]], 1)
        return oa2, oc2

    def assist_ctrl(self, r):
        n = self.n
        if self.assist <= 0:
            return np.zeros((n, 5))
        q = self.qpos(); qv = self.qvel()
        quat, g = self.base_frame()
        gyro = qv[:, 3:6]
        fz = np.clip(1500 * (r['z'] - q[:, 2]) - 100 * qv[:, 2], -0.3 * MG, 0.6 * MG)
        fx = np.clip(60 * (self.v_ref - qv[:, 0]), -60, 60)
        fy = np.clip(-80 * qv[:, 1], -50, 50)
        tx = np.clip(200 * g[:, 1] - 20 * gyro[:, 0], -50, 50)
        ty = np.clip(-200 * (g[:, 0] - np.sin(self.lib.lean)) - 20 * gyro[:, 1], -50, 50)
        return self.assist * np.stack([fz, fx, fy, tx, ty], 1)

    def step(self, a):
        n, m = self.n, self.m
        a = np.clip(a, -4, 4)
        self._schedule()
        dv = np.clip(self.v_cmd - self.v_ref, -self.ACC / self.CTRL_HZ, self.ACC / self.CTRL_HZ)
        self.v_ref = self.v_ref + dv
        rate = 1.0 / (self.lib.T(self.v_ref) * self.CTRL_HZ)
        self.alpha[:] = 1.0
        self.ph = (self.ph + rate) % 1.0
        r = self.lib.sample(self.ph, self.v_ref)
        self._r = r
        qbase = r['q'].copy()
        tgt = qbase.copy()
        tgt[:, :12] += 0.25 * a[:, :12]
        self.kp_scale = np.clip(1 + 0.5 * a[:, 12:24], *KP_RANGE)
        self.kd_scale = np.clip(1 + 0.5 * a[:, 24:36], *KD_RANGE)
        kp = KP[:12] * self.kp_scale * self.kp_dr
        kd = KD[:12] * self.kd_scale * self.kp_dr
        ctrl = tgt.copy(); ctrl[:, :12] = kp * tgt[:, :12]
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
        over = np.clip(np.abs(qd_sub) - QD_LIM, 0, None)                  # (n, nsub, 12)
        self.qd_over = (over ** 2).sum(2).mean(1)
        self.qd_viol = (over > 0).any(2).mean(1)                           # fraction of substeps over the limit
        self.qd_peak = np.abs(qd_sub).max((1, 2))
        self.state = st[:, -1].copy(); self.sdata = sd[:, -1].copy()
        if self.stage >= 2 and self.pushes:
            k = np.where(self.rng.random(n) < 1.0 / 200)[0]
            if len(k):
                vo = 1 + m.nq
                self.state[k, vo:vo + 2] += self.rng.uniform(-self.push_mag, self.push_mag, (len(k), 2))
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
        return rew.astype(np.float32), term, trunc, info

    def reward(self, a, r, qbase, touch):
        q = self.qpos(); qv = self.qvel()
        ql = q[:, self.qadr[:12]]
        r_q = np.exp(-1.5 * ((ql - qbase[:, :12]) ** 2).sum(1))
        ank = np.stack([self.s('left_ankle_rel'), self.s('right_ankle_rel')], 1)
        r_ank = np.exp(-20.0 * ((ank - r['ank']) ** 2).sum((1, 2)))
        fx = np.stack([self.s('left_foot_x'), self.s('right_foot_x')], 1)
        pitch = np.arcsin(np.clip(-fx[:, :, 2], -1, 1))
        r_pitch = np.exp(-4.0 * ((pitch - r['pitch']) ** 2).sum(1))
        quat, g = self.base_frame()
        vlin = qv[:, 0:3]
        rel = (vlin[:, 0] - self.v_ref) / np.maximum(self.v_ref, 1.0)
        r_vel = np.exp(-self.kvel * rel ** 2 - 4.0 * vlin[:, 1] ** 2)       # kvel 25: 10 % error -> 0.78, 20 % -> 0.37
        r_vz = np.exp(-10.0 * (vlin[:, 2] - r['vroot'][:, 2]) ** 2)
        r_up = np.exp(-20.0 * ((g[:, 0] - np.sin(self.lib.lean)) ** 2 + g[:, 1] ** 2))
        gyro = qv[:, 3:6]
        r_rate = np.exp(-0.5 * (gyro[:, 0] ** 2 + gyro[:, 1] ** 2) - 1.0 * gyro[:, 2] ** 2)
        r_h = np.exp(-200.0 * (q[:, 2] - r['z']) ** 2)
        heel = touch[:, [0, 3]] > 5.0
        fore = (touch[:, [1, 4]] > 5.0) | (touch[:, [2, 5]] > 5.0)
        hc = r['contact'][:, :, 0] > 0.5; fc = r['contact'][:, :, 1] > 0.5
        r_contact = 0.5 * ((heel == hc).astype(float) + (fore == fc).astype(float)).mean(1)
        anyc = heel | fore
        flight = ~anyc.any(1); flight_ref = ~(hc | fc).any(1)
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
        thr = 2.5 + 0.2 * np.clip(self.v_ref - 2.0, 0, None)                # human peak GRF grows with speed
        p_impact = np.clip(impact - thr, 0, None)
        a_rate = ((a - self.last_a) ** 2).sum(1)
        a_acc = ((a - 2 * self.last_a + self.last_a2) ** 2).sum(1)
        rew = (0.15 * r_q + 0.15 * r_ank + 0.05 * r_pitch + 0.05 * r_h + 0.10 * r_contact
               + 0.40 * r_vel + 0.15 * r_up + 0.10 * r_rate + 0.2 * r_hs
               + 0.10 * r_vz + 0.3 * (flight & flight_ref) - 0.1 * (flight & ~flight_ref)
               - self.w_impact * p_impact - 0.1 * slip - self.w_qd * self.qd_over
               - self.w_energy * self.P_elec - 0.005 * a_rate - 0.002 * a_acc)
        info = dict(r_q=r_q, r_ank=r_ank, r_pitch=r_pitch, r_vel=r_vel, r_up=r_up, r_rate=r_rate, r_vz=r_vz,
                    r_contact=r_contact, hs_good=hs_good, hs_bad=hs_bad, vx=vlin[:, 0], v_ref=self.v_ref.copy(),
                    verr_rel=np.abs(rel), impact=impact, flight=flight.astype(float),
                    P_elec=self.P_elec, kp_mean=self.kp_scale.mean(1), kd_mean=self.kd_scale.mean(1),
                    torso_tilt=np.linalg.norm(g[:, :2], axis=1), qd_viol=self.qd_viol, qd_peak=self.qd_peak)
        return rew, info
