# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Speed-command walking: K1 + passive MP toes + human-gait imitation + learned relaxation (variable impedance).

Reference: library of human-derived gait cycles at 0.30 ... 1.35 m/s (retarget_speed.py, walk-ratio scaling:
step length and cadence both ~ sqrt(speed)). The reference speed v_ref follows the command v_cmd with a
human-like acceleration limit; gait phase advances at 1/T(v_ref); references of neighbouring speeds are blended
at the same phase (all cycles start at right heel strike, so blending is phase-consistent).

Action (36) as in k1env_eco: 12 position residuals, 12 Kp scales, 12 Kd scales.
Observation: eco observation + [v_cmd, v_ref] inserted after alpha.
"""
import os
import numpy as np
import mujoco
from k1env import ACT_JOINTS, DEFAULT_POSE, KP, KD, quat_rot_inv
from k1env_eco import K1EcoBatch, KM, TLIM, KP_RANGE, KD_RANGE

HERE = os.path.dirname(os.path.abspath(__file__))
V_MIN, V_MAX = 0.30, 1.65     # fast-walk extension: commanded range up to 1.65 m/s (library to 1.95 for the gait choice)
ACC = 0.6          # m/s^2, reference acceleration limit (comfortable human range ~0.5-1)


class RefLib:
    def __init__(self, model, path=os.path.join(HERE, 'ref_lib.npz')):
        L = np.load(path)
        self.speeds = L['speeds'].astype(float); self.Tc = L['T'].astype(float)
        Qs = L['qpos']; S, M, _ = Qs.shape
        self.S, self.M = S, M
        d = mujoco.MjData(model)
        qa = [model.jnt_qposadr[model.joint(j).id] for j in ACT_JOINTS]
        self.Qfull = Qs
        self.q = Qs[:, :, qa]
        self.z = Qs[:, :, 2]; self.y = Qs[:, :, 1]
        self.contact = np.stack([L['cl'], L['cr']], 2).astype(float)          # (S,M,2,2)
        ank = np.zeros((S, M, 2, 3)); pitch = np.zeros((S, M, 2))
        for si in range(S):
            for i in range(M):
                d.qpos[:] = Qs[si, i]
                mujoco.mj_kinematics(model, d)
                pel = d.xpos[model.body('pelvis').id]
                for k, s in enumerate(('left', 'right')):
                    b = model.body(f'{s}_ankle_roll_link').id
                    ank[si, i, k] = d.xpos[b] - pel
                    pitch[si, i, k] = np.arcsin(-d.xmat[b].reshape(3, 3)[2, 0])
        self.ank, self.pitch = ank, pitch
        dt = (self.Tc / M)[:, None, None]
        dQ = (np.roll(Qs, -1, 1) - Qs)
        self.qd = dQ[:, :, qa] / dt
        vroot = dQ[:, :, 0:3] / dt
        vroot[:, :, 0] += self.speeds[:, None]                                 # add progression
        self.vroot = vroot

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
                    vroot=g(self.vroot), contact=self.contact[sn, i0])


class K1SpeedBatch(K1EcoBatch):
    nact = 37      # 12 position residuals, 12 Kp scales, 12 Kd scales, 1 gait-speed choice

    def __init__(self, n, stage=2, nthread=2, seed=0, dt=0.005, randomize=False, ep_len=500, w_energy=0.0015,
                 v_range=(V_MIN, V_MAX)):
        self.v_range = v_range
        self._lib_ready = False
        super().__init__(n, stage=stage, nthread=nthread, seed=seed, dt=dt, randomize=randomize, ep_len=ep_len,
                         w_energy=w_energy)
        self.lib = RefLib(self.m)
        self._lib_ready = True
        self.v_cmd = np.zeros(n); self.v_ref = np.zeros(n); self.next_evt = np.zeros(n, int)
        self.v_lib = np.zeros(n)
        self.yaw_t = np.zeros(n)          # target heading (straight walking = hold this heading)
        self.last_a = np.zeros((n, self.nact)); self.last_a2 = np.zeros((n, self.nact))
        self.scripted = False
        self.kv_walk = 15.0     # stage A used 6.0 (systematic 15 % under-speed), stage B: 15.0
        self.w_im, self.w_vel = 0.45, 0.50   # stage A-E: 0.70 / 0.35 (imitation dominated -> ~10 % under-speed)
        self.reset(np.arange(n))

    # ---------------- reset ----------------
    def reset(self, ids):
        if not getattr(self, '_lib_ready', False):
            return super().reset(ids)
        m = self.m; k = len(ids)
        if k == 0:
            return
        d = self.datas[0]
        walk_start = self.rng.random(k) < 0.6
        ph = self.rng.random(k)
        v0 = self.rng.uniform(*self.v_range, k)
        r = self.lib.sample(ph, v0)
        for j, i in enumerate(ids):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = m.key_qpos[0]
            if walk_start[j]:
                d.qpos[self.qadr] = r['q'][j]
                d.qpos[0] = 0.0; d.qpos[1] = r['y'][j]; d.qpos[2] = r['z'][j] + 0.005
                d.qvel[self.dadr] = r['qd'][j]
                d.qvel[0:3] = r['vroot'][j]
            else:
                d.qpos[self.qadr] = DEFAULT_POSE
                d.qpos[2] = self.ref.z_stand + 0.003
            d.qpos[self.qadr[:12]] += self.rng.normal(0, 0.02, 12)
            mujoco.mj_forward(m, d)
            st = np.zeros(self.nstate); mujoco.mj_getState(m, d, st, self.spec_state)
            self.state[i] = st; self.warm[i] = 0; self.sdata[i] = d.sensordata
        self.ph[ids] = np.where(walk_start, ph, 0.0)
        self.alpha[ids] = walk_start.astype(float)
        self.cmd[ids] = walk_start.astype(float)
        self.v_cmd[ids] = v0; self.v_ref[ids] = v0
        self.t[ids] = 0
        self.last_a[ids] = 0; self.last_a2[ids] = 0
        self.kp_scale[ids] = 1; self.kd_scale[ids] = 1
        self.air[ids] = False; self.air_time[ids] = 0
        self.yaw_t[ids] = self.rng.uniform(-0.4, 0.4, k)      # start with a heading error to correct
        self.next_evt[ids] = np.where(walk_start, self.rng.integers(75, 250, k), self.rng.integers(25, 100, k))
        self.switch_t[ids] = -5; self.stop_t[ids] = -5          # base schedule unused

    def _schedule(self):
        if self.scripted:
            return
        ev = self.t == self.next_evt
        if not ev.any():
            return
        idx = np.where(ev)[0]
        for i in idx:
            u = self.rng.random()
            if self.cmd[i] < 0.5:                       # standing -> start walking at a random speed
                self.cmd[i] = 1.0; self.v_cmd[i] = self.rng.uniform(*self.v_range)
                if self.alpha[i] == 0:
                    self.v_ref[i] = self.v_cmd[i]
            elif u < getattr(self, 'P_STOP', 0.15):      # stop
                self.cmd[i] = 0.0
            elif u < 0.35:                              # big step change (extremes)
                self.v_cmd[i] = self.rng.choice(self.v_range)
            else:
                self.v_cmd[i] = self.rng.uniform(*self.v_range)
            if self.rng.random() < 0.3:                 # occasionally move the heading target (correction practice)
                self.yaw_t[i] = self.yaw()[i] + self.rng.uniform(-0.4, 0.4)
            self.next_evt[i] = self.t[i] + self.rng.integers(75, 250)

    def yaw(self):
        q = self.qpos()
        w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def heading_err(self):
        e = self.yaw() - self.yaw_t
        return np.arctan2(np.sin(e), np.cos(e))

    # ---------------- observation ----------------
    def obs(self):
        quat, g = self.base_frame()
        qv = self.qvel()
        q = self.qpos()[:, self.qadr[:12]] - DEFAULT_POSE[:12]
        qd = qv[:, self.dadr[:12]]
        tw = 2 * np.pi * self.ph
        actor = np.concatenate([qv[:, 3:6] * 0.25, g, self.cmd[:, None], self.alpha[:, None],
                                self.v_cmd[:, None], self.v_ref[:, None], np.clip(self.heading_err(), -1, 1)[:, None],
                                np.sin(tw)[:, None], np.cos(tw)[:, None], q, qd * 0.05, self.last_a], 1)
        touch = np.concatenate([self.s(f'{s}_{p}_touch') for s in ('left', 'right') for p in ('heel', 'meta', 'toe')], 1)
        priv = np.concatenate([self.s('base_linvel'), self.qpos()[:, 2:3], np.minimum(touch / 200.0, 2.0),
                               self.qpos()[:, self.mpadr], self.s('left_ankle_rel'), self.s('right_ankle_rel')], 1)
        return actor.astype(np.float32), np.concatenate([actor, priv], 1).astype(np.float32)

    # ---------------- step ----------------
    def step(self, a):
        n, m = self.n, self.m
        a = np.clip(a, -4, 4)
        self._schedule()
        dv = np.clip(self.v_cmd - self.v_ref, -ACC / self.CTRL_HZ, ACC / self.CTRL_HZ)
        self.v_ref = np.where(self.alpha > 0, self.v_ref + dv, self.v_cmd)
        # gait-speed choice: the policy picks which human-derived gait (step length + cadence) to imitate,
        # around the commanded speed (+-30 %). Velocity is still rewarded against v_ref.
        self.v_lib = np.clip(self.v_ref * (1.0 + 0.15 * np.clip(a[:, 36], -2, 2)), 0.25, 1.95)
        Tv = self.lib.T(self.v_lib)
        rate = 1.0 / (Tv * self.CTRL_HZ)
        self.alpha = np.clip(self.alpha + np.where(self.cmd > 0.5, rate, -rate), 0, 1)
        moving = self.alpha > 0
        self.ph = np.where(moving, (self.ph + rate) % 1.0, self.ph)
        r = self.lib.sample(self.ph, self.v_lib)
        al = self.alpha[:, None]
        qbase = (1 - al) * DEFAULT_POSE[None] + al * r['q']
        tgt = qbase.copy()
        tgt[:, :12] += 0.25 * a[:, :12]
        self.kp_scale = np.clip(1 + 0.5 * a[:, 12:24], *KP_RANGE)
        self.kd_scale = np.clip(1 + 0.5 * a[:, 24:36], *KD_RANGE)
        kp = KP[:12] * self.kp_scale * self.kp_dr
        kd = KD[:12] * self.kd_scale * self.kp_dr
        ctrl = tgt.copy(); ctrl[:, :12] = kp * tgt[:, :12]
        ctrl = np.concatenate([ctrl, kp, kd], 1)
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
        self.state = st[:, -1].copy(); self.sdata = sd[:, -1].copy()
        if self.stage >= 2 and self.pushes:
            kk = np.where(self.rng.random(n) < 1.0 / 200)[0]
            if len(kk):
                vo = 1 + m.nq
                self.state[kk, vo:vo + 2] += self.rng.uniform(-0.35, 0.35, (len(kk), 2))
        touch = np.zeros((n, 6))
        for k, nm in enumerate([f'{s}_{p}_touch' for s in ('left', 'right') for p in ('heel', 'meta', 'toe')]):
            touch[:, k] = sd[:, :, self.sens[nm][0]].max(1)
        self.t += 1
        rew, info = self.reward_speed(a, r, qbase, touch)
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

    def reward_speed(self, a, r, qbase, touch):
        q = self.qpos(); qv = self.qvel(); al = self.alpha
        ql = q[:, self.qadr[:12]]
        r_q = np.exp(-2.0 * ((ql - qbase[:, :12]) ** 2).sum(1))
        ank = np.stack([self.s('left_ankle_rel'), self.s('right_ankle_rel')], 1)
        ank_t = (1 - al)[:, None, None] * self.ref.ank_stand[None] + al[:, None, None] * r['ank']
        r_ank = np.exp(-40.0 * ((ank - ank_t) ** 2).sum((1, 2)))
        fx = np.stack([self.s('left_foot_x'), self.s('right_foot_x')], 1)
        pitch = np.arcsin(np.clip(-fx[:, :, 2], -1, 1))
        r_pitch = np.exp(-4.0 * ((pitch - al[:, None] * r['pitch']) ** 2).sum(1))
        quat, g = self.base_frame()
        vw = qv[:, 0:3]
        c, s_ = np.cos(self.yaw_t), np.sin(self.yaw_t)
        vlin = np.stack([c * vw[:, 0] + s_ * vw[:, 1], -s_ * vw[:, 0] + c * vw[:, 1], vw[:, 2]], 1)  # target-heading frame
        e_head = self.heading_err()
        r_head = np.exp(-4.0 * e_head ** 2)
        v_t = al * self.v_ref
        # relative (percentage) speed error: same penalty for 10 % error at 0.3 m/s and at 1.35 m/s
        kv = np.where(al < 0.05, 20.0, self.kv_walk * (1.05 / np.maximum(self.v_ref, 0.3)) ** 2)
        r_vel = np.exp(-kv * ((vlin[:, 0] - v_t) ** 2 + vlin[:, 1] ** 2))
        r_up = np.exp(-20.0 * (g[:, :2] ** 2).sum(1))
        r_yaw = np.exp(-2.0 * qv[:, 5] ** 2)
        z_t = (1 - al) * self.ref.z_stand + al * r['z']
        r_h = np.exp(-200.0 * (q[:, 2] - z_t) ** 2)
        heel = touch[:, [0, 3]] > 5.0
        fore = (touch[:, [1, 4]] > 5.0) | (touch[:, [2, 5]] > 5.0)
        hc = r['contact'][:, :, 0] > 0.5; fc = r['contact'][:, :, 1] > 0.5
        match = 0.5 * ((heel == hc).astype(float) + (fore == fc).astype(float)).mean(1)
        flat = 0.5 * (heel.astype(float) + fore.astype(float)).mean(1)
        r_contact = np.where(al > 0.9, match, np.where(al < 0.05, flat, 0.5))
        anyc = heel | fore
        touchdown = self.air & anyc
        hs_good = (touchdown & heel & ~fore).sum(1) + 0.5 * (touchdown & heel & fore).sum(1)
        hs_bad = (touchdown & ~heel & fore).sum(1)
        r_hs = (hs_good - hs_bad) * (al > 0.5)
        self.air_time = np.where(anyc, 0, self.air_time + 1.0 / self.CTRL_HZ)
        self.air = (~anyc) & (self.air_time > 0.06) | (self.air & ~anyc)
        fv = np.stack([self.s('left_foot_linvel'), self.s('right_foot_linvel')], 1)
        slip = (np.linalg.norm(fv[:, :, :2], axis=2) ** 2 * anyc).sum(1)
        a_rate = ((a - self.last_a) ** 2).sum(1)
        a_acc = ((a - 2 * self.last_a + self.last_a2) ** 2).sum(1)
        # plain electrical power (stage A-D used power/speed, which made slow walking 'cheaper' when not moving)
        e_scale = 1.0
        rew = (self.w_im * (0.20 * r_q + 0.25 * r_ank + 0.10 * r_pitch) + self.w_vel * r_vel + 0.10 * r_up + 0.05 * r_yaw
               + 0.05 * r_h + 0.15 * r_contact + 0.3 * r_hs + 0.15 * r_head
               - 0.1 * slip - self.w_energy * e_scale * self.P_elec - 0.005 * a_rate - 0.002 * a_acc)
        info = dict(r_q=r_q, r_ank=r_ank, r_vel=r_vel, r_up=r_up, r_contact=r_contact, hs_bad=hs_bad,
                    vx=vlin[:, 0], verr=np.abs(vlin[:, 0] - v_t) * (al > 0.99), P_elec=self.P_elec,
                    P_neg=self.P_elec * 0, kp_mean=self.kp_scale.mean(1), kd_mean=self.kd_scale.mean(1),
                    v_cmd=self.v_cmd.copy(), head_err=np.abs(e_head), v_lib_ratio=self.v_lib / np.maximum(self.v_ref, 0.1))
        return rew, info
