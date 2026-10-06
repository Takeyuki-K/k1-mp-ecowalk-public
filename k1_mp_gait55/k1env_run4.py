# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5.5 running env = v5 running (forefoot landing, motor torque-speed model, stand/walk -> run entries)
+ turning while running + hard braking. Idea & direction: Takeyuki-K.

Turning (turnphys.py):
  * yaw-rate command w_cmd / reference w_ref (ramped), heading target integrates w_ref with a 0.5 rad leash,
    velocity rewarded in the target-heading frame (v5 running rewarded world-x speed and penalised any yaw rate)
  * speed governor v * w <= A_LAT_RUN: a large turn command lowers the speed target, the yaw-rate reference is
    limited by the current speed -> the robot slows down first and then turns harder
  * lean into the turn: trunk target phi = atan(v w / g) (plus the forward running lean), partial feed-forward on the
    hip/ankle rolls
Hard braking ("急停止"):
  * brake flag: the speed reference falls at ACC_BRAKE (4 m/s^2 instead of 1.5) to the hand-over speed 1.8 m/s
  * shaping that follows the user's description: land with the foot ahead of the pelvis, trunk upright (no forward
    lean), absorb with the knee (impact threshold raised, the policy can lower Kp), let the body rise instead of
    dropping (downward pelvis velocity penalised, upward allowed); imitation of the steady running cycle is reduced
    while braking
Observation = v5 running observation + [heading_err, w_cmd, w_ref, brake] after v_cmd/v_ref (index 10) -> 82.
"""
import numpy as np
from k1env import DEFAULT_POSE, KP, KD
from k1env_eco import KP_RANGE, KD_RANGE, TLIM, KM
from k1env_sprint import MG, QD_LIM
from k1env_run2 import K1Run2Batch, V_RUN_MIN, yaw_of
from motor import motor_limit
from turnphys import governed_targets, lean_angle, apply_lean_ff, A_LAT_RUN

W_MAX_RUN = 1.0
W_ACC = 1.5
ACC_BRAKE = 4.0
BRAKE_END = V_RUN_MIN + 0.05


class K1Run4Batch(K1Run2Batch):
    LEAN_FF = 0.5
    YAW_FF = 1.0
    P_TURN = 0.6
    P_BRAKE = 0.15

    def __init__(self, n, **kw):
        self._turn_ready = False
        self.w_cmd = np.zeros(n); self.w_ref = np.zeros(n); self.yaw_t = np.zeros(n)
        self.brake = np.zeros(n); self.phi = np.zeros(n)
        self.a_lat = A_LAT_RUN
        self.w_max = W_MAX_RUN
        super().__init__(n, **kw)
        self._turn_ready = True
        self.reset(np.arange(n))

    # ---------------- helpers ----------------
    def yaw(self):
        q = self.qpos()
        w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def heading_err(self):
        e = self.yaw() - self.yaw_t
        return np.arctan2(np.sin(e), np.cos(e))

    # ---------------- reset / commands ----------------
    def reset(self, ids):
        super().reset(ids)
        if not getattr(self, '_turn_ready', False) or len(ids) == 0:
            return
        self.yaw_t[ids] = self.yaw()[ids]
        self.w_cmd[ids] = 0.0; self.w_ref[ids] = 0.0; self.brake[ids] = 0.0
        # some running starts already turning (otherwise every turn would start from straight running)
        k = len(ids)
        tr = (self.rng.random(k) < 0.3) & (self.entry[ids] == 0)
        for j in np.where(tr)[0]:
            i = ids[j]
            self.w_cmd[i] = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.15, self.w_max)
            _, w_t = governed_targets(self.v_cmd[i:i + 1], self.w_cmd[i:i + 1], self.v_ref[i:i + 1], self.a_lat)
            self.w_ref[i] = w_t[0] * self.rng.random()

    def _schedule(self):
        if self.scripted:
            return
        ev = np.where(self.t == self.next_evt)[0]
        for i in ev:
            if self.brake[i] < 0.5 and self.rng.random() < self.P_BRAKE and self.v_ref[i] > 2.3:
                self.brake[i] = 1.0; self.v_cmd[i] = V_RUN_MIN; self.w_cmd[i] = 0.0
            elif self.brake[i] < 0.5:
                self.v_cmd[i] = self._sample_v(1)[0]
                self.w_cmd[i] = (self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.15, self.w_max)
                                 if self.rng.random() < self.P_TURN else 0.0)
                if self.w_cmd[i] == 0 and self.rng.random() < 0.3:
                    self.yaw_t[i] = self.yaw()[i] + self.rng.uniform(-0.3, 0.3)
            self.next_evt[i] = self.t[i] + self.rng.integers(100, 300)

    # ---------------- observation ----------------
    def obs(self):
        oa, oc = super().obs()
        if not getattr(self, '_turn_ready', False):
            return oa, oc
        na = oa.shape[1]
        ins = np.stack([np.clip(self.heading_err(), -1, 1), self.w_cmd, self.w_ref, self.brake], 1).astype(np.float32)
        oa2 = np.concatenate([oa[:, :10], ins, oa[:, 10:]], 1)
        oc2 = np.concatenate([oc[:, :10], ins, oc[:, 10:na], oc[:, na:]], 1)
        return oa2, oc2

    # ---------------- step ----------------
    def step(self, a):
        n, m = self.n, self.m
        a = np.clip(a, -4, 4)
        self._schedule()
        v_tgt, w_tgt = governed_targets(self.v_cmd, self.w_cmd, self.v_ref, self.a_lat, v_floor=V_RUN_MIN)
        self.v_gov = v_tgt
        acc = np.where(self.brake > 0.5, ACC_BRAKE, self.ACC)
        dv = np.clip(v_tgt - self.v_ref, -acc / self.CTRL_HZ, self.ACC / self.CTRL_HZ)
        self.v_ref = self.v_ref + dv
        self.brake = np.where((self.brake > 0.5) & (self.v_ref <= BRAKE_END), 0.0, self.brake)
        dw = np.clip(w_tgt - self.w_ref, -W_ACC / self.CTRL_HZ, W_ACC / self.CTRL_HZ)
        self.w_ref = self.w_ref + dw
        self.yaw_t = self.yaw_t + self.alpha * self.w_ref / self.CTRL_HZ
        lead = np.arctan2(np.sin(self.yaw_t - self.yaw()), np.cos(self.yaw_t - self.yaw()))
        self.yaw_t = self.yaw() + np.clip(lead, -0.5, 0.5)
        v_lib = np.maximum(self.v_ref, V_RUN_MIN)
        rate = 1.0 / (self.lib.T(v_lib) * self.CTRL_HZ)
        self.alpha = np.clip(self.alpha + rate, 0, 1)
        self.ph = (self.ph + rate) % 1.0
        r = self.lib.sample(self.ph, v_lib)
        self._r = r
        al = self.alpha[:, None]
        qbase = (1 - al) * DEFAULT_POSE[None] + al * r['q']
        self.phi = lean_angle(self.v_ref, self.w_ref) * self.alpha
        apply_lean_ff(qbase, self.phi, self.LEAN_FF)
        # stride-wise hip-yaw feed-forward (same pattern as the v3 in-place turn, T4): the pelvis has to yaw by
        # w * T/2 during each stance; the stance hip yaw unwinds while the swing leg re-opens toward the turn.
        # Added after run r55a: with heading rewards alone the policy reached only ~35 % of the commanded yaw rate.
        Tv = self.lib.T(v_lib)
        dpsi = 0.5 * self.w_ref * Tv * self.alpha * self.YAW_FF
        cph = np.cos(2 * np.pi * self.ph)
        qbase[:, 8] += 0.5 * dpsi * cph
        qbase[:, 2] -= 0.5 * dpsi * cph
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
        for sub in range(self.nsub):                   # motor torque-speed model every 5 ms (as v5)
            q = st[:, 1:1 + nq][:, self.qadr[:12]]
            qd = st[:, 1 + nq:1 + nq + nv][:, self.dadr[:12]]
            tau = kp * (tgt[:, :12] - q) - kd * qd
            f, tdir, direct = motor_limit(tau, qd)
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
        info['cad'] = np.ones(n)
        return rew.astype(np.float32), term, trunc, info

    def reward(self, a, r, qbase, touch):
        if not getattr(self, '_turn_ready', False):
            return super().reward(a, r, qbase, touch)
        q = self.qpos(); qv = self.qvel(); al = self.alpha
        b = self.brake
        ql = q[:, self.qadr[:12]]
        r_q = np.exp(-1.5 * ((ql - qbase[:, :12]) ** 2).sum(1))
        ank = np.stack([self.s('left_ankle_rel'), self.s('right_ankle_rel')], 1)
        ank_t = (1 - al)[:, None, None] * self.ref.ank_stand[None] + al[:, None, None] * r['ank']
        r_ank = np.exp(-20.0 * ((ank - ank_t) ** 2).sum((1, 2)))
        fx = np.stack([self.s('left_foot_x'), self.s('right_foot_x')], 1)
        pitch = np.arcsin(np.clip(-fx[:, :, 2], -1, 1))
        r_pitch = np.exp(-4.0 * ((pitch - al[:, None] * r['pitch']) ** 2).sum(1))
        quat, g = self.base_frame()
        vw = qv[:, 0:3]
        c, s_ = np.cos(self.yaw_t), np.sin(self.yaw_t)
        vlin = np.stack([c * vw[:, 0] + s_ * vw[:, 1], -s_ * vw[:, 0] + c * vw[:, 1], vw[:, 2]], 1)
        rel = (vlin[:, 0] - self.v_ref) / np.maximum(self.v_ref, 1.0)
        r_vel = np.exp(-self.kvel * rel ** 2 - 4.0 * vlin[:, 1] ** 2 / (1.0 + 4.0 * np.abs(self.w_ref)))
        e_head = self.heading_err()
        r_head = np.exp(-8.0 * e_head ** 2)
        gyro = qv[:, 3:6]
        r_yaw = np.exp(-6.0 * (qv[:, 5] - al * self.w_ref) ** 2)
        # vertical: imitate the running bounce; while braking only penalise dropping (rising is allowed)
        r_vz_run = np.exp(-10.0 * (vlin[:, 2] - al * r['vroot'][:, 2]) ** 2)
        r_vz_brk = np.exp(-10.0 * np.clip(-vlin[:, 2], 0, None) ** 2)
        r_vz = (1 - b) * r_vz_run + b * r_vz_brk
        # trunk: running forward lean + lean into the turn; braking: upright (no forward lean)
        lean_fwd = al * np.sin(self.lib.lean) * (1 - b)
        r_up = np.exp(-20.0 * ((g[:, 0] - lean_fwd) ** 2 + (g[:, 1] - np.sin(self.phi)) ** 2))
        r_rate = np.exp(-0.5 * (gyro[:, 0] ** 2 + gyro[:, 1] ** 2) - 1.0 * (gyro[:, 2] - self.w_ref) ** 2)
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
        # braking touchdown: foot ahead of the pelvis (pelvis-frame x of the landing ankle), up to 0.25 m
        r_place = (touchdown * np.clip(ank[:, :, 0] / 0.25, -1.0, 1.0)).sum(1) * b
        self.air_time = np.where(anyc, 0, self.air_time + 1.0 / self.CTRL_HZ)
        self.air = (~anyc) & (self.air_time > 0.06) | (self.air & ~anyc)
        fv = np.stack([self.s('left_foot_linvel'), self.s('right_foot_linvel')], 1)
        slip = (np.linalg.norm(fv[:, :, :2], axis=2) ** 2 * anyc).sum(1)
        f_foot = np.stack([touch[:, :3].sum(1), touch[:, 3:].sum(1)], 1)
        impact = f_foot.max(1) / MG
        thr = 2.5 + 0.2 * np.clip(self.v_ref - 2.0, 0, None) + 0.5 * b
        p_impact = np.clip(impact - thr, 0, None)
        a_rate = ((a - self.last_a) ** 2).sum(1)
        a_acc = ((a - 2 * self.last_a + self.last_a2) ** 2).sum(1)
        im = 1.0 - 0.7 * b
        rew = (im * (0.15 * r_q + 0.15 * r_ank + 0.05 * r_pitch + 0.10 * r_contact + 0.3 * r_strike)
               + 0.05 * r_h + 0.40 * r_vel + 0.15 * r_up + 0.10 * r_rate + 0.10 * r_vz
               + 0.20 * r_yaw + 0.15 * r_head + 0.5 * r_place
               + (1 - b) * (0.3 * (flight & flight_ref) - 0.1 * (flight & ~flight_ref & (al > 0.9)))
               - self.w_impact * p_impact - 0.1 * slip - self.w_qd * self.qd_over
               - self.w_energy * self.P_elec - 0.005 * a_rate - 0.002 * a_acc)
        info = dict(r_q=r_q, r_ank=r_ank, r_pitch=r_pitch, r_vel=r_vel, r_up=r_up, r_rate=r_rate, r_vz=r_vz,
                    r_contact=r_contact, hs_good=fore_first + mid, hs_bad=heel_first, vx=vlin[:, 0],
                    v_ref=self.v_ref.copy(), verr_rel=np.abs(rel), impact=impact, flight=flight.astype(float),
                    P_elec=self.P_elec, kp_mean=self.kp_scale.mean(1), kd_mean=self.kd_scale.mean(1),
                    torso_tilt=np.linalg.norm(g[:, :2], axis=1), qd_viol=self.qd_viol, qd_peak=self.qd_peak,
                    fore_first=fore_first, mid=mid, heel_first=heel_first, entry=self.entry.copy(),
                    werr=np.abs(qv[:, 5] - al * self.w_ref) * (np.abs(self.w_ref) > 0.1),
                    w_ref=np.abs(self.w_ref), head_err=np.abs(e_head), lean=np.abs(self.phi), brake=b.copy(),
                    r_place=r_place)
        return rew, info
