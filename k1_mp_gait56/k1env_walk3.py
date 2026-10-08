# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5.5 walking env = v5 walking (motor model, run -> walk hand-over, 0.3-1.65 m/s) + the v3 turning skills.

Ported from k1_mp_turn/k1env_turn.py (v3), unchanged in intent:
  * yaw-rate command w_cmd, reference w_ref (1.5 rad/s^2 ramp), heading target integrates w_ref with a 0.5 rad leash
  * v = 0 reference (stepping in place) and the in-place turning pattern (T4), so in-place turns are possible
  * safe stop: decelerate to stepping in place, then feet together (alpha -> 0)
New in v5.5 (turnphys.py):
  * speed governor v * w <= A_LAT_WALK, lean target / partial lean feed-forward
Observation = v5 walking observation + [w_cmd, w_ref] after heading_err (index 11) -> 76, same layout as v3.
"""
import os
import numpy as np
from k1env import DEFAULT_POSE, KP, KD
from k1env_eco import KP_RANGE, KD_RANGE, TLIM, KM
from k1env_speed import RefLib, V_MIN, V_MAX
from k1env_walk2 import K1Walk2Batch, yaw_of
from turnphys import governed_targets, lean_angle, apply_lean_ff, A_LAT_WALK

HERE = os.path.dirname(os.path.abspath(__file__))
W_MAX = 1.0
W_ACC = 1.5
ACC = 0.6
ACC_BRAKE_WALK = 1.5   # hard braking while walking (stage w55c): 2.5x the comfortable human deceleration
MGF = 35.7 * 9.81


class K1Walk3Batch(K1Walk2Batch):
    nact = 37
    LEAN_FF = 0.5
    LONG_INPLACE = os.environ.get('LONG_INPLACE', '0') == '1'

    def __init__(self, n, **kw):
        self._turn_ready = False
        super().__init__(n, **kw)
        self.lib = RefLib(self.m, path=os.path.join(HERE, 'ref_lib_turn.npz'))   # v = 0 ... 1.95
        self.w_cmd = np.zeros(n); self.w_ref = np.zeros(n)
        self.impact = np.zeros(n)
        self.a_lat = A_LAT_WALK
        self.use_brake = os.environ.get('WALK_BRAKE', '0') == '1'      # adds the brake input (obs 77)
        self.brake = np.zeros(n)
        self.P_BRAKE = float(os.environ.get('P_BRAKE_WALK', '0.15'))
        self.w_yaw = 0.20; self.w_head = 0.25
        self._turn_ready = True
        self.reset(np.arange(n))

    # ---------------- reset / commands ----------------
    def reset(self, ids):
        super().reset(ids)
        if not getattr(self, '_turn_ready', False) or len(ids) == 0:
            return
        self.w_cmd[ids] = 0.0; self.w_ref[ids] = 0.0; self.brake[ids] = 0.0
        # standing starts begin from v_ref 0 (v3 semantics: ramp up from stepping in place)
        st = self.alpha[ids] == 0
        self.v_ref[ids[st]] = 0.0
        if self.use_brake:      # hard stop right at the run -> walk hand-over (gait manager: brake while running)
            hb = (self.entry[ids] == 1) & (self.rng.random(len(ids)) < 0.3)
            self.brake[ids[hb]] = 1.0; self.cmd[ids[hb]] = 0.0; self.v_cmd[ids[hb]] = 0.0

    def _new_motion(self, i):
        u = self.rng.random()
        sgn = 1.0 if self.rng.random() < float(os.environ.get('TURN_POS_P', '0.5')) else -1.0   # w56g: bias to the weak side
        if os.environ.get('TURN_MIX') == 'inplace':       # v3 T6 curriculum: mostly in-place turning (15/20/55/10 %)
            u = 0.1 if u < 0.15 else (0.4 if u < 0.35 else (0.7 if u < 0.90 else 0.95))
        if u < 0.25:
            self.v_cmd[i] = self.rng.uniform(V_MIN, V_MAX); self.w_cmd[i] = 0.0
        elif u < 0.60:                          # walking turn, including commands the governor has to slow down
            self.v_cmd[i] = self.rng.uniform(V_MIN, V_MAX); self.w_cmd[i] = sgn * self.rng.uniform(0.15, W_MAX)
        elif u < 0.90:                          # in-place turn
            self.v_cmd[i] = 0.0; self.w_cmd[i] = sgn * (self.rng.uniform(0.2, W_MAX) if self.rng.random() > float(os.environ.get('P_FAST_IP', '0')) else self.rng.uniform(0.7, W_MAX))
        else:                                   # stepping in place
            self.v_cmd[i] = 0.0; self.w_cmd[i] = 0.0

    def _schedule(self):
        if self.scripted:
            return
        ev = self.t == self.next_evt
        if not ev.any():
            return
        for i in np.where(ev)[0]:
            if self.brake[i] > 0.5:
                self.next_evt[i] = self.t[i] + self.rng.integers(75, 250)
                continue
            if self.cmd[i] < 0.5:
                self.cmd[i] = 1.0; self._new_motion(i)
            elif self.use_brake and self.v_ref[i] > 0.6 and self.rng.random() < self.P_BRAKE:
                self.brake[i] = 1.0; self.cmd[i] = 0.0
            elif self.rng.random() < getattr(self, 'P_STOP', 0.15):
                self.cmd[i] = 0.0
            else:
                self._new_motion(i)
            if self.w_cmd[i] == 0 and self.rng.random() < 0.3:
                self.yaw_t[i] = self.yaw()[i] + self.rng.uniform(-0.4, 0.4)
            long_ip = self.LONG_INPLACE and self.v_cmd[i] == 0 and self.w_cmd[i] != 0 and self.cmd[i] > 0.5
            # w55e: in-place turns of up to 12 s (w55c/d fell after ~7 s of continuous in-place turning)
            self.next_evt[i] = self.t[i] + (self.rng.integers(250, 600) if long_ip else self.rng.integers(75, 250))

    # ---------------- observation ----------------
    def obs(self):
        oa, oc = super().obs()
        if not getattr(self, '_turn_ready', False):
            return oa, oc
        na = oa.shape[1]
        ins = [self.w_cmd, self.w_ref] + ([self.brake] if self.use_brake else [])
        ins = np.stack(ins, 1).astype(np.float32)
        oa2 = np.concatenate([oa[:, :11], ins, oa[:, 11:]], 1)
        oc2 = np.concatenate([oc[:, :11], ins, oc[:, 11:na], oc[:, na:]], 1)
        return oa2, oc2

    # ---------------- step ----------------
    def step(self, a):
        n, m = self.n, self.m
        a = np.clip(a, -4, 4)
        self._schedule()
        walking = self.cmd > 0.5
        v_c = np.where(walking, self.v_cmd, 0.0); w_c = np.where(walking, self.w_cmd, 0.0)
        v_tgt, w_tgt = governed_targets(v_c, w_c, self.v_ref, self.a_lat)
        self.v_gov = v_tgt
        dec = np.where(self.brake > 0.5, ACC_BRAKE_WALK, ACC)
        dv = np.clip(v_tgt - self.v_ref, -dec / self.CTRL_HZ, ACC / self.CTRL_HZ)
        self.v_ref = np.where(self.alpha > 0, self.v_ref + dv, 0.0)
        dw = np.clip(w_tgt - self.w_ref, -W_ACC / self.CTRL_HZ, W_ACC / self.CTRL_HZ)
        self.w_ref = np.where(self.alpha > 0, self.w_ref + dw, 0.0)
        self.yaw_t = self.yaw_t + self.alpha * self.w_ref / self.CTRL_HZ
        lead = np.arctan2(np.sin(self.yaw_t - self.yaw()), np.cos(self.yaw_t - self.yaw()))
        self.yaw_t = self.yaw() + np.clip(lead, -0.5, 0.5)
        settled = (self.v_ref < 0.05) & (np.abs(self.w_ref) < 0.05)
        self.brake = np.where(self.alpha <= 0, 0.0, self.brake)           # braking ends when the feet are together
        self.v_lib = np.clip(self.v_ref * (1.0 + 0.15 * np.clip(a[:, 36], -2, 2)), 0.0, 1.95)
        Tv = self.lib.T(self.v_lib)
        rate = 1.0 / (Tv * self.CTRL_HZ)
        close = settled & self._close_ok(settled, walking)                  # hook (v5.6: re-place the feet first)
        self.alpha = np.clip(self.alpha + np.where(walking, rate, np.where(close, -rate, 0.0)), 0, 1)
        moving = self.alpha > 0
        self.ph = np.where(moving, (self.ph + rate * self._rate_mult(Tv)) % 1.0, self.ph)   # hook (v5.6e: rhythm)
        r = self.lib.sample(self.ph, self.v_lib)
        al = self.alpha[:, None]
        qbase = (1 - al) * DEFAULT_POSE[None] + al * r['q']
        # v3 in-place turning pattern (T4)
        k_ip = np.clip(1.0 - self.v_ref / 0.4, 0.0, 1.0) * self.alpha
        dpsi = self._ip_dpsi(Tv, k_ip)                                      # hook (v5.6e)
        cph = np.cos(2 * np.pi * self.ph)
        qbase[:, 8] += 0.5 * dpsi * cph
        qbase[:, 2] -= 0.5 * dpsi * cph
        wide = 0.05 * np.clip(np.abs(self.w_ref) / 0.5, 0, 1) * k_ip
        qbase[:, 1] += wide; qbase[:, 7] -= wide
        # v5.5 lean feed-forward (small while walking: v*w <= 1 m/s^2 -> <= 6 deg)
        self.phi = lean_angle(self.v_ref, self.w_ref) * self.alpha
        apply_lean_ff(qbase, self.phi, self.LEAN_FF)
        tgt = qbase.copy()
        tgt[:, :12] += 0.25 * a[:, :12]
        self._extra_targets(tgt, a)                                          # hook (v5.6: arm residuals)
        self.kp_scale = np.clip(1 + 0.5 * a[:, 12:24], *KP_RANGE)
        self.kd_scale = np.clip(1 + 0.5 * a[:, 24:36], *KD_RANGE)
        self._stiff_floor()                                                 # hook (v5.6.3: no limp joints standing)
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
        self.P_leg = ((self.tau_sub ** 2 / KM ** 2).sum(2) + np.clip(p_mech, 0, None).sum(2)).mean(1)
        self.P_elec = self.P_leg + self._extra_power(st, tgt)               # hook (v5.6: arm motors)
        self.state = st[:, -1].copy(); self.sdata = sd[:, -1].copy()
        self.sd_out_all = sd
        if self.stage >= 2 and self.pushes:
            kk = np.where(self.rng.random(n) < float(os.environ.get('PUSH_P', '0.005')))[0]      # w56h: PUSH_P
            if len(kk):
                vo = 1 + m.nq
                self.state[kk, vo:vo + 2] += self.rng.uniform(-0.35, 0.35, (len(kk), 2))
        touch = np.zeros((n, 6))
        for k, nm in enumerate([f'{s}_{p}_touch' for s in ('left', 'right') for p in ('heel', 'meta', 'toe')]):
            touch[:, k] = sd[:, :, self.sens[nm][0]].max(1)
        self.impact = np.maximum(touch[:, 0:3].sum(1), touch[:, 3:6].sum(1))
        self.t += 1
        rew, info = self.reward_turn(a, r, qbase, touch)
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

    # ---------------- hooks (no-ops in v5.5) ----------------
    def _close_ok(self, settled, walking):
        return np.ones(self.n, bool)

    def _rate_mult(self, Tv):
        return 1.0

    def _stiff_floor(self):
        pass

    def _ip_dpsi(self, Tv, k_ip):
        return 0.5 * self.w_ref * Tv * k_ip

    def _extra_targets(self, tgt, a):
        pass

    def _extra_power(self, st, tgt):
        return 0.0

    def _extra_reward(self, a, g, ank, touch):
        return 0.0, {}

    def reward_turn(self, a, r, qbase, touch):
        """v5 walking reward (heading frame velocity, eco power) + v3 yaw-rate / heading terms + lean target."""
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
        vlin = np.stack([c * vw[:, 0] + s_ * vw[:, 1], -s_ * vw[:, 0] + c * vw[:, 1], vw[:, 2]], 1)
        e_head = self.heading_err()
        r_head = np.exp(-8.0 * e_head ** 2)
        v_t = al * self.v_ref
        kv = np.where(al < 0.05, 20.0, self.kv_walk * (1.05 / np.maximum(self.v_ref, 0.3)) ** 2)
        # lateral velocity: on a curve the pelvis frame lags the path, allow v*w-proportional side slip tolerance
        r_vel = np.exp(-kv * ((vlin[:, 0] - v_t) ** 2 + vlin[:, 1] ** 2 / (1.0 + 4.0 * np.abs(self.w_ref))))
        r_up = np.exp(-20.0 * (g[:, 0] ** 2 + (g[:, 1] - np.sin(self.phi)) ** 2))
        b = self.brake
        r_yaw = np.exp(-12.0 * (qv[:, 5] - al * self.w_ref) ** 2)
        p_impact = np.clip(self.impact - 1.3 * MGF, 0, None) / MGF
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
        r_hs = (hs_good - hs_bad) * (al > 0.5) * (self.v_ref > 0.2)     # no heel-strike demand when stepping in place
        r_place = (touchdown * np.clip(ank[:, :, 0] / 0.15, -1.0, 1.0)).sum(1) * b
        r_nodrop = np.exp(-10.0 * np.clip(-vlin[:, 2], 0, None) ** 2) * b
        self.air_time = np.where(anyc, 0, self.air_time + 1.0 / self.CTRL_HZ)
        self.air = (~anyc) & (self.air_time > 0.06) | (self.air & ~anyc)
        fv = np.stack([self.s('left_foot_linvel'), self.s('right_foot_linvel')], 1)
        slip = (np.linalg.norm(fv[:, :, :2], axis=2) ** 2 * anyc).sum(1)
        a_rate = ((a - self.last_a) ** 2).sum(1)
        a_acc = ((a - 2 * self.last_a + self.last_a2) ** 2).sum(1)
        rew = (self.w_im * (0.20 * r_q + 0.25 * r_ank + 0.10 * r_pitch) + self.w_vel * r_vel + 0.10 * r_up
               + self.w_yaw * r_yaw + 0.05 * r_h + 0.15 * r_contact + 0.3 * r_hs + self.w_head * r_head
               + 0.3 * r_place + 0.05 * r_nodrop
               - 0.1 * slip - 0.15 * p_impact - self.w_energy * self.P_elec - 0.005 * a_rate - 0.002 * a_acc)
        ex, exinfo = self._extra_reward(a, g, ank, touch)                   # hook (v5.6: level pelvis, home stance)
        rew = rew + ex
        info = dict(**exinfo, brake=b.copy(), r_place=r_place, r_q=r_q, r_ank=r_ank, r_vel=r_vel, r_up=r_up, r_contact=r_contact, hs_bad=hs_bad,
                    vx=vlin[:, 0], verr=np.abs(vlin[:, 0] - v_t) * (al > 0.99), v_cmd=self.v_cmd.copy(),
                    head_err=np.abs(e_head), v_lib_ratio=self.v_lib / np.maximum(self.v_ref, 0.1),
                    P_elec=self.P_elec, kp_mean=self.kp_scale.mean(1), kd_mean=self.kd_scale.mean(1),
                    werr=np.abs(qv[:, 5] - al * self.w_ref) * (np.abs(self.w_ref) > 0.1),
                    w_ref=np.abs(self.w_ref), impact=self.impact / MGF, lean=np.abs(self.phi))
        return rew, info
