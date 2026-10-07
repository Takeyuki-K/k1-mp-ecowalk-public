# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5.6 walking env: level pelvis, higher swing foot, policy-controlled arm swing, feet re-placed before standing.
Idea & direction: Takeyuki-K. Based on the v5.5 walking env (K1Walk3Batch, hooks in k1env_walk3.py).

User instructions -> implementation
 1. keep the pelvis as level as possible while walking
      reference ref_lib_56.npz (make_ref56.py): narrower stance 16 cm, lateral pelvis travel from the linear inverted
      pendulum, pelvis kept level by the legs;  reward r_level = exp(-60 (g_y - sin(phi))^2)  (phi = turn lean, v5.5)
 2. a higher swing foot is allowed (a little higher than a human)
      reference: +~2.5 cm swing clearance (hip/knee/ankle flexion, foot parallel)
 3. arm swing may help the balance (reward design is hard)
      4 extra actions: shoulder pitch / roll residuals (+-0.3 rad per unit, both sides) on top of the reference arm
      swing. No arm-specific reward: the policy uses the arms only if it helps the existing rewards (level pelvis,
      heading, speed, energy). Arm motor power is added to the energy term so the arms are not "free".
 4. after an in-place turn the robot stopped in a half step and could fall later
      stop rule: when stopping, keep stepping in place until two steps (one per foot) have been made with zero yaw
      rate, then close the legs;  reward r_home (feet at the standing position) while standing.
Observation: v5.5 walking + 4 last-action entries (81). Actions 41.
"""
import os
import numpy as np
from k1env import ACT_JOINTS, KP, KD
from k1env_speed import RefLib
from k1env_walk3 import K1Walk3Batch

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = [ACT_JOINTS.index(j) for j in ('left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
                                     'right_shoulder_pitch_joint', 'right_shoulder_roll_joint')]
ARM_ALL = list(range(13, 23))
ARM_SCALE = 0.3
KM_ARM = 2.2          # assumed motor constant of the arm motors (QC060 class), as the ankle roll in v1
TLIM_ARM = 20.0


class K1Walk4Batch(K1Walk3Batch):
    nact = 41
    W_LEVEL = 0.35
    ROLL_DEADBAND = float(os.environ.get('ROLL_DEADBAND', '0.0'))   # sin(roll) free band; w56b: 0.045 (~2.6 deg)
    IP_WIDE = os.environ.get('IP_WIDE', '0') == '1'                 # w56b: original 21 cm stance when stepping in place
    W_HOME = 0.30
    HOME_STEPS = 2
    ALIGN_CHECK = os.environ.get('ALIGN_CHECK', '0') == '1'
    PLACE_FF = os.environ.get('PLACE_FF', '0') == '1'

    def __init__(self, n, **kw):
        self._v56 = False
        super().__init__(n, **kw)
        self.lib = RefLib(self.m, path=os.path.join(HERE, 'ref_lib_56.npz'))
        self.home_cnt = np.zeros(n, int)
        self.P_arm = np.zeros(n)
        self._prev_ph = self.ph.copy()
        self.ds_dx = np.zeros(n); self.ds_dyaw = np.zeros(n); self.corr = np.zeros((n, 4))
        self.aadr = np.array([self.m.jnt_qposadr[self.m.joint(ACT_JOINTS[i]).id] for i in ARM_ALL])
        self.aadr_d = np.array([self.m.jnt_dofadr[self.m.joint(ACT_JOINTS[i]).id] for i in ARM_ALL])
        self._v56 = True
        self.reset(np.arange(n))

    def reset(self, ids):
        super().reset(ids)
        if getattr(self, '_v56', False) and len(ids):
            self.home_cnt[ids] = 0
            self._prev_ph[ids] = self.ph[ids]
            self.ds_dx[ids] = 0; self.ds_dyaw[ids] = 0; self.corr[ids] = 0

    # ---- 4. re-place the feet before closing the legs ----
    def foot_offsets(self):
        """front-back offset [m], separation [m] and relative yaw [rad] of the feet in the pelvis frame"""
        a = self.s('left_ankle_rel'); b = self.s('right_ankle_rel')
        fl = self.s('left_foot_x'); fr = self.s('right_foot_x')
        dyaw = np.arctan2(fl[:, 1], fl[:, 0]) - np.arctan2(fr[:, 1], fr[:, 0])
        return a[:, 0] - b[:, 0], a[:, 1] - b[:, 1], np.arctan2(np.sin(dyaw), np.cos(dyaw))

    def _close_ok(self, settled, walking):
        """stage w56c: keep stepping in place (zero yaw rate) until the feet are actually re-aligned
        (|front-back| < 4 cm and |relative yaw| < 5 deg, checked after at least 2 steps), at most 8 steps."""
        ph = self.ph
        crossed = ((self._prev_ph < 0.5) & (ph >= 0.5)) | (ph < self._prev_ph)       # one step = half a cycle
        self._prev_ph = ph.copy()
        stepping = settled & ~walking & (self.alpha >= 1.0)
        self.home_cnt = np.where(stepping, self.home_cnt + crossed.astype(int),
                                 np.where(self.alpha >= 1.0, 0, self.home_cnt))
        if self.PLACE_FF:      # w56d: judge the alignment at the last double support (not mid-swing)
            dx, dyaw = self.ds_dx, self.ds_dyaw
        else:
            dx, _, dyaw = self.foot_offsets()
        aligned = (np.abs(dx) < 0.04) & (np.abs(dyaw) < np.radians(5))
        if self.ALIGN_CHECK:
            ok = ((self.home_cnt >= self.HOME_STEPS) & aligned) | (self.home_cnt >= 8)
        else:
            ok = self.home_cnt >= self.HOME_STEPS
        return ok | (self.alpha < 1.0)

    def _place_ff(self, tgt):
        """stage w56d: foot re-placement feed-forward while stepping in place to stop. The front-back offset and the
        relative yaw of the feet are measured at the last double support; the leg that swings next is placed beside
        the stance foot (hip pitch with ankle compensation: 0.63 m per rad, hip yaw: ~1 deg per deg; signs from
        forward kinematics). Added because the w56c policy kept a ~10 cm stagger while stepping in place."""
        touch = np.concatenate([self.s(f'{s}_{p}_touch') for s in ('left', 'right') for p in ('heel', 'meta', 'toe')], 1)
        ds = (touch[:, :3].sum(1) > 5) & (touch[:, 3:].sum(1) > 5)
        dx, _, dyaw = self.foot_offsets()
        self.ds_dx = np.where(ds, dx, self.ds_dx); self.ds_dyaw = np.where(ds, dyaw, self.ds_dyaw)
        stopping = (self.cmd < 0.5) & (self.v_ref < 0.05) & (np.abs(self.w_ref) < 0.05) & (self.alpha > 0)
        c = self.lib.sample(self.ph, self.v_lib)['contact']                  # (n, leg, heel/fore)
        swl = ~(c[:, 0] > 0.5).any(1); swr = ~(c[:, 1] > 0.5).any(1)
        kx = np.clip(self.ds_dx / 0.63, -0.25, 0.25); ky = np.clip(self.ds_dyaw, -0.3, 0.3)
        want = np.zeros((self.n, 4))                                            # hipL, yawL, hipR, yawR
        want[:, 0] = np.where(swl, kx, 0.0); want[:, 1] = np.where(swl, -ky, 0.0)
        want[:, 2] = np.where(swr, -kx, 0.0); want[:, 3] = np.where(swr, ky, 0.0)
        want *= stopping[:, None]
        self.corr += 0.2 * (want - self.corr)
        tgt[:, 0] += self.corr[:, 0]; tgt[:, 4] -= self.corr[:, 0]; tgt[:, 2] += self.corr[:, 1]
        tgt[:, 6] += self.corr[:, 2]; tgt[:, 10] -= self.corr[:, 2]; tgt[:, 8] += self.corr[:, 3]

    # ---- 3. arm residuals ----
    def _extra_targets(self, tgt, a):
        tgt[:, ARM] += ARM_SCALE * a[:, 37:41]
        if self.PLACE_FF:
            self._place_ff(tgt)
        if self.IP_WIDE:
            # in-place turning was learned on the 21 cm stance (v3/v5.5); the 16 cm stance of ref_lib_56 made
            # -1 rad/s in-place turns fall (w56a). Undo the narrowing while stepping in place, fade out by 0.4 m/s.
            k_ip = np.clip(1.0 - self.v_ref / 0.4, 0.0, 1.0) * self.alpha
            dl = np.arctan2((0.209 - 0.16) / 2, 0.70) * k_ip
            tgt[:, 1] += dl; tgt[:, 5] -= dl; tgt[:, 7] -= dl; tgt[:, 11] += dl

    def _extra_power(self, st, tgt):
        m = self.m
        q = st[:, :, 1:1 + m.nq][:, :, self.aadr]
        qd = st[:, :, 1 + m.nq:1 + m.nq + m.nv][:, :, self.aadr_d]
        kp = KP[ARM_ALL]; kd = KD[ARM_ALL]
        tau = np.clip(kp * (tgt[:, None, ARM_ALL] - q) - kd * qd, -TLIM_ARM, TLIM_ARM)
        self.P_arm = ((tau ** 2 / KM_ARM ** 2).sum(2) + np.clip(tau * qd, 0, None).sum(2)).mean(1)
        return self.P_arm

    # ---- 1. level pelvis, 4. home stance ----
    def _extra_reward(self, a, g, ank, touch):
        roll_err = g[:, 1] - np.sin(self.phi)
        # user decision after w56a: human-like sway allowed -> no penalty inside +-ROLL_DEADBAND (energy decides),
        # steep outside (w56a strictly level: roll 3.8 deg p2p but hip-roll losses 22 -> 39 W)
        roll_err = np.sign(roll_err) * np.clip(np.abs(roll_err) - self.ROLL_DEADBAND, 0, None)
        r_level = np.exp(-60.0 * roll_err ** 2)
        standing = (self.alpha <= 0) & (self.cmd < 0.5)
        r_home = np.exp(-200.0 * ((ank - self.ref.ank_stand[None]) ** 2).sum((1, 2))) * standing
        p_arm = (a[:, 37:41] ** 2).sum(1)
        # stage w56c: while stepping in place to stop, reward re-aligned feet (front-back offset, relative yaw)
        dx, _, dyaw = self.foot_offsets()
        stopping = (self.cmd < 0.5) & (self.alpha > 0) & (self.v_ref < 0.05) & (np.abs(self.w_ref) < 0.05)
        r_align = np.exp(-60.0 * dx ** 2 - 15.0 * dyaw ** 2) * stopping
        ex = self.W_LEVEL * r_level + self.W_HOME * r_home + 0.3 * r_align - 0.01 * p_arm
        return ex, dict(r_level=r_level, roll_deg=np.degrees(np.abs(np.arcsin(np.clip(g[:, 1], -1, 1)))),
                        r_home=r_home, r_align=r_align, P_arm=self.P_arm, P_leg=self.P_leg, arm_act=np.abs(a[:, 37:41]).mean(1))
