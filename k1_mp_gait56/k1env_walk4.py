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
import os, json
import numpy as np
import mujoco
from k1env import ACT_JOINTS, KP, KD, DEFAULT_POSE
from k1env_speed import RefLib
from k1env_walk3 import K1Walk3Batch
from legkin import LegFK

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
    ALIGN_MAX = int(os.environ.get('ALIGN_MAX', '6'))
    ALIGN_CHECK = os.environ.get('ALIGN_CHECK', '0') == '1'
    PLACE_FF = os.environ.get('PLACE_FF', '0') == '1'
    PLACE_GAIN = float(os.environ.get('PLACE_GAIN', '1.0'))
    # ---- stage w56e (user: stop quickly, re-stance after ~1 s; in-place turning with the human rhythm) ----
    IP_RHYTHM = os.environ.get('IP_RHYTHM', '0') == '1'      # in-place turning: step period / yaw per step from mocap
    RESTANCE = os.environ.get('RESTANCE_ENV', '0') == '1'     # re-stance state machine inside the env (training)
    RS_DELAY = 1.0; RS_MAX = int(os.environ.get('RS_MAX', '2'))
    RS_DX = float(os.environ.get('RS_DX', '0.025')); RS_DYAW = np.radians(float(os.environ.get('RS_DYAW_DEG', '4'))); RS_DSEP = float(os.environ.get('RS_DSEP', '0.03'))
    RS_OBS = os.environ.get('RS_OBS', '0') == '1'             # w56i: re-stance flag + stance error (FK) as inputs
    W_RSPLACE = float(os.environ.get('W_RSPLACE', '0.0'))     # w56i: reward for each re-stance touchdown at the standing position
    P_STAGGER = float(os.environ.get('P_STAGGER', '0.0'))     # standing resets with staggered feet
    W_LIFT = float(os.environ.get('W_LIFT', '0.0'))           # reward: swing foot really lifted (in-place stepping)
    W_PIVOT = float(os.environ.get('W_PIVOT', '0.0'))         # penalty: loaded foot spinning about the vertical
    PLACE_LAT = os.environ.get('PLACE_LAT', '0') == '1'
    PLACE_GAIN_YAW = float(os.environ.get('PLACE_GAIN_YAW', os.environ.get('PLACE_GAIN', '1.0')))   # yaw overshot with 1.5
    RS_FF_UP = os.environ.get('RS_FF_UP', '0') == '1'         # re-placement already while the re-stance steps start      # re-placement also corrects the stance width
    SYM_REF = os.environ.get('SYM_REF', '0') == '1'          # v5.6.3: left/right symmetric reference (make_sym_ref.py)
    KP_STAND_MIN = float(os.environ.get('KP_STAND_MIN', '0'))  # v5.6.3: stiffness floor while standing (x nominal)
    KD_STAND_MIN = float(os.environ.get('KD_STAND_MIN', '0'))
    LIFT_PEN = float(os.environ.get('LIFT_PEN', '0.0'))       # w56f: penalty weight for a loaded reference-swing foot

    def __init__(self, n, **kw):
        self._v56 = False
        super().__init__(n, **kw)
        self.lib = RefLib(self.m, path=os.path.join(HERE, 'ref_lib_56_sym.npz' if self.SYM_REF else 'ref_lib_56.npz'))
        self.home_cnt = np.zeros(n, int)
        self.P_arm = np.zeros(n)
        self._prev_ph = self.ph.copy()
        self.ds_dx = np.zeros(n); self.ds_dyaw = np.zeros(n); self.corr = np.zeros((n, 12)); self.ds_dy = np.zeros(n)
        # per-robot stop settings (the gait manager changes them for the re-stance after standing, gait56.py)
        self.home_req = np.full(n, self.HOME_STEPS); self.align_req = np.full(n, self.ALIGN_CHECK)
        self.place_on = np.full(n, self.PLACE_FF)
        self.rs = np.zeros(n, int); self.rs_n = np.zeros(n, int); self.stand_t = np.zeros(n)
        self.foot_yaw_prev = None
        self.unl_t = np.zeros((n, 2))
        rh = json.load(open(os.path.join(HERE, 'mocap', 'turn_rhythm.json')))
        # human step period per turn rate (median of each bin), scaled to K1 by Froude similarity sqrt(0.70 / 0.90)
        self.rh_w = np.array([np.mean(b['w_rad_s']) if b['w_rad_s'][1] < 2 else 1.1 for b in rh['bins']])
        self.rh_T = np.array([b['step_period_s'] for b in rh['bins']]) * np.sqrt(0.70 / 0.90)
        c0 = self.lib.contact[0]
        self.f_ds0 = float(((c0[:, 0] > 0.5).any(1) & (c0[:, 1] > 0.5).any(1)).mean())
        self.legfk = LegFK(self.m)
        self.sep0 = float(self.legfk.foot_offsets(DEFAULT_POSE[None, :12])[1][0])
        self.aadr = np.array([self.m.jnt_qposadr[self.m.joint(ACT_JOINTS[i]).id] for i in ARM_ALL])
        self.aadr_d = np.array([self.m.jnt_dofadr[self.m.joint(ACT_JOINTS[i]).id] for i in ARM_ALL])
        self._v56 = True
        self.reset(np.arange(n))

    def reset(self, ids):
        super().reset(ids)
        if getattr(self, '_v56', False) and len(ids):
            self.home_cnt[ids] = 0
            self._prev_ph[ids] = self.ph[ids]
            self.ds_dx[ids] = 0; self.ds_dyaw[ids] = 0; self.corr[ids] = 0; self.ds_dy[ids] = self.sep0 if hasattr(self, "sep0") else 0.25
            self.rs[ids] = 0; self.rs_n[ids] = 0; self.stand_t[ids] = 0; self.unl_t[ids] = 0
            self.home_req[ids] = self.HOME_STEPS; self.align_req[ids] = self.ALIGN_CHECK; self.place_on[ids] = self.PLACE_FF
            if self.RESTANCE:
                self._rs_off(ids)
            if self.P_STAGGER > 0:
                self._stagger(ids)
            if self.foot_yaw_prev is not None:
                fx = np.stack([self.s('left_foot_x')[ids], self.s('right_foot_x')[ids]], 1)
                self.foot_yaw_prev[ids] = np.arctan2(fx[:, :, 1], fx[:, :, 0])

    # ---- 4. re-place the feet before closing the legs ----
    def foot_offsets(self):
        """front-back offset [m], separation [m] and relative yaw [rad] of the feet (left minus right), in the frame of
        the feet's mean heading. Forward kinematics of the 12 leg joint angles only (legkin.py) - computable from
        joint encoders on a real robot; no simulator world poses. (Before the fix, pelvis-frame ankle positions were
        rotated by the world-frame foot heading, which was wrong whenever the robot did not face the world x axis.)"""
        return self.legfk.foot_offsets(self.qpos()[:, self.qadr[:12]])

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
        if self.PLACE_LAT:
            aligned &= np.abs(self.ds_dy - self.sep0) < 0.03
        ok = np.where(self.align_req, ((self.home_cnt >= self.home_req) & aligned) | (self.home_cnt >= self.ALIGN_MAX),
                      self.home_cnt >= self.home_req)
        return ok | (self.alpha < 1.0)

    def _stagger(self, ids):
        """w56e: some standing starts get a staggered stance (feet planted front-back / turned), as left after a quick
        stop, so that the re-stance is trained. Joint angles from inverse kinematics (legkin), pelvis height kept."""
        m = self.m; d = self.datas[0]
        sel = [i for i in ids if self.alpha[i] == 0 and self.rng.random() < self.P_STAGGER]
        for i in sel:
            dx = self.rng.uniform(-0.15, 0.15); dyaw = self.rng.uniform(-0.35, 0.35)
            q = DEFAULT_POSE[None, :12].repeat(1, 0).copy()
            for side, k0, sg in (('left', 0, 1.0), ('right', 6, -1.0)):
                dq, _ = self.legfk.ik_shift(side, q[:, k0:k0 + 6], np.array([[sg * dx / 2, 0, 0]]), np.array([sg * dyaw / 2]))
                q[:, k0:k0 + 6] += dq
            mujoco.mj_resetData(m, d)
            d.qpos[:] = m.key_qpos[0]; d.qpos[self.qadr] = DEFAULT_POSE; d.qpos[self.qadr[:12]] = q[0]
            d.qpos[2] = self.ref.z_stand + 0.003
            mujoco.mj_forward(m, d)
            st = np.zeros(self.nstate); mujoco.mj_getState(m, d, st, self.spec_state)
            self.state[i] = st; self.sdata[i] = d.sensordata

    # ---- re-stance (w56e): stop quickly, re-place the feet after standing RS_DELAY s without a command ----
    def _rs_off(self, ids):
        self.home_req[ids] = 0; self.align_req[ids] = False; self.place_on[ids] = False

    def misaligned(self):
        dx, dy, dyaw = self.foot_offsets()
        return (np.abs(dx) > self.RS_DX) | (np.abs(dyaw) > self.RS_DYAW) | (np.abs(dy - self.sep0) > self.RS_DSEP)

    def restance_update(self, idle):
        """idle: robots without a user command (no walking / running wanted). Returns the robots the re-stance
        controls this step (their cmd / v_cmd / w_cmd are set here). Used by the gait manager and in training."""
        mis = self.misaligned()
        standing = idle & (self.alpha == 0) & (self.cmd < 0.5) & (self.rs == 0)
        self.stand_t = np.where(standing, self.stand_t + 1.0 / self.CTRL_HZ, np.where(self.rs == 0, 0.0, self.stand_t))
        go = standing & (self.stand_t >= self.RS_DELAY) & (self.rs_n < self.RS_MAX) & mis
        for i in np.where(go)[0]:
            self.rs[i] = 1; self.rs_n[i] += 1; self.brake[i] = 0.0
            self.home_req[i] = 2; self.align_req[i] = True; self.place_on[i] = True
            self.yaw_t[i] = self.yaw()[i]
        s1 = (self.rs == 1) & idle
        self.cmd[s1] = 1.0; self.v_cmd[s1] = 0.0; self.w_cmd[s1] = 0.0
        up = s1 & (self.alpha >= 1.0)
        self.cmd[up] = 0.0; self.rs[up] = 2                     # stop rule: >= 2 steps with the re-placement
        done = (self.rs == 2) & (self.alpha == 0)
        self.rs[done] = 0; self.stand_t[done] = 0.0; self._rs_off(np.where(done)[0])
        return (self.rs > 0) & idle

    def restance_cancel(self, ids):
        self.rs[ids] = 0; self.rs_n[ids] = 0; self.stand_t[ids] = 0
        if self.RESTANCE:
            self._rs_off(ids)
        else:
            self.home_req[ids] = self.HOME_STEPS; self.align_req[ids] = self.ALIGN_CHECK; self.place_on[ids] = self.PLACE_FF

    def _schedule(self):
        if self.RESTANCE and not self.scripted:
            busy = self.restance_update((self.cmd < 0.5) | (self.rs > 0))
            busy &= self.brake < 0.5
            # keep events away while re-placing the feet; a new command afterwards ends the standing period
            self.next_evt = np.where(busy & (self.next_evt <= self.t + 1), self.t + 25, self.next_evt)
            ev = (self.t == self.next_evt) & (self.rs == 0)
            if ev.any():
                self.rs_n[ev] = 0; self.stand_t[ev] = 0
        super()._schedule()

    def obs(self):
        oa, oc = super().obs()
        if not (self.RS_OBS and getattr(self, '_v56', False)):
            return oa, oc
        dx, dy, dyaw = self.foot_offsets()
        ex = np.stack([(self.rs > 0).astype(float), np.clip(dx / 0.1, -2, 2), np.clip((dy - self.sep0) / 0.1, -2, 2),
                       np.clip(dyaw / 0.3, -2, 2)], 1).astype(np.float32)
        na = oa.shape[1]
        return np.concatenate([oa, ex], 1), np.concatenate([oc[:, :na], ex, oc[:, na:]], 1)

    # ---- in-place turning rhythm (w56e) ----
    def _ip_active(self):
        k_ip = np.clip(1.0 - self.v_ref / 0.4, 0.0, 1.0) * self.alpha
        return k_ip, np.clip((np.abs(self.w_ref) - 0.05) / 0.1, 0, 1)

    def _step_T(self):
        return np.interp(np.abs(self.w_ref), self.rh_w, self.rh_T)

    def _rate_mult(self, Tv):
        if not self.IP_RHYTHM:
            return 1.0
        k_ip, aw = self._ip_active()
        T0 = Tv / 2; Ts = self._step_T()
        Tsw = T0 * (1 - self.f_ds0); Tds = T0 * self.f_ds0
        c = self.lib.sample(self.ph, self.v_lib)['contact']
        ds = (c[:, 0] > 0.5).any(1) & (c[:, 1] > 0.5).any(1)
        m_ds = Tds / np.maximum(Ts - Tsw, 1e-3)               # slower double support -> longer pause between steps
        m = np.where(Ts >= T0, np.where(ds, m_ds, 1.0), T0 / Ts)
        return 1.0 + (k_ip * aw) * (m - 1.0)

    def _ip_dpsi(self, Tv, k_ip):
        if not self.IP_RHYTHM:
            return 0.5 * self.w_ref * Tv * k_ip
        _, aw = self._ip_active()
        T0 = Tv / 2
        Teff = T0 + aw * (self._step_T() - T0)
        return self.w_ref * Teff * k_ip                      # yaw per step = w * step period (human rhythm)

    def _place_ff(self, tgt):
        """foot re-placement feed-forward while stepping in place to stop (user decision, option 2).
        * offsets (front-back, relative yaw) of the feet at the last double support (forward kinematics, latched;
          used by ALIGN_CHECK). Double support = foot contact sensors (heel / ball / toe on each foot;
          user decision: the real robot will get foot contact sensors).
        * the swing foot is placed beside the stance foot: forward kinematics of the stance leg (measured angles) and
          of the swing leg (current joint target) give the swing target seen from the stance foot; its front-back
          offset and relative yaw are removed with inverse kinematics of the swing leg (legkin.ik_shift; height,
          sideways distance, pitch and roll of the foot kept). Gain PLACE_GAIN (the policy partly fights it)."""
        touch = np.concatenate([self.s(f'{s}_{p}_touch') for s in ('left', 'right') for p in ('heel', 'meta', 'toe')], 1)
        ds = (touch[:, :3].sum(1) > 5) & (touch[:, 3:].sum(1) > 5)
        q12 = self.qpos()[:, self.qadr[:12]]
        dx, dy, dyaw = self.legfk.foot_offsets(q12)
        self.ds_dx = np.where(ds, dx, self.ds_dx); self.ds_dyaw = np.where(ds, dyaw, self.ds_dyaw)
        self.ds_dy = np.where(ds, dy, self.ds_dy)
        stopping = ((self.cmd < 0.5) | ((self.rs > 0) & self.RS_FF_UP)) & (self.v_ref < 0.05) & (np.abs(self.w_ref) < 0.05) & (self.alpha > 0) & self.place_on
        c = self.lib.sample(self.ph, self.v_lib)['contact']                  # reference schedule (n, leg, heel/fore)
        swl = ~(c[:, 0] > 0.5).any(1) & stopping; swr = ~(c[:, 1] > 0.5).any(1) & stopping
        want = np.zeros((self.n, 12))
        for side, sw, k0, o0 in (('left', swl, 0, 6), ('right', swr, 6, 0)):
            if not sw.any():
                continue
            i = np.where(sw)[0]
            po, Ro = self.legfk.fk('right' if side == 'left' else 'left', q12[i, o0:o0 + 6])   # stance foot (measured)
            ps, Rs = self.legfk.fk(side, tgt[i, k0:k0 + 6])                                    # swing foot (target)
            d = np.einsum('nji,nj->ni', Ro, ps - po)                    # swing target seen from the stance foot
            Rrel = np.swapaxes(Ro, 1, 2) @ Rs
            ryaw = np.arctan2(Rrel[:, 1, 0], Rrel[:, 0, 0])
            sx = np.clip(-self.PLACE_GAIN * d[:, 0], -0.15, 0.15); sy = np.clip(-self.PLACE_GAIN_YAW * ryaw, -0.3, 0.3)
            dp = Ro[:, :, 0] * sx[:, None]                               # along the stance foot's heading
            if self.PLACE_LAT:      # also the standing width, so that closing the legs does not push the feet apart
                want_y = (1.0 if side == 'left' else -1.0) * self.sep0
                sl = np.clip(self.PLACE_GAIN * (want_y - d[:, 1]), -0.08, 0.08)
                dp = dp + Ro[:, :, 1] * sl[:, None]
            dq, res = self.legfk.ik_shift(side, tgt[i, k0:k0 + 6], dp, sy)
            want[i, k0:k0 + 6] = np.clip(dq, -0.4, 0.4)
        self.corr += 0.2 * (want - self.corr)
        tgt[:, :12] += self.corr

    def _stiff_floor(self):
        """v5.6.3 (user): while standing the joints must not go limp (the w56j policy set hip-yaw and ankle-roll
        stiffness to 0 to save copper loss). Floor = KP_STAND_MIN x nominal gain, faded out with the gait amplitude
        alpha, so while walking (alpha = 1) a joint may still go fully soft for a moment (inertia-driven motion)."""
        if self.KP_STAND_MIN > 0:
            f = (1.0 - self.alpha)[:, None]
            self.kp_scale = np.maximum(self.kp_scale, self.KP_STAND_MIN * f)
            self.kd_scale = np.maximum(self.kd_scale, self.KD_STAND_MIN * f)

    # ---- 3. arm residuals ----
    def _extra_targets(self, tgt, a):
        tgt[:, ARM] += ARM_SCALE * a[:, 37:41]
        if self.PLACE_FF:
            self._place_ff(tgt)
        else:
            self.corr[:] = 0
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
        # w56e: in-place stepping must really step (the w56c policy shuffled / pivoted with one foot on the ground)
        loaded = np.stack([touch[:, 0:3].sum(1), touch[:, 3:6].sum(1)], 1) > 5.0
        r_lift = np.zeros(self.n); pivot = np.zeros(self.n)
        if self.W_LIFT > 0:
            k_ip, _ = self._ip_active()
            c = self.lib.sample(self.ph, self.v_lib)['contact']
            sw = ~(c > 0.5).any(2)                                              # (n, foot) reference swing
            hz = ank[:, :, 2] - ank[:, ::-1, 2]                                  # foot height above the other foot
            ok = (~loaded) & (hz > 0.03)
            # + for a lifted swing foot, - for a swing foot still carrying load (stage w56f; w56e had only the bonus
            # and kept shuffling the inner foot)
            r_lift = (((sw & ok).sum(1) - self.LIFT_PEN * (sw & loaded).sum(1)) / np.maximum(sw.sum(1), 1)) \
                * (sw.any(1)) * (k_ip > 0.5) * (self.alpha > 0.9)
            ex = ex + self.W_LIFT * r_lift
        fx = np.stack([self.s('left_foot_x'), self.s('right_foot_x')], 1)
        fyaw = np.arctan2(fx[:, :, 1], fx[:, :, 0])
        if self.foot_yaw_prev is not None and self.W_PIVOT > 0:
            wz = np.arctan2(np.sin(fyaw - self.foot_yaw_prev), np.cos(fyaw - self.foot_yaw_prev)) * self.CTRL_HZ
            pivot = (loaded * np.clip(wz, -5, 5) ** 2).sum(1)
            ex = ex - self.W_PIVOT * pivot
        self.foot_yaw_prev = fyaw
        r_rsp = np.zeros(self.n)
        if self.W_RSPLACE > 0:
            td = loaded & (self.unl_t >= 0.1 * self.CTRL_HZ)
            self.unl_t = np.where(loaded, 0, self.unl_t + 1)
            dx, dy, dyaw = self.foot_offsets()
            q = np.exp(-(dx / 0.06) ** 2 - (dyaw / 0.15) ** 2 - ((dy - self.sep0) / 0.06) ** 2)
            r_rsp = td.any(1) * (self.rs > 0) * q
            ex = ex + self.W_RSPLACE * r_rsp
        return ex, dict(r_level=r_level, roll_deg=np.degrees(np.abs(np.arcsin(np.clip(g[:, 1], -1, 1)))),
                        r_home=r_home, r_align=r_align, r_lift=r_lift, pivot=pivot, r_rsp=r_rsp, rs=(self.rs > 0).astype(float), P_arm=self.P_arm, P_leg=self.P_leg, arm_act=np.abs(a[:, 37:41]).mean(1))
