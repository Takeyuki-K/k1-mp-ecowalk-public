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



def under_speed(env, vx, v_t, active, tau=0.5):
    """ankle-spring variant (user: never save power by slowing down): one-sided penalty on the speed deficit of
    the ~tau-s moving average of the measured heading-frame speed vs the moving average of the target speed."""
    k = 1.0 / (tau * env.CTRL_HZ)
    if not hasattr(env, '_vxe') or env._vxe.shape[0] != env.n:
        env._vxe = np.array(vx, float); env._vte = np.array(v_t, float)
    rs = env.t <= 1                                  # just reset
    env._vxe = np.where(rs, vx, env._vxe + k * (vx - env._vxe))
    env._vte = np.where(rs, v_t, env._vte + k * (v_t - env._vte))
    err = env._vte - env._vxe
    if os.environ.get('SPD_SYM', '1') == '1':          # stage C: symmetric (stage B one-sided -> overshoot)
        err = np.abs(err)
    return np.clip(err - float(os.environ.get('SPD_DB', '0.0')), 0, None) * active


class K1Walk4Batch(K1Walk3Batch):
    nact = 41
    W_LEVEL = float(os.environ.get('W_LEVEL', '0.35'))
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
    W_UPRIGHT = float(os.environ.get('W_UPRIGHT', '0.0'))     # ankle-spring variant: upright-torso reward weight
    UP_K = float(os.environ.get('UP_K', '100.0'))
    W_UNDER = float(os.environ.get('W_UNDER', '0.0'))         # ankle-spring variant: under-speed penalty weight
    LIFT_PEN = float(os.environ.get('LIFT_PEN', '0.0'))       # w56f: penalty weight for a loaded reference-swing foot

    def __init__(self, n, **kw):
        self._v56 = False
        super().__init__(n, **kw)
        self.lib = RefLib(self.m, path=os.path.join(HERE, os.environ.get('REF_LIB', 'ref_lib_56_sym.npz' if self.SYM_REF else 'ref_lib_56.npz')))
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
        # stage O (arm relaxation, local): ARM_KPM / ARM_KDM scale the arm servo gains of this (walking) model only;
        # ARM_KP_ACT=1: per-side arm stiffness chosen by the policy through the unused ankle-roll Kp action slots
        # (17 = left, 23 = right; ankle roll has no motor in the spring variant), applied as a sample-and-hold
        # reduction of the servo error at each 50 Hz step: s = clip(ARM_KP0 + 0.5 a, ARM_KP_LO, 1)
        self.arm_kpm = float(os.environ.get('ARM_KPM', '1')); self.arm_kdm = float(os.environ.get('ARM_KDM', '1'))
        self.arm_kp = KP[ARM_ALL] * self.arm_kpm; self.arm_kd = KD[ARM_ALL] * self.arm_kdm
        if self.arm_kpm != 1 or self.arm_kdm != 1:
            ia = [self.m.actuator(ACT_JOINTS[i] + '_motor').id for i in ARM_ALL]
            for mm in {id(x): x for x in list(self.models) + [self.base_model]}.values():
                mm.actuator_gainprm[ia, 0] = self.arm_kp; mm.actuator_biasprm[ia, 1] = -self.arm_kp
                mm.actuator_biasprm[ia, 2] = -self.arm_kd
        self.ARM_KP_ACT = os.environ.get('ARM_KP_ACT', '0') == '1'
        self.ARM_KP0 = float(os.environ.get('ARM_KP0', '0.5')); self.ARM_KP_LO = float(os.environ.get('ARM_KP_LO', '0.15'))
        self.ARM_SWING = float(os.environ.get('ARM_SWING', '1'))
        self.arm_s = np.ones((n, 2))
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
        if self.ARM_SWING != 1.0:     # stage O: scale the reference arm swing about the default arm pose
            d0 = DEFAULT_POSE[ARM_ALL]
            tgt[:, ARM_ALL] = d0 + self.ARM_SWING * (tgt[:, ARM_ALL] - d0)
        tgt[:, ARM] += ARM_SCALE * a[:, 37:41]
        if self.ARM_KP_ACT:
            self.arm_s = np.clip(self.ARM_KP0 + 0.5 * a[:, [17, 23]], self.ARM_KP_LO, 1.0)
            q0 = self.qpos()[:, self.aadr]
            s5 = np.repeat(self.arm_s, 5, axis=1)
            tgt[:, ARM_ALL] = q0 + s5 * (tgt[:, ARM_ALL] - q0)
        if self.PLACE_FF:
            self._place_ff(tgt)
        else:
            self.corr[:] = 0
        if self.IP_WIDE:
            # in-place turning was learned on the 21 cm stance (v3/v5.5); the 16 cm stance of ref_lib_56 made
            # -1 rad/s in-place turns fall (w56a). Undo the narrowing while stepping in place, fade out by 0.4 m/s.
            k_ip = np.clip(1.0 - self.v_ref / 0.4, 0.0, 1.0) * self.alpha
            dl = np.arctan2((0.209 - float(os.environ.get('REF_W', '0.16'))) / 2, 0.70) * k_ip
            tgt[:, 1] += dl; tgt[:, 5] -= dl; tgt[:, 7] -= dl; tgt[:, 11] += dl

    def _extra_power(self, st, tgt):
        m = self.m
        q = st[:, :, 1:1 + m.nq][:, :, self.aadr]
        qd = st[:, :, 1 + m.nq:1 + m.nq + m.nv][:, :, self.aadr_d]
        kp = self.arm_kp; kd = self.arm_kd
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
        # ankle-spring variant (local): upright torso (pelvis/trunk pitch and roll toward vertical, roll toward the
        # physical turn lean phi only), no dead band; weight W_UPRIGHT (0 = v5.6.3)
        # stage D: lateral sway inside +-UP_ROLL_DB is free (v5.6 S3: a strictly level pelvis costs hip-roll copper loss)
        rdb = float(os.environ.get('UP_ROLL_DB', '0.0'))
        re_ = g[:, 1] - np.sin(self.phi); re_ = np.sign(re_) * np.clip(np.abs(re_) - rdb, 0, None)
        tilt2 = g[:, 0] ** 2 + re_ ** 2
        r_upright = np.exp(-self.UP_K * tilt2)
        ex = ex + self.W_UPRIGHT * r_upright
        self._tilt_deg = np.degrees(np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2))
        vw = self.qvel()[:, 0:3]
        vx = np.cos(self.yaw_t) * vw[:, 0] + np.sin(self.yaw_t) * vw[:, 1]
        deficit = under_speed(self, vx, self.alpha * self.v_ref, (self.alpha > 0.9) & (self.v_ref > 0.3) & (self.brake < 0.5))
        ex = ex - self.W_UNDER * deficit
        # stage I (inertia / step-to-step transition, Kuo 2002, Donelan 2002): penalise joint negative mechanical work
        # (collision + braking; free in the electrical model but it has to be re-supplied as positive work) and the
        # effort of the swing leg (hip pitch + knee copper loss and positive work while the foot is unloaded), so the
        # swing becomes pendulum-like.  W_NEGW, W_SWING (0 = off).
        w_nw = float(os.environ.get('W_NEGW', '0.0')); w_sw = float(os.environ.get('W_SWING', '0.0'))
        if (w_nw > 0 or w_sw > 0) and hasattr(self, 'tau_sub'):
            nq_, nv_ = self.m.nq, self.m.nv
            qd_s = self.st_out[:, :, 1 + nq_:1 + nq_ + nv_][:, :, self.dadr[:12]]
            pm = self.tau_sub * qd_s
            if w_nw > 0:
                ex = ex - w_nw * (-np.clip(pm, None, 0)).sum(2).mean(1)
            if w_sw > 0:
                from k1env_eco import KM as _KM
                sw_eff = np.zeros(self.n)
                for side, js in ((0, (0, 3)), (1, (6, 9))):
                    e = sum((self.tau_sub[:, :, j] ** 2 / _KM[j] ** 2 + np.clip(pm[:, :, j], 0, None)).mean(1) for j in js)
                    ld = touch[:, 3 * side:3 * side + 3].sum(1) > 5.0
                    sw_eff = sw_eff + e * (~ld)
                ex = ex - w_sw * sw_eff
        # stage J: landing-impact penalty (peak foot force above 1.05 body weight), W_LAND (0 = off)
        w_ld = float(os.environ.get('W_LAND', '0.0'))
        if w_ld > 0:
            ex = ex - w_ld * np.clip(self.impact / (35.7 * 9.81) - 1.05, 0, None)
        # stage H: extra penalty on hip-roll copper loss (tau^2 / Km^2, Km = 4 Nm/sqrt(W)) on top of the energy term
        w_hr = float(os.environ.get('W_HIPROLL', '0.0'))
        if w_hr > 0 and hasattr(self, 'tau_sub'):
            ex = ex - w_hr * (self.tau_sub[:, :, [1, 7]] ** 2 / 16.0).sum(2).mean(1)
        # stage K: (a) left/right hip-roll copper imbalance (EMA ~1.5 s per side), W_HRSYM; (b) knee copper loss at
        # high commanded walking speed (weight ramps 0 -> 1 over v_ref 0.9 -> 1.4 m/s), W_KNEEHI (0 = off)
        w_hs = float(os.environ.get('W_HRSYM', '0.0')); w_kh = float(os.environ.get('W_KNEEHI', '0.0'))
        if (w_hs > 0 or w_kh > 0) and hasattr(self, 'tau_sub'):
            from k1env_eco import KM as _KM
            if w_hs > 0:
                hr = (self.tau_sub[:, :, [1, 7]] ** 2 / _KM[[1, 7]] ** 2).mean(1)
                if not hasattr(self, '_hr_ema') or self._hr_ema.shape[0] != self.n:
                    self._hr_ema = hr.copy()
                self._hr_ema = 0.985 * self._hr_ema + 0.015 * hr
                ex = ex - w_hs * np.abs(self._hr_ema[:, 0] - self._hr_ema[:, 1])
            if w_kh > 0:
                kc = (self.tau_sub[:, :, [3, 9]] ** 2 / _KM[[3, 9]] ** 2).sum(2).mean(1)
                ramp = np.clip((self.alpha * self.v_ref - 0.9) / 0.5, 0, 1)
                ex = ex - w_kh * ramp * kc
        # stage M (metapen recipe for lateral pelvis sway, 2026-11-19): balance by foot placement, not stiffness.
        #  W_FOOTPL: Gaussian reward on |stance-foot lateral offset from the pelvis| - FOOT_Y (0.055 m), sigma FOOT_SIG,
        #            at touchdown and during single support while walking
        #  W_KPHR:   penalty on hip-roll kp scale above 0.75 (levelness must not be bought with stiffness)
        #  W_ROLLRATE: penalty on pelvis roll angular velocity^2;  W_ARMROLL: shoulder-roll deviation from default pose^2
        w_fp = float(os.environ.get('W_FOOTPL', '0.0')); w_kp = float(os.environ.get('W_KPHR', '0.0'))
        w_rr = float(os.environ.get('W_ROLLRATE', '0.0')); w_ar = float(os.environ.get('W_ARMROLL', '0.0'))
        walking_m = (self.alpha > 0.9) & (self.v_ref > 0.3)
        if w_fp > 0:
            ld2 = np.stack([touch[:, 0:3].sum(1), touch[:, 3:6].sum(1)], 1) > 5.0
            if not hasattr(self, '_prev_ld2') or self._prev_ld2.shape[0] != self.n:
                self._prev_ld2 = ld2.copy()
            single = ld2 & ~ld2[:, ::-1]
            td = ld2 & ~self._prev_ld2
            act = single | td
            fy, sg = float(os.environ.get('FOOT_Y', '0.055')), float(os.environ.get('FOOT_SIG', '0.015'))
            rf = np.exp(-((np.abs(ank[:, :, 1]) - fy) ** 2) / (2 * sg ** 2))
            r_fp = (rf * act).sum(1) / np.maximum(act.sum(1), 1)
            ex = ex + w_fp * r_fp * walking_m
            self._prev_ld2 = ld2.copy()
        if w_kp > 0:
            ex = ex - w_kp * np.clip(self.kp_scale[:, [1, 7]] - 0.75, 0, None).sum(1)
        # stage N: double-support hip-roll squeeze (both hip rolls pushing against each other through the ground:
        # Mc1 +-10 Nm in double support, 39 W vs 19 W for v5.6.3). W_SQZ x min(|tauL|, |tauR|) when signs oppose in DS
        w_sq = float(os.environ.get('W_SQZ', '0.0'))
        if w_sq > 0 and hasattr(self, 'tau_sub'):
            tl_, tr_ = self.tau_sub[:, :, 1], self.tau_sub[:, :, 7]
            sq = np.where(tl_ * tr_ < 0, np.minimum(np.abs(tl_), np.abs(tr_)), 0.0).mean(1)
            ds_ = (touch[:, 0:3].sum(1) > 5.0) & (touch[:, 3:6].sum(1) > 5.0)
            ex = ex - w_sq * sq * ds_
        if w_rr > 0:
            ex = ex - w_rr * self.qvel()[:, 3] ** 2
        if w_ar > 0:
            qs = self.qpos()[:, self.qadr[[14, 19]]] - DEFAULT_POSE[[14, 19]]
            ex = ex - w_ar * (qs ** 2).sum(1)
        # stage O (arm relaxation): W_ARMP extra price per arm watt; W_ARMCLR one-sided penalty on shoulder adduction
        # (arm closing toward the torso / thigh) beyond ARMCLR_M rad inward of the default pose (collision margin)
        w_ap = float(os.environ.get('W_ARMP', '0.0')); w_ac = float(os.environ.get('W_ARMCLR', '0.0'))
        if w_ap > 0:
            ex = ex - w_ap * self.P_arm
        if w_ac > 0:
            qr = self.qpos()[:, self.qadr[[14, 19]]]
            inward = np.stack([DEFAULT_POSE[14] - qr[:, 0], qr[:, 1] - DEFAULT_POSE[19]], 1)
            ex = ex - w_ac * np.clip(inward - float(os.environ.get('ARMCLR_M', '0.05')), 0, None).sum(1)
        self._deficit = deficit
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
        # stage K2 (1.4 m/s diagnosis: weak pre-emptive push-off 5.0 vs 8.6 J, ~1 forefoot re-contact per step):
        # W_PUSH rewards trailing-ankle positive power in late stance (heel unloaded, forefoot loaded) while walking;
        # W_SCUFF penalises a loaded foot during the reference swing phase while walking (0 = off)
        w_pu = float(os.environ.get('W_PUSH', '0.0')); w_sc = float(os.environ.get('W_SCUFF', '0.0'))
        walking = (self.alpha > 0.9) & (self.v_ref > 0.3)
        # stage L: extra push-off bonus at high walking speed, W_PUSHHI x ramp(v_ref 0.9 -> 1.4)
        w_ph = float(os.environ.get('W_PUSHHI', '0.0'))
        if w_ph > 0:
            w_pu = w_pu + w_ph * np.clip((self.alpha * self.v_ref - 0.9) / 0.5, 0, 1)
        if np.any(w_pu > 0) and hasattr(self, 'tau_sub'):
            nq_, nv_ = self.m.nq, self.m.nv
            qd_s = self.st_out[:, :, 1 + nq_:1 + nq_ + nv_][:, :, self.dadr[:12]]
            for side, j in ((0, 4), (1, 10)):
                late = (touch[:, 3 * side] < 5.0) & (touch[:, 3 * side + 1:3 * side + 3].sum(1) > 5.0)
                ex = ex + w_pu * np.clip(self.tau_sub[:, :, j] * qd_s[:, :, j], 0, None).mean(1) * late * walking
        # W_BOUNCE: heel-strike bounce (1.4 m/s: foot unloads for 5-10 ms right after heel strike, then the forefoot
        # slaps = lost heel rocker). Penalise any unloading of a foot during the reference stance phase while walking.
        w_bo = float(os.environ.get('W_BOUNCE', '0.0'))
        if w_bo > 0:
            if not hasattr(self, '_prev_loaded') or self._prev_loaded.shape[0] != self.n:
                self._prev_loaded = loaded.copy()
            cb = self.lib.sample(self.ph, self.v_lib)['contact']
            st_ref = (cb > 0.5).any(2)
            lift = self._prev_loaded & ~loaded & st_ref
            ex = ex - w_bo * lift.sum(1) * walking
            self._prev_loaded = loaded.copy()
        if w_sc > 0:
            c_ = self.lib.sample(self.ph, self.v_lib)['contact']
            sw_ = ~(c_ > 0.5).any(2)
            ex = ex - w_sc * (sw_ & loaded).sum(1) * walking
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
        return ex, dict(v_deficit=deficit, r_upright=r_upright, tilt_deg=self._tilt_deg, r_level=r_level, roll_deg=np.degrees(np.abs(np.arcsin(np.clip(g[:, 1], -1, 1)))),
                        r_home=r_home, r_align=r_align, r_lift=r_lift, pivot=pivot, r_rsp=r_rsp, rs=(self.rs > 0).astype(float), P_arm=self.P_arm, P_leg=self.P_leg, arm_s=self.arm_s.mean(1), arm_act=np.abs(a[:, 37:41]).mean(1))
