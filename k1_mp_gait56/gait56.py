# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5.6 gait manager (v5.6 walking policy: level-ish pelvis, arm residuals, feet re-aligned before standing;
running policy unchanged from v5.5). Original v5.5 docstring:

v5.5 gait manager: walking (v5.5 walk) <-> running (v5.5 run) with a yaw-rate command and hard braking.

Same hand-over rules as v5 (gait.py), extended for turning:
  * the decision walk -> run uses the *running* speed governor: wanted running speed = min(v_user, A_LAT_RUN / |w|).
    If that is >= V_UP the walking policy accelerates to 1.65 m/s (its own governor limits the turn rate meanwhile)
    and hands over at the next right heel strike; turn rate and heading target are carried over.
  * run -> walk when the wanted running speed drops below V_DOWN (slow command or a turn too tight for running):
    the run slows to the hand-over speed and the walking policy takes over at the next right touchdown, again with
    turn rate and heading target carried over.
  * stop():  normal stop (run slows at 1.5 m/s^2, then walking safe stop)
  * brake(): hard braking (run brakes at 4 m/s^2 with the learned braking posture, then walking safe stop)
"""
import os
import numpy as np, torch, mujoco
from k1env import DEFAULT_POSE
from k1env_walk4 import K1Walk4Batch as K1Walk3Batch
from k1env_run4 import K1Run4Batch
from turnphys import A_LAT_RUN
from ppo_run4 import ACEco

V_UP, V_DOWN, V_HAND = 1.8, 1.7, 1.8
V_WALK_MAX = 1.65
# re-stance (user decision after the v5.6 review): stop quickly without re-placing the feet; when the robot has been
# standing for 1 s without a speed command and the feet are not at the standing position (joint-angle FK), step in
# place with the FK + IK re-placement until aligned (or 6 steps), then close the legs again (K1Walk4Batch.restance_*).
RESTANCE = os.environ.get('RESTANCE', '1') == '1'


def _load(path, env, nact):
    oa, oc = env.obs()
    net = ACEco(oa.shape[1], oc.shape[1], nact)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    return net


def _act(net, oa):
    with torch.no_grad():
        return net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)


class Gait56:
    def __init__(self, walk_path, run_path, n=1, seed=5, nthread=1):
        torch.set_num_threads(1)
        K1Walk3Batch.P_RUN = 0.0
        E = os.environ.get
        C = K1Walk3Batch
        # deployment settings (TRAINING_HISTORY.md, REPORT_GAIT56.md S13-S20); each can be overridden for experiments
        C.ALIGN_CHECK = E('ALIGN_CHECK', '0') == '1'; C.IP_WIDE = True; C.ROLL_DEADBAND = 0.045
        C.PLACE_FF = E('PLACE_FF', '1') == '1'; C.PLACE_GAIN = float(E('PLACE_GAIN', '1.5'))
        C.PLACE_GAIN_YAW = float(E('PLACE_GAIN_YAW', '1.0')); C.PLACE_LAT = E('PLACE_LAT', '1') == '1'
        C.RESTANCE = RESTANCE; C.P_STAGGER = 0.0; C.RS_FF_UP = E('RS_FF_UP', '1') == '1'
        C.RS_MAX = int(E('RS_MAX', '1')); C.ALIGN_MAX = int(E('ALIGN_MAX', '4'))
        C.RS_DX = float(E('RS_DX', '0.03')); C.RS_DYAW = np.radians(float(E('RS_DYAW_DEG', '6'))); C.RS_DSEP = float(E('RS_DSEP', '0.04'))
        C.IP_RHYTHM = E('IP_RHYTHM', '1') == '1'
        # v5.6.3: left/right symmetric references for both policies (make_sym_ref.py); SYM_REF=0 for v5.6.2 and older
        os.environ['SYM_REF'] = E('SYM_REF', '1'); C.SYM_REF = os.environ['SYM_REF'] == '1'
        C.KP_STAND_MIN = float(E('KP_STAND_MIN', '0.3')); C.KD_STAND_MIN = float(E('KD_STAND_MIN', '0.5'))
        K1Run4Batch.P_STAND = 0.0; K1Run4Batch.P_WALK = 0.0
        n_in0 = torch.load(walk_path, map_location='cpu')['model']['actor.0.weight'].shape[1]
        K1Walk3Batch.RS_OBS = n_in0 == 85                       # w56i policies see the re-stance state
        self.W = K1Walk3Batch(n, stage=2, randomize=False, seed=seed, ep_len=10 ** 9, nthread=nthread)
        self.R = K1Run4Batch(n, v_lo=2.0, v_hi=2.0, stage=2, randomize=False, seed=seed + 1, ep_len=10 ** 9,
                             nthread=nthread)
        for e in (self.W, self.R):
            e.pushes = False; e.scripted = True
        self.R.assist = 0.0
        # walking policies from stage w55c have the extra brake input (77 inputs)
        n_in = torch.load(walk_path, map_location='cpu')['model']['actor.0.weight'].shape[1]
        self.W.use_brake = n_in == self.W.obs()[0].shape[1] + 1
        self.nw = _load(walk_path, self.W, self.W.nact)
        self.nr = _load(run_path, self.R, 36)
        self.n = n
        # v5.6.3: running speed governor v |w| <= 2.5 m/s^2 at deployment (trained with 3.0). At the 3.0 limit
        # (3 m/s at 1 rad/s, lean target 17 deg) the symmetric running policy over-leaned to 22-27 deg and fell in
        # 54/64 runs of the full profile; with 2.5: 64/64 (REPORT S27). A_LAT_RUN=3.0 restores v5.5 behaviour.
        self.a_lat_run = float(os.environ.get('A_LAT_RUN', '2.5'))
        self.R.a_lat = self.a_lat_run
        self.reset()

    # ------------------------------------------------------------ state
    def reset(self):
        W, m = self.W, self.W.m
        d = mujoco.MjData(m)
        for i in range(self.n):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = m.key_qpos[0]; d.qpos[W.qadr] = DEFAULT_POSE; d.qpos[2] = W.ref.z_stand + 0.002
            mujoco.mj_forward(m, d)
            st = np.zeros(W.nstate); mujoco.mj_getState(m, d, st, W.spec_state)
            W.state[i] = st; W.sdata[i] = d.sensordata
        for k in ('ph', 'alpha', 'cmd', 'v_cmd', 'v_ref', 'w_cmd', 'w_ref'):
            getattr(W, k)[:] = 0
        W.yaw_t[:] = W.yaw()
        W.t[:] = 0; W.last_a[:] = 0; W.last_a2[:] = 0; W.next_evt[:] = 10 ** 9
        self.R.next_evt[:] = 10 ** 9
        self.mode = np.zeros(self.n, int)          # 0 walk/stand, 1 run
        self.v_user = np.zeros(self.n); self.w_user = np.zeros(self.n)
        self.go = np.zeros(self.n, bool)           # walking / running wanted (False = stop requested)
        self.braking = np.zeros(self.n, bool)
        self.calm = np.zeros(self.n, int)          # running cycles completed since the last braking
        self.switches = []
        self.W.restance_cancel(np.arange(self.n))
        self.t = 0
        self._oaw, _ = self.W.obs(); self._oar, _ = self.R.obs()

    @staticmethod
    def _copy(src, dst, i):
        dst.state[i] = src.state[i].copy(); dst.sdata[i] = src.sdata[i].copy()
        dst.last_a[i] = 0; dst.last_a2[i] = 0
        dst.air[i] = False; dst.air_time[i] = 0

    @property
    def env(self):
        """environment that currently simulates robot 0"""
        return self.R if self.mode[0] == 1 else self.W

    # ------------------------------------------------------------ commands
    def command(self, v, w):
        self.v_user[:] = max(0.0, float(v)); self.w_user[:] = float(w)
        self.go[:] = True; self.braking[:] = False
        self.W.brake[:] = 0.0
        self.W.restance_cancel(np.arange(self.n))

    def stop(self):
        self.go[:] = False; self.braking[:] = False

    def brake(self):
        self.go[:] = False; self.braking[:] = True

    def _run_wanted(self, i):
        aw = abs(self.w_user[i])
        return min(self.v_user[i], self.a_lat_run / aw) if aw > 1e-3 else self.v_user[i]

    # ------------------------------------------------------------ step
    def step(self):
        W, R = self.W, self.R
        busy = W.restance_update(~self.go & (self.mode == 0)) if RESTANCE else np.zeros(self.n, bool)
        for i in range(self.n):
            u = self._run_wanted(i) if self.go[i] else 0.0
            if busy[i]:
                self.braking[i] = False
                continue
            if self.mode[i] == 0:
                if self.braking[i] and self.W.use_brake and W.alpha[i] > 0:
                    W.brake[i] = 1.0; W.cmd[i] = 0.0; W.v_cmd[i] = 0.0; W.w_cmd[i] = 0.0   # learned hard stop
                elif not self.go[i]:
                    # as v5: keep walking while slowing down, close the legs only once slow (this is the situation
                    # the walking policy was trained on: a stop event during walking, not at the hand-over itself)
                    if W.cmd[i] > 0.5:
                        W.v_cmd[i] = 0.0; W.w_cmd[i] = 0.0
                        if W.v_ref[i] <= 0.35:
                            W.cmd[i] = 0.0
                elif u >= V_UP:
                    if W.alpha[i] == 0 and W.cmd[i] < 0.5 and abs(self.w_user[i]) < 0.15:   # stand -> run (straight)
                        self._copy(W, R, i)
                        R.alpha[i] = 0.0; R.ph[i] = 0.0; R.v_ref[i] = 0.0; R.v_cmd[i] = u
                        R.w_cmd[i] = self.w_user[i]; R.w_ref[i] = 0.0; R.brake[i] = 0.0; R.yaw_t[i] = W.yaw()[i]
                        self.calm[i] = 0
                        self.mode[i] = 1; self.switches.append((self.t, 'stand->run'))
                    else:
                        W.cmd[i] = 1.0; W.v_cmd[i] = V_WALK_MAX; W.w_cmd[i] = self.w_user[i]
                else:
                    W.cmd[i] = 1.0; W.v_cmd[i] = min(self.v_user[i], V_WALK_MAX); W.w_cmd[i] = self.w_user[i]
            else:
                if self.braking[i]:
                    R.brake[i] = 1.0; R.v_cmd[i] = V_HAND; R.w_cmd[i] = 0.0
                elif not self.go[i] or u < V_DOWN:
                    R.v_cmd[i] = V_HAND; R.w_cmd[i] = self.w_user[i] if self.go[i] else 0.0
                else:
                    R.v_cmd[i] = self.v_user[i]; R.w_cmd[i] = self.w_user[i]
        ph_w, ph_r = W.ph.copy(), R.ph.copy()
        a_w = _act(self.nw, self._oaw); a_r = _act(self.nr, self._oar)
        _, tw, _, iw = W.step(a_w)
        _, tr, _, ir = R.step(a_r)
        term = np.where(self.mode == 0, tw, tr)
        wrap_w = W.ph < ph_w - 0.5; wrap_r = R.ph < ph_r - 0.5
        # after hard braking the body posture differs from steady running (upright trunk, flexed knees); the walking
        # policy was trained on hand-overs from steady running, so run one steady cycle at 1.8 m/s before handing over
        self.calm = np.where(R.brake > 0.5, 0, self.calm + wrap_r.astype(int))
        for i in range(self.n):
            u = self._run_wanted(i) if self.go[i] else 0.0
            if self.mode[i] == 0 and self.go[i] and u >= V_UP and wrap_w[i] and W.v_ref[i] >= 1.55:     # walk -> run
                self._copy(W, R, i)
                R.alpha[i] = 1.0; R.ph[i] = 0.0; R.v_ref[i] = 2.0; R.v_cmd[i] = u
                R.w_ref[i] = W.w_ref[i]; R.w_cmd[i] = self.w_user[i]; R.yaw_t[i] = W.yaw_t[i]; R.brake[i] = 0.0
                self.calm[i] = 0
                self.mode[i] = 1; self.switches.append((self.t, 'walk->run'))
            elif (self.mode[i] == 1 and (not self.go[i] or u < V_DOWN) and wrap_r[i]
                  and R.v_ref[i] <= V_HAND + 0.05 and R.alpha[i] >= 1 and self.calm[i] >= 2):                                   # run -> walk
                self._copy(R, W, i)
                W.cmd[i] = 1.0
                W.alpha[i] = 1.0; W.ph[i] = 0.0; W.v_ref[i] = 1.6
                W.v_cmd[i] = min(self.v_user[i], V_WALK_MAX) if self.go[i] else 0.0
                W.w_ref[i] = float(np.clip(R.w_ref[i], -0.6, 0.6)); W.w_cmd[i] = self.w_user[i] if self.go[i] else 0.0
                W.yaw_t[i] = R.yaw_t[i]
                if self.braking[i] and W.use_brake:
                    W.brake[i] = 1.0; W.cmd[i] = 0.0; W.v_cmd[i] = 0.0; W.w_ref[i] = 0.0
                self.mode[i] = 0; self.switches.append((self.t, 'run->walk' + ('' if self.go[i] else ' (stop)')))
        self._oaw, _ = W.obs(); self._oar, _ = R.obs()
        self.t += 1
        return term, (iw if self.mode[0] == 0 else ir)

    @property
    def rs(self):
        return self.W.rs

    # ------------------------------------------------------------ views
    def qpos(self):
        return np.where(self.mode[:, None] == 0, self.W.qpos(), self.R.qpos())

    def active(self, attr):
        a = np.asarray(getattr(self.W, attr)); b = np.asarray(getattr(self.R, attr))
        return np.where(self.mode.reshape((-1,) + (1,) * (a.ndim - 1)) == 0, a, b)
