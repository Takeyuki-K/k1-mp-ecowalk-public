# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Gait manager: one robot (or a batch) driven by two learned policies, switching like a human does.

  walking policy (eco reward, heel strike, 0.3-1.65 m/s, safe stop)  <->  running policy (mid/forefoot, 2-5 m/s)

Rules (user instruction: walking adjusts in 0.1-0.2 m/s steps; the walk -> run switch may jump 1.6 -> 2.0 m/s):
  * user command >= V_UP (1.8 m/s) while walking: walk accelerates to 1.65 m/s; at the next right heel strike with
    v_ref >= 1.55 m/s the robot hands over to the running policy (running cycle starts at right touchdown, v_ref 2.0)
  * user command >= V_UP while standing: direct stand -> run start (running policy blends from the standing pose)
  * user command < V_DOWN (1.7 m/s) while running: run decelerates to 2.0 m/s; at the next right touchdown it hands
    over to the walking policy (v_ref 1.6 m/s), which then follows the command or stops
  * user command 0 while walking: decelerate to 0.3 m/s, then the walking policy's stop (feet together)
Both policies were trained with these hand-over states (walk_bank.npz / run_bank.npz).
"""
import numpy as np, torch, mujoco
from k1env import DEFAULT_POSE
from k1env_walk2 import K1Walk2Batch
from k1env_run2 import K1Run2Batch
from ppo_eco import AC

V_UP, V_DOWN = 1.8, 1.7
V_HAND = 1.8          # running slows to this reference speed before handing over to walking (G9)
torch.set_num_threads(1)


def load(path, env, nact):
    oa, oc = env.obs()
    net = AC(oa.shape[1], oc.shape[1], nact)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    return net


def act(net, oa):
    with torch.no_grad():
        return net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)


class Gait:
    def __init__(self, N, walk_path, run_path, push=False, seed=5):
        K1Walk2Batch.P_RUN = 0.0
        K1Run2Batch.P_STAND = 0.0; K1Run2Batch.P_WALK = 0.0
        self.W = K1Walk2Batch(N, stage=2, randomize=False, seed=seed, ep_len=10 ** 9, v_range=(0.3, 1.65), nthread=1)
        self.R = K1Run2Batch(N, v_lo=2.0, v_hi=2.0, stage=2, randomize=False, seed=seed + 1, nthread=1, ep_len=10 ** 9)
        for e in (self.W, self.R):
            e.pushes = push; e.scripted = True
        self.R.assist = 0.0
        self.R.push_mag = 0.3
        self.nw = load(walk_path, self.W, self.W.nact)
        self.nr = load(run_path, self.R, 36)
        self.N = N
        self.mode = np.zeros(N, int)          # 0 walk/stand, 1 run
        self.alive = np.ones(N, bool)
        self.switches = []
        self._stand()

    def _stand(self):
        W = self.W; m = W.m; d = mujoco.MjData(m)
        for i in range(self.N):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = m.key_qpos[0]; d.qpos[W.qadr] = DEFAULT_POSE; d.qpos[2] = W.ref.z_stand + 0.002
            mujoco.mj_forward(m, d)
            st = np.zeros(W.nstate); mujoco.mj_getState(m, d, st, W.spec_state)
            W.state[i] = st; W.sdata[i] = d.sensordata
        for k in ('ph', 'alpha', 'cmd', 'v_cmd', 'v_ref', 'yaw_t'):
            getattr(W, k)[:] = 0
        W.t[:] = 0; W.last_a[:] = 0; W.last_a2[:] = 0; W.next_evt[:] = 10 ** 9
        self.R.next_evt[:] = 10 ** 9

    @staticmethod
    def _copy(src, dst, i):
        dst.state[i] = src.state[i].copy(); dst.sdata[i] = src.sdata[i].copy()
        dst.last_a[i] = 0; dst.last_a2[i] = 0
        dst.air[i] = False; dst.air_time[i] = 0

    def _yaw(self, q):
        w, x, y, z = q[3], q[4], q[5], q[6]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def step(self, v_user, t):
        W, R = self.W, self.R
        v_user = np.broadcast_to(np.asarray(v_user, float), (self.N,))
        for i in range(self.N):
            u = v_user[i]
            if self.mode[i] == 0:
                if u >= V_UP:
                    if W.alpha[i] == 0 and W.cmd[i] < 0.5:              # standing -> run
                        self._copy(W, R, i)
                        R.alpha[i] = 0.0; R.ph[i] = 0.0; R.v_ref[i] = 0.0; R.v_cmd[i] = u
                        self.mode[i] = 1; self.switches.append((t, i, 'stand->run'))
                    else:
                        W.cmd[i] = 1.0; W.v_cmd[i] = 1.65
                elif u > 0:
                    W.cmd[i] = 1.0; W.v_cmd[i] = float(np.clip(u, 0.3, 1.65))
                else:
                    if W.cmd[i] > 0.5:
                        W.v_cmd[i] = 0.3
                        if W.v_ref[i] <= 0.35:
                            W.cmd[i] = 0.0
            else:
                R.v_cmd[i] = u if u >= V_DOWN else V_HAND
        ph_w, ph_r = W.ph.copy(), R.ph.copy()
        oaw, _ = W.obs(); oar, _ = R.obs()
        _, tw, _, iw = W.step(act(self.nw, oaw))
        _, tr, _, ir = R.step(act(self.nr, oar))
        term = np.where(self.mode == 0, tw, tr)
        self.alive &= ~term
        wrap_w = W.ph < ph_w - 0.5; wrap_r = R.ph < ph_r - 0.5
        for i in range(self.N):
            u = v_user[i]
            if self.mode[i] == 0 and u >= V_UP and wrap_w[i] and W.v_ref[i] >= 1.55:      # walk -> run
                self._copy(W, R, i)
                R.alpha[i] = 1.0; R.ph[i] = 0.0; R.v_ref[i] = 2.0; R.v_cmd[i] = u
                self.mode[i] = 1; self.switches.append((t, i, 'walk->run'))
            elif self.mode[i] == 1 and u < V_DOWN and wrap_r[i] and R.v_ref[i] <= V_HAND + 0.05 and R.alpha[i] >= 1:  # run -> walk
                self._copy(R, W, i)
                W.cmd[i] = 1.0; W.alpha[i] = 1.0; W.ph[i] = 0.0; W.v_ref[i] = 1.6
                W.v_cmd[i] = float(np.clip(u, 0.3, 1.65)) if u > 0 else 0.3
                W.yaw_t[i] = self._yaw(W.state[i, 1:1 + W.m.nq])
                self.mode[i] = 0; self.switches.append((t, i, 'run->walk'))
        return iw, ir

    def active(self, attr):
        a = np.asarray(getattr(self.W, attr)); b = np.asarray(getattr(self.R, attr))
        return np.where(self.mode.reshape((-1,) + (1,) * (a.ndim - 1)) == 0, a, b)

    def qpos(self):
        return np.where(self.mode[:, None] == 0, self.W.qpos(), self.R.qpos())
