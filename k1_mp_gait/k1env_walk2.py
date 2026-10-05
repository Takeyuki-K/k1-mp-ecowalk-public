# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5 walking env = fast-walk env (eco reward kept) + motor torque-speed limit + run -> walk hand-over starts.

20 % of episodes start from recorded running states (running policy at 1.8-2.3 m/s, right touchdown); the walking
reference starts at right heel strike with v_ref = 1.6 m/s (speed jump allowed by the user), the command is a
walking speed or a stop.
"""
import os
import numpy as np
import mujoco
from k1env_speed import K1SpeedBatch
from motor import MotorRollout

HERE = os.path.dirname(os.path.abspath(__file__))


def yaw_of(q):
    w, x, y, z = q[3], q[4], q[5], q[6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class K1Walk2Batch(K1SpeedBatch):
    P_RUN = float(os.environ.get('P_RUN', '0.2'))     # robustness stage: 0.5
    P_STOP = float(os.environ.get('P_STOP', '0.15'))   # v5 robustness stage: 0.3 (more stops / standing under pushes)

    def __init__(self, n, **kw):
        bank = os.path.join(HERE, 'run_bank.npz')
        self.bank = np.load(bank) if os.path.exists(bank) else None
        self.entry = np.zeros(n, int)
        super().__init__(n, **kw)
        self.roll = MotorRollout(self.roll, self.qadr, self.dadr, self.m.nq, self.m.nv)

    def reset(self, ids):
        super().reset(ids)
        if self.bank is None or not hasattr(self, 'v_ref') or len(ids) == 0:
            return
        m = self.m; d = self.datas[0]
        u = self.rng.random(len(ids))
        self.entry[ids] = 0
        for j, i in enumerate(ids):
            if u[j] >= self.P_RUN:
                continue
            b = self.rng.integers(len(self.bank['qpos']))
            q = self.bank['qpos'][b].copy(); v = self.bank['qvel'][b].copy()
            mujoco.mj_resetData(m, d)
            d.qpos[:] = q; d.qvel[:] = v
            mujoco.mj_forward(m, d)
            st = np.zeros(self.nstate); mujoco.mj_getState(m, d, st, self.spec_state)
            self.state[i] = st; self.sdata[i] = d.sensordata
            self.entry[i] = 1
            self.cmd[i] = 1.0; self.alpha[i] = 1.0; self.ph[i] = 0.0
            self.v_ref[i] = 1.6
            self.v_cmd[i] = self.rng.uniform(1.0, 1.65) if self.rng.random() < 0.7 else self.rng.uniform(0.3, 1.0)
            self.yaw_t[i] = yaw_of(q)
            self.last_a[i] = 0; self.last_a2[i] = 0
            if self.rng.random() < 0.25:              # stop soon after the hand-over
                self.next_evt[i] = self.t[i] + self.rng.integers(50, 120)
