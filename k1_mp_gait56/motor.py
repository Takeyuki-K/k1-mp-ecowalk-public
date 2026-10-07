# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Motor torque-speed limit as a drop-in wrapper around mujoco.rollout.Rollout.

Leg actuation is 3 actuators per joint (see k1env_eco): ctrl_leg = Kp*q_target, Kp, Kd -> tau = ctrl_leg - Kp q - Kd qd.
Before every physics step the wrapper evaluates tau and the joint speed; if the motor cannot deliver tau (motoring
quadrant above the torque-speed curve), all three terms of that joint are scaled by the same factor, so the implicit
damping of the PD stays intact. Curve (K1 URDF: 96.9 Nm / 11.5 rad/s, ankle roll 47.3 Nm / 20.9 rad/s):
full torque up to 60 % of the speed limit, linear to 0 at the limit; braking torque always available.
"""
import numpy as np
from k1env_eco import TLIM
from k1env import ACT_JOINTS

QD_LIM = np.array([20.9 if j.endswith('ankle_roll_joint') else 11.5 for j in ACT_JOINTS[:12]])
KNEE_FRAC = 0.6
NA = len(ACT_JOINTS)            # 23 position / torque channels, then 12 Kp, then 12 Kd, then extras


def motor_limit(tau, qd):
    """returns (scale f for the PD, direct torque, use_direct mask).
    Along the direction of motion the motor can give at most TLIM * (w_max - |w|) / (0.4 w_max), clipped to
    [-TLIM, TLIM]; above w_max this is negative, i.e. back-EMF braking (DC-motor behaviour), so the joint cannot
    keep running faster than the limit. Against the motion (braking) the full torque is available."""
    w = np.abs(qd); s = np.sign(qd)
    lim = TLIM * np.clip((QD_LIM - w) / ((1 - KNEE_FRAC) * QD_LIM), -1.0, 1.0)
    tpar = tau * s
    over = tpar > lim
    f = np.where(over & (lim > 0), lim / np.maximum(np.abs(tau), 1e-6), 1.0)
    direct = over & (lim <= 0)
    return f, lim * s, direct


def motor_scale(tau, qd):
    f, _, direct = motor_limit(tau, qd)
    return np.where(direct, 0.0, f)


class MotorRollout:
    def __init__(self, roll, qadr, dadr, nq, nv):
        self.roll, self.qadr, self.dadr, self.nq, self.nv = roll, qadr, dadr, nq, nv
        self.enabled = True
        self.sat = None

    def rollout(self, models, datas, state0, ctrl, nstep, skip_checks=True, state=None, sensordata=None):
        if not self.enabled:
            return self.roll.rollout(models, datas, state0, ctrl, nstep=nstep, skip_checks=skip_checks,
                                     state=state, sensordata=sensordata)
        n = state0.shape[0]
        st = state0.copy()
        s1 = np.zeros((n, 1, state.shape[2])); d1 = np.zeros((n, 1, sensordata.shape[2]))
        sat = np.zeros(n)
        for k in range(nstep):
            c = ctrl[:, k].copy()
            q = st[:, 1:1 + self.nq][:, self.qadr[:12]]
            qd = st[:, 1 + self.nq:1 + self.nq + self.nv][:, self.dadr[:12]]
            kp = c[:, NA:NA + 12]; kd = c[:, NA + 12:NA + 24]
            tau = c[:, :12] - kp * q - kd * qd
            f, tdir, direct = motor_limit(tau, qd)
            sat += ((f < 0.999) | direct).any(1)
            c[:, :12] *= f; c[:, NA:NA + 12] *= f; c[:, NA + 12:NA + 24] *= f
            c[:, :12] = np.where(direct, tdir, c[:, :12])
            c[:, NA:NA + 12] = np.where(direct, 0.0, c[:, NA:NA + 12])
            c[:, NA + 12:NA + 24] = np.where(direct, 0.0, c[:, NA + 12:NA + 24])
            self.roll.rollout(models, datas, np.ascontiguousarray(st), np.ascontiguousarray(c[:, None]), nstep=1,
                              skip_checks=skip_checks, state=s1, sensordata=d1)
            st = s1[:, 0].copy()
            state[:, k] = s1[:, 0]; sensordata[:, k] = d1[:, 0]
        self.sat = sat / nstep
