# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Leg forward / inverse kinematics from joint angles only (what a real robot has: joint encoders).

The chain parameters (link offsets, joint axes) are read once from the model; they are the same numbers as the
<origin>/<axis> entries of the URDF, so the same code runs on encoder angles of a real K1. No simulator state
(world poses, sensors) is used. Frame: pelvis.

foot_offsets(): front-back offset, separation and relative yaw of the two feet, expressed in the frame of the feet's
mean heading. Only relative quantities of the two legs are used, so the result does not depend on the pelvis
orientation (twist, roll) or on the robot's heading in the world.
"""
import numpy as np

LEG = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')


def _rot(axis, q):
    """batched rotation about a unit axis (x, y or z), q: (n,)"""
    c, s = np.cos(q), np.sin(q)
    R = np.zeros(q.shape + (3, 3))
    k = int(np.argmax(np.abs(axis))); sg = np.sign(axis[k])
    s = s * sg
    i, j = [(1, 2), (2, 0), (0, 1)][k]
    R[..., k, k] = 1; R[..., i, i] = c; R[..., j, j] = c; R[..., i, j] = -s; R[..., j, i] = s
    return R


class LegFK:
    def __init__(self, m):
        self.chain = {}
        for side in ('left', 'right'):
            links = []
            for jn in LEG:
                j = m.joint(f'{side}_{jn}_joint').id
                b = m.jnt_bodyid[j]
                assert np.allclose(m.body_quat[b], [1, 0, 0, 0]) and np.allclose(m.jnt_pos[j], 0)
                links.append((m.body_pos[b].copy(), m.jnt_axis[j].copy()))
            self.chain[side] = links

    def fk(self, side, q):
        """q: (n, 6) joint angles of one leg (LEG order) -> ankle_roll frame position (n,3), rotation (n,3,3)"""
        n = q.shape[0]
        p = np.zeros((n, 3)); R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        for k, (off, ax) in enumerate(self.chain[side]):
            p = p + R @ off
            R = R @ _rot(ax, q[:, k])
        return p, R

    def feet(self, q12):
        pl, Rl = self.fk('left', q12[:, 0:6]); pr, Rr = self.fk('right', q12[:, 6:12])
        return pl, Rl, pr, Rr

    def foot_offsets(self, q12):
        """(dx, dy, dyaw): left minus right, in the frame of the feet's mean heading (feet assumed flat)"""
        pl, Rl, pr, Rr = self.feet(q12)
        Rrel = np.swapaxes(Rr, 1, 2) @ Rl                       # left foot orientation seen from the right foot
        dyaw = np.arctan2(Rrel[:, 1, 0], Rrel[:, 0, 0])
        d = np.einsum('nji,nj->ni', Rr, pl - pr)                # offset in the right-foot frame
        c, s = np.cos(dyaw / 2), np.sin(dyaw / 2)               # rotate to the mean heading
        return c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], dyaw

    def heading(self, q12):
        """mean heading of the feet in the pelvis frame (yaw angle)"""
        pl, Rl, pr, Rr = self.feet(q12)
        yl = np.arctan2(Rl[:, 1, 0], Rl[:, 0, 0]); yr = np.arctan2(Rr[:, 1, 0], Rr[:, 0, 0])
        return yr + 0.5 * np.arctan2(np.sin(yl - yr), np.cos(yl - yr))

    def ik_shift(self, side, q, dp, dyaw, iters=3, eps=1e-4, lam=1e-3):
        """inverse kinematics: joint change dq (n,6) that moves the foot of `side` from its pose at joint angles q by
        dp (n,3, pelvis frame) and rotates it by dyaw about the pelvis z axis; other foot orientation components kept.
        Damped least squares with a finite-difference Jacobian (6x6 per robot), a few Newton iterations."""
        p0, R0 = self.fk(side, q)
        Rz = _rot(np.array([0, 0, 1.0]), dyaw)
        pt = p0 + dp; Rt = Rz @ R0
        qq = q.copy()

        def err(qx):
            p, R = self.fk(side, qx)
            Re = Rt @ np.swapaxes(R, 1, 2)                       # rotation still to go (pelvis frame)
            w = 0.5 * np.stack([Re[:, 2, 1] - Re[:, 1, 2], Re[:, 0, 2] - Re[:, 2, 0], Re[:, 1, 0] - Re[:, 0, 1]], 1)
            return np.concatenate([pt - p, w], 1)

        for _ in range(iters):
            e = err(qq)
            J = np.zeros((q.shape[0], 6, 6))
            for k in range(6):
                dq = np.zeros_like(qq); dq[:, k] = eps
                J[:, :, k] = (e - err(qq + dq)) / eps            # d(pose)/dq
            A = np.swapaxes(J, 1, 2) @ J + lam * np.eye(6)
            qq = qq + np.linalg.solve(A, np.einsum('nji,nj->ni', J, e)[..., None])[..., 0]
        return qq - q, np.abs(err(qq)).max(1)
