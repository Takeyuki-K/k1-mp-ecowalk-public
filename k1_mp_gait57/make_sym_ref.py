# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Left/right symmetric reference libraries for the mirror-symmetric policies (v5.6.3).

The human-derived references are not symmetric: the mirrored pose at phase + 1/2 differs from the pose at phase by
2 deg on average (max 17 deg) for walking and 6 deg (max 31 deg) for running. A mirror-symmetry loss on the policy
would then fight the imitation reward. Each cycle is replaced by the average of itself and its mirror image shifted by
half a cycle:  Q'(ph) = 1/2 (Q(ph) + mirror(Q(ph + 1/2)))   (joints swapped with the signs of mirror.py, base y and
the roll / yaw parts of the base orientation mirrored, base x kept). Contact schedules: left kept, right = left
shifted by half a cycle.
python3 make_sym_ref.py   -> ref_lib_56_sym.npz, ref_sprint_lib_sym.npz
"""
import os
import numpy as np
from k1env import load_model
from mirror import mirror_state

HERE = os.path.dirname(os.path.abspath(__file__))


def sym_lib(src, dst, m):
    L = dict(np.load(os.path.join(HERE, src)))
    Q = L['qpos']; S, M, nq = Q.shape
    assert M % 2 == 0
    h = M // 2
    Qs = Q.copy(); err = []
    for s in range(S):
        for i in range(M):
            j = (i + h) % M
            qm, _ = mirror_state(m, Q[s, j], np.zeros(m.nv))
            q = 0.5 * (Q[s, i] + qm)
            q[0] = Q[s, i, 0]                                          # forward progression unchanged
            quat = q[3:7] / np.linalg.norm(q[3:7]); q[3:7] = quat
            Qs[s, i] = q
            err.append(np.abs(Q[s, i, 7:] - qm[7:]).max())
    for k in ('cl', 'cr'):
        L[k] = L[k].copy()
    cl = L['cl'].copy()
    L['cr'] = np.roll(cl, -h, axis=1)                                   # right(ph) = left(ph + 1/2)
    # base height re-solved so the lowest stance-foot site is as high as in the original frame (as make_ref56.py)
    import mujoco
    d = mujoco.MjData(m)
    sites = {k: [m.site(f'{sd}_{p}_site').id for p in ('heel', 'meta')] for k, sd in ((0, 'left'), (1, 'right'))}
    def low(q, c):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        z = [d.site_xpos[k][2] for leg in (0, 1) if c[leg].any() for k in sites[leg]]
        return min(z) if z else np.nan
    dz = []
    for s in range(S):
        for i in range(M):
            c_new = np.stack([np.atleast_1d(L['cl'][s, i]), np.atleast_1d(L['cr'][s, i])]).astype(bool)
            z0 = low(Q[s, i], c_new); z1 = low(Qs[s, i], c_new)
            if not np.isnan(z0):
                Qs[s, i, 2] += z0 - z1; dz.append(z0 - z1)
    print(f'  base height correction: mean {100 * np.mean(np.abs(dz)):.2f} cm, max {100 * np.max(np.abs(dz)):.2f} cm')
    L['qpos'] = Qs
    np.savez(os.path.join(HERE, dst), **L)
    print(f'{src} -> {dst}: max left/right mismatch before {np.degrees(max(err)):.1f} deg')


def check(dst, m):
    L = np.load(os.path.join(HERE, dst)); Q = L['qpos']; S, M, _ = Q.shape; h = M // 2
    e = max(np.abs(Q[s, i, 7:] - mirror_state(m, Q[s, (i + h) % M], np.zeros(m.nv))[0][7:]).max()
            for s in range(S) for i in range(M))
    c = np.abs(L['cr'].astype(float) - np.roll(L['cl'], -h, axis=1).astype(float)).max()
    print(f'  {dst}: mirror mismatch after {np.degrees(e):.2e} deg, contact mismatch {c}')


if __name__ == '__main__':
    m = load_model()
    for a, b in (('ref_lib_56.npz', 'ref_lib_56_sym.npz'), ('ref_sprint_lib.npz', 'ref_sprint_lib_sym.npz')):
        sym_lib(a, b, m); check(b, m)
