# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Quick robustness check: standing-only, walking, stand->walk->stop survival over 64 envs (no DR)."""
import sys, numpy as np, torch
from k1env import K1Batch
from ppo import AC


def test(path, mode, T=400, n=64, randomize=False, pushes=False):
    e = K1Batch(n, stage=2, randomize=randomize, seed=11)
    e.pushes = pushes
    e.stage = 1 if mode == 'walk' else 2
    if mode != 'walk':
        # all standing starts
        import mujoco
        from k1env import DEFAULT_POSE
        m = e.m; d = mujoco.MjData(m)
        for i in range(n):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = m.key_qpos[0]; d.qpos[e.qadr] = DEFAULT_POSE; d.qpos[2] = e.ref.z_stand + 0.003
            d.qpos[e.qadr[:12]] += e.rng.normal(0, 0.02, 12)
            mujoco.mj_forward(m, d)
            st = np.zeros(e.nstate); mujoco.mj_getState(m, d, st, e.spec_state)
            e.state[i] = st; e.sdata[i] = d.sensordata
        e.alpha[:] = 0; e.cmd[:] = 0; e.ph[:] = 0
    else:
        e.reset(np.arange(n))
    e.stage = 2
    e.switch_t[:] = {'stand': 10 ** 9, 'walk': -1, 'swalk': 75}[mode]
    e.stop_t[:] = 10 ** 9 if mode != 'swalk' else 325
    oa, oc = e.obs()
    net = AC(oa.shape[1], oc.shape[1], 12); net.load_state_dict(torch.load(path)['model']); net.eval()
    alive = np.ones(n, bool)
    for t in range(T):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        r, te, tr, info = e.step(a)
        alive &= ~te
        oa, oc = e.obs()
    return alive.mean()


if __name__ == '__main__':
    p = sys.argv[1]
    for mode in ['stand', 'walk', 'swalk']:
        print(mode, 'survival %.2f' % test(p, mode), flush=True)
    print('swalk DR survival %.2f' % test(p, 'swalk', randomize=True))
    print('stand +pushes survival %.2f' % test(p, 'stand', pushes=True))
    print('swalk DR+pushes survival %.2f' % test(p, 'swalk', randomize=True, pushes=True))
