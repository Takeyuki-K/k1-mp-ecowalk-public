# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Record physics states of the fast-walk policy at right heel strike (gait phase wrap) while walking 1.45-1.65 m/s.
These states are the hand-over points walk -> run for training the running policy (k1_mp_gait).
python3 make_walk_bank.py runs/final/model.pt ../k1_mp_gait/walk_bank.npz"""
import sys
import numpy as np, torch
torch.set_num_threads(1)
from k1env_speed import K1SpeedBatch
from eval_speed import load, act, stand_all
N = 64
env = K1SpeedBatch(N, stage=2, randomize=False, seed=21, ep_len=10 ** 9, v_range=(0.3, 1.65), nthread=1)
env.pushes = False; env.scripted = True
net = load(sys.argv[1], env)
stand_all(env)
rng = np.random.default_rng(0)
env.cmd[:] = 1.0; env.v_cmd[:] = rng.uniform(1.45, 1.65, N)
oa, _ = env.obs()
alive = np.ones(N, bool); Q, V, VR = [], [], []
nq, nv = env.m.nq, env.m.nv
for k in range(600):
    ph0 = env.ph.copy()
    _, term, _, _ = env.step(act(net, oa)); alive &= ~term
    wrap = (env.ph < ph0 - 0.5) & alive & (k > 250)
    for i in np.where(wrap)[0]:
        Q.append(env.qpos()[i].copy()); V.append(env.qvel()[i].copy()); VR.append(env.v_ref[i])
    oa, _ = env.obs()
Q, V = np.array(Q), np.array(V)
Q[:, 0] = 0.0; Q[:, 1] = 0.0                    # translate to origin (keep heading / orientation)
print('states', len(Q), 'alive', alive.mean(), 'speed', V[:, 0].mean())
np.savez(sys.argv[2], qpos=Q, qvel=V, v_ref=np.array(VR))
