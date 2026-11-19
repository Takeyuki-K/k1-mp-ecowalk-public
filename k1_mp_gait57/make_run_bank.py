# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Record physics states of the running policy at right touchdown (phase wrap) while running at the hand-over command (1.75-1.9 m/s)
(hand-over points run -> walk). python3 make_run_bank.py runs/rn1/model.pt run_bank.npz"""
import sys
import numpy as np, torch
torch.set_num_threads(1)
from k1env_run2 import K1Run2Batch
from ppo_eco import AC
K1Run2Batch.P_STAND = 0.0; K1Run2Batch.P_WALK = 0.0
N = 64
env = K1Run2Batch(N, v_lo=2.0, v_hi=2.0, stage=1, seed=31, nthread=1, ep_len=10 ** 9)
env.pushes = False; env.assist = 0.0; env.scripted = True
oa, oc = env.obs()
net = AC(oa.shape[1], oc.shape[1], 36); net.load_state_dict(torch.load(sys.argv[1], map_location='cpu')['model']); net.eval()
env.reset(np.arange(N)); env.v_cmd[:] = np.random.default_rng(0).uniform(1.75, 1.9, N)
oa, _ = env.obs()
alive = np.ones(N, bool); Q, V = [], []
for k in range(500):
    ph0 = env.ph.copy()
    with torch.no_grad():
        a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
    _, term, _, _ = env.step(a); alive &= ~term
    wrap = (env.ph < ph0 - 0.5) & alive & (k > 150)
    for i in np.where(wrap)[0]:
        Q.append(env.qpos()[i].copy()); V.append(env.qvel()[i].copy())
    oa, _ = env.obs()
Q, V = np.array(Q), np.array(V); Q[:, 0] = 0; Q[:, 1] = 0
print('states', len(Q), 'alive', alive.mean(), 'speed', V[:, 0].mean())
np.savez(sys.argv[2], qpos=Q, qvel=V)
