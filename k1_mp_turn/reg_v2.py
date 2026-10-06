# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Straight-walking regression baseline: v2 speed policy (../k1_mp_speed, unchanged) under exactly the same
protocol as eval_turn.py 'straight' (dt 5 ms, 8 robots, start standing, 12 s, measured after 4 s)."""
import os, sys, json
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'k1_mp_speed')
sys.path.insert(0, SP); os.chdir(SP)
import numpy as np, torch
torch.set_num_threads(1)
from k1env_speed import K1SpeedBatch
from eval_speed import load, act, stand_all
res = []
for v in (0.3, 0.6, 0.9, 1.2, 1.35):
    env = K1SpeedBatch(8, stage=2, randomize=False, seed=11, ep_len=10 ** 9, v_range=(0.3, 1.35), nthread=1)
    env.pushes = False; env.scripted = True
    net = load('runs/final/model.pt', env)
    stand_all(env); env.cmd[:] = 1.0; env.v_cmd[:] = v
    alive = np.ones(8, bool); yaw0 = env.yaw().copy(); yaws = []; P = []; xy = []
    oa, _ = env.obs()
    for k in range(600):
        _, term, _, _ = env.step(act(net, oa)); alive &= ~term
        yaws.append(env.yaw().copy()); P.append(env.P_elec.copy()); xy.append(env.qpos()[:, :2].copy())
        oa, _ = env.obs()
    yaws = np.unwrap(np.array(yaws), axis=0); xy = np.array(xy); P = np.array(P)
    sp = np.linalg.norm(xy[-1] - xy[200], axis=1) / 8.0
    r = dict(v_cmd=v, survival=float(alive.mean()), v_meas=float(sp[alive].mean()),
             v_err_pct=float(100 * abs(sp[alive].mean() - v) / v), P_leg=float(P[200:, alive].mean()),
             heading_drift_deg=float(np.degrees(np.abs(yaws[-1] - yaws[0]))[alive].max()))
    print(r, flush=True); res.append(r)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'reg_v2.json'), 'w'), indent=1)
