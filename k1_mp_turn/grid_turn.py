# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Survival / yaw-rate map over (v, w) commands for the turning policy (4 robots, 10 s, from standing)."""
import sys, json
import numpy as np, torch
torch.set_num_threads(1)
import eval_turn as E
env = E.make_env(8); net = E.load(sys.argv[1], env)
res = []
for v in (0.0, 0.15, 0.3, 0.6, 1.0):
    for w in (0.3, 0.6, 0.8, 1.0):
        L = E.run(net, env, v, w, T=10)
        s = E.summarize_turn(L, v, w)
        res.append(dict(v=v, w=w, survival=s['survival'], w_meas=s['w_meas']))
        print(res[-1], flush=True)
json.dump(res, open(sys.argv[2], 'w'), indent=1)
