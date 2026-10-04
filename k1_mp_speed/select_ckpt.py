# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Checkpoint selection: tracking (relative error), heading drift, survival with random commands + pushes."""
import sys, json
import numpy as np
import eval_speed as E
from k1env_speed import K1SpeedBatch


def push_test(path, n=24, T=16.0):
    env = K1SpeedBatch(n, stage=2, randomize=True, seed=21, ep_len=10 ** 9)
    env.pushes = True
    env.reset(np.arange(n))
    net = E.load(path, env)
    alive = np.ones(n, bool)
    oa, _ = env.obs()
    for k in range(int(T * 50)):
        _, term, _, _ = env.step(E.act(net, oa))
        alive &= ~term
        oa, _ = env.obs()
    return float(alive.mean())


if __name__ == '__main__':
    out = {}
    for p in sys.argv[1:]:
        sw = E.sweep(p, [0.3, 0.6, 0.9, 1.2, 1.35], per=3, T=12.0, warm=4.0)
        rel = float(np.mean([s['err'] / s['v_cmd'] for s in sw]))
        yaw = float(np.max([s['yaw_end_deg'] for s in sw]))
        surv = float(np.min([s['survival'] for s in sw]))
        pt = push_test(p)
        out[p] = dict(rel_err=rel, max_yaw_deg=yaw, min_survival=surv, push_DR_random_cmd_survival=pt,
                      P=[round(s['P_elec'], 1) for s in sw])
        print(p, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[p].items()}, flush=True)
    json.dump(out, open('out/select_' + sys.argv[1].split('/')[-1] + '.json', 'w'), indent=1)
