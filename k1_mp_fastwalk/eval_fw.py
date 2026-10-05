# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Fast-walk evaluation: speed sweep 0.30-1.65 m/s (dt 2 ms, 3 robots per speed, 12 s, same protocol as the v2
table) for a policy in this folder or the v2 policy in ../k1_mp_speed (--v2)."""
import os, sys, json
if '--v2' in sys.argv:
    D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'k1_mp_speed')
    sys.path.insert(0, D); os.chdir(D)
import torch
torch.set_num_threads(1)
from eval_speed import sweep
path = sys.argv[1]
res = sweep(path, [0.3, 0.6, 0.9, 1.2, 1.35, 1.5, 1.65], per=3, dt=0.002)
for r in res:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(res, open(sys.argv[2], 'w'), indent=1)
