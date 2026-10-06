# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Walking sweep (same protocol as v2/v4: dt 2 ms, 3 robots, 12 s) with the motor torque-speed limit on."""
import sys, json
import torch
torch.set_num_threads(1)
import eval_speed
from k1env_walk2 import K1Walk2Batch
K1Walk2Batch.P_RUN = 0.0
eval_speed.K1SpeedBatch = K1Walk2Batch
res = eval_speed.sweep(sys.argv[1], [0.3, 0.6, 0.9, 1.2, 1.35, 1.5, 1.65], per=3, dt=0.002)
for r in res:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(res, open(sys.argv[2], 'w'), indent=1)
