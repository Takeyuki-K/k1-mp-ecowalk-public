# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Left/right arm motion (mean and peak-to-peak per joint) for straight walking, in-place turning and stepping in place.
Right-side roll/yaw angles are sign-flipped so that a symmetric gait shows equal numbers. python3 eval_arms56.py <walk.pt>"""
import sys, numpy as np
from eval_gait56 import Gait55, run_profile
from k1env import ACT_JOINTS, DEFAULT_POSE
g = Gait55(sys.argv[1], 'runs/final/run.pt', n=4, nthread=2); W = g.W
J = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow']
idx = {s: [ACT_JOINTS.index(f'{s}_{j}_joint') for j in J] for s in ('left', 'right')}
for name, prof in [('straight 1.0', [(1,'cmd',0,0),(4,'cmd',1.0,0)]), ('inplace +0.6', [(1,'cmd',0,0),(4,'cmd',0,0.6)]), ('stand', [(3,'cmd',0,0)])]:
    run_profile(g, prof)
    Q=[]; R=[]
    for k in range(250):
        g.step(); Q.append(W.qpos()[:, W.qadr].copy())
        R.append(W.last_a[:, 37:41].copy())
    Q=np.degrees(np.array(Q)); R=np.array(R)
    print(f'== {name}')
    for j, jn in enumerate(J):
        l = Q[:, :, idx['left'][j]]; r = Q[:, :, idx['right'][j]]
        sg = -1 if jn in ('shoulder_roll', 'shoulder_yaw') else 1     # mirrored joints
        print(f'{jn:15s} L mean {l.mean():6.1f} amp {np.ptp(l,0).mean():5.1f} | R mean {sg*r.mean():6.1f} amp {np.ptp(r,0).mean():5.1f}   (default L {np.degrees(DEFAULT_POSE[idx["left"][j]]):5.1f})')
    print('arm residual actions mean [Lpitch Lroll Rpitch Rroll]*0.3rad ->deg', np.round(np.degrees(0.3*R.mean((0,1))),1), 'std', np.round(np.degrees(0.3*R.std((0,1))),1))
