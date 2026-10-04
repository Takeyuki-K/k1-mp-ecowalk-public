# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Check: eco env + transferred policy (Kp/Kd scale = 1) reproduces the fixed-gain env + original policy."""
import numpy as np, torch
from k1env import K1Batch
from k1env_eco import K1EcoBatch
from ppo import AC
from ppo_eco import ACEco, transfer_fixed_to_eco

src = torch.load('runs/final/model.pt', map_location='cpu')['model']
envA = K1Batch(4, stage=1, seed=3); envA.pushes = False
envB = K1EcoBatch(4, stage=1, seed=3); envB.pushes = False
envB.state[:] = envA.state; envB.sdata[:] = envA.sdata; envB.ph[:] = envA.ph; envB.alpha[:] = envA.alpha
envB.cmd[:] = envA.cmd; envB.switch_t[:] = envA.switch_t; envB.stop_t[:] = envA.stop_t
oa, oc = envA.obs(); netA = AC(oa.shape[1], oc.shape[1], 12); netA.load_state_dict(src); netA.eval()
ob, ocb = envB.obs(); netB = ACEco(ob.shape[1], ocb.shape[1], 36); transfer_fixed_to_eco(src, netB); netB.eval()
print('obs dims', oa.shape[1], '->', ob.shape[1])
for t in range(250):
    with torch.no_grad():
        aA = netA.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        aB = netB.dist(torch.from_numpy(ob)).mean.numpy().astype(np.float64)
    if t == 0:
        print('action diff (pos part) %.2e, gain part max |a| %.2e' % (np.abs(aA - aB[:, :12]).max(), np.abs(aB[:, 12:]).max()))
    envA.step(aA); envB.step(aB)
    oa, oc = envA.obs(); ob, ocb = envB.obs()
dq = np.abs(envA.qpos() - envB.qpos()).max()
print('after 5 s: max |qpos diff| = %.2e   x positions A %s  B %s' % (dq, envA.qpos()[:, 0].round(3), envB.qpos()[:, 0].round(3)))
print('tau check: eco tau_sub vs servo tau_sub max diff %.2e' % np.abs(envA.tau_sub - envB.tau_sub).max())
