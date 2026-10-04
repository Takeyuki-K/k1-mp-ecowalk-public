# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Contact-sequence quality of the eco policy: stand 2s -> walk 8s -> stop 3s."""
import re, json, numpy as np, torch, mujoco
from k1env import DEFAULT_POSE
from k1env_eco import K1EcoBatch
from ppo_eco import ACEco
import sys
env = K1EcoBatch(1, stage=2, seed=7, ep_len=10 ** 9); env.pushes = False
m = env.m; d = mujoco.MjData(m)
d.qpos[:] = m.key_qpos[0]; d.qpos[env.qadr] = DEFAULT_POSE; d.qpos[2] = env.ref.z_stand + 0.002
mujoco.mj_forward(m, d); st = np.zeros(env.nstate); mujoco.mj_getState(m, d, st, env.spec_state)
env.state[0] = st; env.sdata[0] = d.sensordata; env.ph[:] = 0; env.alpha[:] = 0; env.cmd[:] = 0
env.switch_t[:] = 100; env.stop_t[:] = 500
oa, oc = env.obs(); net = ACEco(oa.shape[1], oc.shape[1], 36)
net.load_state_dict(torch.load(sys.argv[1])['model']); net.eval()
seq = {0: '', 1: ''}; x0 = env.qpos()[0, 0]; mp = []; fell = False; P = []
for t in range(650):
    with torch.no_grad(): a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
    _, te, _, inf = env.step(a)
    if te[0]: fell = True; break
    if 150 <= t < 500: P.append(inf['P_elec'][0])
    for k, s in enumerate(('left', 'right')):
        h, mm, tt = [env.s(f'{s}_{p}_touch')[0, 0] > 5 for p in ('heel', 'meta', 'toe')]
        seq[k] += '-' if not (h or mm or tt) else 'H' if h and not (mm or tt) else 'F' if h else 'T'
    mp.append(env.qpos()[0, env.mpadr].min())
    oa, oc = env.obs()
out = dict(fell=fell, distance=float(env.qpos()[0, 0] - x0), mp_min_deg=float(np.degrees(min(mp))), P_elec_walk=float(np.mean(P)))
for k, s in ((0, 'L'), (1, 'R')):
    w = seq[k][100:520]
    pats = re.findall(r'[HFT]+', w)
    td = [p[0] for p in pats[1:]]
    out[s] = dict(touchdowns=len(td), heel_first=sum(c == 'H' or c == 'F' for c in td),
                  HFT=sum(bool(re.fullmatch(r'H+F+T+', p)) for p in pats[1:-1]), stances=len(pats[1:-1]))
    print(s, w[50:260])
print(json.dumps(out, indent=1))
