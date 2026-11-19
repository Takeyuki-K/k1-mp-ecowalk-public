# per-joint electrical power breakdown, walking 1.0 m/s (gait manager, 8 robots, last 5 s of 9 s)
import sys, json, numpy as np
from eval_gait56 import Gait55, run_profile, HZ
from k1env_eco import KM
from k1env import LEG_JOINTS
w, r, v = sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
g = Gait55(w, r, n=8, nthread=2)
run_profile(g, [(1, 'cmd', 0.0, 0.0), (4 if v < 1.7 else 7, 'cmd', v, 0.0)], seed=3)
cu, mech, tau_abs = [], [], []
for k in range(5 * HZ):
    g.step(); E = g.R if v >= 1.7 else g.W
    t = E.tau_sub; st = E.state; qd = st[:, 1 + E.m.nq:1 + E.m.nq + E.m.nv][:, E.dadr[:12]]
    cu.append((t ** 2 / KM ** 2).mean(1)); mech.append(np.clip(t[:, -1] * qd, 0, None)); tau_abs.append(np.abs(t).mean(1))
cu = np.array(cu).mean((0, 1)); me = np.array(mech).mean((0, 1)); ta = np.array(tau_abs).mean((0, 1))
for j, c, m_, t in zip(LEG_JOINTS, cu, me, ta):
    print(f'{j:28s} copper {c:6.1f} W  mech+ {m_:6.1f} W  |tau| {t:5.1f} Nm')
print('sum copper', round(cu.sum(), 1), 'sum mech+', round(me.sum(), 1))
