import sys, numpy as np
from eval_gait56 import Gait55, run_profile, HZ
w, r, v = sys.argv[1], sys.argv[2], float(sys.argv[3])
g = Gait55(w, r, n=8, nthread=2); E = g.W
run_profile(g, [(1, 'cmd', 0.0, 0.0), (4, 'cmd', v, 0.0)], seed=3)
C = {'ds': [], 'ssL': [], 'ssR': []}; T = []
for k in range(5 * HZ):
    g.step(); t = E.tau_sub  # (n, sub, 12)
    st = E.state; nq = E.m.nq
    tl = np.array([E.s(f'left_{p}_touch')[:, 0] if E.s(f'left_{p}_touch').ndim > 1 else E.s(f'left_{p}_touch') for p in ('heel', 'meta', 'toe')]).sum(0)
    tr = np.array([E.s(f'right_{p}_touch')[:, 0] if E.s(f'right_{p}_touch').ndim > 1 else E.s(f'right_{p}_touch') for p in ('heel', 'meta', 'toe')]).sum(0)
    L, R = tl > 5, tr > 5
    tau = t[:, :, [1, 7]].mean(1)   # (n,2)
    cu = (t[:, :, [1, 7]] ** 2 / 16).mean(1).sum(1)
    for i in range(g.n):
        key = 'ds' if (L[i] and R[i]) else ('ssL' if L[i] else ('ssR' if R[i] else None))
        if key: C[key].append((cu[i], tau[i, 0], tau[i, 1]))
tot = sum(len(x) for x in C.values())
for k, x in C.items():
    x = np.array(x); print(f'{k}: frac {len(x)/tot:.2f}  hiproll copper {x[:,0].mean():5.1f} W  tauL {x[:,1].mean():+5.1f}  tauR {x[:,2].mean():+5.1f} Nm')
