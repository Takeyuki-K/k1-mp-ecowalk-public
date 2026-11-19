# Local (ankle-spring variant): survival while walking under random pushes (same pushes as eval_stop_spring: +-0.3 m/s
# horizontal, ~1 per 3 s per robot), stand 1 s then walk v for T s. n robots x seeds.
# python3 eval_pushwalk.py walk.pt run.pt [--n 64] [--seeds 11 12] [--T 10] [--json out.json]
import argparse, json
import numpy as np
from eval_gait56 import Gait55, run_profile
ap = argparse.ArgumentParser(); ap.add_argument('walk'); ap.add_argument('run'); ap.add_argument('--n', type=int, default=64)
ap.add_argument('--seeds', type=int, nargs='+', default=[11, 12]); ap.add_argument('--T', type=float, default=10.0)
ap.add_argument('--v', type=float, nargs='+', default=[0.6, 1.0, 1.4]); ap.add_argument('--json', default='')
a = ap.parse_args()
g = Gait55(a.walk, a.run, n=a.n, nthread=2)
res = {}
for v in a.v:
    fell = 0; tot = 0
    for s in a.seeds:
        L = run_profile(g, [(1, 'cmd', 0.0, 0.0), (a.T, 'cmd', v, 0.0)], push=True, seed=s)
        fell += int((~L['alive']).sum()); tot += a.n
    res[f'walk{v}_push'] = dict(fell=fell, n=tot, survival=round(1 - fell / tot, 4))
    print(f'walk{v}_push', json.dumps(res[f'walk{v}_push']), flush=True)
if a.json: json.dump(res, open(a.json, 'w'), indent=1)
