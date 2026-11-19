# Local (ankle-spring variant): sudden stop checks. walk at v then 'stop' (hard brake), 32 robots, with and without
# random pushes; survival after the stop and residual speed 3 s after the stop.
# python3 eval_stop_spring.py walk.pt run.pt [--json out.json]
import argparse, json
import numpy as np
from eval_gait56 import Gait55, run_profile
ap = argparse.ArgumentParser(); ap.add_argument('walk'); ap.add_argument('run'); ap.add_argument('--n', type=int, default=32)
ap.add_argument('--json', default=''); a = ap.parse_args()
g = Gait55(a.walk, a.run, n=a.n, nthread=2)
res = {}
for v in (0.6, 1.0, 1.4, 1.65):
    for act, push in (('stop', False), ('brake', False), ('brake', True)):
        prof = [(1, 'cmd', 0.0, 0.0), (5, 'cmd', v, 0.0), (4, act, 0, 0)]
        L = run_profile(g, prof, push=push, seed=5)
        t_stop = 6.0
        before = L['fall_t'] < t_stop; after = L['fall_t'] >= t_stop
        r = dict(fell_before=int(before.sum()), fell_after_stop=int(after.sum()),
                 survival_after_stop=round(float(1 - after.sum() / max(1, (~before).sum())), 3))
        name = f'walk{v}_{act}' + ('_push' if push else '')
        res[name] = r; print(name, json.dumps(r), flush=True)
if a.json:
    json.dump(res, open(a.json, 'w'), indent=1)
