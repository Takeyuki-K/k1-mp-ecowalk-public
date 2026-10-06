# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Smoke test for the released checkpoints: load every final policy into its environment and run a few seconds of
simulation (deterministic actions, no pushes). Checks that code, models and checkpoints fit together; it is NOT a
performance evaluation (use the eval_*.py scripts of each folder for that).
Each folder runs in its own subprocess because the folders contain modules with the same names.
python3 scripts/smoke_eval.py [--steps 150]
"""
import os, sys, subprocess, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = int(sys.argv[sys.argv.index('--steps') + 1]) if '--steps' in sys.argv else 150

COMMON = '''
import sys, json, numpy as np, torch
torch.set_num_threads(1)
def run(env, net, steps):
    env.pushes = False
    oa, _ = env.obs(); alive = np.ones(env.n, bool)
    for _ in range(steps):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        _, term, _, _ = env.step(a); alive &= ~term
        oa, _ = env.obs()
    return float(alive.mean())
def load(net, path):
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval(); return net
'''

CASES = {
    'k1_mp': ('runs/final/model.pt', '''
from k1env import K1Batch; from ppo import AC
env = K1Batch(4, stage=2, nthread=1, seed=1); oa, oc = env.obs()
net = load(AC(oa.shape[1], oc.shape[1], 12), P)'''),
    'k1_mp_eco': ('runs/eco1/model.pt', '''
from k1env_eco import K1EcoBatch; from ppo_eco import AC
env = K1EcoBatch(4, stage=2, nthread=1, seed=1); oa, oc = env.obs()
net = load(AC(oa.shape[1], oc.shape[1], 36), P)'''),
    'k1_mp_speed': ('runs/final/model.pt', '''
from k1env_speed import K1SpeedBatch; from ppo_speed import ACEco
env = K1SpeedBatch(4, stage=2, nthread=1, seed=1); oa, oc = env.obs()
net = load(ACEco(oa.shape[1], oc.shape[1], env.nact), P)'''),
    'k1_mp_turn': ('runs/final/model.pt', '''
from k1env_turn import K1TurnBatch; from ppo_turn import ACEco
env = K1TurnBatch(4, stage=2, nthread=1, seed=1); oa, oc = env.obs()
net = load(ACEco(oa.shape[1], oc.shape[1], env.nact), P)'''),
    'k1_mp_run': ('runs/final/model.pt', '''
from k1env_run import K1RunBatch; from ppo_eco import AC
env = K1RunBatch(4, stage=1, nthread=1, seed=1); env.assist = 0.0; oa, oc = env.obs()
net = load(AC(oa.shape[1], oc.shape[1], 36), P)'''),
    'k1_mp_fastwalk': ('runs/final/model.pt', '''
from k1env_speed import K1SpeedBatch; from ppo_speed import ACEco
env = K1SpeedBatch(4, stage=2, nthread=1, seed=1); oa, oc = env.obs()
net = load(ACEco(oa.shape[1], oc.shape[1], env.nact), P)'''),
    'k1_mp_sprint': ('runs/final/model.pt', '''
from k1env_sprint import K1SprintBatch; from ppo_eco import AC
env = K1SprintBatch(4, v_lo=2.0, v_hi=3.0, stage=1, nthread=1, seed=1); env.assist = 0.0; oa, oc = env.obs()
net = load(AC(oa.shape[1], oc.shape[1], 36), P)'''),
    'k1_mp_gait': ('runs/final/run.pt', '''
from k1env_run2 import K1Run2Batch; from ppo_eco import AC
K1Run2Batch.P_STAND = 0.0
env = K1Run2Batch(4, v_lo=2.0, v_hi=3.0, stage=1, nthread=1, seed=1); env.assist = 0.0; oa, oc = env.obs()
net = load(AC(oa.shape[1], oc.shape[1], 36), P)'''),
}


def main():
    ok = True
    for folder, (ckpt, code) in CASES.items():
        src = COMMON + f"\nP = {ckpt!r}\n" + code + f"\nprint(json.dumps(dict(survival=run(env, net, {STEPS}))))\n"
        env = dict(os.environ)
        p = subprocess.run([sys.executable, '-c', src], cwd=os.path.join(ROOT, folder), capture_output=True, text=True, env=env)
        if p.returncode != 0:
            ok = False
            print(f'FAIL {folder}: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.returncode}')
            continue
        res = json.loads(p.stdout.strip().splitlines()[-1])
        print(f'ok   {folder:16s} {ckpt:24s} survival over {STEPS / 50:.0f} s: {res["survival"]:.2f}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
