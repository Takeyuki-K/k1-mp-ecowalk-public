# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Evaluate the v5 running policy (motor limit on, no pushes unless --push --dr), N robots per test, 12 s.
 steady : start running at 2 m/s, accelerate to the command (1.5 m/s^2), metrics after 5 s
 stand  : standing start with a run command (3 m/s)
 walk   : hand-over from recorded fast-walking states (1.6 m/s, right heel strike), command 3 m/s
Touchdown classes: forefoot first / midfoot (heel and forefoot in the same 20 ms) / heel first.
python3 eval_run2.py runs/rn1/model.pt [--speeds 3,4,5,5.5] [--n 16] [--json out.json]
"""
import sys, json
import numpy as np, torch
from k1env_run2 import K1Run2Batch, K1Run3Batch, MG
from motor import QD_LIM
from ppo_eco import AC

torch.set_num_threads(1)
NACT = 36


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def run(net, mode, v, N, push=False, dr=False, T=600, k0=250):
    K1Run2Batch.P_STAND = 1.0 if mode == 'stand' else 0.0
    K1Run2Batch.P_WALK = 1.0 if mode == 'walk' else 0.0
    v0 = min(v, 2.0)
    env = (K1Run3Batch if NACT == 37 else K1Run2Batch)(N, v_lo=v0, v_hi=v0, stage=2 if push else 1, randomize=dr, seed=9, nthread=1, ep_len=10 ** 9)
    env.pushes = push; env.assist = 0.0; env.scripted = True
    env.reset(np.arange(N))
    env.v_cmd[:] = v
    oa, _ = env.obs()
    alive = np.ones(N, bool)
    L = {k: [] for k in ('vx', 'flight', 'P', 'pitch', 'impact', 'qd_viol', 'sat', 'kp', 'x')}
    air = np.zeros((N, 2), bool); at = np.zeros((N, 2)); tds = []
    t_reach = np.full(N, np.nan)
    for t in range(T):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        rew, term, trunc, info = env.step(a)
        alive &= ~term
        sd = env.sd_out_all
        for k, s in enumerate(('left', 'right')):
            heel = sd[:, :, env.sens[f'{s}_heel_touch'][0]].max(1) > 5
            fore = np.maximum(sd[:, :, env.sens[f'{s}_meta_touch'][0]], sd[:, :, env.sens[f'{s}_toe_touch'][0]]).max(1) > 5
            c = heel | fore
            for i in np.where(air[:, k] & c & alive)[0]:
                tds.append(dict(t=t, i=int(i), heel=bool(heel[i]), fore=bool(fore[i])))
            at[:, k] = np.where(c, 0, at[:, k] + 0.02)
            air[:, k] = (~c) & (at[:, k] > 0.06) | (air[:, k] & ~c)
        quat, g = env.base_frame()
        vx = env.qvel()[:, 0]
        L['vx'].append(vx); L['flight'].append(info['flight']); L['P'].append(env.P_elec.copy())
        L['pitch'].append(np.degrees(np.arcsin(np.clip(g[:, 0], -1, 1)))); L['impact'].append(info['impact'])
        L['qd_viol'].append(info['qd_viol']); L['sat'].append(info['tau_sat']); L['kp'].append(env.kp_scale.mean(1))
        L['x'].append(env.qpos()[:, 0].copy())
        if len(L['x']) > 25:
            vavg = (L['x'][-1] - L['x'][-26]) / 0.5
            t_reach = np.where(np.isnan(t_reach) & (vavg >= 0.9 * v), t / 50, t_reach)
        oa, _ = env.obs()
    L = {k: np.array(x) for k, x in L.items()}
    al = alive
    dur = (T - k0) / 50
    vmean = float(((L['x'][-1] - L['x'][k0]) / dur)[al].mean()) if al.any() else None
    tdv = [d for d in tds if d['t'] >= k0 and al[d['i']]]
    tda = [d for d in tds if al[d['i']]]
    cls = lambda D: (np.mean([d['fore'] and not d['heel'] for d in D]), np.mean([d['fore'] and d['heel'] for d in D]),
                     np.mean([d['heel'] and not d['fore'] for d in D])) if D else (None, None, None)
    ff, mf, hf = cls(tdv)
    cad = len(tdv) / max(al.sum(), 1) / dur * 60
    S = lambda A: float(A[k0:, al].mean()) if al.any() else None
    return dict(mode=mode, v_cmd=v, survival=float(al.mean()), speed=vmean,
                cadence_spm=float(cad), step_length=float(vmean / (cad / 60)) if vmean and cad else None,
                flight_fraction=S(L['flight']), forefoot_first=ff, midfoot=mf, heel_first=hf,
                heel_first_all_touchdowns=cls(tda)[2],
                peak_force_bw_p95=float(np.percentile(L['impact'][k0:, al], 95)) if al.any() else None,
                trunk_pitch_mean_deg=S(L['pitch']), trunk_pitch_std_deg=float(L['pitch'][k0:, al].std()) if al.any() else None,
                joint_speed_over_limit_pct=100 * S(L['qd_viol']) if al.any() else None,
                torque_saturated_pct=100 * S(L['sat']) if al.any() else None,
                P_leg=S(L['P']), CoT=float(S(L['P']) / (35.706 * 9.81 * vmean)) if vmean else None,
                kp_mean=S(L['kp']), t_reach_90pct=float(np.nanmean(t_reach[al])) if al.any() else None)


def main():
    path = sys.argv[1]
    N = int(arg('--n', 8))
    global NACT
    sd = torch.load(path, map_location='cpu')['model']
    NACT = sd['log_std'].shape[0]
    probe = (K1Run3Batch if NACT == 37 else K1Run2Batch)(2, nthread=1)
    oa, oc = probe.obs()
    net = AC(oa.shape[1], oc.shape[1], NACT)
    net.load_state_dict(sd); net.eval()
    res = []
    tests = [(arg('--entry', 'steady'), float(v)) for v in arg('--speeds', '3,4,5,5.5').split(',')]
    if '--no_entry' not in sys.argv:
        tests += [('stand', 3.0), ('walk', 3.0)]
    for mode, v in tests:
        R = run(net, mode, v, N, push='--push' in sys.argv, dr='--dr' in sys.argv)
        R = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in R.items()}
        print(json.dumps(R), flush=True); res.append(R)
    if arg('--json'):
        json.dump(dict(policy=path, results=res), open(arg('--json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
