# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Evaluate the fast-running policy at constant speed commands (8 robots each, 12 s, start running at 2 m/s and
accelerate to the command at 1.5 m/s^2, measured after 5 s). Optional --push --dr for robustness.
Metrics: survival, speed, cadence, step length, flight fraction, heel-first touchdowns, peak foot force,
trunk pitch, joint speed vs the 11.5 rad/s URDF limit, electrical power, CoT.
python3 eval_sprint.py runs/sp1/model.pt [--speeds 2,3,4,5] [--push --dr] [--json out.json] [--rec 5.0:out/rec.npz]
"""
import sys, json
import numpy as np, torch
from k1env_sprint import K1SprintBatch, MG
from ppo_eco import AC

torch.set_num_threads(1)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def run_speed(net, v, push, dr, rec=False, N=8, T=600, k0=250):
    """robots start running at 2.0 m/s (inside the cycle) and accelerate to the command (1.5 m/s^2)"""
    v0 = min(v, 2.0)
    env = K1SprintBatch(N, v_lo=v0, v_hi=v0, stage=2 if push else 1, randomize=dr, seed=7, nthread=1, ep_len=10 ** 9)
    env.pushes = push; env.assist = 0.0; env.scripted = True
    env.reset(np.arange(N))
    env.v_cmd[:] = v
    oa, _ = env.obs()
    alive = np.ones(N, bool)
    L = {k: [] for k in ('vx', 'flight', 'P', 'pitch', 'impact', 'qd_viol', 'qd_peak', 'kp', 'heelL', 'heelR', 'foreL', 'foreR', 'ph')}
    qs = []
    air = np.zeros((N, 2), bool); at = np.zeros((N, 2)); tds = []
    x0 = None
    for t in range(T):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        rew, term, trunc, info = env.step(a)
        alive &= ~term
        if t == k0:
            x0 = env.qpos()[:, 0].copy()
        sd = env.sd_out
        f = {(s, p): sd[:, :, env.sens[f'{s}_{p}_touch'][0]].max(1) for s in ('left', 'right') for p in ('heel', 'meta', 'toe')}
        for k, s in enumerate(('left', 'right')):
            heel = f[s, 'heel'] > 5; fore = np.maximum(f[s, 'meta'], f[s, 'toe']) > 5
            c = heel | fore
            for i in np.where(air[:, k] & c & alive)[0]:
                if t >= k0:
                    tds.append(dict(t=t, i=int(i), heel=bool(heel[i]), fore=bool(fore[i]), x=float(env.s(f'{s}_foot_pos')[i, 0])))
            at[:, k] = np.where(c, 0, at[:, k] + 0.02)
            air[:, k] = (~c) & (at[:, k] > 0.06) | (air[:, k] & ~c)
        quat, g = env.base_frame()
        L['vx'].append(info['vx']); L['flight'].append(info['flight']); L['P'].append(env.P_elec.copy())
        L['pitch'].append(np.degrees(np.arcsin(np.clip(g[:, 0], -1, 1)))); L['impact'].append(info['impact'])
        L['qd_viol'].append(info['qd_viol']); L['qd_peak'].append(info['qd_peak']); L['kp'].append(env.kp_scale.copy())
        L['heelL'].append(f['left', 'heel']); L['heelR'].append(f['right', 'heel'])
        L['foreL'].append(np.maximum(f['left', 'meta'], f['left', 'toe'])); L['foreR'].append(np.maximum(f['right', 'meta'], f['right', 'toe']))
        L['ph'].append(env.ph.copy())
        qs.append(env.qpos()[0].copy())
        oa, _ = env.obs()
    L = {k: np.array(x) for k, x in L.items()}
    al = alive
    dur = (T - k0) / 50
    dist = env.qpos()[:, 0] - x0
    vmean = float(dist[al].mean() / dur) if al.any() else None
    tdv = [d for d in tds if al[d['i']]]
    cadence = len(tdv) / max(al.sum(), 1) / dur * 60
    R = dict(v_cmd=v, survival=float(al.mean()), speed=vmean,
             speed_err_pct=float(100 * abs(vmean - v) / v) if vmean else None,
             cadence_spm=float(cadence), step_length=float(vmean / (cadence / 60)) if vmean and cadence else None,
             flight_fraction=float(L['flight'][k0:, al].mean()) if al.any() else None,
             heel_contact_at_touchdown=float(np.mean([d['heel'] for d in tdv])) if tdv else None,
             heel_only_touchdown=float(np.mean([d['heel'] and not d['fore'] for d in tdv])) if tdv else None,
             peak_force_bw_p95=float(np.percentile(L['impact'][k0:, al], 95)) if al.any() else None,
             trunk_pitch_mean_deg=float(L['pitch'][k0:, al].mean()) if al.any() else None,
             trunk_pitch_std_deg=float(L['pitch'][k0:, al].std()) if al.any() else None,
             joint_speed_over_limit_pct=float(100 * L['qd_viol'][k0:, al].mean()) if al.any() else None,
             joint_speed_peak=float(np.percentile(L['qd_peak'][k0:, al], 99)) if al.any() else None,
             P_leg=float(L['P'][k0:, al].mean()) if al.any() else None,
             CoT=float(L['P'][k0:, al].mean() / (35.706 * 9.81 * vmean)) if vmean else None,
             kp_mean=float(L['kp'][k0:, al].mean()) if al.any() else None)
    recd = None
    if rec:
        recd = dict(qpos=np.array(qs), P=L['P'][:, 0], vx=L['vx'][:, 0], kp=L['kp'][:, 0], flight=L['flight'][:, 0],
                    pitch=L['pitch'][:, 0], heelL=L['heelL'][:, 0], heelR=L['heelR'][:, 0], foreL=L['foreL'][:, 0],
                    foreR=L['foreR'][:, 0], ph=L['ph'][:, 0], v=v, alive0=al[0])
    return R, recd


def main():
    path = sys.argv[1]
    speeds = [float(s) for s in arg('--speeds', '2,3,4,5').split(',')]
    push, dr = '--push' in sys.argv, '--dr' in sys.argv
    env = K1SprintBatch(2, nthread=1)
    oa, oc = env.obs()
    net = AC(oa.shape[1], oc.shape[1], 36)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    rec = arg('--rec')
    res = []
    for v in speeds:
        want = rec is not None and abs(float(rec.split(':')[0]) - v) < 1e-6
        R, recd = run_speed(net, v, push, dr, rec=want, N=int(arg('--n', 8)))
        R = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in R.items()}
        print(json.dumps(R), flush=True); res.append(R)
        if want:
            np.savez(rec.split(':')[1], **recd)
    if arg('--json'):
        json.dump(dict(policy=path, push=push, dr=dr, results=res), open(arg('--json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
