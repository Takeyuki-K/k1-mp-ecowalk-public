# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Evaluate the running policy: 8 robots, no pushes / no randomisation, start inside the running cycle, 10 s.
Metrics: survival, speed, flight fraction (no foot contact during a whole 20 ms step), heel-first touchdowns,
peak foot force per touchdown (in body weights), torso pitch, leg Kp scale over the gait phase (relaxation),
electrical leg power, cost of transport.
python3 eval_run.py runs/run2/model.pt [--json out.json] [--rec out/rec_run.npz] [--push]
"""
import sys, json
import numpy as np, torch
from k1env_run import K1RunBatch, MG
from ppo_eco import AC

torch.set_num_threads(1)


def main():
    path = sys.argv[1]
    N = 24 if '--dr' in sys.argv else 8
    env = K1RunBatch(N, stage=2 if '--push' in sys.argv else 1, randomize='--dr' in sys.argv, seed=5, nthread=1, ep_len=10 ** 9)
    env.pushes = '--push' in sys.argv
    env.assist = 0.0
    if '--hard' in sys.argv:
        env.push_mag = 0.6
    oa, oc = env.obs()
    net = AC(oa.shape[1], oc.shape[1], 36)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    env.reset(np.arange(N))
    T = 800 if '--dr' in sys.argv else 500
    alive = np.ones(N, bool)
    L = {k: [] for k in ('vx', 'flight', 'P', 'pitch', 'kp', 'kd', 'ph', 'fL', 'fR', 'heelL', 'heelR', 'foreL', 'foreR', 'z')}
    qs = []
    air = np.zeros((N, 2), bool); at = np.zeros((N, 2)); tds = []
    oa, _ = env.obs()
    x0 = env.qpos()[:, 0].copy()
    for t in range(T):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        rew, term, trunc, info = env.step(a)
        alive &= ~term
        sd = env.sd_out
        f = {}
        for s in ('left', 'right'):
            for p in ('heel', 'meta', 'toe'):
                f[s, p] = sd[:, :, env.sens[f'{s}_{p}_touch'][0]]          # (N, nsub)
        for k, s in enumerate(('left', 'right')):
            heel = f[s, 'heel'].max(1) > 5; fore = np.maximum(f[s, 'meta'], f[s, 'toe']).max(1) > 5
            c = heel | fore
            down = air[:, k] & c
            for i in np.where(down & alive)[0]:
                tds.append(dict(t=t, i=int(i), side=s, heel=bool(heel[i]), fore=bool(fore[i])))
            at[:, k] = np.where(c, 0, at[:, k] + 0.02)
            air[:, k] = (~c) & (at[:, k] > 0.06) | (air[:, k] & ~c)
        fL = (f['left', 'heel'] + f['left', 'meta'] + f['left', 'toe']).max(1) / MG
        fR = (f['right', 'heel'] + f['right', 'meta'] + f['right', 'toe']).max(1) / MG
        quat, g = env.base_frame()
        L['vx'].append(info['vx']); L['flight'].append(info['flight']); L['P'].append(env.P_elec.copy())
        L['pitch'].append(np.degrees(np.arcsin(np.clip(g[:, 0], -1, 1))))
        L['kp'].append(env.kp_scale.copy()); L['kd'].append(env.kd_scale.copy()); L['ph'].append(env.ph.copy())
        L['fL'].append(fL); L['fR'].append(fR); L['z'].append(env.qpos()[:, 2].copy())
        L['heelL'].append(f['left', 'heel'].max(1)); L['heelR'].append(f['right', 'heel'].max(1))
        L['foreL'].append(np.maximum(f['left', 'meta'], f['left', 'toe']).max(1))
        L['foreR'].append(np.maximum(f['right', 'meta'], f['right', 'toe']).max(1))
        qs.append(env.qpos()[0].copy())
        oa, _ = env.obs()
    L = {k: np.array(v) for k, v in L.items()}
    k0 = 100                                   # skip the first 2 s
    al = alive
    dist = env.qpos()[:, 0] - x0
    # peak force per stance (between touchdown and lift-off) per foot
    peaks = []
    for side, F in (('L', L['fL']), ('R', L['fR'])):
        for i in np.where(al)[0]:
            c = F[k0:, i] > 0.02
            j = 0
            while j < len(c):
                if c[j]:
                    e = j
                    while e < len(c) and c[e]:
                        e += 1
                    peaks.append(F[k0 + j:k0 + e, i].max()); j = e
                else:
                    j += 1
    tdv = [d for d in tds if d['t'] >= k0 and al[d['i']]]
    hf = np.mean([d['heel'] for d in tdv]) if tdv else 0
    hfonly = np.mean([d['heel'] and not d['fore'] for d in tdv]) if tdv else 0
    # Kp scale by phase (legs: 0-5 left, 6-11 right); right leg phase 0 = right touchdown
    ph = L['ph'][k0:, al].ravel(); kp_r = L['kp'][k0:, :, 6:12][:, al].reshape(-1, 6)
    bins = np.linspace(0, 1, 11)
    kp_phase = [float(kp_r[(ph >= bins[b]) & (ph < bins[b + 1])].mean()) for b in range(10)]
    P = L['P'][k0:, al].mean()
    v = L['vx'][k0:, al].mean()
    R = dict(policy=path, survival=float(al.mean()), speed=float(v), distance_10s=float(dist[al].mean()) if al.any() else 0,
             flight_fraction=float(L['flight'][k0:, al].mean()), ref_flight_fraction=0.35,
             touchdowns=len(tdv), heel_contact_at_touchdown=float(hf), heel_only_touchdown=float(hfonly),
             peak_force_bw_mean=float(np.mean(peaks)) if peaks else None,
             peak_force_bw_p95=float(np.percentile(peaks, 95)) if peaks else None,
             torso_pitch_mean_deg=float(L['pitch'][k0:, al].mean()), torso_pitch_std_deg=float(L['pitch'][k0:, al].std()),
             pelvis_z_range=float((L['z'][k0:, al].max(0) - L['z'][k0:, al].min(0)).mean()),
             P_leg=float(P), CoT=float(P / (35.706 * 9.81 * max(v, 1e-3))),
             kp_scale_right_leg_by_phase=kp_phase, kd_scale_mean=float(L['kd'][k0:, al].mean()))
    print(json.dumps(R, indent=1))
    if '--json' in sys.argv:
        json.dump(R, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    if '--rec' in sys.argv:
        np.savez(sys.argv[sys.argv.index('--rec') + 1], qpos=np.array(qs), P=L['P'][:, 0], vx=L['vx'][:, 0],
                 fL=L['fL'][:, 0], fR=L['fR'][:, 0], kp=L['kp'][:, 0], flight=L['flight'][:, 0],
                 pitch=L['pitch'][:, 0], ph=L['ph'][:, 0], heelL=L['heelL'][:, 0], heelR=L['heelR'][:, 0],
                 foreL=L['foreL'][:, 0], foreR=L['foreR'][:, 0], alive0=al[0])


if __name__ == '__main__':
    main()
