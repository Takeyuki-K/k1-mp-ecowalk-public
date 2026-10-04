# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Compare fixed-gain policy vs eco (variable impedance) policy on the SAME electrical model.
python3 eval_eco.py runs/final/model.pt runs/eco1/model.pt
Outputs: out/eco_results.json, out/eco_gain_profile.png, out/eco_power_by_joint.png
"""
import sys, json, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from energy_analysis import analyze
from k1env import K1Batch, DEFAULT_POSE, LEG_JOINTS
from k1env_eco import K1EcoBatch
from ppo import AC
import mujoco

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'


def survival(path, eco, mode, randomize=False, pushes=False, n=64, T=400):
    B = K1EcoBatch if eco else K1Batch
    e = B(n, stage=2, randomize=randomize, seed=11)
    e.pushes = pushes
    if mode == 'walk':
        e.stage = 1; e.reset(np.arange(n)); e.stage = 2
    else:
        m = e.m; d = mujoco.MjData(m)
        for i in range(n):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = m.key_qpos[0]; d.qpos[e.qadr] = DEFAULT_POSE; d.qpos[2] = e.ref.z_stand + 0.003
            d.qpos[e.qadr[:12]] += e.rng.normal(0, 0.02, 12)
            mujoco.mj_forward(m, d)
            st = np.zeros(e.nstate); mujoco.mj_getState(m, d, st, e.spec_state)
            e.state[i] = st; e.sdata[i] = d.sensordata
        e.alpha[:] = 0; e.cmd[:] = 0; e.ph[:] = 0
    e.switch_t[:] = {'stand': 10 ** 9, 'walk': -1, 'swalk': 75}[mode]
    e.stop_t[:] = 325 if mode == 'swalk' else 10 ** 9
    oa, oc = e.obs()
    net = AC(oa.shape[1], oc.shape[1], 36 if eco else 12)
    net.load_state_dict(torch.load(path)['model']); net.eval()
    alive = np.ones(n, bool); Pel = []
    for t in range(T):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        _, te, _, info = e.step(a)
        alive &= ~te
        oa, oc = e.obs()
    return float(alive.mean())


def main(p_fix, p_eco):
    res = {}
    print('energy (steady walking)...', flush=True)
    rf, _ = analyze(p_fix, eco=False)
    re_, rec = analyze(p_eco, eco=True)
    res['fixed'] = rf; res['eco'] = re_
    print('fixed P_elec %.1f W CoT %.3f | eco P_elec %.1f W CoT %.3f' % (rf['P_elec_W'], rf['CoT_elec'],
                                                                          re_['P_elec_W'], re_['CoT_elec']), flush=True)
    rob = {}
    for name, kw in [('stand', dict(mode='stand')), ('walk', dict(mode='walk')), ('stand_walk_stop', dict(mode='swalk')),
                     ('stand_walk_stop_DR', dict(mode='swalk', randomize=True)),
                     ('stand_pushes', dict(mode='stand', pushes=True)),
                     ('stand_walk_stop_DR_pushes', dict(mode='swalk', randomize=True, pushes=True))]:
        rob[name] = dict(fixed=survival(p_fix, False, **kw), eco=survival(p_eco, True, **kw))
        print(name, rob[name], flush=True)
    res['survival_8s_64envs'] = rob
    if not np.isnan(rob['stand']['eco']):
        json.dump(res, open('out/eco_results.json', 'w'), indent=1)

    # ---- gain profile over the gait cycle (left leg) ----
    ph = np.array(rec['ph']); kp = np.array(rec['kp'])
    bins = np.linspace(0, 1, 41); mid = 0.5 * (bins[1:] + bins[:-1])
    names = [('hip pitch', 0), ('hip roll', 1), ('knee', 3), ('ankle pitch', 4)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    # stance shading from the human reference contact
    ref = K1Batch(1, stage=1).ref
    st = ref.contact[:, 0, :].any(1)
    x = np.arange(ref.N) / ref.N
    ax.fill_between(x, 0, 1.6, where=st, color='#e8e7e3', step='mid', lw=0)
    ax.text(0.02, 1.53, 'left foot stance (human ref.)', color=INK2, fontsize=8)
    ax.axhline(1.0, color=INK2, lw=1, ls='--')
    ax.text(0.5, 1.02, 'ROBOTIS nominal', color=INK2, fontsize=8, ha='right')
    ends = []
    for (nm, j), c in zip(names, C):
        y = np.array([kp[(ph >= bins[i]) & (ph < bins[i + 1]), j].mean() if ((ph >= bins[i]) & (ph < bins[i + 1])).any()
                      else np.nan for i in range(40)])
        ax.plot(mid, y, color=c, lw=2, label=nm)
        ends.append([y[~np.isnan(y)][-1], nm])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.07)
    for yv, nm in ends:
        ax.text(1.005, yv, nm, color=INK, fontsize=8, va='center')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.6)
    ax.set_xlabel('gait phase (0 = right heel strike)', color=INK2)
    ax.set_ylabel('Kp / nominal', color=INK2)
    ax.set_title('Eco policy: left-leg stiffness within one stride', color=INK, loc='left', fontsize=11)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#c9c8c3'); ax.spines['bottom'].set_color('#c9c8c3')
    ax.tick_params(colors=INK2); ax.grid(axis='y', color='#ecebe7', lw=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='lower left')
    fig.tight_layout(); fig.subplots_adjust(right=0.86)
    fig.savefig('out/eco_gain_profile.png', facecolor=SURF)

    # ---- electrical power per joint group ----
    groups = [('hip pitch', 'hip_pitch'), ('hip roll', 'hip_roll'), ('hip yaw', 'hip_yaw'), ('knee', 'knee'),
              ('ankle pitch', 'ankle_pitch'), ('ankle roll', 'ankle_roll')]
    def g(r, key):
        return sum(v['pos_W'] + v['copper_W'] for k, v in r['per_joint'].items() if k.endswith(key))
    vf = [g(rf, k) for _, k in groups]; ve = [g(re_, k) for _, k in groups]
    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=150)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    xx = np.arange(len(groups)); w = 0.36
    ax.bar(xx - w / 2 - 0.01, vf, w, color=C[0], label='fixed gain (previous)')
    ax.bar(xx + w / 2 + 0.01, ve, w, color=C[1], label='eco (variable impedance)')
    for i in range(len(groups)):
        ax.text(xx[i] - w / 2, vf[i] + 1, f'{vf[i]:.0f}', ha='center', fontsize=8, color=INK)
        ax.text(xx[i] + w / 2, ve[i] + 1, f'{ve[i]:.0f}', ha='center', fontsize=8, color=INK)
    ax.set_xticks(xx); ax.set_xticklabels([n for n, _ in groups], color=INK2)
    ax.set_ylabel('W (both legs)', color=INK2)
    ax.set_title(f'Leg electrical power, steady walking: {rf["P_elec_W"]:.0f} W -> {re_["P_elec_W"]:.0f} W',
                 color=INK, loc='left', fontsize=11)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#c9c8c3'); ax.spines['bottom'].set_color('#c9c8c3')
    ax.tick_params(colors=INK2); ax.grid(axis='y', color='#ecebe7', lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig('out/eco_power_by_joint.png', facecolor=SURF)
    print('done')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
