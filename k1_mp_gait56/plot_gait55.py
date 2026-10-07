# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Figures for REPORT_GAIT55.md (one robot, no pushes):
  out/fig_gait55_profile.png : top view + speed / yaw rate / lean over the full walk -> run -> turn -> brake profile
  out/fig_gait55_stop.png    : normal stop vs hard braking from 4.5 m/s (speed and pelvis height over time)
python3 plot_gait55.py runs/final/walk.pt runs/final/run.pt
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gait55 import Gait55
import eval_gait55 as E

HERE = os.path.dirname(os.path.abspath(__file__))


def trace(g, prof):
    g.reset(); E.stand_noise(g)
    g._oaw, _ = g.W.obs(); g._oar, _ = g.R.obs()
    L = dict(t=[], xy=[], v=[], vref=[], vcmd=[], w=[], wref=[], lean=[], z=[], mode=[], seg=[])
    k = 0
    for si, (T, act, v, w) in enumerate(prof):
        {'cmd': lambda: g.command(v, w), 'stop': g.stop, 'brake': g.brake}[act]()
        for _ in range(int(T * 50)):
            g.step()
            e = g.env; q = e.qpos()[0]; qv = e.qvel()[0]
            yw = float(e.yaw()[0])
            quat, gv = e.base_frame()
            L['t'].append(k / 50); L['xy'].append(q[:2].copy()); L['z'].append(q[2])
            L['v'].append(np.cos(yw) * qv[0] + np.sin(yw) * qv[1]); L['vref'].append(float(e.v_ref[0]))
            L['vcmd'].append(v if act == 'cmd' else 0.0)
            L['w'].append(qv[5]); L['wref'].append(float(e.w_ref[0]))
            L['lean'].append(np.degrees(np.arcsin(np.clip(gv[0, 1], -1, 1))))
            L['mode'].append(g.mode[0]); L['seg'].append(si)
            k += 1
    return {kk: np.array(vv) for kk, vv in L.items()}


def smooth(x, n=25):
    return np.convolve(x, np.ones(n) / n, mode='same')


def main():
    walk, run = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    g = Gait55(walk, run, n=1, nthread=1)
    L = trace(g, E.PROFILE)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 3, 1)
    run_m = L['mode'] == 1
    ax.plot(L['xy'][~run_m, 0], L['xy'][~run_m, 1], '.', ms=1.5, color='#3a9d6b', label='walking')
    ax.plot(L['xy'][run_m, 0], L['xy'][run_m, 1], '.', ms=1.5, color='#2f78c4', label='running')
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title('top view'); ax.legend(loc='best')
    for i, (lab, keys) in enumerate([('forward speed [m/s]', [('vcmd', 'command', '#999999'), ('vref', 'reference (governed, ramped)', '#8a5cc2'), ('v', 'measured (0.5 s mean)', '#2f78c4')]),
                                     ('yaw rate [rad/s]', [('wref', 'reference', '#8a5cc2'), ('w', 'measured (0.5 s mean)', '#3a9d6b')]),
                                     ('lean into the turn [deg]', [('lean', 'measured (0.5 s mean)', '#d08a2c')])]):
        a = fig.add_subplot(3, 3, (3 * i + 2, 3 * i + 3))
        for k, lab2, col in keys:
            y = L[k]
            if k in ('v', 'w', 'lean'):
                y = smooth(y)
            a.plot(L['t'], y, color=col, lw=1.4, label=lab2)
        if i == 2:
            phi = np.degrees(np.arctan2(L['vref'] * L['wref'], 9.81))
            a.plot(L['t'], phi, '--', color='#777777', lw=1, label='atan(v w / g)')
            a.set_xlabel('time [s]')
        a.set_ylabel(lab); a.legend(loc='upper left', fontsize=7); a.grid(alpha=0.3)
        for s in np.where(np.diff(L['seg']) != 0)[0]:
            a.axvline(L['t'][s], color='#cccccc', lw=0.6)
    fig.suptitle('v5.5: stand -> walk -> walking turn -> run -> running turns -> governor (4.5 m/s, 1 rad/s) -> hard braking')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'out', 'fig_gait55_profile.png'), dpi=130)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for kind, col in (('stop', '#999999'), ('brake', '#c4402f')):
        prof = [(2, 'cmd', 0.0, 0.0), (5, 'cmd', 1.65, 0.0), (6, 'cmd', 4.5, 0.0), (8, kind, 0, 0)]
        L = trace(g, prof)
        k0 = np.where(L['seg'] == 3)[0][0]
        t = L['t'][k0 - 50:] - L['t'][k0]
        axs[0].plot(t, smooth(L['v'])[k0 - 50:], color=col, label=f'{kind}: measured')
        axs[0].plot(t, L['vref'][k0 - 50:], '--', color=col, lw=0.8, label=f'{kind}: reference')
        axs[1].plot(t, L['z'][k0 - 50:], color=col, label=kind)
    axs[0].set_ylabel('forward speed [m/s]'); axs[1].set_ylabel('pelvis height [m]'); axs[1].set_xlabel('time after the command [s]')
    for a in axs:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle('normal stop vs hard braking from 4.5 m/s')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'out', 'fig_gait55_stop.png'), dpi=130)
    print('saved out/fig_gait55_profile.png, out/fig_gait55_stop.png')


if __name__ == '__main__':
    main()
