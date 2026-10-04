# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Figures for the speed-command report (same electrical model, dt 2 ms for the power comparison)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(HERE, 'out')
C_ROB, C_OUR, C_REF = '#2a78d6', '#1baf7a', '#eb6834'
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#ecebe7'
MG = 35.706 * 9.81

ours = json.load(open(os.path.join(O, 'final_sweep_dt002.json')))
ext = json.load(open(os.path.join(O, 'final_eval.json')))['sweep']
rob = json.load(open(os.path.join(O, 'robotis_sweep.json')))
lib = np.load(os.path.join(HERE, 'ref_lib.npz'))


def style(ax, title, xl, yl):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c9c8c3')
    ax.tick_params(colors=INK2); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, loc='left', color=INK, fontsize=11)
    ax.set_xlabel(xl, color=INK2); ax.set_ylabel(yl, color=INK2)


# 1) tracking
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150); fig.patch.set_facecolor(SURF)
style(ax, 'Speed tracking (12 s straight walk, 3 robots per speed)', 'commanded speed (m/s)', 'measured speed (m/s)')
ax.plot([0, 1.7], [0, 1.7], color=INK2, lw=1, ls='--')
ax.text(1.55, 1.62, 'ideal', color=INK2, fontsize=8)
ax.axvspan(1.35, 1.7, color='#f1efe9')
ax.text(1.37, 0.35, 'outside training\nrange (extrapolation)', color=INK2, fontsize=8)
vc = [r['v_cmd'] for r in ext]; vm = [r['v_meas'] for r in ext]
ax.plot(vc, vm, color=C_OUR, lw=2, marker='o', ms=7, markeredgecolor=SURF, markeredgewidth=2)
ax.text(1.66, vm[-1] - 0.16, 'MP + imitation\n+ relaxation', color=INK, fontsize=8, ha='right')
rv = sorted(rob.values(), key=lambda r: r['v_cmd'])
ax.plot([r['v_cmd'] for r in rv], [r['v_meas'] for r in rv], color=C_ROB, lw=2, marker='s', ms=6,
        markeredgecolor=SURF, markeredgewidth=2)
ax.text(0.32, 0.55, 'ROBOTIS walk_default', color=INK, fontsize=8)
ax.set_xlim(0.2, 1.7); ax.set_ylim(0.2, 1.7)
fig.tight_layout(); fig.savefig(os.path.join(O, 'fig_speed_tracking.png'), facecolor=SURF)

# 2) step length & cadence vs speed (two panels, one axis each)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150); fig.patch.set_facecolor(SURF)
sp = lib['speeds']; Tc = lib['T']
ref_step = sp * Tc / 2; ref_cad = 120 / Tc
for ax, key, ref, ttl, yl in [(axs[0], 'step_len', ref_step, 'Step length grows with speed', 'step length (m)'),
                              (axs[1], 'cadence', ref_cad, 'Cadence grows with speed', 'steps / min')]:
    style(ax, ttl, 'speed (m/s)', yl)
    ax.plot(sp[:8], ref[:8], color=C_REF, lw=2, ls='--')
    ax.text(sp[7] + 0.02, ref[7], 'human-derived\nreference', color=INK, fontsize=8, va='center')
    ax.plot([r['v_meas'] for r in ours], [r[key] for r in ours], color=C_OUR, lw=2, marker='o', ms=7,
            markeredgecolor=SURF, markeredgewidth=2)
    ax.text(ours[0]['v_meas'], ours[0][key] * (0.9 if key == 'step_len' else 0.93), 'robot (measured)',
            color=INK, fontsize=8)
    ax.set_xlim(0.2, 1.65)
fig.tight_layout(); fig.savefig(os.path.join(O, 'fig_step_cadence.png'), facecolor=SURF)

# 3) leg power vs speed
fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150); fig.patch.set_facecolor(SURF)
style(ax, 'Leg electrical power vs speed (same physics, dt 2 ms)', 'measured speed (m/s)', 'W (both legs)')
ax.plot([r['v_meas'] for r in rv], [r['P_legs'] for r in rv], color=C_ROB, lw=2, marker='s', ms=6,
        markeredgecolor=SURF, markeredgewidth=2)
ax.plot([r['v_meas'] for r in ours], [r['P_elec'] for r in ours], color=C_OUR, lw=2, marker='o', ms=7,
        markeredgecolor=SURF, markeredgewidth=2)
for r_o, r_r in zip(ours, rv):
    d = (r_o['P_elec'] / r_r['P_legs'] - 1) * 100
    ax.text(r_o['v_meas'], r_o['P_elec'] - 14, f'{d:+.0f}%', color=INK, fontsize=8, ha='center')
ax.text(rv[-1]['v_meas'] - 0.02, rv[-1]['P_legs'] + 8, 'ROBOTIS walk_default', color=INK, fontsize=8, ha='right')
ax.text(ours[-1]['v_meas'] - 0.02, ours[-1]['P_elec'] + 12, 'MP + imitation + relaxation', color=INK, fontsize=8, ha='right')
ax.set_ylim(0, 290); ax.set_xlim(0.2, 1.45)
fig.tight_layout(); fig.savefig(os.path.join(O, 'fig_power_vs_speed.png'), facecolor=SURF)

# table
rows = []
for r_o, r_r in zip(ours, rv):
    rows.append(dict(v_cmd=r_o['v_cmd'], ours_v=round(r_o['v_meas'], 3), ours_err_pct=round(100 * r_o['err'] / r_o['v_cmd'], 1),
                     step=round(r_o['step_len'], 3), cadence=round(r_o['cadence']), yaw_deg=round(r_o['yaw_end_deg'], 1),
                     ours_P_legs=round(r_o['P_elec'], 1), ours_CoT=round(r_o['P_elec'] / (MG * r_o['v_meas']), 3),
                     rob_v=round(r_r['v_meas'], 3), rob_P_legs=round(r_r['P_legs'], 1),
                     rob_CoT=round(r_r['P_legs'] / (MG * r_r['v_meas']), 3),
                     dP_pct=round((r_o['P_elec'] / r_r['P_legs'] - 1) * 100, 1)))
json.dump(rows, open(os.path.join(O, 'speed_table.json'), 'w'), indent=1)
for r in rows:
    print(r)
