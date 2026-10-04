# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('res_robotis.json')); F = json.load(open('res_mp_fixed.json'))['mp_fixed']
E = json.load(open('res_eco.json'))['eco']
MG = 35.706 * 9.81
C = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#ecebe7'
rows = [('(1) ROBOTIS public walk_default\n(original K1, no MP)', R['robotis_v0.9']),
        ('(2) MP joint + human-gait RL\n(fixed gains)', F),
        ('(3) MP joint + eco\n(variable impedance)', E)]
base = R['robotis_v0.9']
table = []
for name, r in rows:
    cot = r['P_total'] / (MG * r['speed'])
    table.append(dict(controller=name.replace('\n', ' '), speed=round(r['speed'], 3), P_total_W=round(r['P_total'], 1),
                      P_legs_W=round(r['P_legs'], 1), legs_copper_W=round(r['legs_copper'], 1),
                      legs_pos_mech_W=round(r['legs_pos_mech'], 1), legs_neg_mech_W=round(r['legs_neg_mech'], 1),
                      CoT_elec=round(cot, 3),
                      CoT_vs_1=f"{(cot / (base['P_total'] / (MG * base['speed'])) - 1) * 100:+.0f}%"))
with open('comparison_table.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
for t in table:
    print(t)


def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c9c8c3')
    ax.tick_params(colors=INK2)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)


# ---- Fig 1: total power and CoT (two separate panels, one axis each) ----
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
fig.patch.set_facecolor(SURF)
labels = ['(1) ROBOTIS\nwalk_default', '(2) MP +\nhuman gait', '(3) MP +\neco']
P = [r['P_total'] for _, r in rows]
CoT = [r['P_total'] / (MG * r['speed']) for _, r in rows]
for ax, vals, ttl, fmt in [(axs[0], P, 'Electrical power while walking (W)', '{:.0f} W'),
                           (axs[1], CoT, 'Cost of transport (lower = more efficient)', '{:.2f}')]:
    style(ax)
    b = ax.bar(range(3), vals, 0.6, color=C)
    for i, v in enumerate(vals):
        txt = fmt.format(v) + ('' if i == 0 else f'\n({(v / vals[0] - 1) * 100:+.0f}%)')
        ax.text(i, v + max(vals) * 0.02, txt, ha='center', va='bottom', fontsize=9, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, color=INK2, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(ttl, loc='left', color=INK, fontsize=11)
fig.suptitle('K1 walking at ~0.9 m/s, same MuJoCo conditions (sim only, Km assumed)', x=0.01, ha='left',
             color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig('fig1_power_cot.png', facecolor=SURF)

# ---- Fig 2: power vs speed (ROBOTIS at 3 speeds, ours at its single speed) ----
fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150); fig.patch.set_facecolor(SURF); style(ax)
rv = [R[k] for k in ('robotis_v0.5', 'robotis_v0.7', 'robotis_v0.9')]
ax.plot([r['speed'] for r in rv], [r['P_total'] for r in rv], color=C[0], lw=2, marker='o', ms=8,
        markeredgecolor=SURF, markeredgewidth=2)
ax.text(rv[0]['speed'], rv[0]['P_total'] + 6, '(1) ROBOTIS walk_default\n(commanded 0.5 / 0.7 / 0.9 m/s)', color=INK, fontsize=8)
for r, c, nm, dy in [(F, C[1], '(2) MP + human gait', 6), (E, C[2], '(3) MP + eco', -14)]:
    ax.plot([r['speed']], [r['P_total']], 'o', color=c, ms=9, markeredgecolor=SURF, markeredgewidth=2)
    ax.text(r['speed'] - 0.01, r['P_total'] + dy, nm, color=INK, fontsize=8, ha='right')
ax.set_xlim(0.4, 1.0); ax.set_ylim(0, 200)
ax.set_xlabel('measured walking speed (m/s)', color=INK2); ax.set_ylabel('electrical power (W)', color=INK2)
ax.set_title('Power vs speed', loc='left', color=INK, fontsize=11)
fig.tight_layout(); fig.savefig('fig2_power_vs_speed.png', facecolor=SURF)

# ---- Fig 3: per-joint breakdown (legs, both sides) ----
keys = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=150); fig.patch.set_facecolor(SURF); style(ax)
x = np.arange(len(keys)); w = 0.26
for i, ((nm, r), c, lab) in enumerate(zip(rows, C, ['(1) ROBOTIS walk_default', '(2) MP + human gait', '(3) MP + eco'])):
    v = [r['legs_' + k] for k in keys]
    ax.bar(x + (i - 1) * (w + 0.01), v, w, color=c, label=lab)
    for xi, vi in zip(x, v):
        ax.text(xi + (i - 1) * (w + 0.01), vi + 0.8, f'{vi:.0f}', ha='center', fontsize=7, color=INK)
ax.set_xticks(x); ax.set_xticklabels([k.replace('_', ' ') for k in keys], color=INK2)
ax.set_ylabel('W (both legs)', color=INK2)
ax.set_title('Where the power goes (legs, ~0.9 m/s)', loc='left', color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig('fig3_per_joint.png', facecolor=SURF)
print('ok')
