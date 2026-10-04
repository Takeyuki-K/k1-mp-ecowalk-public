# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Compose the 3-up comparison video (1920x1080, 50 fps) with live power overlays.
inputs : rec_<k>.npz (power, x) and raw_<k>.mp4 (renders) for k in robotis, mp_fixed, eco
output : K1_3way_energy_comparison.mp4
"""
import numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont

import os
D = os.path.dirname(os.path.abspath(__file__))
KEYS = ['robotis', 'mp_fixed', 'eco']
COL = [(0x39, 0x87, 0xe5), (0xd9, 0x59, 0x26), (0x19, 0x9e, 0x70)]
BG, PANEL, INK, INK2, GRID = (16, 16, 15), (26, 26, 25), (255, 255, 255), (195, 194, 183), (60, 60, 58)
TITLE = ['① ROBOTIS公開 歩行ポリシー', '② MP関節 ＋ 人の歩行模倣', '③ MP関節 ＋ 人の模倣 ＋ 脱力（エコ）']
SUBT = ['ROBOTIS public walk_default · original flat foot · fixed gains',
        'passive MP toe joint · human-gait imitation RL · fixed gains',
        'passive MP toe · imitation · variable impedance (relaxation)']
GROUPS = [('股関節ピッチ hip pitch', 'hip_pitch'), ('股関節ロール hip roll', 'hip_roll'), ('股関節ヨー hip yaw', 'hip_yaw'),
          ('膝 knee', 'knee'), ('足首ピッチ ankle pitch', 'ankle_pitch'), ('足首ロール ankle roll', 'ankle_roll'),
          ('腕・腰 arms + waist', None)]
STAND = 2.0
MG = 35.706 * 9.81
FPS = 50
CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CJKB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
try:
    ImageFont.truetype(CJKB, 10)
except Exception:
    CJKB = CJK


def F(sz, bold=False):
    return ImageFont.truetype(CJKB if bold else CJK, sz)


f_title, f_sub, f_big, f_mid, f_sm, f_xs = F(26, True), F(15), F(44, True), F(20, True), F(16), F(13)

# ---------------- data ----------------
R = [np.load(f'{D}/rec_{k}.npz') for k in KEYS]
T = min(len(r['x']) for r in R)
t = np.arange(T) / FPS


def grouped(r):
    names = list(r['names']); P = r['P'][:T]
    out = []
    for _, key in GROUPS:
        if key is None:
            idx = [i for i, n in enumerate(names) if not any(s in n for s in ('hip', 'knee', 'ankle'))]
        else:
            idx = [i for i, n in enumerate(names) if key in n]
        out.append(P[:, idx].sum(1))
    return np.stack(out, 1), P.sum(1)


def trailing(x, n):
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], 0), 0)
    i = np.arange(1, len(x) + 1); j = np.maximum(0, i - n)
    return (c[i] - c[j]) / (i - j).reshape((-1,) + (1,) * (x.ndim - 1))


G, PT, PT1, G05, E, V = [], [], [], [], [], []
for r in R:
    g, p = grouped(r)
    G.append(g); PT.append(p)
    PT1.append(trailing(p, FPS))            # 1 s average total
    G05.append(trailing(g, FPS // 2))       # 0.5 s average per joint group
    E.append(np.cumsum(np.where(t >= STAND, p, 0)) / FPS)   # J since walk start
    x = r['x'][:T]
    V.append(trailing(np.gradient(x) * FPS, FPS))
steady = (t >= STAND + 4)
summary = [dict(P=PT[i][steady].mean(), v=(R[i]['x'][:T][steady][-1] - R[i]['x'][:T][steady][0]) / (steady.sum() / FPS))
           for i in range(3)]
for s in summary:
    s['cot'] = s['P'] / (MG * s['v'])
print(summary)

# ---------------- layout ----------------
WW, HH = 1920, 1080
CW = 640
HDR, VID, ST = 70, 560, 290
TS_Y = HDR + VID + ST          # 920
TS_H = HH - TS_Y               # 160
BAR_MAX = 60.0


def draw_header(dr, i):
    x0 = CW * i
    dr.rectangle([x0, 0, x0 + CW, HDR], fill=PANEL)
    dr.rectangle([x0, 0, x0 + 8, HDR], fill=COL[i])
    dr.text((x0 + 20, 6), TITLE[i], font=f_title, fill=INK)
    dr.text((x0 + 20, 44), SUBT[i], font=f_xs, fill=INK2)


def draw_stats(dr, i, k):
    x0, y0 = CW * i, HDR + VID
    dr.rectangle([x0, y0, x0 + CW, y0 + ST], fill=PANEL)
    dr.text((x0 + 20, y0 + 8), '関節ごとの消費電力（左右合計, 0.5 s平均）  per-joint power', font=f_xs, fill=INK2)
    bx, bw = x0 + 230, 300
    for j, (lab, _) in enumerate(GROUPS):
        y = y0 + 32 + j * 30
        v = G05[i][k, j]
        dr.text((x0 + 20, y), lab, font=f_sm, fill=INK)
        dr.rectangle([bx, y + 4, bx + bw, y + 22], fill=(44, 44, 42))
        L = int(bw * min(v, BAR_MAX) / BAR_MAX)
        if L > 0:
            dr.rounded_rectangle([bx, y + 4, bx + L, y + 22], radius=4, fill=COL[i])
        dr.text((bx + bw + 10, y + 1), f'{v:5.1f} W', font=f_sm, fill=INK)
    yb = y0 + 32 + 7 * 30 + 6
    dr.line([x0 + 20, yb, x0 + CW - 20, yb], fill=GRID)
    dr.text((x0 + 20, yb + 8), f'歩行開始からの消費エネルギー  energy since walk start:  {E[i][k]:6.0f} J', font=f_sm, fill=INK)


def draw_video_overlay(dr, i, k):
    x0, y0 = CW * i, HDR
    p = PT1[i][k]
    dr.rounded_rectangle([x0 + 14, y0 + 14, x0 + 300, y0 + 104], radius=8, fill=(0, 0, 0, 150))
    dr.text((x0 + 26, y0 + 18), '総消費電力 total power (1 s avg)', font=f_xs, fill=INK2)
    dr.text((x0 + 26, y0 + 36), f'{p:5.0f} W', font=f_big, fill=COL[i])
    st = 'STAND 立位' if t[k] < STAND else 'WALK 歩行'
    dr.rounded_rectangle([x0 + CW - 230, y0 + 14, x0 + CW - 14, y0 + 70], radius=8, fill=(0, 0, 0, 150))
    dr.text((x0 + CW - 220, y0 + 18), f't = {t[k]:5.2f} s   {st}', font=f_xs, fill=INK)
    dr.text((x0 + CW - 220, y0 + 40), f'速度 speed {V[i][k]:4.2f} m/s', font=f_sm, fill=INK)


def draw_timeseries(dr, k):
    y0 = TS_Y
    dr.rectangle([0, y0, WW, HH], fill=BG)
    x0, x1 = 90, WW - 300
    ya, yb = y0 + 22, HH - 30
    pmax = 250.0
    def X(tt): return x0 + (x1 - x0) * tt / t[-1]
    def Y(p): return yb - (yb - ya) * min(p, pmax) / pmax
    for p in (0, 100, 200):
        dr.line([x0, Y(p), x1, Y(p)], fill=GRID)
        dr.text((x0 - 50, Y(p) - 9), f'{p} W', font=f_xs, fill=INK2)
    dr.text((x0, y0 + 2), '総消費電力の推移（1 s平均）  total power over time', font=f_xs, fill=INK2)
    dr.line([X(STAND), ya, X(STAND), yb], fill=GRID)
    dr.text((X(STAND) + 4, yb + 6), '歩行開始 walk start', font=f_xs, fill=INK2)
    for i in range(3):
        pts = [(X(t[j]), Y(PT1[i][j])) for j in range(0, k + 1)]
        if len(pts) > 1:
            dr.line(pts, fill=COL[i], width=3)
        dr.ellipse([X(t[k]) - 5, Y(PT1[i][k]) - 5, X(t[k]) + 5, Y(PT1[i][k]) + 5], fill=COL[i], outline=BG, width=2)
    # direct labels at right, ordered by value
    order = sorted(range(3), key=lambda i: -PT1[i][k])
    ys = []
    for n, i in enumerate(order):
        y = Y(PT1[i][k]) - 10
        if ys and y < ys[-1] + 22:
            y = ys[-1] + 22
        ys.append(y)
        dr.text((x1 + 20, y), f'{["①", "②", "③"][i]} {PT1[i][k]:4.0f} W', font=f_mid, fill=COL[i])
    dr.text((x1 + 20, HH - 24), 'MuJoCo sim · same physics · 0.92 m/s', font=f_xs, fill=INK2)


def summary_card():
    im = Image.new('RGB', (WW, HH), BG)
    dr = ImageDraw.Draw(im, 'RGBA')
    dr.text((90, 70), '定常歩行（約0.9 m/s）の平均消費電力  average power, steady straight walking', font=F(34, True), fill=INK)
    dr.text((90, 120), '同一物理条件・同一速度・同一電力モデル（MuJoCoシミュレーション）', font=F(20), fill=INK2)
    b = summary[0]
    for i in range(3):
        y = 220 + i * 210
        s = summary[i]
        dr.rectangle([90, y, 98, y + 160], fill=COL[i])
        dr.text((120, y), TITLE[i], font=F(30, True), fill=INK)
        dr.text((120, y + 44), SUBT[i], font=F(16), fill=INK2)
        L = int(900 * s['P'] / 200)
        dr.rounded_rectangle([120, y + 80, 120 + L, y + 140], radius=6, fill=COL[i])
        txt = f"{s['P']:.0f} W    CoT {s['cot']:.2f}"
        if i:
            txt += f"    {(s['P'] / b['P'] - 1) * 100:+.0f}% vs ①"
        dr.text((140 + L, y + 88), txt, font=F(32, True), fill=INK)
    dr.text((90, HH - 60), 'Electrical model: Σ τ²/Km² + Σ max(τ·ω, 0), Km assumed (4.0 / 2.2 Nm/√W). '
            'Simulation only. Independent research, not affiliated with ROBOTIS.', font=F(16), fill=INK2)
    return np.array(im)


readers = [imageio.get_reader(f'{D}/raw_{k}.mp4') for k in KEYS]
wr = imageio.get_writer(f'{D}/K1_3way_energy_comparison.mp4', fps=FPS, codec='libx264', quality=8, macro_block_size=1)
for k in range(T):
    im = Image.new('RGB', (WW, HH), BG)
    for i in range(3):
        im.paste(Image.fromarray(readers[i].get_data(k)), (CW * i, HDR))
    dr = ImageDraw.Draw(im, 'RGBA')
    for i in range(3):
        draw_header(dr, i); draw_stats(dr, i, k); draw_video_overlay(dr, i, k)
        if i:
            dr.line([CW * i, 0, CW * i, TS_Y], fill=BG, width=4)
    draw_timeseries(dr, k)
    wr.append_data(np.array(im))
card = summary_card()
for _ in range(FPS * 5):
    wr.append_data(card)
wr.close()
Image.fromarray(card).save(f'{D}/summary_card.png')
print('saved')
