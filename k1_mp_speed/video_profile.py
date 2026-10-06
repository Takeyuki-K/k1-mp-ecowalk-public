# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""2-up video on the same speed profile: ROBOTIS walk_default vs MP + imitation + relaxation (speed command).
python3 video_profile.py render robotis|speed      (raw render, slow: run both in parallel)
python3 video_profile.py compose                    -> out/K1_speed_command_comparison.mp4
"""
import os, sys
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, imageio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W, H = 960, 560
FPS = 50
COL = [(0x39, 0x87, 0xe5), (0x19, 0x9e, 0x70)]
BG, PANEL, INK, INK2, GRID = (16, 16, 15), (26, 26, 25), (255, 255, 255), (195, 194, 183), (60, 60, 58)
MG = 35.706 * 9.81


def render(k):
    import mujoco
    r = np.load(os.path.join(HERE, 'out', f'prof_{k}.npz'))
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, 'ai_sapiens', 'ai_sapiens_description', 'mujoco', 'k1', str(r['xml'])))
    m.vis.global_.offwidth = max(W, m.vis.global_.offwidth); m.vis.global_.offheight = max(H, m.vis.global_.offheight)
    d = mujoco.MjData(m)
    rd = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.0, 90, -8
    wr = imageio.get_writer(os.path.join(HERE, 'out', f'raw_prof_{k}.mp4'), fps=FPS, codec='libx264', quality=9,
                            macro_block_size=1)
    ys = None
    for q in r['qpos']:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        ys = q[1] if ys is None else 0.9 * ys + 0.1 * q[1]
        cam.lookat[:] = [q[0], ys, 0.48]
        rd.update_scene(d, cam)
        wr.append_data(rd.render())
    wr.close()
    print('rendered', k)


def compose():
    from PIL import Image, ImageDraw, ImageFont
    CJK = os.environ.get('K1_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')  # Noto Sans CJK (fonts-noto-cjk)
    CJKB = os.environ.get('K1_FONT_BOLD', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')

    def _truetype(path, size):
        """font with a fallback: without Noto Sans CJK the captions are drawn with Pillow's default font"""
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default(size)
    if not os.path.exists(CJKB):
        CJKB = CJK
    F = lambda s, b=False: _truetype(CJKB if b else CJK, s)
    f_t, f_s, f_big, f_mid, f_sm, f_xs = F(26, True), F(15), F(40, True), F(20, True), F(16), F(13)
    R = [np.load(os.path.join(HERE, 'out', f'prof_{k}.npz')) for k in ('robotis', 'speed')]
    T = min(len(r['qpos']) for r in R)
    t = np.arange(T) / FPS
    vcmd = R[0]['v_cmd'][:T]

    def trailing(x, n):
        c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x], 0), 0)
        i = np.arange(1, len(x) + 1); j = np.maximum(0, i - n)
        return (c[i] - c[j]) / (i - j).reshape((-1,) + (1,) * (x.ndim - 1))
    PT = [trailing(r['P'][:T].sum(1), FPS) for r in R]
    V = [trailing(np.gradient(r['qpos'][:T, 0]) * FPS, FPS) for r in R]
    E = [np.cumsum(r['P'][:T].sum(1)) / FPS for r in R]
    X = [r['qpos'][:T, 0] - r['qpos'][0, 0] for r in R]
    TITLE = ['① ROBOTIS公開 歩行ポリシー（walk_default）', '② MP関節 ＋ 人の模倣 ＋ 脱力（速度指令対応）']
    SUB = ['ROBOTIS public walk_default · original flat foot · fixed gains · velocity command',
           'passive MP toe · human-gait imitation (speed-scaled) · learned relaxation · velocity command']
    readers = [imageio.get_reader(os.path.join(HERE, 'out', f'raw_prof_{k}.mp4')) for k in ('robotis', 'speed')]
    out = os.path.join(HERE, 'out', 'K1_speed_command_comparison.mp4')
    wr = imageio.get_writer(out, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    HDR = 64
    CH_Y = HDR + H                     # 624
    WW, HH = 1920, 1080

    def chart(dr, x0, y0, w, h, series, cols, ymax, k, title, unit, dashed=None):
        dr.rectangle([x0, y0, x0 + w, y0 + h], fill=BG)
        dr.text((x0 + 60, y0 + 4), title, font=f_xs, fill=INK2)
        xa, xb, ya, yb = x0 + 60, x0 + w - 20, y0 + 26, y0 + h - 26
        X_ = lambda tt: xa + (xb - xa) * tt / t[-1]
        Y_ = lambda v: yb - (yb - ya) * min(max(v, 0), ymax) / ymax
        for g in np.linspace(0, ymax, 3):
            dr.line([xa, Y_(g), xb, Y_(g)], fill=GRID)
            dr.text((x0 + 4, Y_(g) - 9), f'{g:.0f}{unit}' if ymax > 10 else f'{g:.1f}{unit}', font=f_xs, fill=INK2)
        if dashed is not None:
            pts = [(X_(t[j]), Y_(dashed[j])) for j in range(0, k + 1, 2)]
            for a_, b_ in zip(pts[::2], pts[1::2]):
                dr.line([a_, b_], fill=INK, width=2)
        for s, c in zip(series, cols):
            pts = [(X_(t[j]), Y_(s[j])) for j in range(0, k + 1)]
            if len(pts) > 1:
                dr.line(pts, fill=c, width=3)
            dr.ellipse([X_(t[k]) - 5, Y_(s[k]) - 5, X_(t[k]) + 5, Y_(s[k]) + 5], fill=c, outline=BG, width=2)

    for k in range(T):
        im = Image.new('RGB', (WW, HH), BG)
        for i in range(2):
            im.paste(Image.fromarray(readers[i].get_data(k)), (W * i, HDR))
        dr = ImageDraw.Draw(im, 'RGBA')
        for i in range(2):
            x0 = W * i
            dr.rectangle([x0, 0, x0 + W, HDR], fill=PANEL)
            dr.rectangle([x0, 0, x0 + 8, HDR], fill=COL[i])
            dr.text((x0 + 20, 4), TITLE[i], font=f_t, fill=INK)
            dr.text((x0 + 20, 40), SUB[i], font=f_xs, fill=INK2)
            dr.rounded_rectangle([x0 + 14, HDR + 14, x0 + 330, HDR + 104], radius=8, fill=(0, 0, 0, 150))
            dr.text((x0 + 26, HDR + 18), '総消費電力 total power (1 s avg)', font=f_xs, fill=INK2)
            dr.text((x0 + 26, HDR + 38), f'{PT[i][k]:5.0f} W', font=f_big, fill=COL[i])
            dr.rounded_rectangle([x0 + W - 300, HDR + 14, x0 + W - 14, HDR + 128], radius=8, fill=(0, 0, 0, 150))
            dr.text((x0 + W - 288, HDR + 18), f't = {t[k]:5.2f} s', font=f_xs, fill=INK)
            dr.text((x0 + W - 288, HDR + 40), f'指令 command  {vcmd[k]:4.2f} m/s', font=f_sm, fill=INK)
            dr.text((x0 + W - 288, HDR + 64), f'実速度 actual {V[i][k]:4.2f} m/s', font=f_sm, fill=COL[i])
            dr.text((x0 + W - 288, HDR + 90), f'累積 energy {E[i][k]:5.0f} J', font=f_sm, fill=INK2)
            if i:
                dr.line([W, 0, W, CH_Y], fill=BG, width=4)
        chart(dr, 0, CH_Y, 960, HH - CH_Y, V, COL, 1.6, k, '速度 speed: 指令（白破線）と実速度  command (white dashed) vs actual', ' ', dashed=vcmd)
        chart(dr, 960, CH_Y, 960, HH - CH_Y - 30, PT, COL, 300, k, '総消費電力 total power (1 s avg)', 'W')
        dr.text((980, HH - 26), 'MuJoCo sim · same physics · same speed profile · electrical model Σ τ²/Km² + Σ max(τω,0), Km assumed',
                font=f_xs, fill=INK2)
        wr.append_data(np.array(im))
    # summary card
    im = Image.new('RGB', (WW, HH), BG); dr = ImageDraw.Draw(im)
    dr.text((90, 70), '同じ速度指令プロファイルでの比較  same speed-command profile (30 s)', font=F(34, True), fill=INK)
    walk = vcmd > 0
    for i in range(2):
        y = 200 + i * 260
        Ewalk = (R[i]['P'][:T].sum(1) * walk).sum() / FPS
        dist = X[i][-1]
        err = np.abs(V[i] - vcmd)[walk & (t > 3)].mean()
        dr.rectangle([90, y, 98, y + 200], fill=COL[i])
        dr.text((120, y), TITLE[i], font=F(30, True), fill=INK)
        dr.text((120, y + 50), f'距離 distance {dist:5.1f} m    消費エネルギー energy {Ewalk:6.0f} J    '
                f'1 mあたり {Ewalk / dist:5.0f} J/m', font=F(26), fill=INK)
        dr.text((120, y + 95), f'平均消費電力 mean power while walking {(R[i]["P"][:T].sum(1))[walk].mean():5.0f} W    '
                f'速度誤差 mean |v − v_cmd| {err:4.2f} m/s', font=F(26), fill=INK)
    dr.text((90, 780), '速度誤差は加減速中の遅れを含みます（②は参照速度の加速度を人に近い0.6 m/s²に制限）。定常歩行での誤差は ② 1〜4 %、① 0〜7 %。',
            font=F(20), fill=INK2)
    dr.text((90, 815), 'Speed error includes acceleration lag (② limits reference acceleration to 0.6 m/s², human-like). '
            'Steady-state error: ② 1-4 %, ① 0-7 %.', font=F(18), fill=INK2)
    dr.text((90, HH - 60), 'Simulation only. Independent research by Takeyuki-K, not affiliated with ROBOTIS. '
            'Km assumed (4.0 / 2.2 Nm/√W).', font=F(16), fill=INK2)
    card = np.array(im)
    for _ in range(FPS * 5):
        wr.append_data(card)
    wr.close()
    im.save(os.path.join(HERE, 'out', 'speed_summary_card.png'))
    print('saved', out)


if __name__ == '__main__':
    if sys.argv[1] == 'render':
        render(sys.argv[2])
    else:
        compose()
