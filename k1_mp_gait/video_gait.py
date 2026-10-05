# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Video of the gait manager: profile 'accel' (stand -> walk, +0.15 m/s steps -> run to top speed -> walk -> stop)
or 'standrun'. Overlays: gait mode, command vs actual speed, foot contact (heel / forefoot), leg stiffness, power.
python3 video_gait.py WALK.pt RUN.pt accel out/K1_walk_run.mp4
"""
import os, sys
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
from eval_gait import run, FPS
from k1env_run2 import MG

W, H = 1280, 600
BG, PANEL, INK, INK2 = (16, 16, 15), (26, 26, 25), (255, 255, 255), (195, 194, 183)
C_WALK, C_RUN, C_HEEL, C_FORE = (0x19, 0x9e, 0x70), (0xe5, 0x5a, 0x39), (0xe5, 0x8a, 0x39), (0x39, 0x87, 0xe5)
CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CJKB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
F = lambda s, b=False: ImageFont.truetype(CJKB if (b and os.path.exists(CJKB)) else CJK, s)


def trailing(x, n):
    c = np.cumsum(np.concatenate([[0], x]))
    i = np.arange(1, len(x) + 1); j = np.maximum(0, i - n)
    return (c[i] - c[j]) / (i - j)


def main():
    walk, runp, prof, out = sys.argv[1:5]
    R, L = run(walk, runp, prof, 1)
    print(R)
    Q = L['qpos'][:, 0]; mode = L['mode'][:, 0]; vu = L['v_user']
    v = trailing(np.gradient(Q[:, 0]) * FPS, 25)
    Pw = trailing(L['P'][:, 0], 50)
    from k1env_run2 import K1Run2Batch
    m = K1Run2Batch(1, nthread=1).m
    m.vis.global_.offwidth = max(W, m.vis.global_.offwidth); m.vis.global_.offheight = max(H, m.vis.global_.offheight)
    d = mujoco.MjData(m); rd = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 3.4, 90, -6
    f_t, f_m, f_s, f_b = F(26, True), F(19), F(14), F(40, True)
    HDR, BOT = 64, 120
    wr = imageio.get_writer(out, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    ly = Q[0, 1]
    sw = [(t, s) for t, i, s in R['switches']]
    for k in range(len(Q)):
        d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
        ly = 0.95 * ly + 0.05 * Q[k, 1]
        cam.lookat[:] = [Q[k, 0], ly, 0.5]
        rd.update_scene(d, cam)
        im = Image.new('RGB', (W, H + HDR + BOT), BG)
        im.paste(Image.fromarray(rd.render()), (0, HDR))
        dr = ImageDraw.Draw(im, 'RGBA')
        col = C_RUN if mode[k] else C_WALK
        dr.rectangle([0, 0, W, HDR], fill=PANEL); dr.rectangle([0, 0, 8, HDR], fill=col)
        dr.text((20, 4), '歩行 ⇄ 走行の自動切替（立位→歩行→走行→歩行→停止）', font=f_t, fill=INK)
        dr.text((20, 40), 'gait manager: eco walking policy (heel strike) ⇄ running policy (mid/forefoot strike) · motor torque-speed limit 11.5 rad/s · MuJoCo',
                font=f_s, fill=INK2)
        dr.rounded_rectangle([14, HDR + 14, 430, HDR + 160], radius=8, fill=(0, 0, 0, 160))
        standing = (mode[k] == 0) and v[k] < 0.05 and vu[k] == 0
        lab = '走行 RUN' if mode[k] else ('停止 STAND' if standing else '歩行 WALK')
        dr.text((26, HDR + 16), lab, font=f_b, fill=col)
        dr.text((26, HDR + 70), f'指令 command  {vu[k]:4.2f} m/s', font=f_m, fill=INK)
        dr.text((26, HDR + 98), f'実速度 actual  {v[k]:4.2f} m/s', font=f_m, fill=col)
        dr.text((26, HDR + 128), f't = {k / FPS:4.1f} s', font=f_s, fill=INK2)
        recent = [s for t, s in sw if 0 <= k / FPS - t < 2.0]
        if recent:
            dr.rounded_rectangle([W - 380, HDR + 14, W - 14, HDR + 70], radius=8, fill=(0, 0, 0, 170))
            dr.text((W - 362, HDR + 22), '切替 ' + recent[-1], font=F(28, True), fill=(0xff, 0xd0, 0x40))
        y0 = HDR + H
        dr.rectangle([0, y0, W, y0 + BOT], fill=PANEL)
        for j, (side, hk, fk) in enumerate((('左 L', 'heelL', 'foreL'), ('右 R', 'heelR', 'foreR'))):
            bx = 20 + j * 390
            dr.text((bx, y0 + 8), f'{side} 接地力 foot force (×体重)', font=f_s, fill=INK2)
            for kk, (key, c, lb) in enumerate(((hk, C_HEEL, 'かかと heel'), (fk, C_FORE, '前足 fore'))):
                val = L[key][k, 0] / MG; yy = y0 + 32 + kk * 28
                dr.text((bx, yy), lb, font=f_s, fill=c)
                dr.rectangle([bx + 90, yy + 2, bx + 330, yy + 20], outline=(70, 70, 68))
                dr.rectangle([bx + 90, yy + 2, bx + 90 + int(240 * min(val, 4) / 4), yy + 20], fill=c)
                dr.text((bx + 340, yy), f'{val:3.1f}', font=f_s, fill=INK)
        bx = 800
        dr.text((bx, y0 + 8), f'脚の剛性 Kp {L["kp"][k, 0]:4.2f}   消費電力 power (1 s) {Pw[k]:5.0f} W', font=f_s, fill=INK)
        dr.text((bx, y0 + 40), 'walking: eco reward · running: stability & speed', font=f_s, fill=INK2)
        dr.text((bx, y0 + 64), 'MuJoCo sim · mocap.cs.cmu.edu (NSF EIA-0196217)', font=f_s, fill=INK2)
        dr.text((bx, y0 + 88), 'Idea: Takeyuki-K · Implementation: Claude', font=f_s, fill=INK2)
        wr.append_data(np.array(im))
    wr.close()
    print('saved', out)


if __name__ == '__main__':
    main()
