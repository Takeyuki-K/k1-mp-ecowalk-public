# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Running video: left = CMU-derived reference (kinematic, what is imitated), right = learned policy (physics).
Overlays: speed, flight indicator, foot forces (heel / forefoot, body weights), leg stiffness (Kp scale = relaxation),
torso pitch. Real time 6 s, then 0.25x slow motion of 2 strides.
python3 video_run.py out/rec_run.npz out/K1_running.mp4
"""
import os, sys
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, imageio, mujoco
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, '..', 'ai_sapiens', 'ai_sapiens_description', 'mujoco', 'k1', 'scene_mp.xml')
W, H = 800, 600
FPS = 50
BG, PANEL, INK, INK2 = (16, 16, 15), (26, 26, 25), (255, 255, 255), (195, 194, 183)
C_REF, C_POL, C_HEEL, C_FORE = (0x9a, 0x9a, 0x95), (0x19, 0x9e, 0x70), (0xe5, 0x8a, 0x39), (0x39, 0x87, 0xe5)
CJK = os.environ.get('K1_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')  # Noto Sans CJK (fonts-noto-cjk)
CJKB = os.environ.get('K1_FONT_BOLD', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')

def _truetype(path, size):
    """font with a fallback: without Noto Sans CJK the captions are drawn with Pillow's default font"""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)
F = lambda s, b=False: _truetype(CJKB if (b and os.path.exists(CJKB)) else CJK, s)


def ref_traj(ph):
    """reference pose at the policy's gait phase (both panels in step)"""
    r = np.load(os.path.join(HERE, 'ref_run.npz'))
    Q, T = r['qpos'], float(r['T'])
    N = len(Q)
    out = []
    cyc = 0
    for k, p in enumerate(ph):
        if k and p < ph[k - 1] - 0.5:
            cyc += 1
        i0 = int(p * N) % N
        q = Q[i0].copy(); q[0] = Q[i0, 0] + cyc * float(r['speed']) * T
        out.append(q)
    return np.array(out)


def main():
    rec = np.load(sys.argv[1]); out = sys.argv[2]
    Qp = rec['qpos']
    n = len(Qp)
    Qr = ref_traj(rec['ph'])
    m = mujoco.MjModel.from_xml_path(XML)
    m.vis.global_.offwidth = max(W, m.vis.global_.offwidth); m.vis.global_.offheight = max(H, m.vis.global_.offheight)
    d = mujoco.MjData(m)
    rd = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 2.6, 90, -6

    def frame(q):
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        cam.lookat[:] = [q[0], q[1] * 0.5, 0.5]
        rd.update_scene(d, cam)
        return rd.render()

    f_t, f_s, f_m, f_b = F(24, True), F(14), F(18), F(34, True)
    # time line: 6 s real time from t=2 s, then slow motion 0.25x of 1.4 s
    k_start = 100
    seq = [(k, 1) for k in range(k_start, min(n, k_start + 6 * FPS))]
    slow0 = k_start + 2 * FPS
    seq += [(k, 4) for k in range(slow0, slow0 + 70)]
    wr = imageio.get_writer(out, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    HDR = 70
    for k, slow in seq:
        A = frame(Qr[k]); B = frame(Qp[k])
        for rep in range(slow):
            im = Image.new('RGB', (2 * W, H + HDR + 150), BG)
            im.paste(Image.fromarray(A), (0, HDR)); im.paste(Image.fromarray(B), (W, HDR))
            dr = ImageDraw.Draw(im, 'RGBA')
            for i, (ti, su, col) in enumerate((
                    ('① 人の走り（CMU mocap → K1へ縮尺・かかと着地に修正）', 'reference: CMU 16_35 jog, scaled to K1, heel-first landing (kinematic, no physics)', C_REF),
                    ('② 学習した走行（物理シミュレーション）', 'learned policy: MP passive toes + human imitation + variable impedance (MuJoCo)', C_POL))):
                x0 = W * i
                dr.rectangle([x0, 0, x0 + W, HDR], fill=PANEL); dr.rectangle([x0, 0, x0 + 8, HDR], fill=col)
                dr.text((x0 + 20, 6), ti, font=f_t, fill=INK); dr.text((x0 + 20, 44), su, font=f_s, fill=INK2)
            if slow > 1:
                dr.rounded_rectangle([W - 110, HDR + H - 56, W + 110, HDR + H - 14], radius=8, fill=(0, 0, 0, 170))
                dr.text((W - 92, HDR + H - 52), 'スロー 0.25× slow', font=f_m, fill=INK)
            # policy overlays (right panel)
            x0 = W
            fl = rec['flight'][k] > 0.5
            dr.rounded_rectangle([x0 + 14, HDR + 14, x0 + 290, HDR + 120], radius=8, fill=(0, 0, 0, 150))
            dr.text((x0 + 26, HDR + 18), f'速度 speed  {rec["vx"][k]:4.2f} m/s', font=f_m, fill=INK)
            dr.text((x0 + 26, HDR + 46), f'体幹前傾 trunk lean {rec["pitch"][k]:+4.1f}°', font=f_m, fill=INK)
            dr.text((x0 + 26, HDR + 76), '空中 FLIGHT' if fl else '接地 stance', font=f_b if fl else f_m,
                    fill=(0xff, 0xd0, 0x40) if fl else INK2)
            # bottom panel: foot force bars + stiffness
            y0 = HDR + H
            dr.rectangle([0, y0, 2 * W, y0 + 150], fill=PANEL)
            MG = 35.706 * 9.81
            for j, (side, hk, fk, kps) in enumerate((('左足 L', 'heelL', 'foreL', slice(0, 6)), ('右足 R', 'heelR', 'foreR', slice(6, 12)))):
                bx = 40 + j * 520
                dr.text((bx, y0 + 8), f'{side} 接地力 foot force (×体重)', font=f_s, fill=INK2)
                for kk, (key, col, lab) in enumerate(((hk, C_HEEL, 'かかと heel'), (fk, C_FORE, '前足 fore'))):
                    v = rec[key][k] / MG
                    yy = y0 + 34 + kk * 30
                    dr.text((bx, yy), lab, font=f_s, fill=col)
                    dr.rectangle([bx + 90, yy + 2, bx + 90 + 300, yy + 20], outline=(70, 70, 68))
                    dr.rectangle([bx + 90, yy + 2, bx + 90 + int(300 * min(v, 3) / 3), yy + 20], fill=col)
                    dr.text((bx + 400, yy), f'{v:3.1f}', font=f_s, fill=INK)
                kp = rec['kp'][k, kps].mean()
                yy = y0 + 100
                dr.text((bx, yy), '剛性 Kp', font=f_s, fill=INK2)
                dr.rectangle([bx + 90, yy + 2, bx + 390, yy + 20], outline=(70, 70, 68))
                dr.rectangle([bx + 90, yy + 2, bx + 90 + int(300 * kp / 1.5), yy + 20], fill=(0x8a, 0x6f, 0xd8))
                dr.text((bx + 400, yy), f'{kp:3.2f}  ' + ('脱力 relaxed' if kp < 0.6 else ''), font=f_s, fill=INK)
            dr.text((1080, y0 + 12), 'Kp scale: 1 = nominal servo stiffness,', font=f_s, fill=INK2)
            dr.text((1080, y0 + 32), '0 = fully relaxed (脱力). Legs mean.', font=f_s, fill=INK2)
            dr.text((1080, y0 + 70), 'MuJoCo simulation only.', font=f_s, fill=INK2)
            dr.text((1080, y0 + 90), 'Mocap: mocap.cs.cmu.edu (NSF EIA-0196217)', font=f_s, fill=INK2)
            dr.text((1080, y0 + 110), 'Idea: Takeyuki-K · Impl.: generated with Claude', font=f_s, fill=INK2)
            wr.append_data(np.array(im))
    wr.close()
    print('saved', out)


if __name__ == '__main__':
    main()
