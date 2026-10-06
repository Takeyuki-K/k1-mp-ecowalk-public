# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Fast-running demo: speed command 2 -> 3 -> 4 -> 5 m/s (one robot, no pushes), side view tracking camera,
then 0.25x slow motion at 5 m/s. Overlays: command / measured speed, flight, foot force (heel / forefoot),
leg joint speed vs the 11.5 rad/s URDF limit, trunk lean.
python3 video_sprint.py runs/final/model.pt out/K1_sprint.mp4 [label]
"""
import os, sys
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, torch, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
from k1env_sprint import K1SprintBatch, MG, QD_LIM
from ppo_eco import AC

torch.set_num_threads(1)
FPS = 50
W, H = 1280, 600
BG, PANEL, INK, INK2 = (16, 16, 15), (26, 26, 25), (255, 255, 255), (195, 194, 183)
COL, C_HEEL, C_FORE, C_QD = (0x19, 0x9e, 0x70), (0xe5, 0x8a, 0x39), (0x39, 0x87, 0xe5), (0xd8, 0x4a, 0x4a)
CJK = os.environ.get('K1_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')  # Noto Sans CJK (fonts-noto-cjk)
CJKB = os.environ.get('K1_FONT_BOLD', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')

def _truetype(path, size):
    """font with a fallback: without Noto Sans CJK the captions are drawn with Pillow's default font"""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)
F = lambda s, b=False: _truetype(CJKB if (b and os.path.exists(CJKB)) else CJK, s)
PROFILE = [(0.0, 2.0), (3.0, 3.0), (6.0, 4.0), (9.0, 5.5), (15.0, None)]


def record(path):
    env = K1SprintBatch(1, v_lo=2.0, v_hi=2.0, stage=1, randomize=False, seed=3, nthread=1, ep_len=10 ** 9)
    env.pushes = False; env.assist = 0.0; env.scripted = True
    oa, oc = env.obs()
    net = AC(oa.shape[1], oc.shape[1], 36)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    env.reset(np.arange(1))
    oa, _ = env.obs()
    R = {k: [] for k in ('qpos', 'vcmd', 'vx', 'flight', 'heelL', 'heelR', 'foreL', 'foreR', 'qdr', 'pitch', 'alive')}
    alive = True
    T = PROFILE[-1][0]
    for k in range(int(T * FPS)):
        t = k / FPS
        env.v_cmd[:] = [v for ts, v in PROFILE if ts <= t and v is not None][-1]
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        rew, term, trunc, info = env.step(a)
        alive &= not term[0]
        sd = env.sd_out
        f = {(s, p): sd[0, :, env.sens[f'{s}_{p}_touch'][0]].max() for s in ('left', 'right') for p in ('heel', 'meta', 'toe')}
        qd = np.abs(env.st_out[0, :, 1 + env.m.nq:1 + env.m.nq + env.m.nv][:, env.dadr[:12]]).max(0) / QD_LIM
        quat, g = env.base_frame()
        R['qpos'].append(env.qpos()[0].copy()); R['vcmd'].append(env.v_cmd[0]); R['vx'].append(env.qvel()[0, 0])
        R['flight'].append(info['flight'][0]); R['heelL'].append(f['left', 'heel']); R['heelR'].append(f['right', 'heel'])
        R['foreL'].append(max(f['left', 'meta'], f['left', 'toe'])); R['foreR'].append(max(f['right', 'meta'], f['right', 'toe']))
        R['qdr'].append(qd.max()); R['pitch'].append(np.degrees(np.arcsin(np.clip(g[0, 0], -1, 1)))); R['alive'].append(alive)
        oa, _ = env.obs()
    return {k: np.array(v) for k, v in R.items()}, env


def trailing(x, n):
    c = np.cumsum(np.concatenate([[0], x]))
    i = np.arange(1, len(x) + 1); j = np.maximum(0, i - n)
    return (c[i] - c[j]) / (i - j)


def render(R, env, out, label):
    m = env.m
    m.vis.global_.offwidth = max(W, m.vis.global_.offwidth); m.vis.global_.offheight = max(H, m.vis.global_.offheight)
    d = mujoco.MjData(m)
    rd = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 3.4, 90, -6
    Q = R['qpos']; n = len(Q)
    vs = trailing(np.gradient(Q[:, 0]) * FPS, 25)
    f_t, f_m, f_s, f_b = F(26, True), F(19), F(14), F(40, True)
    HDR, BOT = 64, 130
    seq = [(k, 1) for k in range(n)] + [(k, 4) for k in range(n - 80, n - 20)]
    wr = imageio.get_writer(out, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    look = Q[0, :2].copy()
    for k, slow in seq:
        d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
        look = np.array([Q[k, 0], 0.9 * look[1] + 0.1 * Q[k, 1]])
        cam.lookat[:] = [look[0], look[1], 0.5]
        rd.update_scene(d, cam)
        frame = rd.render()
        for _ in range(slow):
            im = Image.new('RGB', (W, H + HDR + BOT), BG)
            im.paste(Image.fromarray(frame), (0, HDR))
            dr = ImageDraw.Draw(im, 'RGBA')
            dr.rectangle([0, 0, W, HDR], fill=PANEL); dr.rectangle([0, 0, 8, HDR], fill=COL)
            dr.text((20, 4), f'高速走行 2 → 5 m/s超（CMU模倣・かかと着地・MP関節）{label}', font=f_t, fill=INK)
            dr.text((20, 40), 'fast running, speed command 2 → 5.5 m/s · CMU 09_04 style prior · heel-first landing · passive MP toes · MuJoCo',
                    font=f_s, fill=INK2)
            dr.rounded_rectangle([14, HDR + 14, 400, HDR + 150], radius=8, fill=(0, 0, 0, 160))
            dr.text((26, HDR + 18), f'指令 command  {R["vcmd"][k]:4.2f} m/s', font=f_m, fill=INK)
            dr.text((26, HDR + 46), f'{vs[k]:4.2f} m/s', font=f_b, fill=COL)
            dr.text((210, HDR + 66), '実速度 actual (0.5 s avg)', font=f_s, fill=INK2)
            dr.text((26, HDR + 106), f'体幹前傾 trunk lean {R["pitch"][k]:+4.1f}°   t = {k / FPS:4.1f} s', font=f_s, fill=INK2)
            if R['flight'][k] > 0.5:
                dr.rounded_rectangle([W - 290, HDR + 14, W - 14, HDR + 76], radius=8, fill=(0, 0, 0, 160))
                dr.text((W - 272, HDR + 18), '空中 FLIGHT', font=f_b, fill=(0xff, 0xd0, 0x40))
            if slow > 1:
                dr.rounded_rectangle([W // 2 - 110, HDR + H - 56, W // 2 + 110, HDR + H - 14], radius=8, fill=(0, 0, 0, 170))
                dr.text((W // 2 - 92, HDR + H - 52), 'スロー 0.25× slow', font=f_m, fill=INK)
            y0 = HDR + H
            dr.rectangle([0, y0, W, y0 + BOT], fill=PANEL)
            for j, (side, hk, fk) in enumerate((('左 L', 'heelL', 'foreL'), ('右 R', 'heelR', 'foreR'))):
                bx = 20 + j * 400
                dr.text((bx, y0 + 8), f'{side} 接地力 foot force (×体重)', font=f_s, fill=INK2)
                for kk, (key, col, lab) in enumerate(((hk, C_HEEL, 'かかと heel'), (fk, C_FORE, '前足 fore'))):
                    v = R[key][k] / MG; yy = y0 + 34 + kk * 30
                    dr.text((bx, yy), lab, font=f_s, fill=col)
                    dr.rectangle([bx + 90, yy + 2, bx + 330, yy + 20], outline=(70, 70, 68))
                    dr.rectangle([bx + 90, yy + 2, bx + 90 + int(240 * min(v, 4) / 4), yy + 20], fill=col)
                    dr.text((bx + 340, yy), f'{v:3.1f}', font=f_s, fill=INK)
            bx = 820
            dr.text((bx, y0 + 8), '脚の最大関節速度 / 仕様上限 11.5 rad/s', font=f_s, fill=INK2)
            r = R['qdr'][k]
            dr.rectangle([bx, y0 + 36, bx + 300, y0 + 56], outline=(70, 70, 68))
            dr.rectangle([bx, y0 + 36, bx + int(300 * min(r, 2) / 2), y0 + 56], fill=C_QD if r > 1 else COL)
            dr.line([bx + 150, y0 + 30, bx + 150, y0 + 62], fill=INK, width=2)
            dr.text((bx + 310, y0 + 36), f'{100 * r:4.0f} %', font=f_s, fill=INK)
            dr.text((bx, y0 + 70), 'MuJoCo sim only · mocap.cs.cmu.edu (NSF EIA-0196217)', font=f_s, fill=INK2)
            dr.text((bx, y0 + 92), 'Idea: Takeyuki-K · Impl.: generated with Claude', font=f_s, fill=INK2)
            wr.append_data(np.array(im))
    wr.close()


if __name__ == '__main__':
    R, env = record(sys.argv[1])
    print('alive at end', R['alive'][-1], 'mean speed last 4 s', np.gradient(R['qpos'][-200:, 0]).mean() * FPS)
    np.savez(sys.argv[2].replace('.mp4', '_rec.npz'), **R)
    render(R, env, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else '')
    print('saved', sys.argv[2])
