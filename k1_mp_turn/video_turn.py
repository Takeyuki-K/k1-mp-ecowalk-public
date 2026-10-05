# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Turning demo: one robot, scripted command sequence (stand -> walk -> walking turns -> safe stop ->
in-place turns -> safe stop). Records with the policy, then renders a video with overlays
(command / measured speed and yaw rate, leg stiffness, foot impact, top-view path).
python3 video_turn.py runs/final/model.pt out/K1_turning.mp4
"""
import os, sys
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, torch, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
import eval_turn as E

torch.set_num_threads(1)
FPS = 50
W, H = 1280, 640
BG, PANEL, INK, INK2 = (16, 16, 15), (26, 26, 25), (255, 255, 255), (195, 194, 183)
COL = (0x19, 0x9e, 0x70)
CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CJKB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
F = lambda s, b=False: ImageFont.truetype(CJKB if (b and os.path.exists(CJKB)) else CJK, s)
# (start time s, cmd, v, w, label)
SCRIPT = [(0.0, 0, 0.0, 0.0, '停止 stand'),
          (1.0, 1, 0.6, 0.0, '前進 walk 0.6 m/s'),
          (5.0, 1, 0.6, 0.5, '歩行旋回 左 walking turn L 0.5 rad/s'),
          (10.0, 1, 1.0, -0.6, '歩行旋回 右 walking turn R 1.0 m/s, 0.6 rad/s'),
          (14.0, 1, 1.2, 0.0, '直進 straight 1.2 m/s'),
          (17.0, 0, 0.0, 0.0, '停止指令 STOP command (safe stop)'),
          (21.0, 1, 0.0, 0.8, 'その場旋回 左 in-place turn L 0.8 rad/s'),
          (27.0, 1, 0.0, -0.4, 'その場旋回 右 in-place turn R 0.4 rad/s'),
          (32.0, 0, 0.0, 0.0, '停止指令 STOP command'),
          (35.0, 0, 0.0, 0.0, '')]


def record(path):
    env = E.make_env(1)
    net = E.load(path, env)
    E.stand_all(env)
    T = SCRIPT[-1][0]
    rec = {k: [] for k in ('qpos', 'v', 'w', 'vcmd', 'wcmd', 'cmd', 'kp', 'impact', 'P', 'label', 'alive')}
    oa, _ = env.obs()
    alive = True
    for k in range(int(T * FPS)):
        t = k / FPS
        seg = [s for s in SCRIPT if s[0] <= t][-1]
        env.cmd[:] = seg[1]
        if seg[1]:
            env.v_cmd[:] = seg[2]; env.w_cmd[:] = seg[3]
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        rew, term, trunc, info = env.step(a)
        alive &= not term[0]
        q = env.qpos()[0].copy(); qv = env.qvel()[0]
        yaw = env.yaw()[0]
        rec['qpos'].append(q); rec['v'].append(np.cos(yaw) * qv[0] + np.sin(yaw) * qv[1]); rec['w'].append(qv[5])
        rec['vcmd'].append(seg[2] if seg[1] else 0.0); rec['wcmd'].append(seg[3] if seg[1] else 0.0)
        rec['cmd'].append(seg[1]); rec['kp'].append(env.kp_scale[0].mean()); rec['impact'].append(env.impact[0] / (35.706 * 9.81))
        rec['P'].append(env.P_elec[0]); rec['label'].append(seg[4]); rec['alive'].append(alive)
        oa, _ = env.obs()
    return {k: np.array(v) for k, v in rec.items()}, env


def trailing(x, n):
    c = np.cumsum(np.concatenate([[0], x]))
    i = np.arange(1, len(x) + 1); j = np.maximum(0, i - n)
    return (c[i] - c[j]) / (i - j)


def render(rec, env, out):
    m = env.m
    m.vis.global_.offwidth = max(W, m.vis.global_.offwidth); m.vis.global_.offheight = max(H, m.vis.global_.offheight)
    d = mujoco.MjData(m)
    rd = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 3.2, 120, -18
    Q = rec['qpos']; n = len(Q)
    vs, ws = trailing(rec['v'], 25), trailing(rec['w'], 25)
    look = Q[0, :2].copy()
    f_t, f_m, f_s = F(26, True), F(19), F(14)
    wr = imageio.get_writer(out, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    HDR, BOT = 60, 120
    xy = Q[:, :2]
    lo, hi = xy.min(0) - 0.5, xy.max(0) + 0.5
    sc = 220 / max(hi - lo)
    for k in range(n):
        d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
        look = 0.95 * look + 0.05 * Q[k, :2]
        cam.lookat[:] = [look[0], look[1], 0.45]
        rd.update_scene(d, cam)
        im = Image.new('RGB', (W, H + HDR + BOT), BG)
        im.paste(Image.fromarray(rd.render()), (0, HDR))
        dr = ImageDraw.Draw(im, 'RGBA')
        dr.rectangle([0, 0, W, HDR], fill=PANEL); dr.rectangle([0, 0, 8, HDR], fill=COL)
        dr.text((20, 4), '旋回・その場旋回・安全停止（MP関節＋人の模倣＋脱力）', font=f_t, fill=INK)
        dr.text((20, 38), 'walking turn / in-place turn / safe stop · K1 + passive MP toes + human-gait imitation + learned relaxation · MuJoCo',
                font=f_s, fill=INK2)
        # command box
        dr.rounded_rectangle([14, HDR + 14, 520, HDR + 150], radius=8, fill=(0, 0, 0, 160))
        dr.text((26, HDR + 18), rec['label'][k], font=f_m, fill=(0xff, 0xd0, 0x40))
        dr.text((26, HDR + 52), f'指令 cmd   v {rec["vcmd"][k]:4.2f} m/s   ω {rec["wcmd"][k]:+4.2f} rad/s', font=f_m, fill=INK)
        dr.text((26, HDR + 82), f'実測 meas  v {vs[k]:4.2f} m/s   ω {ws[k]:+4.2f} rad/s', font=f_m, fill=COL)
        dr.text((26, HDR + 114), f't = {k / FPS:5.1f} s', font=f_s, fill=INK2)
        # top view inset
        x0, y0 = W - 260, HDR + 14
        dr.rounded_rectangle([x0, y0, x0 + 246, y0 + 246], radius=8, fill=(0, 0, 0, 160))
        P = lambda p: (x0 + 13 + (p[0] - lo[0]) * sc, y0 + 233 - (p[1] - lo[1]) * sc)
        pts = [P(p) for p in xy[:k + 1:5]]
        if len(pts) > 1:
            dr.line(pts, fill=COL, width=2)
        cx, cy = P(xy[k]); yaw = np.arctan2(2 * (Q[k, 3] * Q[k, 6] + Q[k, 4] * Q[k, 5]), 1 - 2 * (Q[k, 5] ** 2 + Q[k, 6] ** 2))
        dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=INK)
        dr.line([cx, cy, cx + 16 * np.cos(yaw), cy - 16 * np.sin(yaw)], fill=INK, width=2)
        dr.text((x0 + 10, y0 + 4), '軌跡 path (top view)', font=f_s, fill=INK2)
        # bottom bars
        y0 = HDR + H
        dr.rectangle([0, y0, W, y0 + BOT], fill=PANEL)
        for j, (lab, val, vmax, col, fmt) in enumerate((
                ('脚の剛性 leg Kp scale (1 = nominal, 0 = 脱力)', rec['kp'][k], 1.5, (0x8a, 0x6f, 0xd8), '{:.2f}'),
                ('足の接地力 foot force peak (×体重 body weight)', rec['impact'][k], 2.0, (0xe5, 0x8a, 0x39), '{:.2f}'))):
            yy = y0 + 16 + j * 50
            dr.text((20, yy), lab, font=f_s, fill=INK2)
            dr.rectangle([420, yy + 2, 920, yy + 22], outline=(70, 70, 68))
            dr.rectangle([420, yy + 2, 420 + int(500 * min(val, vmax) / vmax), yy + 22], fill=col)
            dr.text((935, yy), fmt.format(val), font=f_m, fill=INK)
        dr.text((1040, y0 + 20), 'MuJoCo simulation only', font=f_s, fill=INK2)
        dr.text((1040, y0 + 44), 'Idea: Takeyuki-K', font=f_s, fill=INK2)
        dr.text((1040, y0 + 68), 'Implementation: Claude', font=f_s, fill=INK2)
        wr.append_data(np.array(im))
    wr.close()


if __name__ == '__main__':
    rec, env = record(sys.argv[1])
    np.savez(sys.argv[2].replace('.mp4', '_rec.npz'), **{k: v for k, v in rec.items() if k != 'label'})
    print('alive at end', rec['alive'][-1])
    render(rec, env, sys.argv[2])
    print('saved', sys.argv[2])
