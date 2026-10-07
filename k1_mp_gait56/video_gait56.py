# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Video of v5.5 through the gait manager (one robot, no pushes): the eval_gait55 PROFILE
(stand -> walk -> walking turn -> run -> running turns -> governor -> hard braking) or 'inplace' / 'brake'.
Overlays: gait mode, command -> governed -> actual speed, yaw rate, lean (measured vs atan(v w / g)), top-view path.
MUJOCO_GL=osmesa python3 video_gait56.py runs/final/walk.pt runs/final/run.pt profile out/K1_v55_turn_run.mp4
"""
import os, sys
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
from gait56 import Gait56 as Gait55
import eval_gait56 as E

W, H = 1280, 600
HDR, BOT = 64, 120
FPS = 25                        # rendered (simulation 50 Hz, every 2nd step)
BG, PANEL, INK, INK2 = (16, 16, 15), (26, 26, 25), (255, 255, 255), (195, 194, 183)
C_WALK, C_RUN, C_BRAKE, C_GOV = (0x19, 0x9e, 0x70), (0xe5, 0x5a, 0x39), (0xe0, 0x3c, 0x3c), (0xff, 0xd0, 0x40)
CJK = os.environ.get('K1_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
CJKB = os.environ.get('K1_FONT_BOLD', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')


def _tt(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


F = lambda s, b=False: _tt(CJKB if (b and os.path.exists(CJKB)) else CJK, s)

PROFILES = {
    'profile': (E.PROFILE, '歩行・走行中の旋回、旋回時の自動減速、急停止 (v5.6)', 150),
    'inplace': (E.INPLACE, 'その場旋回 ±0.6 rad/s (v5.6)', 150),
    'straight': ([(2, 'cmd', 0.0, 0.0), (10, 'cmd', 1.0, 0.0), (5, 'stop', 0, 0)], '直進 1.0 m/s を正面から (v5.6)', 180),
    'inplace_stop': ([(2, 'cmd', 0.0, 0.0), (6, 'cmd', 0.0, 0.6), (6, 'stop', 0, 0)], 'その場旋回 → 停止 (v5.6)', 150),
    'brake': ([(2, 'cmd', 0.0, 0.0), (5, 'cmd', 1.65, 0.0), (6, 'cmd', 4.5, 0.0), (7, 'brake', 0, 0)],
              '4.5 m/s からの急停止 (v5.6、側面)', 90),
}
LABEL = {'cmd': '指令', 'stop': '停止', 'brake': '急停止'}


def record(g, prof):
    g.reset(); E.stand_noise(g)
    g._oaw, _ = g.W.obs(); g._oar, _ = g.R.obs()
    L = {k: [] for k in ('q', 'mode', 'brake', 'vu', 'wu', 'vgov', 'vref', 'v', 'wref', 'w', 'lean', 'phi', 'act', 'P')}
    for T, act, v, w in prof:
        {'cmd': lambda: g.command(v, w), 'stop': g.stop, 'brake': g.brake}[act]()
        for _ in range(int(T * 50)):
            g.step()
            e = g.env; q = e.qpos()[0].copy(); qv = e.qvel()[0]
            yw = float(e.yaw()[0]); _, gv = e.base_frame()
            L['q'].append(q); L['mode'].append(int(g.mode[0]))
            L['brake'].append(bool(g.braking[0]))
            L['vu'].append(v if act == 'cmd' else 0.0); L['wu'].append(w if act == 'cmd' else 0.0)
            L['vgov'].append(float(getattr(e, 'v_gov', e.v_cmd)[0])); L['vref'].append(float(e.v_ref[0]))
            L['v'].append(np.cos(yw) * qv[0] + np.sin(yw) * qv[1])
            L['wref'].append(float(e.w_ref[0])); L['w'].append(qv[5])
            L['lean'].append(np.degrees(np.arcsin(np.clip(gv[0, 1], -1, 1))))
            L['phi'].append(np.degrees(np.arctan2(e.v_ref[0] * e.w_ref[0], 9.81)))
            L['act'].append(LABEL[act]); L['P'].append(float(e.P_elec[0]))
    return {k: (np.array(v_) if k != 'act' else v_) for k, v_ in L.items()}, list(g.switches)


def ema(x, tau, hz=50):
    y = np.zeros_like(x); a = 1.0 / (tau * hz); s = x[0]
    for i, xi in enumerate(x):
        s += a * (xi - s); y[i] = s
    return y


def main():
    walk, runp, name, out = sys.argv[1:5]
    prof, title, az_off = PROFILES[name]
    g = Gait55(walk, runp, n=1, nthread=1)
    L, sw = record(g, prof)
    v = ema(L['v'], 0.3); w = ema(L['w'], 0.3); lean = ema(L['lean'], 0.3); P = ema(L['P'], 1.0)
    Q = L['q']
    yaw = np.unwrap(np.arctan2(2 * (Q[:, 3] * Q[:, 6] + Q[:, 4] * Q[:, 5]), 1 - 2 * (Q[:, 5] ** 2 + Q[:, 6] ** 2)))
    m = g.W.m
    m.vis.global_.offwidth = max(W, m.vis.global_.offwidth); m.vis.global_.offheight = max(H, m.vis.global_.offheight)
    d = mujoco.MjData(m); rd = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation = 4.2, -12
    f_t, f_m, f_s, f_b = F(26, True), F(19), F(14), F(40, True)
    wr = imageio.get_writer(out, fps=FPS, codec='libx264', quality=8, macro_block_size=1)
    look = Q[0, :2].copy(); az = np.degrees(yaw[0]) + az_off
    xy = Q[:, :2]; lo, hi = xy.min(0) - 1, xy.max(0) + 1; span = max(hi - lo)
    MAP = 210
    for k in range(0, len(Q), 50 // FPS):
        d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
        look = 0.9 * look + 0.1 * Q[k, :2]
        az += 0.08 * ((np.degrees(yaw[k]) + az_off) - az)        # camera follows the heading slowly (150: behind-left, 90: side)
        cam.lookat[:] = [look[0], look[1], 0.55]; cam.azimuth = az
        rd.update_scene(d, cam)
        im = Image.new('RGB', (W, H + HDR + BOT), BG)
        im.paste(Image.fromarray(rd.render()), (0, HDR))
        dr = ImageDraw.Draw(im, 'RGBA')
        run = L['mode'][k] == 1
        braking = L['brake'][k] and v[k] > 0.05
        col = C_BRAKE if braking else (C_RUN if run else C_WALK)
        dr.rectangle([0, 0, W, HDR], fill=PANEL); dr.rectangle([0, 0, 8, HDR], fill=col)
        dr.text((20, 4), title, font=f_t, fill=INK)
        dr.text((20, 40), 'walking policy ⇄ running policy · yaw-rate command · speed governor v·|ω| ≤ 3 m/s² · lean target atan(vω/g) · MuJoCo, simulation only',
                font=f_s, fill=INK2)
        # left box: mode and speeds
        dr.rounded_rectangle([14, HDR + 14, 470, HDR + 196], radius=8, fill=(0, 0, 0, 160))
        standing = (not run) and abs(v[k]) < 0.05 and abs(w[k]) < 0.1 and L['vu'][k] == 0 and L['wu'][k] == 0
        lab = '急停止 BRAKE' if braking else ('走行 RUN' if run else ('停止 STAND' if standing else '歩行 WALK'))
        dr.text((26, HDR + 16), lab, font=f_b, fill=col)
        gov = L['vu'][k] > L['vgov'][k] + 0.05 and L['act'][k] == '指令'
        dr.text((26, HDR + 70), f"{L['act'][k]}  v {L['vu'][k]:4.2f} m/s   ω {L['wu'][k]:+4.2f} rad/s", font=f_m, fill=INK)
        dr.text((26, HDR + 98), f'自動減速 governed  {L["vgov"][k]:4.2f} m/s' if gov else ' ', font=f_m, fill=C_GOV)
        dr.text((26, HDR + 126), f'実測 actual  v {v[k]:4.2f} m/s   ω {w[k]:+4.2f} rad/s', font=f_m, fill=col)
        dr.text((26, HDR + 160), f't = {k / 50:4.1f} s', font=f_s, fill=INK2)
        # top view
        x0, y0m = W - MAP - 16, HDR + 14
        dr.rounded_rectangle([x0, y0m, x0 + MAP, y0m + MAP], radius=8, fill=(0, 0, 0, 160))
        pts = [(x0 + 10 + (p[0] - lo[0]) / span * (MAP - 20), y0m + MAP - 10 - (p[1] - lo[1]) / span * (MAP - 20)) for p in xy[:k + 1:10]]
        if len(pts) > 1:
            dr.line(pts, fill=(200, 200, 200, 200), width=2)
        if pts:
            dr.ellipse([pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5], fill=col)
        dr.text((x0 + 8, y0m + 4), '上から見た軌跡 top view', font=f_s, fill=INK2)
        recent = [s for t, s in sw if 0 <= k - t < 100]
        if recent:
            dr.rounded_rectangle([W - 380, HDR + MAP + 24, W - 14, HDR + MAP + 74], radius=8, fill=(0, 0, 0, 170))
            dr.text((W - 362, HDR + MAP + 30), '切替 ' + recent[-1], font=F(26, True), fill=C_GOV)
        # bottom panel: lean gauge and power
        yb = HDR + H
        dr.rectangle([0, yb, W, yb + BOT], fill=PANEL)
        dr.text((20, yb + 8), '旋回内側への体の傾き lean into the turn (roll, 0.3 s mean)', font=f_s, fill=INK2)
        cx, cyb, rr = 160, yb + 104, 70
        dr.arc([cx - rr, cyb - rr, cx + rr, cyb + rr], 200, 340, fill=(90, 90, 88), width=3)
        for val, c, wd in ((L['phi'][k], C_GOV, 3), (lean[k], col, 6)):
            a = np.radians(-90 - val * 2.5)
            dr.line([cx, cyb, cx + rr * np.cos(a), cyb + rr * np.sin(a)], fill=c, width=wd)
        dr.text((260, yb + 40), f'実測 measured {lean[k]:+5.1f}°', font=f_m, fill=col)
        dr.text((260, yb + 70), f'物理 atan(vω/g) {L["phi"][k]:+5.1f}°', font=f_m, fill=C_GOV)
        dr.text((700, yb + 12), f'脚の消費電力 leg power (1 s) {P[k]:5.0f} W', font=f_s, fill=INK)
        dr.text((700, yb + 40), 'hard braking: foot ahead of the pelvis, knee absorbs, trunk upright and rising', font=f_s, fill=INK2)
        dr.text((700, yb + 64), 'MuJoCo sim · mocap.cs.cmu.edu (NSF EIA-0196217) · not tested on a real robot', font=f_s, fill=INK2)
        dr.text((700, yb + 88), 'Idea: Takeyuki-K · Impl.: generated with Claude', font=f_s, fill=INK2)
        wr.append_data(np.array(im))
    wr.close()
    print('saved', out, 'switches', sw)


if __name__ == '__main__':
    main()
