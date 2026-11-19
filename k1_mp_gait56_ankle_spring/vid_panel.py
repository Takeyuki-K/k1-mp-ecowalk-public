# Local (ankle-spring variant): one labelled video panel through the gait manager (1 robot, no pushes).
# python3 vid_panel.py WALK.pt RUN.pt PROFILE VIEW LABEL OUT.mp4     VIEW: front | side | diag
import os, sys
import numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
from gait56 import Gait56 as Gait55
import eval_gait56 as E

PROF = {'walk_front3': [(1.5, 'cmd', 0.0, 0.0), (5, 'cmd', 0.6, 0.0), (5, 'cmd', 1.0, 0.0), (5, 'cmd', 1.4, 0.0)],
        'walk_front': [(1.5, 'cmd', 0.0, 0.0), (6, 'cmd', 1.0, 0.0), (4, 'cmd', 1.4, 0.0)],
        'sequence': [(1.5, 'cmd', 0.0, 0.0), (6, 'cmd', 1.0, 0.0), (8, 'cmd', 3.0, 0.0), (6, 'cmd', 1.0, 0.0), (5, 'stop', 0, 0)]}
AZ = {'front': 180.0, 'side': 90.0, 'diag': 135.0}
W, H, FPS = 640, 480, 25
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
def font(s):
    try: return ImageFont.truetype(FONT, s)
    except OSError: return ImageFont.load_default(s)

walk, runp, pname, view, label, out = sys.argv[1:7]
g = Gait55(walk, runp, n=1, nthread=1)
g.reset(); E.stand_noise(g)
g._oaw, _ = g.W.obs(); g._oar, _ = g.R.obs()
Q, MODE, V, P, CMD = [], [], [], [], []
fell = None
for T, act, v, w in PROF[pname]:
    {'cmd': lambda: g.command(v, w), 'stop': g.stop, 'brake': g.brake}[act]()
    for _ in range(int(T * 50)):
        term, _ = g.step()
        if term[0] and fell is None: fell = len(Q) / 50
        q = g.qpos()[0].copy(); qv = g.active('state')[0, 1 + g.W.m.nq:1 + g.W.m.nq + g.W.m.nv]
        yw = E.yaw_of(q[None])[0]
        Q.append(q); MODE.append(int(g.mode[0])); V.append(np.cos(yw) * qv[0] + np.sin(yw) * qv[1])
        P.append(float((g.R if g.mode[0] == 1 else g.W).P_elec[0])); CMD.append(f'{act} {v:.1f}' if act == 'cmd' else act)
Q = np.array(Q); V = np.array(V); P = np.array(P)
def ema(x, tau):
    y = np.zeros_like(x); s = x[0]; a = 1 / (tau * 50)
    for i, xi in enumerate(x): s += a * (xi - s); y[i] = s
    return y
Ve, Pe = ema(V, 0.3), ema(P, 1.0)
w_, x_, y_, z_ = Q[:, 3], Q[:, 4], Q[:, 5], Q[:, 6]
roll = np.degrees(np.arctan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_ * x_ + y_ * y_)))
yaw = np.unwrap(E.yaw_of(Q))
m = g.W.m
m.vis.global_.offwidth = max(W, m.vis.global_.offwidth); m.vis.global_.offheight = max(H, m.vis.global_.offheight)
d = mujoco.MjData(m); rd = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera(); cam.distance, cam.elevation = (3.0, -8) if view == 'front' else (4.0, -12)
wr = imageio.get_writer(out, fps=FPS, codec='libx264', quality=7, macro_block_size=1)
look = Q[0, :2].copy(); f1, f2 = font(26), font(18)
for k in range(0, len(Q), 2):
    d.qpos[:] = Q[k]; mujoco.mj_forward(m, d)
    look = 0.85 * look + 0.15 * Q[k, :2]
    cam.lookat[:] = [look[0], look[1], 0.5]; cam.azimuth = np.degrees(yaw[0]) + AZ[view]
    rd.update_scene(d, cam)
    im = Image.fromarray(rd.render()); dr = ImageDraw.Draw(im, 'RGBA')
    dr.rectangle([0, 0, W, 92], fill=(0, 0, 0, 170))
    dr.text((10, 4), label, font=f1, fill=(255, 255, 255))
    mode = 'RUN' if MODE[k] == 1 else 'WALK'
    dr.text((10, 40), f't {k/50:4.1f}s  {CMD[k]}  {mode}  v {Ve[k]:4.2f} m/s  P {Pe[k]:4.0f} W', font=f2, fill=(230, 230, 120))
    dr.text((10, 64), f'pelvis roll {roll[k]:+5.1f} deg', font=f2, fill=(150, 220, 255))
    wr.append_data(np.array(im))
wr.close()
print('saved', out, 'frames', len(Q) // 2, 'fell_at', fell)
