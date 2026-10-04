# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Render one recording with an identical side-follow camera -> raw_<name>.mp4 (640x560, 50 fps)."""
import os, sys
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, mujoco, imageio

W, H = 640, 560
k = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
r = np.load(f'{HERE}/rec_{k}.npz')
xml = os.path.join(os.path.dirname(HERE), 'ai_sapiens', 'ai_sapiens_description', 'mujoco', 'k1', os.path.basename(str(r['xml'])))
m = mujoco.MjModel.from_xml_path(xml)
m.vis.global_.offwidth = max(W, m.vis.global_.offwidth); m.vis.global_.offheight = max(H, m.vis.global_.offheight)
d = mujoco.MjData(m)
rd = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.azimuth, cam.elevation = 2.5, 90, -8
wr = imageio.get_writer(f'{HERE}/raw_{k}.mp4', fps=50, codec='libx264', quality=9, macro_block_size=1)
for q in r['qpos']:
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    ys = q[1] if 'ys' not in dir() else 0.9 * ys + 0.1 * q[1]
    cam.lookat[:] = [q[0], ys, 0.48]
    rd.update_scene(d, cam)
    wr.append_data(rd.render())
wr.close()
print('done', k)
