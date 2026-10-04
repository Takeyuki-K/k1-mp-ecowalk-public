# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, '..', 'ai_sapiens', 'ai_sapiens_description', 'mujoco', 'k1', 'scene_mp.xml')


def make_cam(model, dist=2.2, azim=90, elev=-10):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = dist, azim, elev
    return cam


def filmstrip(model, qposes, path, w=240, h=320, azim=90):
    import imageio
    d = mujoco.MjData(model)
    r = mujoco.Renderer(model, h, w)
    cam = make_cam(model, azim=azim)
    frames = []
    for q in qposes:
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.45]
        r.update_scene(d, cam)
        frames.append(r.render())
    img = np.concatenate(frames, 1)
    imageio.imwrite(path, img)
    return path


if __name__ == '__main__':
    import sys
    m = mujoco.MjModel.from_xml_path(XML)
    ref = np.load(os.path.join(HERE, 'ref_gait.npz'))
    Q = ref['qpos']
    idx = np.linspace(0, len(Q) - 1, 8).astype(int)
    filmstrip(m, Q[idx], os.path.join(HERE, 'out', 'ref_filmstrip.png'))
    print('ok')
