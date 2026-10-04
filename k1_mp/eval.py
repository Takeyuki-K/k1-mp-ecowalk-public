# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Evaluate a policy, compute gait metrics and render an mp4.
python3 eval.py runs/s2/model.pt --stand 1.5 --walk 8 --stop 2.5 --out out/k1_mp_walk.mp4
"""
import os, argparse, json
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, torch, mujoco
from k1env import K1Batch, DEFAULT_POSE
from ppo import AC


def run(model_path, stand=1.5, walk=8.0, stop=2.5, render=True, out='out/walk.mp4', walk_start=False,
        w=640, h=480, push=None):
    env = K1Batch(1, stage=2, randomize=False, seed=7, ep_len=10 ** 9)
    env.pushes = False
    # deterministic scripted command
    if walk_start:
        env.stage = 1; env.reset(np.array([0])); env.stage = 2
    else:
        # force standing start
        m = env.m; d = mujoco.MjData(m)
        d.qpos[:] = m.key_qpos[0]; d.qpos[env.qadr] = DEFAULT_POSE; d.qpos[2] = env.ref.z_stand + 0.002
        mujoco.mj_forward(m, d)
        st = np.zeros(env.nstate); mujoco.mj_getState(m, d, st, env.spec_state)
        env.state[0] = st; env.sdata[0] = d.sensordata
        env.ph[:] = 0; env.alpha[:] = 0; env.cmd[:] = 0
    env.switch_t[:] = -1 if walk_start else int(stand * 50)
    if walk_start:
        stand = 0.0
    env.stop_t[:] = int((stand + walk) * 50) if stop > 0 else 10 ** 9
    T = int((stand + walk + max(stop, 0)) * 50)
    oa, oc = env.obs()
    net = AC(oa.shape[1], oc.shape[1], 12)
    net.load_state_dict(torch.load(model_path, map_location='cpu')['model']); net.eval()
    m = env.m; d = mujoco.MjData(m)
    if render:
        import imageio, cv2
        r = mujoco.Renderer(m, h, w)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 2.4, 90, -8
        opt = mujoco.MjvOption()
        writer = imageio.get_writer(out, fps=50, codec='libx264', quality=8, macro_block_size=1)
    logs = []
    first_contact = {0: [], 1: []}
    air = [False, False]; air_t = [0.0, 0.0]
    fell = False
    for t in range(T):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        if push is not None and t == push[0]:
            env.state[0, 1 + m.nq + 1] += push[1]
        rew, term, trunc, info = env.step(a)
        q = env.qpos()[0]
        touch = np.array([env.s(f'{s}_{p}_touch')[0, 0] for s in ('left', 'right') for p in ('heel', 'meta', 'toe')])
        c = touch > 5.0
        for k in range(2):
            anyc = c[3 * k:3 * k + 3].any()
            if air[k] and anyc:
                first_contact[k].append(('heel' if c[3 * k] and not (c[3 * k + 1] or c[3 * k + 2]) else
                                         'heel+fore' if c[3 * k] else 'fore'))
            air_t[k] = 0 if anyc else air_t[k] + 0.02
            air[k] = (not anyc) and air_t[k] > 0.06 or (air[k] and not anyc)
        logs.append(dict(t=t / 50, x=q[0], z=q[2], vx=env.qvel()[0, 0], cmd=env.cmd[0], alpha=env.alpha[0],
                         mpL=q[env.mpadr[0]], mpR=q[env.mpadr[1]], touch=touch.tolist(), power=info['power'][0]))
        if term[0]:
            fell = True
            print('FELL at t=%.2f' % (t / 50))
            break
        if render:
            mujoco.mj_setState(m, d, env.state[0], env.spec_state)
            mujoco.mj_forward(m, d)
            cam.lookat[:] = [d.qpos[0], d.qpos[1] * 0.3, 0.42]
            r.update_scene(d, cam, opt)
            img = r.render().copy()
            phase = []
            for k, s in enumerate(('L', 'R')):
                hc, mc, tc = c[3 * k:3 * k + 3]
                st = 'swing' if not (hc or mc or tc) else 'HEEL' if hc and not (mc or tc) else \
                    'FLAT' if hc else 'TOE' if (tc and not mc) else 'FORE'
                phase.append(f'{s}:{st:5s} MP {np.degrees(q[env.mpadr[k]]):5.1f}deg')
            mode = 'STAND' if env.alpha[0] == 0 else ('WALK' if env.cmd[0] > 0 else 'STOPPING')
            txt = [f'K1 + passive MP toe joints   t={t/50:5.2f}s   {mode}   vx={env.qvel()[0,0]:.2f} m/s'] + phase
            for i, s in enumerate(txt):
                cv2.putText(img, s, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            writer.append_data(img)
        oa, oc = env.obs()
    if render:
        writer.close()
    L = logs
    walk_logs = [l for l in L if l['alpha'] > 0.99]
    res = dict(fell=fell, duration=L[-1]['t'], distance=L[-1]['x'] - L[0]['x'],
               mean_vx_walk=float(np.mean([l['vx'] for l in walk_logs])) if walk_logs else 0,
               touchdowns_L=first_contact[0], touchdowns_R=first_contact[1],
               heel_first_ratio=float(np.mean([s.startswith('heel') for s in first_contact[0] + first_contact[1]]))
               if (first_contact[0] + first_contact[1]) else 0,
               mp_min_deg=float(np.degrees(min(min(l['mpL'], l['mpR']) for l in L))),
               mean_power_W=float(np.mean([l['power'] for l in walk_logs])) if walk_logs else 0)
    mass = sum(m.body_mass)
    if walk_logs and res['mean_vx_walk'] > 0.05:
        res['CoT_mech'] = res['mean_power_W'] / (mass * 9.81 * res['mean_vx_walk'])
    np.save(out.replace('.mp4', '_log.npy'), L, allow_pickle=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--stand', type=float, default=1.5)
    ap.add_argument('--walk', type=float, default=8.0)
    ap.add_argument('--stop', type=float, default=2.5)
    ap.add_argument('--out', default='out/walk.mp4')
    ap.add_argument('--norender', action='store_true')
    ap.add_argument('--walk_start', action='store_true')
    a = ap.parse_args()
    res = run(a.model, a.stand, a.walk, a.stop, not a.norender, a.out, a.walk_start)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith('touchdowns')}, indent=1))
    print('touchdowns L', res['touchdowns_L']); print('touchdowns R', res['touchdowns_R'])
