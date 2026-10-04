# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Final MuJoCo video: stand -> walk -> stop (real time, 2 panels) + slow-motion foot close-up.
python3 final_video.py runs/final/model.pt out/K1_MP_heel_toe_walk.mp4
"""
import os, sys, json
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, torch, mujoco, imageio, cv2
from k1env import DEFAULT_POSE
from k1env_eco import K1EcoBatch as K1Batch
from ppo_eco import ACEco as AC

STAND, WALK, STOP = 2.0, 8.0, 3.0
W, H = 640, 480


def simulate(path):
    env = K1Batch(1, stage=2, randomize=False, seed=7, ep_len=10 ** 9)
    env.pushes = False
    m = env.m; d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]; d.qpos[env.qadr] = DEFAULT_POSE; d.qpos[2] = env.ref.z_stand + 0.002
    mujoco.mj_forward(m, d)
    st = np.zeros(env.nstate); mujoco.mj_getState(m, d, st, env.spec_state)
    env.state[0] = st; env.sdata[0] = d.sensordata
    env.ph[:] = 0; env.alpha[:] = 0; env.cmd[:] = 0
    env.switch_t[:] = int(STAND * 50); env.stop_t[:] = int((STAND + WALK) * 50)
    oa, oc = env.obs()
    net = AC(oa.shape[1], oc.shape[1], 36)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    states, info = [], []
    for t in range(int((STAND + WALK + STOP) * 50)):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        rew, term, trunc, inf = env.step(a)
        states.append(env.st_out[0].copy())
        touch = np.array([env.s(f'{s}_{p}_touch')[0, 0] for s in ('left', 'right') for p in ('heel', 'meta', 'toe')])
        mode = 'STAND' if env.alpha[0] == 0 else ('WALK' if env.cmd[0] > 0 else 'STOP')
        info.append(dict(touch=touch, mode=mode, vx=env.qvel()[0, 0], P=inf['P_elec'][0], kp=env.kp_scale[0].copy(), kd=env.kd_scale[0].copy()))
        if term[0]:
            print('FELL', t / 50); break
        oa, oc = env.obs()
    return env, np.concatenate(states, 0), info


def state_label(c):
    h, mm, t = c
    if not (h or mm or t):
        return 'SWING', (200, 200, 200)
    if h and not (mm or t):
        return 'HEEL STRIKE', (80, 200, 255)
    if h:
        return 'FOOT FLAT', (120, 255, 120)
    return 'TOE (MP bends)', (255, 170, 60)


def put(img, s, y, col=(255, 255, 255), sc=0.55, x=10):
    (tw, th), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
    sub = img[max(0, y - th - 5):y + b + 3, max(0, x - 4):x + tw + 4]
    sub[:] = (sub * 0.35).astype(np.uint8)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)


def main(path, out):
    env, S, info = simulate(path)
    m = env.m; d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    camA = mujoco.MjvCamera(); camA.type = mujoco.mjtCamera.mjCAMERA_FREE
    camA.distance, camA.azimuth, camA.elevation = 2.1, 90, -6
    camB = mujoco.MjvCamera(); camB.type = mujoco.mjtCamera.mjCAMERA_FREE
    camB.distance, camB.azimuth, camB.elevation = 1.0, 90, -5
    lf = m.body('left_ankle_roll_link').id
    mpL, mpR = env.mpadr
    wr = imageio.get_writer(out, fps=50, codec='libx264', quality=8, macro_block_size=1)
    fx = None

    def frame(k, slow=False):
        nonlocal fx
        mujoco.mj_setState(m, d, S[k], env.spec_state)
        mujoco.mj_forward(m, d)
        x = d.qpos[0]
        camA.lookat[:] = [x, 0, 0.50]
        r.update_scene(d, camA); A = r.render().copy()
        fpos = d.xpos[lf]
        fx = fpos[0] if fx is None else 0.85 * fx + 0.15 * fpos[0]
        camB.lookat[:] = [fx + 0.05, fpos[1], 0.16]
        r.update_scene(d, camB); B = r.render().copy()
        it = info[min(k // env.nsub, len(info) - 1)]
        c = it['touch'] > 5.0
        t = k * env.dt
        put(A, f'K1 + MP toes  |  ECO: variable impedance legs', 22, sc=0.6)
        put(A, f't = {t:5.2f} s    {it["mode"]}    vx = {it["vx"]:.2f} m/s', 46)
        put(A, f'leg electrical power ~ {it["P"]:5.0f} W', 70, (255, 230, 120))
        put(A, 'policy: human-gait imitation (stage 1) + RL fine-tune (stage 2)', H - 14, (200, 220, 255), 0.45)
        put(B, 'LEFT FOOT close-up' + ('   (slow motion x0.25)' if slow else ''), 22, sc=0.6)
        for i, (s, j) in enumerate([('L', 0), ('R', 1)]):
            lab, col = state_label(c[3 * j:3 * j + 3])
            ang = np.degrees(d.qpos[mpL if j == 0 else mpR])
            put(B, f'{s}: {lab:15s} MP {ang:6.1f} deg', 48 + 24 * i, col)
        # stiffness bars (Kp scale, left leg): 1.0 = ROBOTIS nominal gain
        names = [('hip pitch', 0), ('hip roll', 1), ('knee', 3), ('ankle pitch', 4)]
        x0, y0 = 10, 120
        put(B, 'LEFT leg stiffness Kp / nominal   (damping Kd / nominal)', y0 - 8, (255, 255, 255), 0.45)
        for i, (nm, j) in enumerate(names):
            y = y0 + 8 + 26 * i
            kpv, kdv = it['kp'][j], it['kd'][j]
            cv2.rectangle(B, (x0 + 95, y), (x0 + 95 + 150, y + 16), (60, 60, 60), -1)
            col = (90, 200, 255) if kpv < 0.5 else (120, 255, 120) if kpv < 0.9 else (255, 180, 80)
            cv2.rectangle(B, (x0 + 95, y), (x0 + 95 + int(100 * kpv), y + 16), col, -1)
            cv2.line(B, (x0 + 195, y - 2), (x0 + 195, y + 18), (255, 255, 255), 1)
            put(B, f'{nm:11s}', y + 14, (255, 255, 255), 0.45, x0)
            put(B, f'{kpv:4.2f} ({kdv:4.2f})', y + 14, (255, 255, 255), 0.45, x0 + 255)
        put(B, 'MP: passive spring | bars: 1.0 = nominal gain (white line), 0 = fully relaxed', H - 14, (200, 220, 255), 0.42)
        return np.concatenate([A, B], 1)

    n = len(S)
    for k in range(0, n, env.nsub):       # real time
        wr.append_data(frame(k))
    # slow motion: one full stride in steady walking, every substep (200 Hz -> 50 fps = x0.25)
    T = env.ref.T
    k0 = int((STAND + 4.0) / env.dt); k1 = k0 + int(1.15 * T / env.dt)
    title = np.zeros((H, 2 * W, 3), np.uint8)
    put(title, 'Slow motion x0.25 : how the stiffness changes within one stride', H // 2, sc=0.8, x=60)
    for _ in range(75):
        wr.append_data(title)
    fx = None
    for k in range(k0, min(k1, n)):
        wr.append_data(frame(k, slow=True))
    wr.close()
    print('saved', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
