"""Spring-tuning diagnostic: straight walk at a commanded speed, deployment env vars (K1_ANKROLL_SPRING/DAMP) set.
Reports per-metre energy, hip roll/pitch load, posture, touchdown/rocker impact.
Usage: spring_diag.py <walk.pt> [speeds] [n] [tag]   (run from k1_mp_gait57/)"""
import os, sys, json
import numpy as np, torch, mujoco
os.environ.setdefault('WALK_BRAKE', '1')
sys.path.insert(0, os.getcwd())
import eval56
from k1env import ACT_JOINTS as AJ
from k1env_walk4 import K1Walk4Batch as C

HP = [AJ.index('left_hip_pitch_joint'), AJ.index('right_hip_pitch_joint')]
HR = [AJ.index('left_hip_roll_joint'), AJ.index('right_hip_roll_joint')]
AR = [AJ.index('left_ankle_roll_joint'), AJ.index('right_ankle_roll_joint')]
KM = np.array([2.2 if j in AR else 4.0 for j in range(12)])
SENS = [f'{s}_{p}_touch' for s in ('left', 'right') for p in ('heel', 'meta', 'toe')]


def run(env, net, v, T=12.0, skip=4.0):
    eval56.stand(env); env.cmd[:] = 1; env.v_cmd[:] = v; env.w_cmd[:] = 0
    oa, _ = env.obs(); alive = np.ones(env.n, bool)
    mg = float(env.m.body_subtreemass[1]) * 9.81
    dt_sub = env.m.opt.timestep
    acc = {k: [] for k in ('P', 'hr_tau2', 'hp_tau2', 'hr_cu', 'hp_cu', 'roll', 'ank')}
    peaks, rates = [], []
    cur_pk = np.zeros((env.n, 2)); cur_rt = np.zeros((env.n, 2)); loaded = np.zeros((env.n, 2), bool)
    p0 = p1 = None
    for k in range(int(T * 50)):
        _, term, _, _ = env.step(eval56.act(net, oa)); oa, _ = env.obs(); alive &= ~term
        if k < skip * 50:
            continue
        q = env.qpos()
        if p0 is None: p0 = q[:, :2].copy()
        p1 = q[:, :2].copy()
        _, g = env.base_frame()
        acc['roll'].append(np.degrees(np.arcsin(np.clip(g[:, 1], -1, 1))))
        acc['ank'].append(np.degrees(q[:, env.qadr[AR]]))
        acc['P'].append(env.P_leg.copy() if hasattr(env, 'P_leg') else env.P_elec.copy())
        tau = env.tau_sub
        acc['hr_tau2'].append((tau[:, :, HR] ** 2).mean((1, 2))); acc['hp_tau2'].append((tau[:, :, HP] ** 2).mean((1, 2)))
        acc['hr_cu'].append((tau[:, :, HR] ** 2 / 16.0).sum(2).mean(1)); acc['hp_cu'].append((tau[:, :, HP] ** 2 / 16.0).sum(2).mean(1))
        # foot normal force per substep (sum of heel/meta/toe touch sensors)
        sd = env.sd_out  # (n, nsub, nsensordata)
        F = np.stack([sum(sd[:, :, env.sens[SENS[3 * s + j]][0]] for j in range(3)) for s in (0, 1)], 2)  # (n, nsub, 2)
        dF = np.diff(np.concatenate([np.zeros((env.n, 1, 2)) if k == skip * 50 else run.lastF[:, None, :], F], 1), axis=1) / dt_sub
        run.lastF = F[:, -1, :]
        Fm = F.max(1); now = Fm > 5.0
        cur_pk = np.where(now, np.maximum(cur_pk, Fm), cur_pk)
        cur_rt = np.where(now, np.maximum(cur_rt, dF.max(1)), cur_rt)
        off = loaded & ~now  # lift-off -> close the stance
        for i, s in zip(*np.where(off & alive[:, None])):
            peaks.append(cur_pk[i, s] / mg); rates.append(cur_rt[i, s] / mg)
        cur_pk[off] = 0; cur_rt[off] = 0
        loaded = now
    a = alive
    v_act = np.linalg.norm(p1 - p0, axis=1) / ((T - skip) - 1 / 50)
    A = {k: np.array(x) for k, x in acc.items()}
    R = A['roll'][:, a]
    P = float(A['P'][:, a].mean())
    return dict(v_cmd=v, survival=float(a.mean()), v_meas=round(float(v_act[a].mean()), 3),
                P_leg_W=round(P, 1), J_per_m=round(P / float(v_act[a].mean()), 1),
                hiproll_tau_rms=round(float(np.sqrt(A['hr_tau2'][:, a].mean())), 2),
                hippitch_tau_rms=round(float(np.sqrt(A['hp_tau2'][:, a].mean())), 2),
                hiproll_cu_W=round(float(A['hr_cu'][:, a].mean()), 1), hippitch_cu_W=round(float(A['hp_cu'][:, a].mean()), 1),
                roll_rms_deg=round(float(np.sqrt((R ** 2).mean())), 2), roll_p99_deg=round(float(np.percentile(np.abs(R), 99)), 2),
                ankroll_rms_deg=round(float(np.sqrt((A['ank'][:, a] ** 2).mean())), 2),
                grf_peak_bw=round(float(np.mean(peaks)), 3) if peaks else None,
                grf_peak_p95_bw=round(float(np.percentile(peaks, 95)), 3) if peaks else None,
                load_rate_bw_s=round(float(np.mean(rates)), 1) if rates else None, n_steps=len(peaks))


run.lastF = None


def main():
    path = sys.argv[1]
    speeds = [float(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['1.0'])]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    tag = sys.argv[4] if len(sys.argv) > 4 else 'sd'
    C.P_RUN = 0.0
    env = C(n, stage=2, randomize=False, seed=31, ep_len=10 ** 9, nthread=2)
    env.pushes = False; env.scripted = True
    net = eval56.load(path, env)
    for v in speeds:
        r = run(env, net, v); r.update(tag=tag, K=os.environ.get('K1_ANKROLL_SPRING', '0'), D=os.environ.get('K1_ANKROLL_DAMP', '-'))
        print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
