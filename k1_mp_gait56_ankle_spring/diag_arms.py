# Local (ankle-spring variant): arm power / yaw momentum / arm clearance diagnostic for straight walking.
# python3 diag_arms.py WALK.pt RUN.pt [--n 16] [--v 1.0 1.4] [--json out.json]
import argparse, json, os
import numpy as np, mujoco
from eval_gait56 import Gait55, run_profile, yaw_of, HZ
from k1env import ACT_JOINTS, KP, KD
import k1env_walk4 as W4

ap = argparse.ArgumentParser(); ap.add_argument('walk'); ap.add_argument('run')
ap.add_argument('--n', type=int, default=16); ap.add_argument('--v', type=float, nargs='+', default=[0.6, 1.0, 1.4])
ap.add_argument('--meas', type=float, default=6.0); ap.add_argument('--json', default=''); ap.add_argument('--seed', type=int, default=3)
a = ap.parse_args()
g = Gait55(a.walk, a.run, n=a.n, nthread=2); E = g.W; m = E.m
d = mujoco.MjData(m)
AJ = W4.ARM_ALL; names = [ACT_JOINTS[i].replace('_joint', '') for i in AJ]
qa = np.array([m.jnt_qposadr[m.joint(ACT_JOINTS[i]).id] for i in AJ]); da = np.array([m.jnt_dofadr[m.joint(ACT_JOINTS[i]).id] for i in AJ])
# clearance: arm geoms (visual meshes of upper arm / forearm / hand) vs torso, pelvis, thighs, shanks
gb = lambda gi: m.body(int(m.geom_bodyid[gi])).name
ARMB = ('shoulder_roll_link', 'shoulder_yaw_link', 'elbow_link', 'wrist_roll_rubber_hand')
BODB = ('torso_link', 'pelvis', 'hip_pitch_link', 'hip_roll_link', 'hip_yaw_link', 'knee_link')
vis = [i for i in range(m.ngeom) if m.geom_group[i] == 2 and m.geom_type[i] == 7]
armg = [i for i in vis if any(gb(i).endswith(k) for k in ARMB)]
bodg = [i for i in vis if any(gb(i).endswith(k) or gb(i) == k for k in BODB)]
pairs = [(i, j) for i in armg for j in bodg if not (gb(i).endswith('shoulder_roll_link') and gb(j) == 'torso_link' and False)]
ft = np.zeros(6)
def clearance(qpos):
    d.qpos[:] = qpos; mujoco.mj_kinematics(m, d); mujoco.mj_collision  # noqa
    out = {}
    for i, j in pairs:
        dist = mujoco.mj_geomDistance(m, d, i, j, 0.3, ft)
        key = gb(i).replace('left_', '').replace('right_', '') + '-' + gb(j).replace('left_', '').replace('right_', '') + \
              ('(same side)' if gb(j) in ('torso_link', 'pelvis') or gb(i).split('_')[0] == gb(j).split('_')[0] else '(other)')
        out[key] = min(out.get(key, 9), dist)
    return out
res = []
for v in a.v:
    run_profile(g, [(1, 'cmd', 0.0, 0.0), (4, 'cmd', v, 0.0)], seed=a.seed)
    alive = np.ones(a.n, bool)
    P, PA, CU, MECH, Q, LZ, WZ, V, CL = [], [], [], [], [], [], [], [], {}
    for k in range(int(a.meas * HZ)):
        term, _ = g.step(); alive &= ~term
        st = E.state; qp = st[:, 1:1 + m.nq]; qv = st[:, 1 + m.nq:1 + m.nq + m.nv]
        P.append(E.P_elec.copy()); PA.append(E.P_arm.copy())
        kp = KP[AJ] * float(os.environ.get('ARM_KPM', '1')); kd = KD[AJ]
        tau = np.clip(kp * (E.ctrl[:, AJ] - qp[:, qa]) - kd * qv[:, da], -W4.TLIM_ARM, W4.TLIM_ARM)
        CU.append(tau ** 2 / W4.KM_ARM ** 2); MECH.append(np.clip(tau * qv[:, da], 0, None)); Q.append(np.degrees(qp[:, qa]))
        yw = yaw_of(qp); V.append(np.cos(yw) * qv[:, 0] + np.sin(yw) * qv[:, 1]); WZ.append(qv[:, 5])
        lz = np.zeros(a.n)
        for i in range(a.n):
            d.qpos[:] = qp[i]; d.qvel[:] = qv[i]; mujoco.mj_forward(m, d); mujoco.mj_subtreeVel(m, d)
            lz[i] = d.subtree_angmom[1, 2]
            if k % 5 == 0 and i < 8:
                for kk, dist in clearance(qp[i]).items():
                    CL[kk] = min(CL.get(kk, 9), dist)
        LZ.append(lz)
    al = alive
    P, PA, CU, MECH, Q, LZ, WZ, V = [np.array(x)[:, al] for x in (P, PA, CU, MECH, Q, LZ, WZ, V)]
    dlz = np.diff(LZ, axis=0) * HZ
    r = dict(v_cmd=v, survival=float(al.mean()), v=round(float(V.mean()), 3), P_W=round(float(P.mean()), 1), P_arm=round(float(PA.mean()), 2),
             arm_copper={n_: round(float(CU[:, :, j].mean()), 2) for j, n_ in enumerate(names)},
             arm_mech={n_: round(float(MECH[:, :, j].mean()), 2) for j, n_ in enumerate(names)},
             arm_p2p={n_: round(float((Q[:, :, j].max(0) - Q[:, :, j].min(0)).mean()), 1) for j, n_ in enumerate(names)},
             arm_mean={n_: round(float(Q[:, :, j].mean()), 1) for j, n_ in enumerate(names)},
             Lz_p2p=round(float((LZ.max(0) - LZ.min(0)).mean()), 3), Lz_rms=round(float(np.sqrt(((LZ - LZ.mean(0)) ** 2).mean())), 3),
             yaw_moment_rms=round(float(np.sqrt((dlz ** 2).mean())), 2), yawrate_rms_dps=round(float(np.degrees(np.sqrt((WZ ** 2).mean()))), 2),
             clearance_cm={k_: round(100 * v_, 1) for k_, v_ in sorted(CL.items(), key=lambda t: t[1])[:6]})
    res.append(r); print(json.dumps(r), flush=True)
if a.json: json.dump(res, open(a.json, 'w'), indent=1)
