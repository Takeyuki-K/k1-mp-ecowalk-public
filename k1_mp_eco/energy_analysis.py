# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Per-joint energy breakdown of a policy during steady walking.
Electrical model (assumption, editable): P_elec = sum( tau^2 / Km^2 ) + sum( max(tau*qd, 0) )
  - copper loss with motor constant at the joint output Km [Nm/sqrt(W)]
  - positive mechanical work drawn from the battery; negative work dissipated (no regeneration)
Also reports negative work per joint = energy a passive damper/spring could absorb instead.
"""
import sys, json, numpy as np, torch, mujoco
from k1env import K1Batch, ACT_JOINTS, LEG_JOINTS
from ppo import AC

# Km at joint output. QC080 (hips/knee/ankle pitch, 96.9 Nm peak) / QC060 (ankle roll, 47.3 Nm).
# Rough QDD-class values; replace with measured motor data.
KM = np.array([4.0 if 'ankle_roll' not in j else 2.2 for j in LEG_JOINTS])


def analyze(path, T=8.0, warm=2.0, eco=False):
    if eco:
        from k1env_eco import K1EcoBatch as B
        from ppo_eco import ACEco as N
    else:
        B, N = K1Batch, AC
    env = B(1, stage=1, randomize=False, seed=5, ep_len=10 ** 9)
    env.pushes = False
    oa, oc = env.obs()
    net = N(oa.shape[1], oc.shape[1], env.nact if eco else 12)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    m = env.m
    da = env.dadr[:12]
    rec = dict(tau=[], qd=[], ph=[], kp=[], x=[])
    for t in range(int(T * 50)):
        with torch.no_grad():
            a = net.dist(torch.from_numpy(oa)).mean.numpy().astype(np.float64)
        _, term, _, info = env.step(a)
        if term[0]:
            print('fell'); break
        if t >= warm * 50:
            # substep-resolved actuator torque & velocity
            nq, nv = m.nq, m.nv
            for k in range(env.nsub):
                st = env.st_out[0, k]
                qv = st[1 + nq:1 + nq + nv]
                rec['qd'].append(qv[da])
            rec['tau'].extend(env.tau_sub[0])
            rec['ph'].append(env.ph[0]); rec['x'].append(env.qpos()[0, 0])
            if eco:
                rec['kp'].append(env.kp_scale[0].copy())
        oa, oc = env.obs()
    tau = np.array(rec['tau']); qd = np.array(rec['qd'])
    dt = env.dt
    dur = len(tau) * dt
    dist = rec['x'][-1] - rec['x'][0]
    p = tau * qd
    pos = np.clip(p, 0, None).sum(0) * dt / dur
    neg = -np.clip(p, None, 0).sum(0) * dt / dur
    cu = (tau ** 2 / KM ** 2).sum(0) * dt / dur
    mass = sum(m.body_mass)
    v = dist / dur
    P_elec = (pos + cu).sum()
    res = dict(speed=v, P_elec_W=P_elec, P_pos_mech_W=pos.sum(), P_neg_mech_W=neg.sum(), P_copper_W=cu.sum(),
               CoT_elec=P_elec / (mass * 9.81 * v), CoT_posmech=pos.sum() / (mass * 9.81 * v),
               per_joint={j.replace('_joint', ''): dict(pos_W=round(float(pos[i]), 1), neg_W=round(float(neg[i]), 1),
                                                        copper_W=round(float(cu[i]), 1),
                                                        tau_rms=round(float(np.sqrt((tau[:, i] ** 2).mean())), 1))
                          for i, j in enumerate(LEG_JOINTS)})
    if eco:
        res['kp_scale_mean'] = dict(zip([j.replace('_joint', '') for j in LEG_JOINTS],
                                        np.round(np.mean(rec['kp'], 0), 2).tolist()))
    return res, rec


if __name__ == '__main__':
    eco = '--eco' in sys.argv
    res, _ = analyze(sys.argv[1], eco=eco)
    print(json.dumps(res, indent=1))
