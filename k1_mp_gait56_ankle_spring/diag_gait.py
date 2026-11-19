# Gait-mechanics diagnostics for walking (gait manager, 8 robots, stand 1 s, walk v for 4 s, measure 6 s at 200 Hz):
# touchdown type, CoP progression along the foot (touch-sensor weighted), ankle push-off timing vs opposite touchdown,
# swing vs stance hip-pitch effort, collision negative work, per-joint +/- mechanical work and copper loss.
# python3 diag_gait.py WALK.pt RUN.pt [v] [--json out.json]
import sys, json, numpy as np, mujoco
from eval_gait56 import Gait55, run_profile, HZ
from k1env_eco import KM
from k1env import LEG_JOINTS
args = [a for a in sys.argv[1:] if not a.startswith('--')]
w, r = args[0], args[1]; v = float(args[2]) if len(args) > 2 else 1.0
out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else ''
g = Gait55(w, r, n=8, nthread=2); E = g.W; m = E.m; nq, nv = m.nq, m.nv
d = mujoco.MjData(m); d.qpos[:] = m.key_qpos[0]; mujoco.mj_kinematics(m, d)
def xin(side, site):     # site x in ankle_roll_link frame (default pose)
    b = m.body(f'{side}_ankle_roll_link').id; R = d.xmat[b].reshape(3, 3)
    return float((R.T @ (d.site_xpos[m.site(f'{side}_{site}_touch').id] - d.xpos[b]))[0])
XS = np.array([[xin(s, p) for p in ('heel', 'meta', 'toe')] for s in ('left', 'right')])   # (2,3)
SN = [[E.sens[f'{s}_{p}_touch'][0] for p in ('heel', 'meta', 'toe')] for s in ('left', 'right')]
run_profile(g, [(1, 'cmd', 0.0, 0.0), (4, 'cmd', v, 0.0)], seed=3)
TAU, QD, F, = [], [], []
alive = np.ones(8, bool)
for k in range(6 * HZ):
    term, _ = g.step(); alive &= ~term
    TAU.append(E.tau_sub.copy()); QD.append(E.st_out[:, :, 1 + nq:1 + nq + nv][:, :, E.dadr[:12]].copy())
    F.append(np.stack([E.sd_out[:, :, SN[0]], E.sd_out[:, :, SN[1]]], 2))   # (n,nsub,2,3)
TAU = np.concatenate(TAU, 1)[alive]; QD = np.concatenate(QD, 1)[alive]; F = np.concatenate(F, 1)[alive]   # (n,T,...)
dt = 1.0 / (HZ * E.nsub); T = TAU.shape[1]; MG = 35.7 * 9.81
P = TAU * QD
res = dict(v=v, n_alive=int(alive.sum()))
res['joint'] = {j: dict(pos_W=round(float(np.clip(P[:, :, i], 0, None).mean()), 2), neg_W=round(float(-np.clip(P[:, :, i], None, 0).mean()), 2),
                        copper_W=round(float((TAU[:, :, i] ** 2).mean() / KM[i] ** 2), 2)) for i, j in enumerate(LEG_JOINTS)}
res['total'] = dict(pos_W=round(float(np.clip(P, 0, None).sum(2).mean()), 1), neg_W=round(float(-np.clip(P, None, 0).sum(2).mean()), 1),
                    copper_W=round(float((TAU ** 2 / KM ** 2).sum(2).mean()), 1))
load = F.sum(3) > 5.0                       # (n,T,2) foot loaded
td_type = dict(heel=0, flat=0, fore=0); cop_bins = np.zeros((10, 2)); coll = []; push = []; imp = []
hip_sw, hip_st = [], []; air = []
for i in range(load.shape[0]):
    for s in range(2):
        L = load[i, :, s]
        on = np.where(L[1:] & ~L[:-1])[0] + 1; off = np.where(~L[1:] & L[:-1])[0] + 1
        for t0 in on:
            if t0 < 40 or t0 > T - 40: continue
            fh, fm, ft = F[i, t0:t0 + 3, s].mean(0)
            if fh > 2 * (fm + ft): td_type['heel'] += 1
            elif (fm + ft) > 2 * fh:
                td_type['fore'] += 1
                prev = off[off < t0]; air.append(int(t0 - prev[-1]) if len(prev) else -1)
            else: td_type['flat'] += 1
            imp.append(F[i, t0:t0 + 20, s].sum(1).max() / MG)
            coll.append(-np.clip(P[i, t0:t0 + 20], None, 0).sum() * dt)          # negative joint work, 0.1 s after touchdown [J]
            # trailing (other) leg ankle-pitch positive work 0.15 s before vs 0.15 s after this touchdown
            a = 4 if s == 1 else 10
            pa = np.clip(P[i, :, a], 0, None)
            push.append((pa[t0 - 30:t0].sum() * dt, pa[t0:t0 + 30].sum() * dt))
            nxt = off[off > t0]
            if len(nxt):
                t1 = nxt[0]; ft_ = F[i, t0:t1, s]; tot = ft_.sum(1)
                ok = tot > 5
                if ok.sum() > 10:
                    cop = (ft_ * XS[s]).sum(1)[ok] / tot[ok]; ph = np.linspace(0, 1, ok.sum(), endpoint=False)
                    b = (ph * 10).astype(int)
                    for k in range(10):
                        if (b == k).any(): cop_bins[k] += [cop[b == k].mean(), 1]
        hp = 0 if s == 0 else 6
        hip_sw.append(np.abs(TAU[i, ~L, hp]).mean() if (~L).any() else np.nan); hip_st.append(np.abs(TAU[i, L, hp]).mean())
res['touchdown'] = td_type; res['fore_td_airtime_steps_200Hz'] = sorted(air)[::max(1, len(air) // 15)]
res['impact_peak_mg'] = round(float(np.mean(imp)), 2)
res['collision_negwork_J_per_step'] = round(float(np.mean(coll)), 2)
pu = np.array(push); res['trail_ankle_push_J'] = dict(before_td=round(float(pu[:, 0].mean()), 2), after_td=round(float(pu[:, 1].mean()), 2))
res['cop_x_cm_by_stance_decile'] = [round(100 * c / max(nn, 1), 1) for c, nn in cop_bins]
res['foot_sites_x_cm'] = (100 * XS[0]).round(1).tolist()
res['hip_pitch_tau_swing_vs_stance_Nm'] = [round(float(np.nanmean(hip_sw)), 1), round(float(np.mean(hip_st)), 1)]
hsw = ~load[:, :, 0]; res['hip_pitch_swing_P_W'] = dict(pos=round(float(np.clip(P[:, :, 0][hsw], 0, None).mean()), 2), neg=round(float(-np.clip(P[:, :, 0][hsw], None, 0).mean()), 2))
print(json.dumps(res, indent=1))
if out: json.dump(res, open(out, 'w'), indent=1)
