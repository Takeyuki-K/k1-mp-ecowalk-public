"""Diagnostic: pelvis roll, foot placement, hip-roll copper, power for a walking checkpoint.
Run from k1_mp_gait56/ with deployment env vars set. Usage: diag.py <walk.pt> [speeds] [n] [tag]"""
import os, sys, json
import numpy as np, torch, mujoco
os.environ.setdefault('WALK_BRAKE','1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../repo/k1_mp_gait56')
sys.path.insert(0, os.getcwd())
import eval56
from k1env_walk4 import K1Walk4Batch as C

def main():
    path = sys.argv[1]
    speeds = [float(x) for x in (sys.argv[2].split(',') if len(sys.argv)>2 else ['0.6','1.0','1.4'])]
    n = int(sys.argv[3]) if len(sys.argv)>3 else 8
    tag = sys.argv[4] if len(sys.argv)>4 else 'diag'
    C.P_RUN = 0.0
    env = C(n, stage=2, randomize=False, seed=31, ep_len=10**9, nthread=2)
    env.pushes=False; env.scripted=True
    net = eval56.load(path, env)
    out=[]
    for v in speeds:
        eval56.stand(env); env.cmd[:]=1; env.v_cmd[:]=v; env.w_cmd[:]=0
        oa,_ = env.obs(); alive=np.ones(env.n,bool)
        rolls=[]; plegs=[]; phips=[]; kproll=[]; lever=[]; pelvis_y=[]; sroll=[]; aroll=[]; pcorr=[]; pank=[]
        td_width=[]; td_off=[]  # touchdown: foot separation, |foot lat - pelvis lat|
        prev_load=np.zeros((env.n,2),bool)
        p0=None
        for k in range(int(12*50)):
            _,term,_,_ = env.step(eval56.act(net,oa)); oa,_=env.obs(); alive &= ~term
            if k < 4*50: 
                # update prev_load anyway
                t = np.stack([env.sdata[:, env.sens[f'left_{p}_touch'][0]] for p in ('heel','meta','toe')],1) if False else None
                continue
            quat,g = env.base_frame(); yaw = env.yaw(); q = env.qpos()
            if p0 is None: p0 = q[:, :2].copy()
            p1 = q[:, :2].copy()
            rolls.append(np.degrees(np.arcsin(np.clip(g[:,1],-1,1))))
            from k1env import ACT_JOINTS as _AJ
            sroll.append(env.qpos()[:, env.qadr[[_AJ.index('left_shoulder_roll_joint'), _AJ.index('right_shoulder_roll_joint')]]])
            aroll.append(env.qpos()[:, env.qadr[[_AJ.index('left_ankle_roll_joint'), _AJ.index('right_ankle_roll_joint')]]])
            plegs.append(getattr(env,'P_leg',env.P_elec).copy()); phips.append(getattr(env,'P_arm',np.zeros(env.n)).copy())
            # hip-roll copper from tau_sub (joints 1,7; KM=4.0)
            tau = env.tau_sub  # (n, nsub, 12)
            # corrected electrical power: ankle-roll channels (idx 5,11) separated out.
            # For the passive-spring variant their tau is fictitious (no motor) -> excluded.
            _m = env.m
            _qd = env.st_out[:, :, 1+_m.nq:1+_m.nq+_m.nv][:, :, env.dadr[:12]]
            _KM = np.array([2.2 if j in (5, 11) else 4.0 for j in range(12)])
            _cu = tau**2 / _KM**2; _mech = np.clip(tau*_qd, 0, None)
            _ank = (_cu[:, :, [5, 11]].sum(2) + _mech[:, :, [5, 11]].sum(2)).mean(1)
            pcorr.append(((_cu.sum(2) + _mech.sum(2)).mean(1)) - _ank); pank.append(_ank)
            phips_roll = ((tau[:,:,[1,7]]**2/16.0).sum(2)).mean(1)
            # store separately via rolls list hack: use dedicated list
            # (append to lever placeholder lists defined outside loop scope)
            # touchdown detection from touch sensors
            touch = np.zeros((env.n,6))
            for kk,nm in enumerate([f'{s}_{p}_touch' for s in ('left','right') for p in ('heel','meta','toe')]):
                touch[:,kk]=env.sd_out[:, :, env.sens[nm][0]].max(1) if hasattr(env,'sd_out') else env.sdata[:, env.sens[nm][0]]
            load = np.stack([touch[:,0:3].sum(1), touch[:,3:6].sum(1)],1) > 5.0
            # positions in heading frame
            fl = env.s('left_foot_pos'); fr = env.s('right_foot_pos')
            cy, sy = np.cos(yaw), np.sin(yaw)
            def lat(px,py): return -sy*px + cy*py
            pel_lat = lat(q[:,0], q[:,1]); l_lat = lat(fl[:,0], fl[:,1]); r_lat = lat(fr[:,0], fr[:,1])
            pelvis_y.append(pel_lat)
            # single-support lever: |pelvis - stance foot|
            ss = load[:,0] ^ load[:,1]
            lv = np.where(load[:,0], np.abs(pel_lat-l_lat), np.abs(pel_lat-r_lat))
            lever.append(np.where(ss, lv, np.nan))
            new_td = load & ~prev_load
            for side in (0,1):
                ids = np.where(new_td[:,side] & alive)[0]
                for i in ids:
                    td_width.append(abs(l_lat[i]-r_lat[i]))
                    td_off.append(abs((l_lat[i] if side==0 else r_lat[i]) - pel_lat[i]))
            prev_load = load
            # hip roll copper list stored in a global-ish way
            rolls_roll = phips_roll
            # append to dedicated lists (defined in outer scope via closure trick)
            hip_copper.append(rolls_roll); kp_roll.append(env.kp_scale[:, [1,7]].mean(1))
        R=np.array(rolls); 
        v_act = np.linalg.norm(p1 - p0, axis=1) / ((12*50 - 4*50 - 1) / 50.0)
        res=dict(tag=tag, v_cmd=v, survival=float(alive.mean()), v_meas=round(float(v_act[alive].mean()),3), J_per_m=round(float(np.array(plegs)[:,alive].mean()/v_act[alive].mean()),1),
                 roll_p2p_deg=round(float(np.mean(np.percentile(R[:,alive],95,0)-np.percentile(R[:,alive],5,0))),2) if alive.any() else None,
                 roll_rms_deg=round(float(np.sqrt((R[:,alive]**2).mean())),2) if alive.any() else None,
                 roll_absmean_deg=round(float(np.abs(R[:,alive]).mean()),2) if alive.any() else None,
                 P_leg_W=round(float(np.array(plegs)[:,alive].mean()),1) if alive.any() else None,
                 P_leg_corr_W=round(float(np.array(pcorr)[:,alive].mean()),1) if alive.any() and pcorr else None,
                 P_ankleroll_W=round(float(np.array(pank)[:,alive].mean()),1) if alive.any() and pank else None,
                 hiproll_copper_W=round(float(np.array(hip_copper)[:,alive].mean()),1) if alive.any() else None,
                 kp_hiproll_scale=round(float(np.array(kp_roll)[:,alive].mean()),3) if alive.any() else None,
                 lever_ss_cm=round(float(np.nanmean(np.array(lever)[:,alive]))*100,1) if alive.any() else None,
                 step_width_td_cm=round(float(np.mean(td_width))*100,1) if td_width else None,
                 touchdown_off_pelvis_cm=round(float(np.mean(td_off))*100,1) if td_off else None,
                 n_touchdowns=len(td_width),
                 arm_roll_p2p_deg=round(float(np.mean([np.mean(np.percentile(np.degrees(np.array(sroll))[:,alive,s],95,0)-np.percentile(np.degrees(np.array(sroll))[:,alive,s],5,0)) for s in (0,1)])),1) if alive.any() and sroll else None,
                 arm_roll_dev_rms_deg=round(float(np.sqrt((((np.array(sroll)[:,alive,:]-np.array([0.315,-0.315]))**2).mean()))*57.2958),1) if alive.any() and sroll else None,
                 ank_roll_rms_deg=round(float(np.sqrt((np.array(aroll)[:,alive,:]**2).mean())*57.2958),1) if alive.any() and aroll else None,
                 ank_roll_absmax_deg=round(float(np.abs(np.array(aroll)[:,alive,:]).max()*57.2958),1) if alive.any() and aroll else None)
        out.append(res); print(json.dumps(res), flush=True)
        # reset per-speed lists
        hip_copper.clear(); kp_roll.clear()
    json.dump(out, open(f'/tmp/diag_{tag}.json','w'), indent=1)

hip_copper=[]; kp_roll=[]
if __name__=='__main__':
    main()
