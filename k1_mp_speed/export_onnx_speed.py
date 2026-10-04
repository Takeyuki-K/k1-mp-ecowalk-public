# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Export the speed-command policy to ONNX + deployment spec + reference library tables.
Deployment loop (50 Hz):
  v_ref  <- v_ref + clip(v_cmd - v_ref, +-0.6*0.02)               (when walking)
  a      = policy(obs)  (37 outputs)
  v_lib  = clip(v_ref * (1 + 0.15*clip(a[36],-2,2)), 0.25, 1.65)   gait choice
  phase += 0.02 / T(v_lib) ;  alpha ramps 0->1 (start) / 1->0 (stop) at the same rate
  q_ref  = blend of library cycles at v_lib and phase ;  q_base = (1-alpha)*default + alpha*q_ref
  target = q_base ; target[:12] += 0.25*a[:12]
  Kp = Kp_nom*clip(1+0.5*a[12:24], 0, 1.5) ; Kd = Kd_nom*clip(1+0.5*a[24:36], 0.05, 3)  -> Goal Position / MIT P/D Gain
"""
import sys, os, json
import numpy as np, torch, torch.nn as nn
from k1env import ACT_JOINTS, DEFAULT_POSE, KP, KD
from k1env_speed import K1SpeedBatch, ACC, V_MIN, V_MAX
from k1env_eco import KP_RANGE, KD_RANGE, KM
from ppo_speed import ACEco


class Actor(nn.Module):
    def __init__(self, net):
        super().__init__(); self.norm = net.na_norm; self.actor = net.actor

    def forward(self, obs):
        return self.actor(self.norm(obs))


def main(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    env = K1SpeedBatch(1, stage=2)
    oa, oc = env.obs()
    net = ACEco(oa.shape[1], oc.shape[1], env.nact)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    act = Actor(net).eval()
    f = os.path.join(outdir, 'k1_mp_speed_policy.onnx')
    torch.onnx.export(act, torch.zeros(1, oa.shape[1]), f, input_names=['obs'], output_names=['action'],
                      dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}}, opset_version=17, dynamo=False)
    import onnx
    onnx.checker.check_model(f)
    spec = dict(control_hz=50, command_range_mps=[V_MIN, V_MAX], ref_acceleration_limit=ACC,
                obs_layout=[['base_ang_vel(local) x0.25', 3], ['projected_gravity', 3], ['walk_flag', 1], ['alpha', 1],
                            ['v_cmd', 1], ['v_ref', 1], ['heading_error(clip +-1 rad)', 1], ['sin(2pi*phase)', 1],
                            ['cos(2pi*phase)', 1], ['leg_q - default (12)', 12], ['leg_qd x0.05 (12)', 12],
                            ['last_action (37)', 37]],
                action_layout='[0:12] position residual x0.25 rad, [12:24] Kp scale, [24:36] Kd scale, [36] gait choice',
                gait_choice='v_lib = clip(v_ref*(1+0.15*clip(a36,-2,2)), 0.25, 1.65)',
                gain_law=f'Kp = Kp_nom*clip(1+0.5a, {KP_RANGE}); Kd = Kd_nom*clip(1+0.5a, {KD_RANGE})',
                joints=ACT_JOINTS, default_pose=DEFAULT_POSE.tolist(), kp_nom=KP.tolist(), kd_nom=KD.tolist(),
                library_speeds=env.lib.speeds.tolist(), library_cycle_time_s=env.lib.Tc.tolist(),
                energy_model_Km=KM.tolist())
    json.dump(spec, open(os.path.join(outdir, 'deploy_spec.json'), 'w'), indent=1)
    M = env.lib.M
    for si, v in enumerate(env.lib.speeds):
        np.savetxt(os.path.join(outdir, f'ref_{v:.2f}mps.csv'),
                   np.concatenate([np.arange(M)[:, None] / M, env.lib.q[si]], 1), delimiter=',',
                   header='phase,' + ','.join(ACT_JOINTS), comments='', fmt='%.6f')
    print('exported', outdir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
