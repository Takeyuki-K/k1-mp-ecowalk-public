# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Export trained actor to ONNX + deployment spec (obs layout, gains, reference joint table).
Deployment (50 Hz):
  alpha  : ramps 0->1 in T_cycle s after walk command (and 1->0 on stop); phase advances by 1/(T_cycle*50) while alpha>0
  q_base = (1-alpha)*default_pose + alpha*ref_table(phase)        (23 joints)
  target = q_base;  target[:12] += 0.25 * clip(policy(obs), -4, 4)
  PD on motors with gains in deploy_spec.json. MP joints are passive (no motor, no sensor).
"""
import sys, json, os, numpy as np, torch, torch.nn as nn
from k1env import K1Batch, ACT_JOINTS, DEFAULT_POSE, KP, KD
from ppo import AC


class Actor(nn.Module):
    def __init__(self, net):
        super().__init__(); self.norm = net.na_norm; self.actor = net.actor

    def forward(self, obs):
        return self.actor(self.norm(obs))


def main(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    env = K1Batch(1, stage=2)
    oa, oc = env.obs()
    net = AC(oa.shape[1], oc.shape[1], 12)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    act = Actor(net).eval()
    x = torch.zeros(1, oa.shape[1])
    torch.onnx.export(act, x, os.path.join(outdir, 'k1_mp_walk_policy.onnx'), input_names=['obs'],
                      output_names=['action'], dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    spec = dict(
        control_hz=50, action_scale=0.25, action_clip=4.0, policy_joints=ACT_JOINTS[:12], all_joints=ACT_JOINTS,
        default_pose=DEFAULT_POSE.tolist(), kp=KP.tolist(), kd=KD.tolist(), T_cycle=env.ref.T,
        walk_speed=env.ref.speed,
        obs_layout=[['base_ang_vel(local) x0.25', 3], ['projected_gravity', 3], ['walk_cmd (0/1)', 1],
                    ['alpha', 1], ['sin(2pi*phase)', 1], ['cos(2pi*phase)', 1],
                    ['leg_q - default (12)', 12], ['leg_qd x0.05 (12)', 12], ['last_action (12)', 12]],
        mp_joint=dict(type='passive torsion spring', stiffness_Nm_per_rad=2.63, damping=0.04,
                      range_rad=[-1.0, 0.05], location_in_ankle_roll_link=[0.075, 0, -0.050]))
    json.dump(spec, open(os.path.join(outdir, 'deploy_spec.json'), 'w'), indent=1)
    ph = np.arange(env.ref.N) / env.ref.N
    np.savetxt(os.path.join(outdir, 'ref_joint_table.csv'), np.concatenate([ph[:, None], env.ref.q], 1),
               delimiter=',', header='phase,' + ','.join(ACT_JOINTS), comments='', fmt='%.6f')
    # check ONNX vs torch
    import onnx
    onnx.checker.check_model(os.path.join(outdir, 'k1_mp_walk_policy.onnx'))
    print('exported to', outdir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
