# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Joint-speed audit of the v5 running policy (added for the public release; does not change any policy or result).
Same protocol as eval_run2.py 'walk' mode (16 robots, hand-over from recorded fast-walking states, 12 s, metrics after
5 s). For every 2 ms physics step and leg joint it records |joint speed| and the direction of the torque the motor model
actually applied, and reports:
  * time with any leg joint above its speed limit (11.5 rad/s, ankle roll 20.9 rad/s)
  * of those over-limit samples, the fraction where the motor torque still pushed in the direction of motion
    (motoring) - 0 means the motor was always braking, i.e. the joint was driven beyond the limit by impact / inertia
  * per joint p99 and peak speed, and how far above the limit the joint went
python3 audit_joint_speed.py runs/final/run.pt [--speeds 2,3,4,5,5.5] [--json out/joint_speed_audit.json]
"""
import sys, json
import numpy as np, torch
import k1env_run2 as E
import eval_run2 as V
from k1env import ACT_JOINTS
from motor import QD_LIM
from ppo_eco import AC

torch.set_num_threads(1)
REC = []
_orig = E.motor_limit


def _recording_motor_limit(tau, qd):
    f, tdir, direct = _orig(tau, qd)
    applied = np.where(direct, tdir, f * tau)
    REC.append((np.abs(qd).copy(), applied * qd > 0))
    return f, tdir, direct


def main():
    E.motor_limit = _recording_motor_limit
    path = sys.argv[1]
    speeds = [float(x) for x in V.arg('--speeds', '2,3,4,5,5.5').split(',')]
    probe = E.K1Run2Batch(2, nthread=1); oa, oc = probe.obs()
    net = AC(oa.shape[1], oc.shape[1], 36)
    net.load_state_dict(torch.load(path, map_location='cpu')['model']); net.eval()
    names = list(ACT_JOINTS[:12])
    res = []
    for v in speeds:
        REC.clear()
        r = V.run(net, 'walk', v, 16)
        nsub = len(REC) // 600
        W = np.stack([a for a, _ in REC])[250 * nsub:]
        M = np.stack([b for _, b in REC])[250 * nsub:]
        over = W > QD_LIM
        mag = (W - QD_LIM)[over]
        row = dict(v_cmd=v, speed=r['speed'], survival=r['survival'],
                   time_any_joint_over_limit_pct=100 * float(over.any(2).mean()),
                   over_limit_samples_with_motoring_torque=float((over & M).sum() / max(over.sum(), 1)),
                   excess_over_limit_p50=float(np.percentile(mag, 50)) if over.any() else 0.0,
                   excess_over_limit_p99=float(np.percentile(mag, 99)) if over.any() else 0.0,
                   joint_limit={n: float(QD_LIM[i]) for i, n in enumerate(names)},
                   joint_over_limit_pct={n: 100 * float(over[:, :, i].mean()) for i, n in enumerate(names)},
                   joint_speed_p99={n: float(np.percentile(W[:, :, i], 99)) for i, n in enumerate(names)},
                   joint_speed_peak={n: float(W[:, :, i].max()) for i, n in enumerate(names)})
        res.append(row)
        print(f"cmd {v}: {r['speed']:.2f} m/s, any joint over limit {row['time_any_joint_over_limit_pct']:.1f} % of the time, "
              f"motoring while over {row['over_limit_samples_with_motoring_torque']:.3f}, excess p50/p99 "
              f"{row['excess_over_limit_p50']:.2f}/{row['excess_over_limit_p99']:.2f} rad/s", flush=True)
    out = V.arg('--json')
    if out:
        json.dump(dict(policy=path, protocol='eval_run2 walk mode, 16 robots, 12 s, metrics after 5 s', results=res),
                  open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
