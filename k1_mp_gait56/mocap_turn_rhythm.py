# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""In-place turning rhythm from the user's monocular motion capture (IMG_2938, MediaPipe -> 21-joint skeleton).

Only the turning RHYTHM is used (user decision): the leg joint angles of this capture are not usable for imitation
(foot heights move together with the facing direction, both feet move +-20 cm front-back together = depth error,
left/right thigh/shank lengths differ by 6 cm, 64 % of the frames have a low-visibility joint).
The pelvis heading (hip line) is the most robust quantity of a monocular estimate, so the rhythm is taken from it:
turning happens in bursts (one burst = one step), separated by pauses in double support.

python3 mocap_turn_rhythm.py <motion_and_confidence.npz>  -> mocap/turn_rhythm.json
"""
import sys, os, json
import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))


def main(path):
    d = np.load(path)
    J = list(d['joint_names']); P = d['fk_positions']; fps = float(d['fps'])
    h = P[:, J.index('LeftUpLeg')] - P[:, J.index('RightUpLeg')]
    yaw = np.unwrap(np.arctan2(-h[:, 2], h[:, 0]))
    rate = savgol_filter(yaw, 7, 2, deriv=1) * fps
    act = np.abs(rate) > np.radians(20)                 # turning burst
    bursts = []
    i = 0
    while i < len(act):
        if act[i]:
            j = i
            while j < len(act) and act[j]:
                j += 1
            if j - i >= 4:
                bursts.append((i, j, yaw[j - 1] - yaw[i]))
            i = j
        else:
            i += 1
    b = np.array([(i / fps, (j - i) / fps, dy) for i, j, dy in bursts])
    start, dur, dyaw = b[:, 0], b[:, 1], b[:, 2]
    gap = np.diff(start)
    ok = (np.abs(dyaw) > np.radians(8)) & (np.abs(dyaw) < np.radians(90))
    period = np.r_[gap, np.nan]
    w_avg = dyaw / period
    sel = ok & np.isfinite(period) & (period < 3.0)
    out = dict(source=os.path.basename(path), n_bursts=int(len(b)), n_used=int(sel.sum()),
               yaw_per_step_deg_median=float(np.degrees(np.median(np.abs(dyaw[sel])))),
               yaw_per_step_deg_p25_p75=[float(x) for x in np.degrees(np.percentile(np.abs(dyaw[sel]), [25, 75]))],
               burst_duration_s_median=float(np.median(dur[sel])),
               step_period_s_min_p10=float(np.percentile(period[sel], 10)),
               bins=[])
    for lo, hi in ((0.0, 0.4), (0.4, 0.7), (0.7, 1.0), (1.0, 2.0)):
        k = sel & (np.abs(w_avg) >= lo) & (np.abs(w_avg) < hi)
        if k.sum() >= 3:
            out['bins'].append(dict(w_rad_s=[lo, hi], n=int(k.sum()),
                                    yaw_per_step_deg=float(np.degrees(np.median(np.abs(dyaw[k])))),
                                    step_period_s=float(np.median(period[k])),
                                    burst_s=float(np.median(dur[k]))))
    print(json.dumps(out, indent=1))
    os.makedirs(os.path.join(HERE, 'mocap'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'mocap', 'turn_rhythm.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
