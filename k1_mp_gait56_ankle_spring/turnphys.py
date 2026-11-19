# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""v5.5 turning physics shared by walking and running (idea & direction: Takeyuki-K).

1. Speed governor ("slow down to turn"): a turn at speed v and yaw rate w needs a centripetal acceleration
   a = v * w, supplied only by foot friction and limited by how far the body can lean. The governor keeps
   v * w <= A_LAT: the yaw-rate reference is limited by the *current* speed (w <= A_LAT / v_ref), and the speed
   target is lowered to A_LAT / |w_cmd| so that the requested turn rate becomes reachable as the robot slows down.
   Both references are still ramped by the usual acceleration limits, so nothing jumps.
2. Lean into the turn: steady turning needs the centre of mass on the inside of the support, lean angle
   phi = atan(v * w / g). Used (a) as the trunk-orientation target in the reward and (b) as a partial feed-forward
   on the reference (both hip rolls -k*phi, both ankle rolls +k*phi: feet move to the outside of the turn, soles stay
   flat). Sign conventions were checked by forward kinematics (left turn w > 0 -> lean left -> g_body_y = +sin phi).
"""
import numpy as np

G = 9.81
A_LAT_WALK = 1.0       # m/s^2 (v3 trained up to v 1.0 / w 1.0)
A_LAT_RUN = 3.0        # m/s^2 -> 17 deg lean; assumption, well inside friction 1.0 (about 9.8 m/s^2)
HIP_ROLL = (1, 7)
ANKLE_ROLL = (5, 11)


def governed_targets(v_cmd, w_cmd, v_ref, a_lat, v_floor=0.0):
    """(speed target, yaw-rate limit) for the reference ramps."""
    aw = np.abs(w_cmd)
    v_tgt = np.where(aw > 1e-3, np.minimum(v_cmd, a_lat / np.maximum(aw, 1e-3)), v_cmd)
    v_tgt = np.maximum(v_tgt, np.minimum(v_cmd, v_floor))
    w_lim = a_lat / np.maximum(v_ref, a_lat)            # |w| <= 1 when v_ref <= a_lat
    w_tgt = np.clip(w_cmd, -w_lim, w_lim)
    return v_tgt, w_tgt


def lean_angle(v_ref, w_ref, max_lean=0.35):
    return np.clip(np.arctan2(v_ref * w_ref, G), -max_lean, max_lean)


def apply_lean_ff(qbase, phi, k):
    """feet to the outside of the turn, soles flat (qbase: (n, 23) joint targets, modified in place)"""
    f = (k * phi)[:, None]
    qbase[:, list(HIP_ROLL)] -= f
    qbase[:, list(ANKLE_ROLL)] += f
    return qbase
