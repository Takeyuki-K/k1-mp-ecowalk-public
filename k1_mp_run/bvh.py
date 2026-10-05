# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Minimal BVH reader + forward kinematics for the CMU mocap BVH conversion (Y-up, ZYX euler channels).
Returns world positions of joints and end sites in metres (CMU unit: 1 = 0.056444 m), converted to
robot-style axes later by the retargeting code."""
import numpy as np

CMU_UNIT = 0.056444


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 'X':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'Y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load(path):
    lines = open(path).read().split('\n')
    joints = []          # dict(name, parent, offset, channels)
    stack = []
    i = 0
    while not lines[i].strip().startswith('MOTION'):
        t = lines[i].strip().split()
        if not t:
            i += 1; continue
        if t[0] in ('ROOT', 'JOINT'):
            joints.append(dict(name=t[1], parent=stack[-1] if stack else -1, offset=None, ch=[]))
            cur = len(joints) - 1
        elif t[0] == 'End':
            joints.append(dict(name=joints[stack[-1]]['name'] + '_End', parent=stack[-1], offset=None, ch=[]))
            cur = len(joints) - 1
        elif t[0] == '{':
            stack.append(cur)
        elif t[0] == '}':
            stack.pop()
        elif t[0] == 'OFFSET':
            joints[cur]['offset'] = np.array([float(v) for v in t[1:4]])
        elif t[0] == 'CHANNELS':
            joints[cur]['ch'] = t[2:2 + int(t[1])]
        i += 1
    nfr = int(lines[i + 1].split()[-1]); dt = float(lines[i + 2].split()[-1])
    data = np.array([[float(v) for v in l.split()] for l in lines[i + 3:i + 3 + nfr] if l.strip()])
    # forward kinematics
    nj = len(joints)
    P = np.zeros((len(data), nj, 3))
    for f, row in enumerate(data):
        k = 0
        R = [None] * nj
        for j, jt in enumerate(joints):
            pos = jt['offset'].copy()
            rot = np.eye(3)
            for c in jt['ch']:
                v = row[k]; k += 1
                if c.endswith('position'):
                    pos = pos * 0 + pos  # keep offset
                    pos[{'X': 0, 'Y': 1, 'Z': 2}[c[0]]] = v + (jt['offset'][{'X': 0, 'Y': 1, 'Z': 2}[c[0]]] if False else 0)
                else:
                    rot = rot @ _rot(c[0], v)
            if jt['parent'] < 0:
                R[j] = rot; P[f, j] = pos
            else:
                p = jt['parent']
                P[f, j] = P[f, p] + R[p] @ jt['offset']
                R[j] = R[p] @ rot
    names = [j['name'] for j in joints]
    return dict(names=names, P=P * CMU_UNIT, dt=dt)
