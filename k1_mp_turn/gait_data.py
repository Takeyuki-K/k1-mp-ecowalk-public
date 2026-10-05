"""Load open-source gait TRC (Duarte BMC dataset) -> hip/ankle joint centres, heel, MT, per leg.
Coordinates: X forward, Y up, Z lateral (right = +Z in this lab). Units -> metres.
Hip joint centre (HJC) for files without virtual markers: Bell/Harrington regression from ASIS/PSIS.
Ankle joint centre = midpoint of lateral & medial malleolus.
"""
import numpy as np


def load_trc(path):
    lines = open(path).read().splitlines()
    names = [n for n in lines[3].split('\t')[2:] if n.strip()]
    rate = float(lines[2].split()[0])
    rows = []
    for l in lines[5:]:
        p = l.split('\t')
        if len(p) < 5:
            continue
        rows.append([float(v) if v.strip() else np.nan for v in p[2:2 + 3 * len(names)]])
    n = max(len(r) for r in rows)
    arr = np.full((len(rows), n), np.nan)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r
    mk = {}
    for i, nm in enumerate(names):
        if 3 * i + 3 <= n:
            mk[nm] = arr[:, 3 * i:3 * i + 3] / 1000.0
    return mk, rate


def to_robot_axes(p):
    """lab (X fwd, Y up, Z right) -> robot world (x fwd, y left, z up)"""
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)


def joint_centres(mk):
    g = lambda n: to_robot_axes(mk[n])
    rasis, lasis, rpsis, lpsis = g('R.ASIS'), g('L.ASIS'), g('R.PSIS'), g('L.PSIS')
    if 'V_R.Hip_JC' in mk and not np.isnan(mk['V_R.Hip_JC']).all():
        rhip, lhip = g('V_R.Hip_JC'), g('V_L.Hip_JC')
    else:  # Harrington 2007 regression
        midasis = 0.5 * (rasis + lasis)
        midpsis = 0.5 * (rpsis + lpsis)
        y = lasis - rasis; pw = np.linalg.norm(y, axis=1, keepdims=True); y /= pw
        x = midasis - midpsis; x -= (x * y).sum(1, keepdims=True) * y; x /= np.linalg.norm(x, axis=1, keepdims=True)
        z = np.cross(x, y)
        pd = np.linalg.norm(midasis - midpsis, axis=1, keepdims=True)
        o = midasis
        def hjc(side):
            px = -0.24 * pd - 0.0099
            pz = -0.30 * pw - 0.0109
            py = side * (0.33 * pw + 0.0073)
            return o + px * x + py * y + pz * z
        rhip, lhip = hjc(-1), hjc(+1)
    out = {}
    for s, S in (('r', 'R'), ('l', 'L')):
        out[s] = dict(hip=rhip if s == 'r' else lhip,
                      ankle=0.5 * (g(f'{S}.Ankle') + g(f'{S}.Ankle.Medial')),
                      heel=g(f'{S}.Heel'),
                      mt=(g(f'{S}.MT1') + g(f'{S}.MT5')) / 2)
    out['pelvis'] = 0.25 * (rasis + lasis + rpsis + lpsis)
    return out
