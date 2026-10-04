# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""Generate K1 + passive MP(toe) joint model (MJCF + URDF) from ROBOTIS ai_sapiens K1.

MP joint design
- Location: x=+0.075 m, z=-0.050 m in ankle_roll_link frame (72% of foot length from heel,
  15 mm above sole - same as human MTP head height).
- Axis: +y (pitch). Toe dorsiflexion (toe up) is negative rotation. Range [-1.0, 0.15] rad.
- Passive: no motor. Torsional spring (stiffness K, ref 0 = flat) + small damping.
  K is chosen so that  K * theta_max < m*g * l_toe  -> spring alone can never lift the robot,
  but it bends naturally when the CoP moves onto the toes.
- Foot contact: rounded heel (capsule r=15mm) -> metatarsal-head capsule -> flat toe pad box.
"""
import os, re, copy
import numpy as np
import trimesh
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
DESC = os.path.join(ROOT, '..', 'ai_sapiens', 'ai_sapiens_description')
MESH = os.path.join(DESC, 'meshes', 'k1_rev1')

MP_X, MP_Z = 0.075, -0.050
SOLE_Z = -0.065
TOTAL_MASS = 35.706
L_TOE = 0.025          # MP -> toe pad centre of pressure [m]
THETA_MAX = 1.0        # dorsiflexion limit [rad]
# spring alone must not lift the foot even in double support (half body weight per foot):
#   K*theta_max / l_toe <= 0.6 * (m g / 2)
K_MP = round(0.6 * 0.5 * TOTAL_MASS * 9.81 * L_TOE / THETA_MAX, 2)   # ~2.6 Nm/rad
D_MP = 0.04


def split_meshes():
    info = {}
    for side in ['left', 'right']:
        src = os.path.join(MESH, f'{side}_ankle_roll_link.stl')
        m = trimesh.load(src)                      # mm units
        mpx = MP_X * 1000
        rear = m.slice_plane([mpx - 1.0, 0, 0], [-1, 0, 0], cap=True)   # 1 mm gap
        toe = m.slice_plane([mpx + 1.0, 0, 0], [1, 0, 0], cap=True)
        toe.apply_translation([-MP_X * 1000, 0, -MP_Z * 1000])
        rear.export(os.path.join(MESH, f'{side}_ankle_roll_link_rear.stl'))
        toe.export(os.path.join(MESH, f'{side}_toe_link.stl'))
        vr, vt = abs(rear.volume), abs(toe.volume)
        info[side] = dict(frac=vt / (vr + vt), m_toe=round(max(0.03, 0.6144995 * vt / (vr + vt) * 0.6), 4), toe_com=(toe.center_mass * 1e-3).tolist(),
                          rear_com=(rear.center_mass * 1e-3).tolist())
    return info


def build_mjcf(info):
    src = os.path.join(DESC, 'mujoco', 'k1', 'k1.xml')
    tree = ET.parse(src)
    root = tree.getroot()
    root.set('model', 'k1_mp')
    asset = root.find('asset')
    for side in ['left', 'right']:
        ET.SubElement(asset, 'mesh', name=f'{side}_ankle_roll_link_rear',
                      file=f'{side}_ankle_roll_link_rear.stl', scale='0.001 0.001 0.001')
        ET.SubElement(asset, 'mesh', name=f'{side}_toe_link',
                      file=f'{side}_toe_link.stl', scale='0.001 0.001 0.001')
    # default class for MP joint
    dflt = root.find('default')
    mp = ET.SubElement(dflt, 'default', {'class': 'mp_passive'})
    ET.SubElement(mp, 'joint', type='hinge', axis='0 1 0', range=f'{-THETA_MAX} 0.05',
                  stiffness=str(K_MP), springref='0', damping=str(D_MP), armature='0.0001',
                  solreflimit='0.004 1')
    foot = ET.SubElement(dflt, 'default', {'class': 'foot'})
    ET.SubElement(foot, 'geom', contype='1', conaffinity='1', group='3', density='0',
                  friction='1.0 0.02 0.001', condim='3', rgba='0.9 0.4 0.2 0.5', priority='1')

    for body in root.iter('body'):
        name = body.get('name')
        if name not in ('left_ankle_roll_link', 'right_ankle_roll_link'):
            continue
        side = name.split('_')[0]
        inert = body.find('inertial')
        m_tot = float(inert.get('mass'))
        frac = info[side]['frac']
        m_toe = info[side]['m_toe']   # toe is mostly shell, light
        m_rear = m_tot - m_toe
        # keep rear inertia (slightly conservative), shift COM back a little
        com = np.array([float(v) for v in inert.get('pos').split()])
        toe_com_world = np.array(info[side]['toe_com']) + np.array([MP_X, 0, MP_Z])
        com_rear = (com * m_tot - toe_com_world * m_toe) / m_rear
        inert.set('mass', f'{m_rear:.6f}')
        inert.set('pos', ' '.join(f'{v:.6e}' for v in com_rear))
        # replace geoms
        for g in list(body.findall('geom')):
            body.remove(g)
        ET.SubElement(body, 'geom', {'class': 'visual', 'type': 'mesh',
                                     'mesh': f'{side}_ankle_roll_link_rear', 'material': 'dark'})
        r_h = 0.015
        ET.SubElement(body, 'geom', {'class': 'foot', 'name': f'{side}_heel', 'type': 'capsule',
                                     'fromto': f'-0.047 -0.028 {SOLE_Z + r_h} -0.047 0.028 {SOLE_Z + r_h}',
                                     'size': f'{r_h}'})
        r_m = 0.008
        ET.SubElement(body, 'geom', {'class': 'foot', 'name': f'{side}_meta', 'type': 'capsule',
                                     'fromto': f'0.066 -0.038 {SOLE_Z + r_m} 0.066 0.038 {SOLE_Z + r_m}',
                                     'size': f'{r_m}'})
        ET.SubElement(body, 'site', name=f'{side}_heel_site', pos=f'-0.047 0 {SOLE_Z}', size='0.008', group='4')
        ET.SubElement(body, 'site', name=f'{side}_meta_site', pos=f'0.066 0 {SOLE_Z}', size='0.008', group='4')
        # toe body
        toe = ET.SubElement(body, 'body', name=f'{side}_toe_link', pos=f'{MP_X} 0 {MP_Z}')
        tc = info[side]['toe_com']
        ET.SubElement(toe, 'inertial', mass=f'{m_toe:.4f}', pos=' '.join(f'{v:.4e}' for v in tc),
                      diaginertia='2.0e-5 1.5e-5 2.5e-5')
        ET.SubElement(toe, 'joint', {'class': 'mp_passive', 'name': f'{side}_mp_joint'})
        ET.SubElement(toe, 'geom', {'class': 'visual', 'type': 'mesh', 'mesh': f'{side}_toe_link',
                                    'material': 'blue'})
        ET.SubElement(toe, 'geom', {'class': 'foot', 'name': f'{side}_toe', 'type': 'box',
                                    'pos': f'0.027 0 {SOLE_Z - MP_Z + 0.004}', 'size': '0.024 0.034 0.004'})
        ET.SubElement(toe, 'site', name=f'{side}_toe_site', pos=f'0.03 0 {SOLE_Z - MP_Z}', size='0.008', group='4')

    # sensors: toe joint
    sens = root.find('sensor')
    for side in ['left', 'right']:
        ET.SubElement(sens, 'jointpos', name=f'{side}_mp_pos', joint=f'{side}_mp_joint')
    # keyframe length: add 2 zeros (toe joints are inserted right after each ankle roll)
    key = root.find('keyframe').find('key')
    q = key.get('qpos').split()
    # order: free(7), left leg 6 + toe, right leg 6 + toe, waist, arms 10
    q = q[:7] + q[7:13] + ['0'] + q[13:19] + ['0'] + q[19:]
    key.set('qpos', ' '.join(q))
    ET.indent(tree, space='  ')
    out = os.path.join(DESC, 'mujoco', 'k1', 'k1_mp.xml')
    tree.write(out)
    scene = open(os.path.join(DESC, 'mujoco', 'k1', 'scene.xml')).read().replace('k1.xml', 'k1_mp.xml')
    scene = scene.replace('model="k1 scene"', 'model="k1_mp scene"')
    scene = scene.replace('<geom name="floor" size="0 0 0.05"', '<geom name="floor" size="0 0 0.05" friction="1.0 0.02 0.001"')
    open(os.path.join(DESC, 'mujoco', 'k1', 'scene_mp.xml'), 'w').write(scene)
    return out


def build_urdf(info):
    src = os.path.join(DESC, 'urdf', 'k1_rev1', 'k1.urdf')
    txt = open(src).read()
    for side in ['left', 'right']:
        txt = txt.replace(f'meshes/k1_rev1/{side}_ankle_roll_link.stl', f'meshes/k1_rev1/{side}_ankle_roll_link_rear.stl')
        tc = info[side]['toe_com']
        add = f'''
  <!-- ===== Passive MP (metatarsophalangeal) joint : spring-return, no actuator ===== -->
  <link name="{side}_toe_link">
    <inertial>
      <origin xyz="{tc[0]:.4f} {tc[1]:.4f} {tc[2]:.4f}" rpy="0 0 0"/>
      <mass value="{info[side]['m_toe']}"/>
      <inertia ixx="2.0e-5" ixy="0" ixz="0" iyy="1.5e-5" iyz="0" izz="2.5e-5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="package://ai_sapiens_description/meshes/k1_rev1/{side}_toe_link.stl" scale="0.001 0.001 0.001"/></geometry>
    </visual>
    <collision>
      <origin xyz="0.027 0 {SOLE_Z - MP_Z + 0.004}" rpy="0 0 0"/>
      <geometry><box size="0.048 0.068 0.008"/></geometry>
    </collision>
  </link>
  <joint name="{side}_mp_joint" type="revolute">
    <origin xyz="{MP_X} 0 {MP_Z}" rpy="0 0 0"/>
    <parent link="{side}_ankle_roll_link"/>
    <child link="{side}_toe_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="{-THETA_MAX}" upper="0.05" effort="0" velocity="20"/>
    <!-- passive torsion spring: stiffness {K_MP} Nm/rad (ref 0 rad), damping {D_MP} Nms/rad -->
    <dynamics damping="{D_MP}" friction="0"/>
  </joint>
'''
        txt = txt.replace('</robot>', add + '</robot>')
    for side in ['left', 'right']:
        i = txt.find(f'<link name="{side}_ankle_roll_link">')
        j = txt.find('</link>', i)
        blk = txt[i:j]
        mold = float(re.search(r'<mass value="([0-9.eE+-]+)"', blk).group(1))
        blk = re.sub(r'<mass value="[0-9.eE+-]+"', f'<mass value="{mold - info[side]["m_toe"]:.6f}"', blk)
        txt = txt[:i] + blk + txt[j:]
    # MuJoCo-compatible spring annotation (URDF has no spring tag)
    txt = txt.replace('<robot ', f'<!-- MP joints: passive spring K={K_MP} Nm/rad (see k1_mp.xml) -->\n<robot ', 1)
    out = os.path.join(DESC, 'urdf', 'k1_rev1', 'k1_mp.urdf')
    open(out, 'w').write(txt)
    return out


if __name__ == '__main__':
    info = split_meshes()
    print('split info', info)
    print('K_MP =', K_MP)
    print(build_mjcf(info))
    print(build_urdf(info))
