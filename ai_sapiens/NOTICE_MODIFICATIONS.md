# Modifications to ROBOTIS ai_sapiens files (Apache-2.0, Section 4(b))

Source: https://github.com/ROBOTIS-GIT/ai_sapiens (ai_sapiens_description).
Only the files needed for this project are included. Modified/added by Takeyuki-K, 2026:

| file | status |
|---|---|
| `mujoco/k1/k1.xml`, `mujoco/k1/scene.xml`, `urdf/k1_rev1/k1.urdf`, `meshes/k1_rev1/*.stl` (original names) | unmodified copies |
| `mujoco/k1/k1_mp.xml`, `mujoco/k1/scene_mp.xml` | **added** – generated from `k1.xml` by `k1_mp/gen_model.py`: foot split at the MP plane, passive spring MP joints, new foot contact geometry, toe mass/inertia |
| `urdf/k1_rev1/k1_mp.urdf` | **added** – generated from `k1.urdf` (same changes) |
| `meshes/k1_rev1/{left,right}_ankle_roll_link_rear.stl`, `{left,right}_toe_link.stl` | **added** – original foot meshes split at the MP plane |
