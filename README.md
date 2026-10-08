# K1 MP Eco-Walk

**English** | [日本語](README.ja.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23178357.svg)](https://doi.org/10.5281/zenodo.23178357)

**An energy-efficient walking / running / turning policy for the ROBOTIS AI Sapiens K1 humanoid with passive spring
toe (MP) joints — current version v5.6.3. MuJoCo simulation only.**

Idea & direction: **Takeyuki-K** · Implementation generated with Claude (Anthropic) under Takeyuki-K's direction

> **Simulation only.** Everything on this page was produced and measured in the MuJoCo physics simulator with a
> modified K1 model (passive spring toe joints added). **Nothing has been tested on a real robot.** Power values come
> from a motor model with assumed constants (see [Power model](#power-model)). Independent personal research,
> **not affiliated with or endorsed by ROBOTIS.**
>
> **Licences:** project code and original assets **Apache-2.0** · human-gait-derived reference trajectories
> **CC BY 4.0** (Marcos Duarte & Renato Naville Watanabe, BMC) · CMU motion-capture files: free use, acknowledgement
> below. See [NOTICE](NOTICE) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## What the current policy (v5.6.3) does

One walking policy and one running policy, switched automatically by a gait manager, driven by two commands:
forward speed *v* and yaw rate *ω* (plus "stop" and "hard brake"). All clips: MuJoCo, real time, one robot.

| | |
|---|---|
| **Stop, then re-place the feet.** Stops quickly without first aligning the feet; 1 s later, with no command, it steps back to the initial standing stance. ![stop](media/K1_v563_stop.gif) | **Turning on the spot** (+0.6 → −1.0 rad/s), both feet lifted and placed in a human turning rhythm (taken from a motion capture). ![in-place turning](media/K1_v563_inplace.gif) |
| **Walking** 0.6 → 1.0 → 1.4 m/s, heel-to-toe over the passive toe joint, relaxed joints in swing. ![walking](media/K1_v563_walk.gif) | **Walking with turns** 1.0 m/s at ±0.6 rad/s. ![walking turns](media/K1_v563_walk_turn.gif) |
| **Running** 2.5 → 3.5 → 4.5 m/s, automatic walk → run → walk hand-over. ![running](media/K1_v563_run.gif) | **Running with turns**: ±0.5 rad/s at 3 m/s, then 4.5 m/s with 0.6 rad/s — the speed governor slows down and the body leans into the turn. ![running turns](media/K1_v563_run_turn.gif) |

Full-resolution MP4s: `media/K1_v563_{stop,inplace,walk,walk_turn,run,run_turn}.mp4`.

| capability (simulation) | v5.6.3 |
|---|---|
| walking speed | 0.3 – 1.65 m/s (walk → run hand-over above) |
| running speed | up to 4.5 m/s tested (policy trained to 5.5 m/s command inside a modelled motor torque–speed envelope) |
| turning | yaw rate up to 1 rad/s while walking, running and on the spot; governor v·\|ω\| ≤ 1 m/s² walking, ≤ 2.5 m/s² running |
| leaning into a running turn | 80–85 % of the physical lean atan(vω/g), left and right turns symmetric |
| stopping | quick stop (from in-place turning 1.7 s, from 1.2 m/s walking 3.3 s, from 4.5 m/s running 5.9 s); hard braking 4.5 m/s → standing 3.8 s; re-stance 1 s after standing |
| robustness | full test course (walk, turns, run, governed turns, braking; 54 s): 64/64 robots; with random pushes (±0.3 m/s about every 3 s) 62/64 |
| symmetry | walking and running policies trained with a left/right mirror-symmetry loss |
| standing | no limp joints while standing (stiffness floor), feet do not creep |

## Energy: re-measured against the v1.0 comparison

This project started as an attempt to **reduce the electrical power of walking** (passive toe joints × human-gait
imitation × learned relaxation). The very first result (v1.0) was measured against the public ROBOTIS `walk_default`
policy under identical simulation conditions. The same measurement was repeated for the current policy
(`k1_compare/compare3.py`, same script, same physics, same power model, straight walking at about 0.9 m/s):

![same-condition power comparison: ROBOTIS walk_default, v1.0 eco policy, v5.6.3](media/K1_v1_v563_energy.gif)

| controller (straight walking, ≈ 0.9 m/s) | speed | electrical power (total) | of which legs | CoT (electrical) | CoT vs ① |
|---|---|---|---|---|---|
| ① ROBOTIS public `walk_default` (original flat foot) | 0.901 m/s | 179.1 W | 172.4 W | 0.568 | – |
| ③ v1.0: MP joint + imitation + relaxation (walking only) | 0.917 m/s | 114.1 W | 111.9 W | 0.355 | −37 % |
| ④ **v5.6.3** (current; walking policy of the walk / run / turn system) | 0.888 m/s | 125.5 W | 119.1 W | 0.403 | **−29 %** |

Same comparison with the strict static friction used since v5.6.3 (`COMPARE_NOSLIP=10`): ① 177.8 W (CoT 0.560),
v1.0 117.8 W (0.355, −37 %), v5.6.3 126.3 W (0.408, −27 %).

- This comparison belongs to the **walking-power reduction** part of the project: it compares the electrical cost of
  steady straight walking only. It is not a ranking of the controllers' overall quality — the ROBOTIS policy was
  designed for other goals, and power is not its training target.
- v5.6.3 uses about 10 % more than v1.0: v1.0 could only walk straight; v5.6.3 also turns, runs, stops, recovers
  from pushes, keeps the pelvis near level and is left/right symmetric. It still needs **27–29 % less** electrical
  cost of transport than ① in this simulation.
- Videos: `media/K1_v1_v563_energy_comparison.mp4` (live per-joint power; recorded with the same 0.92 m/s command for
  all three, as the v1.0 video — its summary card gives power ratios); numbers: `k1_compare/results/res_*.json`.

## How it works (short)
1. **Passive spring toe (MP) joints** added to K1 — the foot rolls over the toe without a motor.
2. **Human-gait imitation** — reference motions retargeted from human gait data (BMC) and CMU running captures.
3. **Learned relaxation** — every 20 ms the policy also chooses joint stiffness and damping and is rewarded for low
   electrical power; joints may go fully soft for a moment while walking or running (inertia-driven motion), but not
   while standing.
4. **Gait manager** — walk ⇄ run hand-over, speed governor for turns, lean into turns, hard braking, quick stop and
   re-stance (feet placed with joint-angle forward / inverse kinematics; double support from foot contact sensors).
5. **Left/right mirror symmetry** of both policies; references made symmetric.
6. **Strict static friction** in the simulator (no-slip pass), so feet only move when the robot really steps.

The full development path (v1 → v5.6.3), every decision with its reason, rejected stages and all earlier results:
**[history/README.md](history/README.md)** and the reports in each folder (latest:
[k1_mp_gait56/REPORT_GAIT56.md](k1_mp_gait56/REPORT_GAIT56.md), training commands:
[k1_mp_gait56/TRAINING_HISTORY.md](k1_mp_gait56/TRAINING_HISTORY.md)).

## Limitations
- **Simulation only**; no sim-to-real work has been done. Motor constants, torque limits and the torque–speed
  envelope are modelled / assumed values.
- The re-stance needs foot contact sensors on the real robot (the simulation uses touch sensors under heel, ball and toe).
- After an in-place stop the re-stance can leave the feet 6–8 cm apart front-back.
- The tightest running turn is limited to v·|ω| ≤ 2.5 m/s² (at 3.0 the running policy over-leaned and fell).
- Walking at 0.6–1.0 m/s costs a few per cent more than v1.0/v5.5; turning on the spot costs ~100–120 W because the
  feet are really lifted (earlier policies spun a foot on the floor).

## Power model
At every 2 ms physics step for every joint: `τ = clip(Kp(q* − q) − Kd·q̇, motor limit)`

`P = Σ τ²/Km² + Σ max(τ·q̇, 0)` — copper (Joule) loss + positive mechanical work, no regeneration;
Km = 4.0 Nm/√W (hip, knee, ankle pitch, waist) and 2.2 Nm/√W (ankle roll, arms) are **assumed**.
CoT = P / (m g v), m = 35.7 kg.

## Evaluate / reproduce
```bash
pip install -r requirements.txt            # see history/README.md §Reproduce for the full environment
bash scripts/fetch_external.sh             # pinned upstream data (BMC gait data, ROBOTIS ai_sapiens)
cd k1_mp_gait56
python3 eval_gait56.py runs/final/walk.pt runs/final/run.pt [--push --n 16]   # end-to-end course
python3 eval_power56.py runs/final/walk.pt runs/final/run.pt                  # walking / turning / running power
python3 eval_inplace56.py runs/final/walk.pt runs/final/run.pt                # in-place turning
python3 eval_runturn56.py runs/final/walk.pt runs/final/run.pt                # lean in running turns
MUJOCO_GL=osmesa python3 video_gait56.py runs/final/walk.pt runs/final/run.pt v_run_turn out.mp4
cd ../k1_compare && python3 compare3.py robotis && python3 compare3.py eco && python3 compare3.py v563
```
Policies: `k1_mp_gait56/runs/final/walk.pt`, `k1_mp_gait56/runs/final/run.pt`.

## Repository layout (main parts)
| path | content |
|---|---|
| `k1_mp_gait56/` | **current version** (v5.6 – v5.6.3): environments, training, gait manager `gait56.py`, evaluations, report |
| `k1_compare/` | same-condition power comparison with the ROBOTIS public policy |
| `ai_sapiens/` | K1 model (ROBOTIS, Apache-2.0) + K1 with MP joints |
| `k1_mp*/` | earlier stages (v1 – v5.5), see [history/README.md](history/README.md) |
| `history/` | development history README (English / Japanese) |
| `media/` | videos and GIFs |

## Authorship
**Idea and direction — Takeyuki-K** (concept: passive spring MP toe joints × imitation of human walking × relaxation of
the actuators for an energy-efficient humanoid; all specifications and evaluation criteria — full list in
[history/README.md](history/README.md#authorship)).
**Implementation — generated with Claude (Anthropic)** under the direction, selection, testing and integration of
Takeyuki-K. This describes how the work was made; it is not a statement on copyright ownership of AI-generated
content, which differs between jurisdictions.

## Use this idea
You are free to use, modify and build on this work (code: Apache-2.0; data files keep their own licences, see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)).
**If this work or its idea inspired yours, please credit `Takeyuki-K` and link this repository** (GitHub "Cite this repository" uses [CITATION.cff](CITATION.cff)).
Archived on Zenodo: **DOI [10.5281/zenodo.23178357](https://doi.org/10.5281/zenodo.23178357)** — please cite this DOI.

**No patents, open for everyone.** The author does not intend to patent this idea. It is published so that anyone can
use it, find its problems and improve it. The public, dated release (Zenodo DOI) also serves as a defensive publication.
Redistributions must keep the [NOTICE](NOTICE) file (Apache-2.0 §4(d)).

## Credits
- **Robot model**: ROBOTIS AI Sapiens K1, [ROBOTIS-GIT/ai_sapiens](https://github.com/ROBOTIS-GIT/ai_sapiens) (Apache-2.0), commit `bdc40f1` — modified (passive spring MP toe joints added), see [ai_sapiens/NOTICE_MODIFICATIONS.md](ai_sapiens/NOTICE_MODIFICATIONS.md).
- **Human gait data**: Marcos Duarte and Renato Naville Watanabe, "Notes on Scientific Computing for Biomechanics and Motor Control" (BMC), [BMClab/BMC](https://github.com/BMClab/BMC), DOI [10.5281/zenodo.4599319](https://doi.org/10.5281/zenodo.4599319), commit `50a05ae`, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — retargeted to the robot; the derived reference files stay under CC BY 4.0 (list in [NOTICE](NOTICE)).
- **Running motion data**: CMU Graphics Lab Motion Capture Database, [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu) (subject 16 trial 35, subject 9 trial 4), BVH conversion by B. Hahne — free for research and commercial use. *The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.*
- **In-place turning rhythm**: extracted from a video provided by the author (monocular pose estimate); only the derived rhythm
  (`k1_mp_gait56/mocap/turn_rhythm.json`) is included, not the recording.
- **Software used** (not redistributed): see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

"ROBOTIS" and "AI Sapiens" may be trademarks of their respective owner; they are used here only to identify the robot model.
