# Licences and third-party material

This repository is **not** under a single licence. Summary:

| what | licence | where |
|---|---|---|
| Project code (`*.py`, `*.sh`, CI, Dockerfile), reports, figures, videos, trained policies (`*.pt`, `*.onnx`), hand-over state banks (`*_bank.npz`), result files | **Apache-2.0** | [LICENSE](LICENSE), [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) |
| K1 robot model and meshes in `ai_sapiens/` (original files unmodified, derived files marked) | **Apache-2.0** (ROBOTIS) | [ai_sapiens/LICENSE](ai_sapiens/LICENSE), [ai_sapiens/NOTICE_MODIFICATIONS.md](ai_sapiens/NOTICE_MODIFICATIONS.md) |
| Human-gait-derived reference trajectories (`ref_gait.npz`, `ref_lib.npz`, `ref_*.csv`, `ref_joint_table.csv` in the walking folders) | **CC BY 4.0** (Marcos Duarte and Renato Naville Watanabe, BMC) | [LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt), [NOTICE](NOTICE) item 2 |
| CMU motion-capture files (`*/data/cmu/`) and running references derived from them (`ref_run.npz`, `ref_sprint_lib.npz`) | free for research and commercial use, no restrictions (acknowledgement requested) | `*/data/cmu/READMEFIRST.txt`, [NOTICE](NOTICE) item 3 |

The exact file lists are in [NOTICE](NOTICE). The trained policies were trained against the CC BY 4.0 / CMU-derived
reference motions; the reference data themselves keep their own licences when redistributed.

## Upstream sources (pinned)
| source | used for | commit | how it is obtained |
|---|---|---|---|
| [ROBOTIS-GIT/ai_sapiens](https://github.com/ROBOTIS-GIT/ai_sapiens) (Apache-2.0) | K1 model (copied into `ai_sapiens/`), ROBOTIS `walk_default` policy for the comparison (not redistributed) | `bdc40f126c4a1551422dce44dce144974f7a2f2b` | `scripts/fetch_external.sh` → `external/ai_sapiens` |
| [BMClab/BMC](https://github.com/BMClab/BMC) (non-software content CC BY 4.0; code MIT), DOI [10.5281/zenodo.4599319](https://doi.org/10.5281/zenodo.4599319) | `data/walk2.trc` (not redistributed) | `50a05aebcc814e5341860f1b270652e3ea590d20` | `scripts/fetch_external.sh` → `k1_mp/data/bmc` |
| [una-dinosauria/cmu-mocap](https://github.com/una-dinosauria/cmu-mocap) (CMU database, BVH conversion by B. Hahne) | `16_35.bvh`, `09_04.bvh` (included) | `09a07f54f3bbb58797325f009282d0b2048a2871` | included in the repository |

## Software used (not redistributed)
MuJoCo (Apache-2.0), PyTorch (BSD-style), NumPy / SciPy / NetworkX / Shapely (BSD), trimesh / Rtree / PyYAML /
ONNX Runtime (MIT), ONNX / OpenCV (Apache-2.0), imageio (BSD-2), matplotlib (PSF-based), Pillow (MIT-CMU),
Noto Sans CJK font (SIL OFL 1.1, used for video captions), FFmpeg (encoding tool), OSMesa / Mesa (MIT, rendering).

## AI-assisted implementation
The implementation was generated with Claude (Anthropic) under the direction, selection, testing and integration of
Takeyuki-K (see README, "Authorship"). This statement describes how the work was made; it is not a statement about
copyright ownership, whose treatment for AI-generated content differs between jurisdictions.
