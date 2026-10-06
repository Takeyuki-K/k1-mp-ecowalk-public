#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
# Fetch third-party material that is NOT redistributed in this repository, at the exact upstream commits that were
# used to produce the published results (pinned for reproducibility).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Human gait data: Marcos Duarte & Renato Naville Watanabe, "Notes on Scientific Computing for Biomechanics and
# Motor Control" (BMC), https://github.com/BMClab/BMC, DOI 10.5281/zenodo.4599319. Non-software content: CC BY 4.0.
# Only data/walk2.trc is used (k1_mp/retarget.py, k1_mp_*/retarget_speed.py).
BMC_URL="https://github.com/BMClab/BMC.git"
BMC_COMMIT="50a05aebcc814e5341860f1b270652e3ea590d20"

# ROBOTIS ai_sapiens (Apache-2.0): only needed for the comparison with the public walk_default policy (k1_compare/,
# k1_mp_speed/robotis_sweep.py). The policy is NOT redistributed here. The robot model files in ai_sapiens/ of this
# repository were taken from the same commit.
ROBOTIS_URL="https://github.com/ROBOTIS-GIT/ai_sapiens.git"
ROBOTIS_COMMIT="bdc40f126c4a1551422dce44dce144974f7a2f2b"

fetch_pinned() {   # url commit dir [sparse paths...]
  local url=$1 commit=$2 dir=$3; shift 3
  if [ -d "$dir/.git" ]; then
    echo "exists: $dir ($(git -C "$dir" rev-parse HEAD))"; return
  fi
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$url"
  if [ $# -gt 0 ]; then
    git -C "$dir" sparse-checkout set --no-cone "$@"
  fi
  git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$commit"
  git -C "$dir" checkout -q FETCH_HEAD
  echo "fetched $url @ $(git -C "$dir" rev-parse HEAD)"
}

fetch_pinned "$BMC_URL" "$BMC_COMMIT" "$ROOT/k1_mp/data/bmc" /data/walk2.trc
fetch_pinned "$ROBOTIS_URL" "$ROBOTIS_COMMIT" "$ROOT/external/ai_sapiens"
echo "done"
