#!/usr/bin/env bash
# Fetch third-party material that is NOT redistributed in this repository.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# 1) Human gait data (CC BY 4.0) - needed only to re-run k1_mp/retarget.py
if [ ! -d "$ROOT/k1_mp/data/bmc" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/duartexyz/BMC.git "$ROOT/k1_mp/data/bmc"
  (cd "$ROOT/k1_mp/data/bmc" && git sparse-checkout set data)
fi
# 2) ROBOTIS ai_sapiens upstream (Apache-2.0) - needed only for the comparison with the
#    public walk_default policy (k1_compare/), which is NOT included here.
if [ ! -d "$ROOT/external/ai_sapiens" ]; then
  git clone --depth 1 https://github.com/ROBOTIS-GIT/ai_sapiens.git "$ROOT/external/ai_sapiens"
fi
echo "done"
