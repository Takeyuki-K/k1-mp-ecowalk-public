#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
# End-to-end smoke test (CPU, ~5 min; --full adds the running references, ~30 min).
# Checks that the code runs from start to end, NOT that training reaches the published performance.
#   1. pinned upstream data (scripts/fetch_external.sh)
#   2. regenerate the K1+MP model and the reference motions in a scratch copy and compare them with the
#      committed files (scripts/generated_files.sha256)
#   3. MP spring test, eco-env equivalence test
#   4. two-iteration training runs of the first stages (fixed MP, eco, speed)
#   5. load every released checkpoint and simulate 3 s (scripts/smoke_eval.py)
# usage: scripts/smoke_test.sh [--full] [--strict]   (--strict: checksum mismatch = failure)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FULL=0; STRICT=0
for a in "$@"; do [ "$a" = "--full" ] && FULL=1; [ "$a" = "--strict" ] && STRICT=1; done
# MUJOCO_GL is only needed for rendering (video scripts set it themselves). Exporting MUJOCO_GL=osmesa before
# training was observed to crash CUDA builds of PyTorch (triton import), so it is not set here.
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
step() { echo; echo "=== $*"; }

step "versions"
python3 -c "import sys, mujoco, torch, numpy; print('python', sys.version.split()[0], '| mujoco', mujoco.__version__, '| torch', torch.__version__, '| numpy', numpy.__version__)"

step "fetch pinned upstream data"
"$ROOT/scripts/fetch_external.sh"

step "regenerate model and references in a scratch copy"
mkdir -p "$TMP/repo"
(cd "$ROOT" && tar --exclude=.git --exclude=media --exclude=external -cf - .) | (cd "$TMP/repo" && tar -xf -)
cd "$TMP/repo/k1_mp" && python3 gen_model.py > /dev/null && python3 retarget.py > /dev/null
cd "$TMP/repo/k1_mp_speed" && python3 retarget_speed.py > /dev/null
cd "$TMP/repo/k1_mp_turn" && python3 retarget_speed.py > /dev/null
cd "$TMP/repo/k1_mp_fastwalk" && python3 retarget_speed.py > /dev/null
cd "$TMP/repo/k1_mp_run" && python3 retarget_run.py > /dev/null
if [ $FULL = 1 ]; then
  cd "$TMP/repo/k1_mp_sprint" && python3 retarget_sprint.py > /dev/null
  cd "$TMP/repo/k1_mp_gait" && python3 retarget_sprint.py > /dev/null
  SUMS="$(cat "$ROOT/scripts/generated_files.sha256")"
else
  SUMS="$(grep -v ref_sprint_lib "$ROOT/scripts/generated_files.sha256")"
fi
cd "$TMP/repo"
if echo "$SUMS" | sha256sum -c --quiet; then
  echo "regenerated files are byte-identical to the committed ones"
else
  echo "WARNING: some regenerated files differ (floating-point / library-version differences are possible)"
  [ $STRICT = 1 ] && exit 1
fi

step "MP spring test / eco equivalence test"
cd "$TMP/repo/k1_mp" && python3 test_mp.py | tail -2
cd "$TMP/repo/k1_mp_eco" && python3 test_eco_equiv.py | tail -2

step "two-iteration training runs"
cd "$TMP/repo/k1_mp" && python3 ppo.py --stage 1 --iters 2 --n 16 --out "$TMP/t_fixed" | tail -1
cd "$TMP/repo/k1_mp_eco" && python3 ppo_eco.py --stage 2 --iters 2 --n 16 --init ../k1_mp/runs/final/model.pt --from_fixed --out "$TMP/t_eco" | tail -1
cd "$TMP/repo/k1_mp_speed" && python3 ppo_speed.py --iters 2 --n 16 --init ../k1_mp_eco/runs/eco1/model.pt --from_eco_full --out "$TMP/t_speed" | tail -1

step "released checkpoints"
python3 "$ROOT/scripts/smoke_eval.py" --steps 150
echo; echo "smoke test passed"
