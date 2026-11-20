#!/bin/bash
# F1 = adopted walking setting: walk_Qup075.pt with the passive ankle-roll spring K=20 Nm/rad, D=0.3 Nms/rad (zero-shot),
# relaxed arms (ARM_KPM=0.25) and the 11 cm stance reference. Extra arguments are passed to eval_spring.py.
cd "$(dirname "$0")"
ANKLE_SPRING=1 ANK_K=20 ANK_D=0.3 ARM_KPM=0.25 REF_LIB=ref_lib_56_w11_sym.npz REF_W=0.11 \
python3 eval_spring.py runs/final_spring/walk_Qup075.pt runs/final_spring/run_K10_D0.6.pt --n 32 --meas 10 --straight "$@"
