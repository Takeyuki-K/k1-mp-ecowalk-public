# K1 ankle-roll passive spring variant (local only, nothing pushed)
Code: /workspace/k1-mp-ecowalk-public/k1_mp_gait56_ankle_spring (untracked copy of k1_mp_gait56; v5.6.3 dir untouched)
  ANKLE_SPRING=1 ANK_K ANK_D ANK_ARM (k1env.py): ankle-roll motor Kp=Kd=0, joint stiffness/damping, armature 0.002
  W_UPRIGHT/UP_K/UP_ROLL_DB (walk4, run4), RUN_LEAN_SCALE (run4), W_UNDER/SPD_DB/SPD_SYM (speed-tracking guard), W_HIPROLL (walk4)
Baseline: baseline/spring_eval_v563.json (eval_spring.py, ANKLE_SPRING=0), baseline/power_v563.json (eval_power56.py)
Best: best/walk_K10_D0.6_it1000.pt (= train/walk_K10/model_1000.pt), best/run_K10_D0.6_it1400.pt (= train/run_K10/model_1400.pt)
Full eval: evals/full/K10_w1000_r1400.json ; compare: python3 fullcmp.py evals/full/K10_w1000_r1400.json
Quick evals: python3 summarize.py ; training progress: python3 train/progress.py ; per-run cmd/env in train/<tag>/{cmd,env}.txt
Video: video/compare_v563_vs_spring_K10.mp4 (top v5.6.3, bottom spring; walk policy = walk_K10e it200, run = run_K10 it1400)

## Stage I (2026-11-19 15:2x JST): walking, inertia / step-to-step transition (running frozen: best/run_K10_D0.6_it1400.pt)
Gait diagnostics: diag_gait.py -> baseline/gait_v563_1.0.json, evals/gait_best_1.0.json (walk 1.0 m/s, 8 robots, 6 s at 200 Hz)
New reward options (k1env_walk4.py): W_NEGW (joint negative mechanical power), W_SWING (swing-leg hip-pitch+knee copper + positive work)
Runs (init best/walk_K10_D0.6_it1000.pt, w_energy 0.0025, W_UNDER 1.5 / SPD_DB 0.02 two-sided, W_UPRIGHT 0.05, UP_ROLL_DB 0.045):
  walkI_neg (W_NEGW 0.0015), walkI_swing (W_SWING 0.005), walkI_both (both), walkI_neg2 (W_NEGW 0.003)
Stopped (not improving): walk_K10h, walk_K20h (W_HIPROLL 0.002), walk_K10, walk_K10e, walk_K10g (under-tracked), run_K10f, run_K20
walkI_both stopped 15:40 JST (it500 +10% P, worsening); walkI_neg stopped earlier.

## Stage J (2026-11-19 15:41 JST)
Rocker research summary: heel rocker (heel strike, controlled plantarflexion = collision absorption), ankle rocker (tibia rolls over
the planted foot, inverted-pendulum vault, ideally near-passive), forefoot rocker (heel rise, ankle push-off = main positive work,
best timed just before contralateral heel strike to cut collision loss), toe rocker (MP extension -> K1 passive toe spring);
swing leg should be a mostly passive pendulum. K1 already: heel-first touchdown, smooth CoP, pre-emptive push-off. Losses: hip-roll
copper (no ankle-roll motor now), knee negative work at loading response, hip motor actively driving the swing.
New options:
  W_LAND (k1env_walk4.py): -W_LAND*clip(peak foot force/mg - 1.05, 0)  (landing impact; v5.6.3 already has 0.15 above 1.3 mg)
  HR_Q_SCALE, ANK_Y_SCALE (k1env_walk3.py reward_turn): weight of hip-roll joint imitation error and lateral ankle-position
  imitation error (1 = v5.6.3) -> step-width freedom
Runs (same base as stage I): walkJ_land (W_SWING 0.005 + W_LAND 0.3), walkJ_width (HR_Q_SCALE 0.3, ANK_Y_SCALE 0.3)
Precise eval: precise.sh (32 robots x seeds 3,11, 10 s window, straight walk 0.6/1.0/1.4) -> evals/precise, precise_sum.py.
  Finding: per-robot spread is <0.3% (deterministic policy converges to one limit cycle); the +-3-5% scatter between
  checkpoints is real policy-to-policy difference, not measurement noise -> pick checkpoints on all three speeds.
walkJ_land stopped (worse at every speed). walkJ_straight: W_SWING 0.005 + WALK_MIX=straight (50% straight walking commands; k1env_walk3.py)
Results 16:0x JST (precise eval, dP@v = power vs v5.6.3 interpolated at the same actual speed):
  best_w1000: 0.6 -0.9 / 1.0 +1.8 / 1.4 +10.9 %   I_swing_it1400: -2.1 / +0.0 / +12.0 % (tilt 3.34/3.34/3.10)
  J_width_it800: +3.1 / +4.0 / +9.2 %, tilt 2.52/3.32/3.52 (better) but hip-roll copper +4.7 W at 1.0 -> width freedom is spent
  on uprightness, not on hip-roll unloading (hip-roll torque is set by upper-body COM vs stance hip, not by step width).
  J_land: worse everywhere (stopped). I_neg2_it1400: +4.5/+4.0/+13.9 (no).
  1.4 m/s gap (best vs v5.6.3, diag_power): hip-roll copper +7 W, knee copper +6 W, hip-pitch +2.7 W, ankle-roll motor saving -7.8 W.

## Stage K (2026-11-19 16:08 JST): attack the 1.4 m/s gap (init best/walk_cand_Iswing_it1400.pt, W_SWING 0.005, same base)
  W_HRSYM (k1env_walk4.py): |EMA(left hip-roll copper) - EMA(right)| (EMA ~1.5 s)  -> walkK_hrsym (0.01)
  W_KNEEHI: knee copper loss x ramp(v_ref 0.9->1.4)                                 -> walkK_knee (0.004)
Stage K results: K_hrsym (it200-400: +2.7..+6.5 % @1.0, +14..+17 % @1.4) and K_knee (it300: +5.1 / +14.2 %) worse -> stopped.
  J_straight (it100-500: +12..+13 % @1.4) no help. J_width it1000: tilt 2.4-2.9 deg (best) but +8..+14 % power.
  I_swing it1400 full eval: all 12 cases survive, in-place turn 90.0 W vs 100.4 (v5.6.3), run unchanged, profile survival 1.0.
1.4 m/s diag_gait (v5.6.3 vs I_swing it1400): trailing-ankle push-off before opposite touchdown 8.6 vs 5.0 J;
  touchdowns heel 120 / fore 0 vs heel 121 / fore 110 (one forefoot re-contact per step); copper 88.8 vs 104.5 W
  (right hip roll 10.9 -> 18.1, right knee 6.0 -> 12.3); hip-pitch positive work 29.7 -> 34.1 W.
Stage K2: W_PUSH (late-stance ankle positive power bonus) -> walkK_push 0.002; W_SCUFF (loaded foot in reference swing) -> walkK_scuff 0.1
walkK_scuff, walkJ_straight stopped. Bounce found: fore 'touchdowns' have 5-10 ms air time after heel strike -> W_BOUNCE (unloading during reference stance): walkK_bounce 0.5, walkK_pushbounce (+W_PUSH 0.002)
16:35 JST: new candidate best/walk_cand_Kpushbounce_it200.pt (precise dP@v -0.4 / -1.3 / +8.5 % at 0.6/1.0/1.4; tilt 3.55/3.73/3.38).
  Full eval: 12/12 cases survive, profile survival 1.0, walk turns 134.2/136.9 W (v5.6.3 138.9/133.5 at lower speed),
  in-place 75.3 W (v5.6.3 100.4) but drifts 0.19 m/s (v5.6.3 0.13); ankle roll max 14.7 deg in walk turns.
  1.4 diag: copper 97.5 W (I_swing 104.5, v5.6.3 88.8); 'fore' re-contacts 195 at 0.91 mg peak (light chatter, not the cause).
  PPO fine-tunes oscillate +-5 % between checkpoints -> keep precise-evaluating every 100 it.

## Stage L (2026-11-19 16:4x JST): init best/walk_cand_Kpushbounce_it200.pt, base = K_pushbounce (W_SWING .005, W_PUSH .002, W_BOUNCE .5)
  walkL_up: W_UPRIGHT 0.15 (from 0.05), HR_Q_SCALE 0.6, ANK_Y_SCALE 0.6 (half-way to J_width)
  walkL_pushhi: W_PUSHHI 0.004 (extra push-off bonus ramping in over v_ref 0.9 -> 1.4)   walkK_push stopped.
16:57 JST: K_bounce/K_pushbounce finished. K_bounce it800 (+3.6/-0.4/+5.7 %, tilt 2.96/3.31/3.28) -> best/walk_cand_Kbounce_it800.pt.
  L_pushhi worse (+13..+15 % @1.4) -> stopped. L_up it300: tilt 2.83/2.88/2.56 but +9.5/+4.7/+12.0 % -> upright costs power.
  New: walkL_up2 (Kpb200 init, W_UPRIGHT 0.10, normal tracking weights), walkL_bup (Kbounce800 init, W_UPRIGHT 0.10, W_PUSH 0)
17:1x JST: best balanced candidate best/walk_cand_Lbup_it300.pt (K_bounce it800 + W_UPRIGHT 0.10, no push bonus):
  precise dP@v -1.0 / +0.8 / +6.9 % at 0.6/1.0/1.4, tilt 2.83/3.17/2.95 (v5.6.3 3.16/3.21/3.24 -> more upright at all three).
  Full eval 12/12 + profile survive; walk turns 141.7/139.3 W at 1.00/0.99 m/s (v5.6.3 138.9/133.5 at 0.93/0.97);
  in-place 72.7 W (v5.6.3 100.4) with 0.17 m/s creep; running unchanged.
  L_up (W_UPRIGHT .15 + HR_Q/ANK_Y .6) and L_up2 (W_UPRIGHT .10 from Kpb200): tilt 2.4-2.9 but +4..+17 % -> stopped.
  Pareto (power vs tilt): Kpb200 (lowest power, tilt worse), Lbup300 (balanced), J_width/L_up (most upright, +8..+14 %).

## Stage M (2026-11-19 18:53 JST): metapen recipe for lateral pelvis sway (user request)
  ref: make_ref56.py REF_W=0.11 REF_OUT=ref_lib_56_w11.npz + make_sym_ref.py -> ref_lib_56_w11_sym.npz (LIPM lateral travel
       re-computed for the 11 cm base); env REF_LIB/REF_W select it (eval_ckpt.sh / stop_ckpt.sh pass them through!)
  rewards (k1env_walk4.py): W_LEVEL env (0.35 default), ROLL_DEADBAND 0, UP_ROLL_DB 0, W_FOOTPL (Gaussian on |stance ankle y|
       - 0.055, sigma 1.5 cm, touchdown + single support), W_KPHR (hip-roll kp_scale > 0.75; the policy outputs kp/kd scales),
       W_HIPROLL (hip-roll copper), W_ROLLRATE (pelvis roll rate^2), W_ARMROLL (shoulder roll - default pose)^2
  eval: eval_spring.py now also roll_p2p, arm_roll_p2p (shoulder roll p2p), hiproll_kp; eval_stop_spring.py (stop / brake / brake+push
       from walk 0.6-1.65) via stop_ckpt.sh
  a: W_LEVEL .7 FOOTPL .3 ROLLRATE .5 ARMROLL 1, w_energy .0025; a2: LEVEL 1, FOOTPL .5, ROLLRATE 1; b: as a, w_energy .0005
  -> roll rms 3.0 -> 0.8-1.1 within 300 it, power +12..+19 %, M_a under-tracks at it400 (0.947 @1.0) -> stopped; arms unchanged (ARMROLL too weak)
  c (power restored): c1 from M_a2 it300, c2 from M_b it200: w_energy .004, W_UNDER 3.0, W_ARMROLL 5
  19:17 JST: c1 it400 -> best/walk_cand_Mc1_it400.pt: roll rms 0.59/1.01/1.12 deg, p2p 1.9/2.9/2.9, arm p2p 6.4/8.6/11.3 deg,
    tilt 0.78/1.29/1.63, power +16/+15/+18 % at matched speed. Full eval 12/12 + profile OK; walk turns 157/161 W (+13..+20 %);
    in-place 114.5 W (+14 %). Stop / brake from walk 0.6-1.65: 100 %. Brake + random pushes: 0.875/0.875/0.906/0.844
    (v5.6.3 0.93/0.93/0.93/0.97, Lbup300 0.94/0.94/0.94/0.88) -> metapen's "sudden stop weakens" seen with pushes.
  d1/d2: power-restoration from Mc1_it400 with w_energy .007 (d2 also W_LEVEL .6, ROLLRATE .5); e: P_BRAKE_WALK 0.5, w_energy .005
  19:29 JST: d1/d2 (w_energy .007): power falls only by slowing down (d2 it200 0.922 m/s @1.0, d1 it300 0.939) even with
    W_UNDER 3 -> stopped (violates no-slowing rule). Matched-speed power stays +11..+16 %: the level pelvis costs power here.
    walkM_e (brake share 0.5) still running; precise_queue_m.sh evaluates walkM_* automatically.

## Stage N (2026-11-19 21:1x JST): hold M_c1 posture, cut power (user decision)
  diag_power @1.0: hip-roll copper v5.6.3 31.2 W, Lbup300 34.8 W, Mc1_400 56.5 W -> the whole +26 W gap is hip roll
  (level pelvis => upper-body COM stays over the midline, lever to the stance hip ~half hip width; v5.6.3 leaned 3 deg over
  the stance leg). Hip pitch mech+ also +5 W @1.0, +10 W @1.4. Arms not the issue.
  walkM_e (P_BRAKE_WALK 0.5): posture regressed (tilt 2.3 @1.0), pushed-brake survival noisy (0.84-0.94) -> not folded in as a
  checkpoint; brake share 0.4 kept in stage N.
  Runs (init Mc1_400; W_UNDER 5, SPD_DB 0.015, P_BRAKE_WALK 0.4):
    N_en: w_energy .005 (posture terms fixed)
    N_db: roll free band 1 deg (ROLL_DEADBAND = UP_ROLL_DB = 0.017), ROLLRATE .5, w_energy .004
    N_fy45: N_db + FOOT_Y 0.045 (swing leg closer to the midline -> shorter hip-roll lever)
    N_hr: N_db + W_HIPROLL .003
  21:2x JST: N_db / N_en it300-400: power only -1..-4 % with posture worse (roll rms up to 1.4-1.5, tilt 1.9-2.5) -> stopped.
  diag_hrphase.py @1.0 (hip-roll torque by phase): v5.6.3 single support stance hip 21-24 Nm, double support +-8 Nm (19 W);
    Mc1 single support 30-31 Nm, double support +-10 Nm opposing (squeeze, 39.5 W).
  New W_SQZ (k1env_walk4.py): min(|tauL|,|tauR|) when hip-roll torques oppose in double support.
    walkN_sqz (W_SQZ .02, posture terms fixed), walkN_sqzfy (+ FOOT_Y .045)
  21:3x JST: stage N result - no candidate keeps the Mc1 posture with less power. N_sqz it200 +12.4/+12.1/+16.3 % but roll rms
    0.99/1.35/1.43 (>1.2). N_fy45, N_hr, N_sqzfy: worse. Every fine-tune from Mc1 drifts the posture (roll 1.2-1.7) while
    power drops only 2-4 %. Geometry: hip-roll joint 12.5 cm lateral of the pelvis centre, leg 8.4 kg, upper body ~19 kg ->
    with a level pelvis the stance hip must hold ~30 Nm (v5.6.3 21-24 Nm by leaning ~3 deg); hip-roll copper 56 vs 31 W @1.0
    is the whole gap. Reward shaping cannot remove that lever. Best remains best/walk_cand_Mc1_it400.pt.

## Stage O (2026-11-19 23:3x JST): arm relaxation from M_c1 it400 (user: 腕の脱力, M_c1基準, no collisions)
  Code: k1env_walk4.py ARM_KPM/ARM_KDM (arm servo gain scale, walking model only), ARM_KP_ACT (policy arm stiffness via
  unused ankle-roll Kp slots 17/23, sample-and-hold), ARM_SWING (ref arm swing scale), W_ARMP (extra arm W price),
  W_ARMCLR/ARMCLR_M (one-sided shoulder adduction penalty). diag_arms.py (arm power per joint, Lz, yaw moment, mesh-hull
  clearance arm vs torso/pelvis/thigh). train/launch_O.sh, evalO.sh. eval_ckpt.sh/stop_ckpt.sh pass ARM_* through.
  Baseline arm power (W, 0.6/1.0/1.4): v563 4.2/5.7/8.9, Mc1 2.9/5.4/11.1 -> upper bound of savings ~3-5 %.
  Clearance (hull, min hand-hip): v563 -3.8/-5.2/-5.7 cm (overlap), Mc1 +3.7/+2.8/+2.3, Ok25 +4.5/+3.8/+2.8 cm.
  Precise dP@v vs v563 (0.6/1.0/1.4): Mc1 +16.0/+14.7/+17.6; zero-shot Mc1 kp x0.25 +15.3/+13.6/+16.3;
    O_k25 it400 (ARM_KPM .25, W_ARMP .004) +13.0/+10.8/+12.7, roll rms .82/.99/.98, tilt .91/1.30/1.47 -> best/walk_cand_Ok25_it400.pt (needs ARM_KPM=0.25)
    O_act it400 +13.8/+11.5/+11.6 (tilt .6 1.41, roll 1.4 1.25: fails posture), O_k50 400 +16.7/+14.9/+16.4, O_k25sw 200 tilt 2.0-2.9 fail,
    O_ctrl (same rewards, gain x1) it400 +11.7/+14.8/+25.8 tilt 1.4-2.3 -> gain from arms, not the reward set.
  Ok25 full 12 cases + profile all survive; pushed hard brake 90.0/86.2/93.8/90.6 % but 2-3 falls while pushed walking at 0.6/1.0 (Mc1 0).
  Video: video/compare_arm_relax_front.mp4 (M_c1 | O_k25).

## Stage P/Q (2026-11-20 00:0x JST): push robustness of O_k25 (ARM_KPM .25, 11 cm ref)
  New: eval_pushwalk.py + pushwalk.sh (64 robots x seeds 11,12, walk 10 s under eval pushes). PUSH_V env (k1env_walk3,
  training push size), launch.sh PUSH_P overridable, launch_O.sh INIT / W_UPRIGHT overridable, stop_ckpt.sh absolute paths + N.
  Pushed-walk falls /128 (0.6/1.0/1.4): v563 3/2/1, Mc1 2/1/2, Ok25 4/3/1, Pk25p2_200 1/0/0, Qup075_100 1/0/0.
  Pushed hard brake n64 survival-after-stop: v563 96.7/98.4/96.7/98.4 (3 falls before stop each), Mc1 92.1/96.8/96.8/96.8,
    Ok25 96.9/95.3/98.4/95.2, Pk25p2_200 96.8/96.8/98.4/95.2, Qup075_100 98.4/96.9/100/98.4 (0 falls before stop).
  Precise dP@v: P_k25_p2 it200 (PUSH_P .02 V .35) +13.3/+10.8/+12.4 tilt 1.16/1.35/1.69 roll .99/1.04/1.00
    Q_up075 it100 (from Pk25p2_200, W_UPRIGHT .075) +12.9/+7.9/+9.5 tilt 1.18/1.20/1.57 roll .80/.94/.92 -> best/walk_cand_Qup075_it100.pt
    (only miss: tilt at 0.6 = Mc1 +0.40). p2v/p3/k20 runs worse power or posture.
  Qup075_100: full 12 + profile all survive, arm 1.8/2.7/4.4 W, min arm clearance >= 3.9 cm.

## Phases 2-4 (2026-11-20 JST): summary (experimental code not included in this branch)
Precise dP@v vs v5.6.3 at the same actual speed (0.6/1.0/1.4 m/s); "vs F1" in points.
- Phase 2 B, ankle-roll spring sweep on the Q_up075 checkpoint (zero-shot): power falls monotonically with K up to K20; D 0.3-1.0 matters little.
  **F1 = K20 D0.3: +12.1/+6.7/+8.7 %**, v 0.628/1.044/1.387, tilt 1.20/1.17/1.62, roll .82/.93/.93, pushed-walk falls 0/0/0,
  pushed hard brake 98.4/96.9/100/98.4 %, 12 cases + profile all survive, run policy -0.4..-0.9 % power -> adopted (eval_F1.sh).
  Hip-roll kp: nominal is best (higher -> pelvis drifts, lower -> more power or falls). Hip-roll reference x0.75 + stance +0.02 rad:
  +9.9/+5.6/+8.8 % but mostly from walking 2-3 % faster; absolute power about equal.
- Phase 2 A, waist roll axis (+0.7 kg QC060-class actuator + bracket): +20.9/+15.6/+18.9 % (same mass, axis locked: +17.4/+12.0/+12.7 %).
  Better torso posture (turns 2.4 -> ~1 deg) but no energy benefit.
- Phase 3, clutched parallel ankle-pitch spring (Collins-type): best physically valid version (engage at foot-flat, K40, 0.1 kg near the knee)
  -3.7/-0.5/+1.7 vs F1; 0.2 kg is a net loss. Spring cycles only 0.1-0.7 J/stride vs 17-24 J ankle-motor work (K1 pushes off above
  the engagement angle). A fixed-angle clutch model that engaged pre-loaded was found non-physical and discarded.
- Phase 4, allow pelvis roll up to 1.5 deg (fine-tune from F1): best -2.5/+0.8/-1.4 vs F1 with roll 1.52 deg and tilt at the limit;
  strict-pass version +0.1/-0.6/-1.1 (within fine-tune noise). Policies leaned over the stance leg; stance hip-roll torque -1..-2 Nm of ~30.
- Conclusion: v5.6.3's saving comes from relaxed hip-roll holding (leaning ~3 deg over the stance leg), not from the rocker function.
  Posture was prioritised for camera / IMU on hardware; F1 is the walking setting.
