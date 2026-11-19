"""Summarise a sweep dir: one row per (K, D)."""
import json, sys, glob, os, re
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, 'diag_K*_D*.jsonl'))):
    K, D = re.findall(r'K(\d+(?:\.\d+)?)_D(\d+(?:\.\d+)?)', f)[0]
    ds = [json.loads(l) for l in open(f) if l.startswith('{')]
    ef = os.path.join(d, f'eval_K{K}_D{D}.jsonl')
    ev = [json.loads(l) for l in open(ef) if l.startswith('{')] if os.path.exists(ef) else []
    if len(ds) < 2:
        continue
    a, b = ds[0], ds[1]
    surv = '/'.join(f"{e['survival']:.2f}".rstrip('0').rstrip('.') for e in ev) if ev else '-'
    minsurv = min([e['survival'] for e in ev]) if ev else float('nan')
    pf = os.path.join(d, f'push_K{K}_D{D}.jsonl')
    pv = [json.loads(l) for l in open(pf) if l.startswith('{')] if os.path.exists(pf) else []
    surv += '  push ' + ('/'.join(f"{e['survival']:.3g}" for e in pv) if pv else '-')
    rows.append((float(K), float(D), a, b, surv, minsurv))
rows.sort()
hdr = 'K     D    | v1.0  J/m1.0 J/m1.4 | HRtau HPtau HRcu HPcu | roll  p99 | ank  | GRFpk p95  rate | survival(course/ip/stop/brake)'
print(hdr)
for K, D, a, b, surv, ms in rows:
    print(f"{K:<5g} {D:<4g} | {a['v_meas']:.2f}  {a['J_per_m']:6.1f} {b['J_per_m']:6.1f} | {a['hiproll_tau_rms']:5.1f} {a['hippitch_tau_rms']:5.1f} {a['hiproll_cu_W']:4.1f} {a['hippitch_cu_W']:4.1f} | "
          f"{a['roll_rms_deg']:.2f} {a['roll_p99_deg']:.2f} | {a['ankroll_rms_deg']:4.1f} | {a['grf_peak_bw']:.3f} {a['grf_peak_p95_bw']:.3f} {a['load_rate_bw_s']:5.0f} | {surv}")
