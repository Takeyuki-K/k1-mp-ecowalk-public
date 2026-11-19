"""eval_gait56 with the ankle-roll spring also applied to the RUN env models."""
import os, sys, runpy
sys.path.insert(0, os.getcwd())
import k1env_run4, k1env_eco
_orig = k1env_run4.K1Run4Batch.__init__
def _init(self, *a, **k):
    _orig(self, *a, **k)
    K = float(os.environ['K1_ANKROLL_SPRING']); D = float(os.environ.get('K1_ANKROLL_DAMP', '0.3'))
    ms = {id(m): m for m in list(getattr(self, 'models', [])) + [self.m]}
    for m in ms.values(): k1env_eco._apply_ankle_spring(m, K, D)
    print('# spring applied to run env models:', len(ms), file=sys.stderr)
k1env_run4.K1Run4Batch.__init__ = _init
sys.argv = ['eval_gait56.py'] + sys.argv[1:]
runpy.run_path('eval_gait56.py', run_name='__main__')
