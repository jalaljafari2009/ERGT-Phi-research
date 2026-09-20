"""Reproducible isolated-kernel acceptance plus preserved native regressions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.research_paths import workspace_path
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    out = ROOT/'runs/m1'
    out.mkdir(parents=True,exist_ok=True)
    env = {**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    summary = []
    # Invalidate prior success before a new acceptance attempt.
    status_path = workspace_path(ROOT, "manifests/m1_status.json")
    status_path.write_text(json.dumps({'status':'running','complete':False}))
    suites = [
        ('kernel',ROOT,['tests/test_phase_core.py']),
        ('m0_regression',ROOT,['tests/test_m0_parity.py','tests/test_m0_checkpoint.py','tests/test_m0_contracts.py']),
        ('reference',ROOT/'reference',['tests']),
    ]
    for name,cwd,paths in suites:
        start = time.perf_counter()
        print(f'Starting {name}',flush=True)
        xml = out/f'{name}-tests.xml'
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            proc = subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*paths,f'--junitxml={xml}'],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            status_path.write_text(json.dumps({'status':'failed','complete':False,'suite':name,'log':str(out/f'{name}.log')},indent=2))
            print((out/f'{name}.log').read_text(encoding='utf-8'),flush=True)
            raise SystemExit(proc.returncode)
        tree = ET.parse(xml)
        counts = {k:sum(int(s.attrib.get(k,0)) for s in tree.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
        assert not any(counts[k] for k in ('failures','errors','skipped'))
        summary.append({'suite':name,'seconds':time.perf_counter()-start,**counts})
        print(json.dumps(summary[-1]),flush=True)
    import torch
    from dataclasses import asdict
    from ergt_phi.phase_core import KernelConfig, prepare_patch, solve_phase_mirror, energy
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = KernelConfig()
    anchor = torch.tensor([-.2,.1,.3],dtype=torch.float64)
    ctx,_ = prepare_patch(anchor,torch.tensor([.4,.6],dtype=torch.float64),torch.tensor([[0,1],[1,2]]),torch.tensor([[.7,.3],[.2,.8]],dtype=torch.float64),torch.tensor([0.,torch.pi],dtype=torch.float64),cfg)
    start = time.perf_counter()
    result = solve_phase_mirror(ctx,cfg)
    demo = dict(config=asdict(cfg),certificate=asdict(result.certificate),iterations=result.iterations,
                anchor=anchor.tolist(),theta=result.theta.tolist(),base=ctx.base.tolist(),adhesion=result.adhesion.tolist(),
                energy_before=float(energy(ctx.anchor,ctx.base,ctx,cfg)),energy_after=float(energy(result.theta,result.adhesion,ctx,cfg)),
                residual=float(result.residual),error_bound_ignoring_roundoff=float(result.error_bound_ignoring_roundoff),
                single_toy_solve_seconds=time.perf_counter()-start,performance_claim=False)
    (out/'example.json').write_text(json.dumps(demo,indent=2))
    files = ['ergt_phi/phase_core.py','tests/test_phase_core.py','scripts/run_m1.py','docs/MATHEMATICAL_SPEC.md']
    status = dict(status='passed',complete=True,stage='M1_isolated_kernel',tests=summary,
                  config=asdict(cfg),c0_policy='explicit_research_default_0.5_not_fixed_by_spec',
                  device='cpu',dtype='float64_math_checks_and_float32_domain_checks',torch=torch.__version__,
                  native_phase_integration=False,anchor_calibration=False,scientific_validation=False,
                  certificate_scope='fixed_context_analytic_bound_ignoring_roundoff',
                  source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (workspace_path(ROOT, "manifests/m1_example.json")).write_text(json.dumps(demo,indent=2))
    status_path.write_text(json.dumps(status,indent=2))
    print('M1 ACCEPTANCE PASSED',flush=True)


if __name__ == '__main__': main()
