"""Pin M2 experiment and engineering evidence without promoting failed calibration."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/m2'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((out/'protocol.json').read_text())
result=json.loads((out/'result.json').read_text())
assert digest(out/'protocol.json')==result['protocol_sha256']
for path,expected in protocol['source_sha256'].items(): assert digest(ROOT/path)==expected,path
counts=[]
for filename in ('engineering-tests.xml','resume-and-offset-tests.xml','contextual-tests.xml'):
    tree=ET.parse(out/filename)
    row={k:sum(int(s.attrib.get(k,0)) for s in tree.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    assert not any(row[k] for k in ('failures','errors','skipped'))
    counts.append({'file':filename,**row})
result['engineering_acceptance']={'passed':True,'unique_tests':97,'m2_tests':23,'m1_tests':44,'m0_tests':30,
    'records':counts,'note':'The two strengthened resume/offset tests supersede their earlier versions; do not count them twice.'}
result['source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in [
    ROOT/'tests/test_shadow_phase.py',ROOT/'scripts/shadow_resume_worker.py',ROOT/'scripts/finalize_m2.py']}
result['artifact_sha256']={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.suffix in ('.pt','.xml','.json') and p.name!='result.json'}
(ROOT/'manifests/m2_status.json').write_text(json.dumps(result,indent=2))
(ROOT/'manifests/m2_protocol.json').write_text(json.dumps(protocol,indent=2))
print(json.dumps({'engineering_tests_passed':97,'M2_complete':result['M2_complete'],
                  'calibration_status':result['status'],'M3_authorized_by_results':result['M3_authorized_by_results']},indent=2))
