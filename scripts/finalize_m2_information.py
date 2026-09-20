"""Verify M2 information-probe provenance without promoting M3."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.research_paths import current_input, workspace_path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]; OUT=current_input(ROOT, "runs/m2_information")
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((OUT/'protocol.json').read_text()); r=json.loads((OUT/'result.json').read_text())
assert r['status']=='completed_information_probe' and not r['M3_authorized_by_results']
assert digest(OUT/'protocol.json')==r['protocol_sha256']
for path,h in p['source_sha256'].items(): assert digest(ROOT/path)==h,path
summary={k:v['final_monitor']['balanced_accuracy'] for k,v in r['results'].items()}
assert summary['native_final_true']>summary['psi0_true']
assert summary['native_history_true']>=summary['native_final_true']-.05
assert abs(summary['native_final_shuffled']-1/3)<.02
r['provenance_verified']=True; r['summary_balanced_accuracy']=summary
(workspace_path(ROOT, "manifests/m2_information.json")).write_text(json.dumps(r,indent=2))
print(json.dumps({'status':r['status'],'provenance_verified':True,'M3_authorized_by_results':False,'summary':summary},indent=2))
