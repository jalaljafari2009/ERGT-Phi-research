"""Verify M2 diagnostic provenance and record its non-promotion decision."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.research_paths import current_input, workspace_path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]; OUT=current_input(ROOT, "runs/m2_diagnostic")
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((OUT/'protocol.json').read_text()); r=json.loads((OUT/'result.json').read_text())
assert r['status']=='completed_diagnostic' and not r['M3_authorized_by_results']
assert digest(OUT/'protocol.json')==r['protocol_sha256']
for path,h in p['source_sha256'].items(): assert digest(ROOT/path)==h,path
summary={k:v['final_monitor']['balanced_accuracy'] for k,v in r['results'].items()}
assert summary['pair_true'] > summary['contextual_true'] > summary['token_true']
assert abs(summary['pair_shuffled']-1/3) < .02
r['provenance_verified']=True; r['M3_authorized_by_results']=False; r['summary_balanced_accuracy']=summary
(workspace_path(ROOT, "manifests/m2_diagnostic.json")).write_text(json.dumps(r,indent=2))
print(json.dumps({'status':r['status'],'provenance_verified':True,'M3_authorized_by_results':False,'summary':summary},indent=2))
