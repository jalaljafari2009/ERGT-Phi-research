"""Verify the native-proposal information probe and keep M3 gated."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/m2_proposal'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((OUT/'protocol.json').read_text());r=json.loads((OUT/'result.json').read_text())
assert r['status']=='completed_proposal_probe' and not r['M3_authorized_by_results']
assert digest(OUT/'protocol.json')==r['protocol_sha256']
for path,h in p['source_sha256'].items():assert digest(ROOT/path)==h,path
true=r['results']['proposal_true']['final_monitor'];shuffled=r['results']['proposal_shuffled']['final_monitor']
assert true['balanced_accuracy']>=.99 and abs(shuffled['balanced_accuracy']-1/3)<.02
r['provenance_verified']=True;r['M3_authorized_by_results']=False
(ROOT/'manifests/m2_proposal.json').write_text(json.dumps(r,indent=2))
print(json.dumps({'status':r['status'],'provenance_verified':True,'proposal_monitor_balanced_accuracy':true['balanced_accuracy'],'shuffled_monitor_balanced_accuracy':shuffled['balanced_accuracy'],'M3_authorized_by_results':False},indent=2))
