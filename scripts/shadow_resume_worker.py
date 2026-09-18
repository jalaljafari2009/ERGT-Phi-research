"""Small M2 full-state resume fixture, exercised in separate Python processes."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import random
import numpy as np
from ergt_phi.shadow_phase import PhaseAnchorNetwork,CalibrationConfig,calibration_step
from ergt_phi.checkpoint import save,load


def main():
    mode,directory=sys.argv[1],Path(sys.argv[2])
    directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    seed=937 if mode=='resume' else 81
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    phase=PhaseAnchorNetwork(4,2,3,CalibrationConfig(hidden_dim=8))
    opt=torch.optim.AdamW(phase.parameters(),lr=.002)
    contract={'stage':'M2_resume_fixture','seed':81}
    progress={'step':0,'cursor':0,'frozen':False}
    if mode=='resume': progress=load(directory/'cut.pt',phase,opt,contract=contract)
    generator=torch.Generator().manual_seed(17)
    psi=torch.randn(2,5,4,generator=generator)
    mask=torch.ones(2,5,dtype=torch.bool)
    events=torch.tensor([[0,0,1],[0,1,2],[1,2,3],[1,3,4]])
    labels=torch.tensor([0,1,2,0])
    target=1 if mode=='first' else 3
    while progress['step']<target:
        factor=float(torch.rand(()))+random.random()+float(np.random.rand())+.5
        calibration_step(phase,opt,(psi*factor,mask,events,labels),contract=contract,progress=progress)
        progress['cursor']+=2
    name={'first':'cut.pt','resume':'resumed.pt','continuous':'continuous.pt'}[mode]
    save(directory/name,phase,opt,contract=contract,progress=progress)


if __name__=='__main__': main()
