"""Training-partition-only calibration diagnostics; never answer qualification."""
import torch
from torch.nn import functional as F
from .shadow_phase import relation_logits
from .shadow_data import pack_records


@torch.no_grad()
def evaluate_calibration(phase, records, indices):
    confusion = torch.zeros(phase.relations,phase.relations,dtype=torch.long)
    ce,total = 0.,0
    cos_sum = torch.zeros(phase.worlds,dtype=torch.float64)
    sin_sum = torch.zeros_like(cos_sum)
    phase_count = 0
    phase.eval()
    for start in range(0,len(indices),phase.config.batch_size):
        psi,mask,events,labels = pack_records(records,indices[start:start+phase.config.batch_size])
        anchors = phase(psi,mask)
        logits = relation_logits(anchors,phase.offsets,events,phase.config.temperature)
        ce += float(F.cross_entropy(logits,labels,reduction='sum')); total += labels.numel()
        prediction = logits.argmax(-1)
        confusion += torch.bincount(labels*phase.relations+prediction,minlength=phase.relations**2).reshape(phase.relations,phase.relations)
        # Collapse diagnostic on supervised endpoints only, indexing AFTER forward.
        b,i,j = events.unbind(-1)
        endpoint_phases = torch.cat((anchors[b,:,i],anchors[b,:,j]),dim=0).double()
        cos_sum += endpoint_phases.cos().sum(0); sin_sum += endpoint_phases.sin().sum(0)
        phase_count += len(endpoint_phases)
    counts = confusion.sum(1)
    recall = confusion.diag().double()/counts.clamp_min(1)
    offsets = phase.offsets
    difference = offsets[:,:,None]-offsets[:,None,:]
    distances = torch.atan2(difference.sin(),difference.cos()).abs()
    eye = torch.eye(phase.relations,dtype=torch.bool)[None]
    separation = float(distances.masked_fill(eye,float('inf')).min())
    resultant = torch.sqrt(cos_sum.square()+sin_sum.square())/phase_count
    return {'ce':ce/total,'accuracy':float(confusion.diag().sum())/total,
            'balanced_accuracy':float(recall.mean()),'per_class_recall':recall.tolist(),
            'class_counts':counts.tolist(),'confusion_matrix':confusion.tolist(),'events':total,
            'minimum_offset_separation':separation,'maximum_offset_displacement':float((offsets-phase.initial_offsets).abs().max()),
            'phase_resultant':float(resultant.mean()),'phase_resultant_per_world':resultant.tolist()}
