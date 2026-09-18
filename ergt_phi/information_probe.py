"""Information-only probes; they never modify or enter native inference."""
import random
import torch
from torch import nn
from torch.nn import functional as F


class LinearRelationProbe(nn.Module):
    def __init__(self, feature_dim, relations=3, seed=24092026):
        super().__init__()
        if feature_dim<=0 or relations<=1: raise ValueError('invalid probe shape')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.net=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,relations))

    def forward(self,x):
        if x.ndim!=2: raise ValueError('probe expects [E,F]')
        return self.net(x)


def history_pair_features(records,indices,mode='final'):
    rows=[]; labels=[]
    for record_index in indices:
        direct=records[record_index].get('pair_features')
        if direct is not None:
            rows.extend(list(direct)); labels.extend(list(records[record_index]['labels']))
            continue
        history=records[record_index].get('history')
        if history is None:
            history=records[record_index]['psi'].unsqueeze(0)
        for (i,j),label in zip(records[record_index]['events'],records[record_index]['labels']):
            if mode=='final': selected=history[-1:]
            elif mode=='all': selected=history
            else: raise ValueError('unknown history mode')
            left,right=selected[:,i],selected[:,j]
            rows.append(torch.cat((left,right,(left-right).abs(),left*right),dim=-1).reshape(-1))
            labels.append(label)
    if not rows: raise ValueError('empty information probe')
    return torch.stack(rows),torch.stack(labels).long()


@torch.no_grad()
def probe_metrics(model,x,y):
    model.eval(); logits=model(x); pred=logits.argmax(-1)
    confusion=torch.bincount(y*3+pred,minlength=9).reshape(3,3)
    counts=confusion.sum(1); recall=confusion.diag().double()/counts.clamp_min(1)
    return {'accuracy':float(confusion.diag().sum())/len(y),'balanced_accuracy':float(recall.mean()),'per_class_recall':recall.tolist(),'class_counts':counts.tolist(),'confusion_matrix':confusion.tolist(),'ce':float(F.cross_entropy(logits,y)),'events':len(y)}


def train_probe(train_x,train_y,monitor_x,monitor_y,*,seed=24092026,epochs=40,lr=.003,shuffle=False):
    model=LinearRelationProbe(train_x.shape[1],seed=seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    initial=probe_metrics(model,monitor_x,monitor_y)
    for epoch in range(epochs):
        order=list(range(len(train_y))); random.Random(seed+epoch).shuffle(order)
        for start in range(0,len(order),256):
            index=torch.tensor(order[start:start+256],dtype=torch.long)
            y=train_y[index]
            if shuffle:
                generator=torch.Generator().manual_seed(seed+100000+epoch*1000+start)
                y=y[torch.randperm(len(y),generator=generator)]
            optimizer.zero_grad(set_to_none=True); loss=F.cross_entropy(model(train_x[index]),y); loss.backward(); optimizer.step()
    return {'initial_monitor':initial,'final_train':probe_metrics(model,train_x,train_y),'final_monitor':probe_metrics(model,monitor_x,monitor_y),'epochs':epochs,'shuffled_labels':shuffle}
