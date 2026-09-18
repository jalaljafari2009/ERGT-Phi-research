"""Pair-endpoint diagnostic head; loss-only, never a native inference input."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PairDiagnosticConfig:
    hidden_dim: int = 128
    temperature: float = .2
    learning_rate: float = .002
    weight_decay: float = .0001
    epochs: int = 20
    batch_size: int = 16
    seed: int = 23092026

    def __post_init__(self):
        if min(self.hidden_dim,self.epochs,self.batch_size) <= 0:
            raise ValueError('positive diagnostic dimensions and budgets required')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')


class PairRelationNetwork(nn.Module):
    """A supervised diagnostic only: endpoint pair features -> relation logits."""
    def __init__(self, feature_dim, config=PairDiagnosticConfig(), relations=3):
        super().__init__()
        if feature_dim <= 0 or relations <= 1: raise ValueError('invalid feature dimensions')
        self.config,self.feature_dim,self.relations = config,feature_dim,relations
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.net=nn.Sequential(nn.LayerNorm(4*feature_dim),nn.Linear(4*feature_dim,config.hidden_dim),nn.SiLU(),nn.Linear(config.hidden_dim,relations))

    def pair_features(self, features, events):
        if features.ndim != 3 or events.ndim != 2 or events.shape[1] != 3 or events.dtype != torch.long:
            raise ValueError('features/events shape mismatch')
        if events.numel():
            b,i,j=events.unbind(-1)
            if bool((events<0).any()) or bool((b>=features.shape[0]).any()) or bool((i>=features.shape[1]).any()) or bool((j>=features.shape[1]).any()):
                raise ValueError('event outside feature domain')
            left,right=features[b,i],features[b,j]
            return torch.cat((left,right,(left-right).abs(),left*right),dim=-1)
        return features.new_empty((0,4*features.shape[-1]))

    def forward(self, features, events):
        return self.net(self.pair_features(features,events))


def pair_loss(model,features,events,labels):
    if labels.numel()==0: raise ValueError('empty pair supervision')
    return F.cross_entropy(model(features,events),labels)
