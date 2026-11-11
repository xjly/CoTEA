"""Frozen discrete-plus-periodic time lookup, with no additional prediction branch."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .root_mask_context import RootMaskedModel
from .interval_context import IntervalModel


class HybridTime(nn.Module):
    def __init__(self, original, coordinates, alpha=2.):
        super().__init__()
        self.original=original
        self.original.requires_grad_(False)
        values=np.asarray(coordinates,dtype=np.float64)
        dim=original.embedding_dim
        assert dim%2==0 and len(values)==original.num_embeddings
        # Scales use input timestamps only; days for DICEWS, years for YAGO.
        span=float(np.ptp(values[1:]))
        periods=np.geomspace(14.,max(28.,2*span),dim//2)
        phase=2*np.pi*(values-values[1])[:,None]/periods[None,:]
        basis=np.concatenate([np.sin(phase),np.cos(phase)],1)*math.sqrt(2.)
        basis[0]=0
        self.register_buffer('basis',torch.tensor(basis,dtype=next(original.parameters()).dtype))
        self.register_buffer('periods',torch.tensor(periods,dtype=torch.float64))
        self.alpha=float(alpha)

    def forward(self, ids):
        return (self.original(ids)+self.alpha*self.basis[ids])/math.sqrt(1+self.alpha**2)


def make_model(protocol,events,train,coordinates,relation_alias,hybrid,dim=512,heads=8):
    model=(IntervalModel(protocol,events,train,dim=dim,heads=heads,time_mode='split')
           if events.shape[2]==5 else RootMaskedModel(protocol,events,train,dim=dim,heads=heads))
    # Cache truncation must not incidentally change which relation vectors are zero.
    model.encoder.relations.alias.copy_(torch.as_tensor(relation_alias))
    if hybrid:
        model.encoder.times=HybridTime(model.encoder.times,coordinates)
    return model


@torch.no_grad()
def time_diagnostics(encoder,coordinates):
    device=next(encoder.parameters()).device
    tokens=torch.arange(1,len(coordinates),device=device)
    vectors=encoder.times(tokens).float()
    normalized=F.normalize(vectors,dim=-1)
    gaps=np.diff(coordinates[1:])
    similarities=(normalized[:-1]*normalized[1:]).sum(-1).cpu().numpy()
    nearest=gaps==gaps[gaps>0].min()
    result={'mean_time_norm':float(vectors.norm(dim=-1).mean()),
            'nearest_numeric_gap':float(gaps[gaps>0].min()),
            'nearest_gap_mean_cosine':float(similarities[nearest].mean()),
            'all_adjacent_token_mean_cosine':float(similarities.mean())}
    if isinstance(encoder.times,HybridTime):
        result.update(alpha=encoder.times.alpha,period_min=float(encoder.times.periods.min()),
                      period_max=float(encoder.times.periods.max()),time_trainable=False)
    return result
