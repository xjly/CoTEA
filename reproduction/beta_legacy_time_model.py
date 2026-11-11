"""BETA time-slot controls with the original root-masked Transformer context."""
import torch
from torch import nn

from .seed_relation_context import SeedRelationModel


class YearOnlyTime(nn.Module):
    """Reuse the original frozen year table; missing time remains token zero."""
    def __init__(self, calendar):
        super().__init__()
        self.year=calendar.year
        self.register_buffer('year_index',calendar.components[:,0].clone())
        self.embedding_dim=calendar.embedding_dim
        self.num_embeddings=calendar.num_embeddings
        self.requires_grad_(False)

    def forward(self, ids):
        return self.year(self.year_index[ids])


def make_model(protocol,events,train,counts,coordinates,arm,dim=512,heads=8):
    if events.shape[-1]!=5 or 'ymd_components' not in protocol:
        raise ValueError('Expected original BETA interval cache and calendar components')
    # Construct the same complete model first: every other parameter and random
    # draw is identical, including the original context projection and gate.
    model=SeedRelationModel(protocol,events,train,counts,coordinates,'induced',dim,heads)
    if arm=='no_period':model.encoder.times=model.encoder.times.original
    elif arm=='year_only':model.encoder.times=YearOnlyTime(model.encoder.times.original)
    elif arm!='hybrid':raise ValueError(arm)
    assert not hasattr(model.encoder,'graph_context')
    return model
