"""Four-slot intra-event and ordered inter-event Transformer encoders (alpha=1)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EventChainAligner(nn.Module):
    def __init__(self, entities, relations, times, max_length=64, dim=256, layers=4,
                 heads=8, dropout=0.2, positions=True):
        super().__init__()
        self.dim = dim
        self.positions = positions
        self.objects = nn.Embedding(entities + 1, dim, padding_idx=0)
        self.relations = nn.Embedding(2 * relations + 1, dim, padding_idx=0)
        self.times = nn.Embedding(times + 1, dim, padding_idx=0)
        self.frequency = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.slot_position = nn.Parameter(torch.randn(4, dim) * 0.02)
        self.event_position = nn.Parameter(torch.randn(max_length, dim) * 0.02)
        self.empty = nn.Parameter(torch.randn(dim) * 0.02)

        def encoder():
            # Separately initialize layers rather than cloning identical initial weights.
            return nn.ModuleList([nn.TransformerEncoderLayer(
                dim, heads, dim_feedforward=4 * dim, dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True) for _ in range(layers)])

        self.intra = encoder()
        self.inter = encoder()
        self.intra_norm = nn.LayerNorm(dim)
        self.inter_norm = nn.LayerNorm(dim)

    def forward(self, events, central_ids=None):
        batch, length, _ = events.shape
        valid = events[:, :, 0] > 0
        real = events[valid]
        # A nonempty input allows creation in the autocast activation dtype.
        if len(real):
            slots = torch.stack([self.objects(real[:, 0]), self.relations(real[:, 1]),
                                 self.times(real[:, 2]),
                                 self.frequency(torch.log1p(real[:, 3].float())[:, None])], dim=1)
            slots = slots + self.slot_position
            time_missing = torch.zeros((len(real), 4), dtype=torch.bool, device=events.device)
            time_missing[:, 2] = real[:, 2] == 0
            # Additive attention masks keep the memory-efficient SDPA path available.
            for layer in self.intra:
                slots = layer(slots, src_key_padding_mask=time_missing)
            slots = self.intra_norm(slots)
            slot_valid = (~time_missing)[:, :, None]
            event_vectors = (slots * slot_valid).sum(1) / slot_valid.sum(1).clamp_min(1)
            sequence = event_vectors.new_zeros(batch, length, self.dim)
            sequence[valid] = event_vectors
        else:
            sequence = self.empty.new_zeros(batch, length, self.dim)
        if self.positions:
            sequence = sequence + self.event_position[:length].to(sequence.dtype)
        empty = ~valid.any(1)
        attention_valid = valid.clone()
        attention_valid[empty, 0] = True  # Avoid all-masked attention producing NaN.
        for layer in self.inter:
            sequence = layer(sequence, src_key_padding_mask=~attention_valid)
        sequence = self.inter_norm(sequence)
        pooled = (sequence * valid[:, :, None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        pooled = torch.where(empty[:, None], self.empty, pooled)
        return F.normalize(pooled.float(), dim=-1, eps=1e-8)
