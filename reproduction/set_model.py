"""Unordered neural event sets with end-to-end one-hop object context.

Absolute timestamps are input slots/group keys. There are no event positions,
predecessors, intervals, temporal transition modules, or extra retrieval scores.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .chain_only import ChainEncoder
from .controlled import mask_self_objects


KINDS = ("base", "neighbor", "relation", "both")


def day_membership(times, valid):
    """Unordered equality groups, represented once at their first occurrence.

    Representative indices have no features or weights of their own. Reordering
    input rows only permutes these groups; it cannot encode chronological order.
    """
    same = (times[:, :, None] == times[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    earlier = torch.ones(times.shape[1], times.shape[1], device=times.device, dtype=torch.bool).tril(-1)
    present = valid & ~(same & earlier).any(-1)
    membership = same & present[:, :, None]
    return membership, present


class SetEncoder(ChainEncoder):
    def __init__(self, protocol, events, train, kind="base", dim=256, heads=8):
        if kind not in KINDS:
            raise ValueError(kind)
        # Preserve the earlier common backbone initialization. Remove all unused
        # sequence modules/parameters/buffers rather than keeping disabled heads.
        super().__init__(protocol, events, train, depth=2, dropout=0., freeze_inputs=True,
                         relative=False, dim=dim, heads=heads)
        del self.inter, self.event_position, self.relative_table, self.gap_permutations
        self.pool_norm = self.inter_norm
        del self.inter_norm
        self.positions = False
        self.kind = kind
        self.use_neighbor = kind in ("neighbor", "both")
        self.use_relation = kind in ("relation", "both")
        self.intervention = "observed"
        self.context_project = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.context_project.weight)
        self.context_gate = nn.Linear(2 * dim, 1)
        nn.init.zeros_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)
        self.relation_gate = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 64),
                                           nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.relation_gate[-1].weight)
        nn.init.zeros_(self.relation_gate[-1].bias)
        self.context_project.requires_grad_(self.use_neighbor)
        self.context_gate.requires_grad_(self.use_neighbor)
        self.relation_gate.requires_grad_(self.use_relation)
        # This permutation changes only read-only diagnostic message endpoints.
        rng = np.random.default_rng(912026)
        n_source = protocol["entities"][0]
        permutation = np.concatenate([rng.permutation(n_source),
                                      n_source + rng.permutation(len(events) - n_source)])
        self.register_buffer("diagnostic_context_permutation", torch.as_tensor(permutation), persistent=False)

    def extra_parameters(self):
        return [p for module in (self.context_project, self.context_gate, self.relation_gate)
                for p in module.parameters() if p.requires_grad]

    def event_vectors(self, events, central_ids, context=None):
        value = mask_self_objects(events, central_ids, self.entities)
        valid = events[:, :, 0] > 0
        real = value[valid]
        if not len(real):
            return self.empty.new_zeros(*valid.shape, self.dim), valid, value
        objects = self.objects(real[:, 0])
        relations, times = self.relations(real[:, 1]), self.times(real[:, 2])
        if context is not None:
            # Raw IDs route within each observed graph; only training pairs
            # define cross-graph alias tokens. Self-loop identities stay masked.
            raw_objects = events[:, :, 0][valid] - 1
            source_centers = central_ids[:, None].expand_as(valid)[valid]
            if self.intervention == "shuffle_context":
                raw_objects = self.diagnostic_context_permutation[raw_objects]
            # Unit-normalized entity vectors are rescaled to the lookup-slot
            # variance (roughly one per coordinate), before the learned map.
            message = self.context_project(context[raw_objects] * math.sqrt(self.dim))
            gate = torch.sigmoid(self.context_gate(torch.cat([relations, times], -1)))
            nonself = (events[:, :, 0][valid] - 1 != source_centers)[:, None]
            objects = objects + message * gate * nonself
        slots = torch.stack([objects, relations, times,
                             self.frequency(torch.log1p(real[:, 3].float())[:, None])], 1)
        slots = slots + self.slot_position
        missing = torch.zeros(len(real), 4, dtype=torch.bool, device=events.device)
        missing[:, 2] = real[:, 2] == 0
        for layer in self.intra:
            slots = layer(slots, src_key_padding_mask=missing)
        slots = self.intra_norm(slots)
        vector = (slots * (~missing)[:, :, None]).sum(1) / (~missing).sum(1)[:, None]
        sequence = vector.new_zeros(*valid.shape, self.dim)
        sequence[valid] = vector
        return sequence, valid, value

    def pool(self, sequence, events, valid, conditioned=False):
        membership, day_valid = day_membership(events[:, :, 2], valid)
        logits = torch.zeros(valid.shape, dtype=torch.float32, device=sequence.device)
        if conditioned and self.intervention != "uniform_pool":
            real = valid.nonzero(as_tuple=True)
            if len(real[0]):
                relation = self.relations(events[:, :, 1][valid])
                raw = self.relation_gate(torch.cat([sequence[valid], relation], -1)).squeeze(-1)
                logits[real] = 4 * torch.tanh(raw.float() / 4)
        with torch.autocast(sequence.device.type, enabled=False):
            # Subtracting the maximum is a numerical normalization only.
            maximum = logits.float().masked_fill(~valid, -torch.inf).amax(1, keepdim=True)
            maximum = torch.where(valid.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
            weights = torch.exp(logits.float() - maximum) * valid
            group = membership.float()
            denominator = torch.bmm(group, weights[:, :, None]).squeeze(-1)
            count = group.sum(-1)
            days = torch.bmm(group, sequence.float() * weights[:, :, None]) / denominator.clamp_min(1e-12)[:, :, None]
            day_weight = denominator / count.clamp_min(1)
        days = self.pool_norm(days)
        pooled = (days * day_weight[:, :, None]).sum(1) / day_weight.sum(1).clamp_min(1e-12)[:, None]
        pooled = torch.where(day_valid.any(1)[:, None], pooled, self.empty)
        return F.normalize(pooled.float(), dim=-1, eps=1e-8)

    def base(self, events, central_ids):
        # Shared learned encoder, no recursive graph messages or relation gate.
        sequence, valid, _ = self.event_vectors(events, central_ids)
        return self.pool(sequence, events, valid, conditioned=False)

    def forward(self, events, central_ids, context=None):
        use_context = self.use_neighbor and self.intervention != "disable_context"
        if use_context and context is None:
            raise ValueError("A current learned context bank is required; use set_training.encode_all")
        sequence, valid, _ = self.event_vectors(events, central_ids, context if use_context else None)
        return self.pool(sequence, events, valid, conditioned=self.use_relation)


class SetModel(nn.Module):
    def __init__(self, protocol, events, train, **options):
        super().__init__()
        self.n_source = protocol["entities"][0]
        self.encoder = SetEncoder(protocol, events, train, **options)

    def forward(self, events, central_ids, context=None):
        return self.encoder(events, central_ids, context)

    def matrix_scores(self, queries, candidates, left, right):
        with torch.autocast(left.device.type, enabled=False):
            return left.float() @ right.float().T
