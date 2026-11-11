"""Standalone event-chain encoders and exact full-candidate gradient replay.

There are no fixed feature vectors, similarity matrices, teachers, or score bypasses.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .controlled import ControlledAligner, mask_self_objects


def configure_attention():
    # The same math backend is used in gradient-enabled caching and replay.
    # In particular, replay must not switch dropout kernels/masks between passes.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


class ChainEncoder(ControlledAligner):
    def __init__(self, protocol, events, train, depth=4, dropout=0.2,
                 freeze_inputs=False, relative=False, dim=256, heads=8):
        # Always initialize the legacy four-layer model before selecting depth:
        # common parameters are identical across same-seed depth controls.
        super().__init__(protocol, events, train, "shared_namespace", dim, 4, heads, dropout)
        self.intra = nn.ModuleList(list(self.intra)[:depth])
        self.inter = nn.ModuleList(list(self.inter)[:depth])
        self.heads = heads
        self.relative = relative
        self.gap_intervention = "observed"
        rng = np.random.default_rng(732026)
        permutations = np.tile(np.arange(events.shape[1]), (len(events), 1))
        for row, facts in zip(permutations, events):
            positions = np.flatnonzero(facts[:, 0] > 0)
            row[positions] = rng.permutation(positions)
        self.register_buffer("gap_permutations", torch.from_numpy(permutations), persistent=False)
        self.relative_table = nn.Embedding(25, heads)
        nn.init.zeros_(self.relative_table.weight)
        self.relative_table.weight.requires_grad_(relative)
        if freeze_inputs:
            for module in (self.objects, self.relations, self.times, self.frequency):
                module.requires_grad_(False)
            self.slot_position.requires_grad_(False)
            self.event_position.requires_grad_(False)

    def gap_bias(self, events, valid, central_ids):
        times = events[:, :, 2]
        if self.gap_intervention == "shuffled":
            # Change only time-gap/event association. The time slot and event
            # facts themselves remain untouched; coherent row permutation is
            # not a valid ablation of a relative-time model.
            times = times.gather(1, self.gap_permutations[central_ids])
        delta = times[:, :, None] - times[:, None, :]
        magnitude = torch.floor(torch.log2(delta.abs().float() + 1)).long().clamp_max(12)
        bucket = 12 + delta.sign().long() * magnitude
        bias = self.relative_table(bucket)
        present = valid & (events[:, :, 2] > 0)
        bias = bias * (present[:, :, None] & present[:, None, :])[:, :, :, None]
        if not self.relative or self.gap_intervention == "off":
            bias = bias * 0
        return bias.permute(0, 3, 1, 2).reshape(-1, events.shape[1], events.shape[1])

    def forward(self, events, central_ids=None):
        if central_ids is None:
            raise ValueError("Central IDs are required for seed/self-identity masking")
        value = mask_self_objects(events, central_ids, self.entities)
        batch, length, _ = value.shape
        valid = value[:, :, 0] > 0
        real = value[valid]
        if len(real):
            slots = torch.stack([self.objects(real[:, 0]), self.relations(real[:, 1]),
                                 self.times(real[:, 2]), self.frequency(torch.log1p(real[:, 3].float())[:, None])], 1)
            slots = slots + self.slot_position
            missing = torch.zeros((len(real), 4), dtype=torch.bool, device=value.device)
            missing[:, 2] = real[:, 2] == 0
            for layer in self.intra:
                slots = layer(slots, src_key_padding_mask=missing)
            slots = self.intra_norm(slots)
            event_vector = (slots * (~missing)[:, :, None]).sum(1) / (~missing).sum(1)[:, None]
            sequence = event_vector.new_zeros(batch, length, self.dim)
            sequence[valid] = event_vector
        else:
            sequence = self.empty.new_zeros(batch, length, self.dim)
        if self.positions:
            sequence = sequence + self.event_position[:length].to(sequence.dtype)
        empty = ~valid.any(1)
        attention_valid = valid.clone()
        attention_valid[empty, 0] = True
        padding = torch.zeros(valid.shape, device=value.device).masked_fill(~attention_valid, -torch.inf)
        relative_bias = self.gap_bias(value, valid, central_ids)
        for layer in self.inter:
            sequence = layer(sequence, src_mask=relative_bias, src_key_padding_mask=padding)
        sequence = self.inter_norm(sequence)
        pooled = (sequence * valid[:, :, None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        pooled = torch.where(empty[:, None], self.empty, pooled)
        return F.normalize(pooled.float(), dim=-1, eps=1e-8)


class ChainModel(nn.Module):
    def __init__(self, protocol, events, train, **options):
        super().__init__()
        self.n_source = protocol["entities"][0]
        self.encoder = ChainEncoder(protocol, events, train, **options)

    def forward(self, events, central_ids):
        return self.encoder(events, central_ids)

    def matrix_scores(self, queries, candidates, left, right):
        with torch.autocast(left.device.type, enabled=False):
            return left.float() @ right.float().T

    def pair_scores(self, queries, candidates, left, right):
        with torch.autocast(left.device.type, enabled=False):
            return torch.einsum("bd,bkd->bk", left.float(), right.float())


def full_candidate_loss(embedding, train, n_source, temperature=0.07):
    """Only TRAINING golds; every opposite-graph entity is in each denominator."""
    device = embedding.device
    pairs = torch.as_tensor(train, device=device)
    with torch.autocast(device.type, enabled=False):
        forward = embedding[pairs[:, 0]].float() @ embedding[n_source:].float().T
        reverse = embedding[pairs[:, 1]].float() @ embedding[:n_source].float().T
        return (F.cross_entropy(forward / temperature, pairs[:, 1] - n_source)
                + F.cross_entropy(reverse / temperature, pairs[:, 0])) / 2


def cached_full_backward(model, events, train, chunk=128, temperature=0.07, amp=True,
                         verify_replay=False):
    """Exact chain-rule gradient with bounded activations and RNG-mask replay.

    No stop-gradient candidate bank: embeddings of all entities receive their exact
full-candidate loss derivatives, then every encoder chunk is replayed for backward.
"""
    device = next(model.parameters()).device
    states, values = [], []
    for start in range(0, len(events), chunk):
        ids = np.arange(start, min(start + chunk, len(events)))
        states.append((torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None))
        # At real BF16 shapes, no_grad can choose different forward kernels.
        # Build then immediately discard each chunk's graph so both passes take
        # the identical grad-enabled path without retaining all activations.
        with torch.enable_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                enabled=amp and device.type == "cuda"):
            cached = model(torch.as_tensor(events[ids], dtype=torch.long, device=device),
                           torch.as_tensor(ids, device=device))
        values.append(cached.detach())
        del cached
    outer_cpu = torch.get_rng_state()
    outer_cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    embedding = torch.cat(values).detach().requires_grad_(True)
    loss = full_candidate_loss(embedding, train, model.n_source, temperature)
    loss.backward()
    gradients = embedding.grad.detach()
    max_replay_error = 0.0
    try:
        for batch, start in enumerate(range(0, len(events), chunk)):
            torch.set_rng_state(states[batch][0])
            if device.type == "cuda":
                torch.cuda.set_rng_state(states[batch][1], device)
            ids = np.arange(start, min(start + chunk, len(events)))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                replay = model(torch.as_tensor(events[ids], dtype=torch.long, device=device),
                               torch.as_tensor(ids, device=device))
            if verify_replay:
                error = (replay.detach() - values[batch]).abs().max().item()
                max_replay_error = max(max_replay_error, error)
                if error > 1e-6:
                    raise AssertionError(f"Cached/replayed forward mismatch: {error}")
            replay.backward(gradients[ids])
    finally:
        torch.set_rng_state(outer_cpu)
        if device.type == "cuda":
            torch.cuda.set_rng_state(outer_cuda, device)
    return loss.item(), max_replay_error
