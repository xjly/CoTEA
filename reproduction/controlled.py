"""Single-factor controls for cross-graph inputs and temporal-geometry preservation."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import EventChainAligner

VARIANTS = ("baseline", "no_object", "no_relation", "time_only", "seed_shared",
            "object_profile", "relation_profile", "frozen_time", "time_preserve", "histogram",
            "object_profile_shuffled", "relation_profile_shuffled", "shared_namespace",
            "shared_namespace_time", "shared_context")


def time_profiles(events, times):
    """Fixed features from observed input facts; no entity-alignment labels."""
    result = np.zeros((len(events) + 1, times + 1), dtype=np.float32)
    valid = (events[:, :, 0] > 0) & (events[:, :, 2] > 0)
    rows, positions = np.nonzero(valid)
    np.add.at(result, (rows + 1, events[rows, positions, 2]), events[rows, positions, 3])
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result


def seed_alias(entities, train):
    alias = np.arange(entities + 1, dtype=np.int64)
    for source, target in train:
        alias[target + 1] = source + 1
    return alias


def relation_profiles(events, relations, train):
    """Common axes are TRAINING seed identities, not held-out alignments."""
    result = np.zeros((2 * relations + 1, len(train)), dtype=np.float32)
    for anchor, pair in enumerate(train):
        for entity in pair:
            selected = events[entity]
            valid = selected[:, 0] > 0
            np.add.at(result[:, anchor], selected[valid, 1], selected[valid, 3])
    result = np.log1p(result)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result


class TokenControl(nn.Module):
    def __init__(self, original, mode, alias=None, features=None):
        super().__init__()
        self.original = original
        self.mode = mode
        if alias is not None:
            self.register_buffer("alias", torch.from_numpy(alias))
        if features is not None:
            self.register_buffer("features", torch.from_numpy(features), persistent=False)
            # Preserve the baseline's RNG stream for dropout and all other parameters.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(732026)
                self.project = nn.Sequential(nn.Linear(features.shape[1], original.embedding_dim),
                                             nn.GELU(), nn.Linear(original.embedding_dim, original.embedding_dim))

    def forward(self, ids):
        if self.mode == "zero":
            return torch.zeros((*ids.shape, self.original.embedding_dim), dtype=self.original.weight.dtype, device=ids.device)
        if self.mode == "alias":
            return self.original(self.alias[ids])
        if self.mode == "profile":
            return self.project(self.features[ids])
        return self.original(ids)


def normalized_histogram(events, bins):
    valid = (events[:, :, 0] > 0) & (events[:, :, 2] > 0)
    values = events[:, :, 3].float() * valid
    result = values.new_zeros(len(events), bins)
    result.scatter_add_(1, events[:, :, 2], values)
    return F.normalize(result, dim=-1, eps=1e-12)


def neighbor_time_profiles(events, times):
    from scipy import sparse
    own = time_profiles(events, times)
    rows, positions = np.nonzero(events[:, :, 0] > 0)
    adjacency = sparse.coo_matrix((events[rows, positions, 3].astype(np.float32),
                                   (rows + 1, events[rows, positions, 0])),
                                  shape=(len(events) + 1, len(events) + 1)).tocsr()
    context = np.asarray(adjacency @ own)
    context /= np.maximum(np.linalg.norm(context, axis=1, keepdims=True), 1e-12)
    return context


def mask_self_objects(events, central_ids, entities):
    value = events.clone()
    mask = value[:, :, 0] == central_ids[:, None] + 1
    # Keep these as real events; the extra object token maps to a zero vector.
    value[:, :, 0][mask] = entities + 1
    return value


class ControlledAligner(EventChainAligner):
    def __init__(self, protocol, events, train, variant, dim, layers, heads, dropout,
                 positions=True, time_weight=0.8):
        super().__init__(sum(protocol["entities"]), protocol["shared_relations"], protocol["time_tokens"],
                         protocol["max_length"], dim, layers, heads, dropout, positions)
        self.variant = variant
        self.time_weight = time_weight
        self.time_bins = protocol["time_tokens"] + 1
        self.entities = sum(protocol["entities"])
        if variant in ("no_object", "time_only"):
            self.objects = TokenControl(self.objects, "zero")
        if variant in ("no_relation", "time_only"):
            self.relations = TokenControl(self.relations, "zero")
        if variant == "seed_shared":
            self.objects = TokenControl(self.objects, "alias", alias=seed_alias(sum(protocol["entities"]), train))
        if variant in ("object_profile", "object_profile_shuffled"):
            features = time_profiles(events, protocol["time_tokens"])
            if variant.endswith("_shuffled"):
                rng = np.random.default_rng(402026)
                for begin, end in [(1, protocol["entities"][0] + 1),
                                   (protocol["entities"][0] + 1, len(features))]:
                    features[begin:end] = features[rng.permutation(np.arange(begin, end))]
            self.objects = TokenControl(self.objects, "profile", features=features)
        if variant in ("relation_profile", "relation_profile_shuffled"):
            features = relation_profiles(events, protocol["shared_relations"], train)
            if variant.endswith("_shuffled"):
                n_source = protocol["entities"][0]
                source = set(events[:n_source, :, 1][events[:n_source, :, 0] > 0].tolist())
                target = set(events[n_source:, :, 1][events[n_source:, :, 0] > 0].tolist())
                rng = np.random.default_rng(402026)
                # Keep supplied common schema labels and relation direction intact.
                for parity in (0, 1):
                    exclusive = np.array(sorted(r for r in target - source if r % 2 == parity), dtype=np.int64)
                    features[exclusive] = features[rng.permutation(exclusive)]
            self.relations = TokenControl(self.relations, "profile", features=features)
        if variant == "frozen_time":
            self.times.weight.requires_grad_(False)
        if variant.startswith("shared_namespace") or variant == "shared_context":
            alias = np.zeros(self.entities + 2, dtype=np.int64)
            for source, target in train:
                alias[source + 1] = source + 1
                alias[target + 1] = source + 1
            self.objects = TokenControl(self.objects, "alias", alias=alias)
            n_source = protocol["entities"][0]
            source = set(events[:n_source, :, 1][events[:n_source, :, 0] > 0].tolist())
            target = set(events[n_source:, :, 1][events[n_source:, :, 0] > 0].tolist())
            alias_rel = np.zeros(2 * protocol["shared_relations"] + 1, dtype=np.int64)
            for token in source & target:
                alias_rel[token] = token
            self.relations = TokenControl(self.relations, "alias", alias=alias_rel)
            if variant == "shared_context":
                self.register_buffer("neighbor_profiles", torch.from_numpy(neighbor_time_profiles(
                    events, protocol["time_tokens"])), persistent=False)

    def forward(self, events, central_ids=None):
        namespace = self.variant.startswith("shared_namespace") or self.variant == "shared_context"
        if namespace:
            if central_ids is None:
                raise ValueError("The shared namespace encoder needs central entity IDs to mask self-loop identities")
            value = mask_self_objects(events, central_ids, self.entities)
        else:
            value = events
        learned = super().forward(value)
        if self.variant not in ("time_preserve", "shared_namespace_time", "shared_context"):
            return learned
        temporal = normalized_histogram(events, self.time_bins)
        if self.variant == "shared_context":
            context = self.neighbor_profiles[central_ids + 1]
            vector = torch.cat([learned * math.sqrt(1 - self.time_weight),
                                temporal * math.sqrt(self.time_weight / 2),
                                context * math.sqrt(self.time_weight / 2)], dim=-1)
        else:
            vector = torch.cat([learned * math.sqrt(1 - self.time_weight),
                                temporal * math.sqrt(self.time_weight)], dim=-1)
        return F.normalize(vector, dim=-1, eps=1e-12)


class HistogramAligner(nn.Module):
    """Positive shared time weights; starts at the EXACT time-frequency baseline."""
    def __init__(self, times):
        super().__init__()
        self.log_weight = nn.Parameter(torch.zeros(times + 1))

    def forward(self, events, central_ids=None):
        value = normalized_histogram(events, len(self.log_weight))
        value = value * torch.exp(0.5 * self.log_weight.clamp(-8, 8))
        return F.normalize(value, dim=-1, eps=1e-12)


def build_model(args, protocol, events, splits):
    variant = getattr(args, "variant", "baseline")
    if variant == "histogram":
        return HistogramAligner(protocol["time_tokens"])
    if variant != "baseline":
        return ControlledAligner(protocol, events, splits["train"], variant, args.dim, args.layers,
                                 args.heads, args.dropout, args.positions, args.time_weight)
    return EventChainAligner(sum(protocol["entities"]), protocol["shared_relations"], protocol["time_tokens"],
                             protocol["max_length"], args.dim, args.layers, args.heads, args.dropout, args.positions)


def temporal_hard_pool(events, train, target_pool, times, width=64):
    from scipy import sparse
    valid = (events[:, :, 0] > 0) & (events[:, :, 2] > 0)
    rows, positions = np.nonzero(valid)
    matrix = sparse.coo_matrix((events[rows, positions, 3].astype(np.float32),
                               (rows, events[rows, positions, 2])),
                              shape=(len(events), times + 1)).tocsr()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(1)).ravel())
    matrix = sparse.diags(1 / np.maximum(norms, 1e-12)).dot(matrix).tocsr()
    lookup = {int(value): i for i, value in enumerate(target_pool)}
    width = min(width, len(target_pool) - 1)
    result = np.empty((len(train), width), dtype=np.int64)
    for start in range(0, len(train), 64):
        selected = train[start:start + 64]
        score = (matrix[selected[:, 0]] @ matrix[target_pool].T).toarray()
        for row, positive in enumerate(selected[:, 1]):
            score[row, lookup[int(positive)]] = -np.inf
            columns = np.argpartition(-score[row], width - 1)[:width]
            result[start + row] = target_pool[columns]
    return result


def mixed_negatives(positive, pool, hard_rows, count, hard_count, rng):
    if not 0 <= hard_count <= count:
        raise ValueError("hard_count must be between zero and total negatives")
    result = np.empty((len(positive), count), dtype=np.int64)
    for row, value in enumerate(positive):
        chosen = list(map(int, rng.choice(hard_rows[row], hard_count, replace=False)))
        used = {int(value), *chosen}
        while len(chosen) < count:
            candidate = int(rng.choice(pool))
            if candidate not in used:
                used.add(candidate)
                chosen.append(candidate)
        result[row] = chosen
    return result
