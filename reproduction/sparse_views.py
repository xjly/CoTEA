"""Training-only sparse event queries with exact full-bank candidate gradients."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from .chain_only import full_candidate_loss
from .set_training import cached_full_backward as full_backward


def schedule(step):
    """Fixed conservative curriculum, declared before reading this round's results."""
    if step <= 20:
        return 0., ()
    weight = .1 * min(1., (step - 20) / 20)
    sizes = (8,) if step <= 40 else (4, 8) if step <= 60 else (2, 4, 8)
    return weight, sizes


def sample_views(events, train, seed, step):
    weight, sizes = schedule(step)
    if not sizes:
        return None, None, {"weight": 0., "sizes": [], "queries": 0}
    ids = np.concatenate([train[:, 0], train[:, 1]])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20261029, step]))
    views = np.zeros((len(ids), 8, 4), dtype=events.dtype)
    retained = []
    for index, entity in enumerate(ids):
        valid = np.flatnonzero(events[entity, :, 0] > 0)
        k = min(int(rng.choice(sizes)), len(valid))
        positions = np.sort(rng.choice(valid, k, replace=False)) if k else []
        views[index, :k] = events[entity, positions]
        retained.append(k)
    values, counts = np.unique(retained, return_counts=True)
    stats = {"weight": weight, "sizes": list(sizes), "queries": len(ids),
             "retained_event_counts": {str(int(value)): int(count) for value, count in zip(values, counts)},
             "views_sha256": hashlib.sha256(views.tobytes()).hexdigest(),
             "mean_retained_events": float(np.mean(retained)),
             "provenance": "only original cached facts of the training query entities; frequencies unchanged"}
    return views, ids, stats


def sparse_loss(queries, full, train, n_source, temperature=.07):
    """Sparse sources/targets query complete opposite-graph candidate embeddings."""
    pairs = torch.as_tensor(train, device=full.device)
    n = len(pairs)
    if len(queries) != 2 * n:
        raise ValueError("Expected source views then target views for every training pair")
    with torch.autocast(full.device.type, enabled=False):
        forward = queries[:n].float() @ full[n_source:].float().T
        reverse = queries[n:].float() @ full[:n_source].float().T
        return (F.cross_entropy(forward / temperature, pairs[:, 1] - n_source)
                + F.cross_entropy(reverse / temperature, pairs[:, 0])) / 2


def cached_sparse_backward(model, events, train, views=None, query_ids=None, weight=0.,
                           chunk=128, amp=True, verify_replay=False):
    if model.encoder.use_neighbor:
        raise ValueError("This experiment isolates sparse views on base/relation models, without neighbor context")
    if weight == 0:
        loss, errors = full_backward(model, events, train, chunk=chunk, amp=amp, verify_replay=verify_replay)
        return loss, {"full": loss, "sparse": None, "weight": 0.}, {"full": errors["final"], "sparse": None}
    if views is None or query_ids is None or len(query_ids) != 2 * len(train):
        raise ValueError("Missing sparse training queries")
    expected_ids = np.concatenate([train[:, 0], train[:, 1]])
    if not np.array_equal(query_ids, expected_ids):
        raise ValueError("Sparse views may only use the training-pair entities, in source/target order")
    device = next(model.parameters()).device

    def rng_state():
        return torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None

    def restore(state):
        torch.set_rng_state(state[0])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state[1], device)

    def inputs(data, ids, start):
        value = torch.as_tensor(data[start:start+chunk], dtype=torch.long, device=device)
        return value, torch.as_tensor(ids[start:start+chunk], dtype=torch.long, device=device)

    def cache(data, ids):
        parts, states = [], []
        for start in range(0, len(data), chunk):
            states.append(rng_state())
            value, central = inputs(data, ids, start)
            with torch.enable_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                    enabled=amp and device.type == "cuda"):
                output = model(value, central)
            parts.append(output.detach())
            del output
        return parts, states

    full_ids = np.arange(len(events))
    full_parts, full_states = cache(events, full_ids)
    sparse_parts, sparse_states = cache(views, query_ids)
    outer_rng = rng_state()
    full = torch.cat(full_parts).detach().requires_grad_(True)
    queries = torch.cat(sparse_parts).detach().requires_grad_(True)
    complete_loss = full_candidate_loss(full, train, model.n_source)
    auxiliary_loss = sparse_loss(queries, full, train, model.n_source)
    loss = complete_loss + weight * auxiliary_loss
    loss.backward()
    errors = {"full": 0., "sparse": 0.}

    def replay(data, ids, parts, states, derivatives, name):
        for index, start in enumerate(range(0, len(data), chunk)):
            restore(states[index])
            value, central = inputs(data, ids, start)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                output = model(value, central)
            if verify_replay:
                error = float((output.detach()-parts[index]).abs().max())
                errors[name] = max(errors[name], error)
                if error > 1e-6:
                    raise AssertionError(f"{name} cache/replay mismatch: {error}")
            output.backward(derivatives[start:start+chunk])

    try:
        # Full candidates receive derivatives from BOTH objectives. Replay is
        # connected through the exact chain rule, not a frozen teacher bank.
        replay(events, full_ids, full_parts, full_states, full.grad.detach(), "full")
        replay(views, query_ids, sparse_parts, sparse_states, queries.grad.detach(), "sparse")
    finally:
        restore(outer_rng)
    return float(loss), {"full": float(complete_loss), "sparse": float(auxiliary_loss), "weight": weight}, errors
