"""Exact two-stage chain rule for learned one-hop neighbors; no stale bank."""
from __future__ import annotations

import numpy as np
import torch

from .chain_only import full_candidate_loss


def tensor_chunk(events, start, chunk, device):
    ids = torch.arange(start, min(start + chunk, len(events)), device=device)
    value = torch.as_tensor(events[start:start + chunk], dtype=torch.long, device=device)
    return value, ids


def needs_context(model):
    return model.encoder.use_neighbor and model.encoder.intervention != "disable_context"


@torch.inference_mode()
def encode_all(model, events, chunk=128, amp=True):
    model.eval()
    device = next(model.parameters()).device
    context = None
    if needs_context(model):
        parts = []
        for start in range(0, len(events), chunk):
            value, ids = tensor_chunk(events, start, chunk, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                parts.append(model.encoder.base(value, ids))
        context = torch.cat(parts)
    parts = []
    for start in range(0, len(events), chunk):
        value, ids = tensor_chunk(events, start, chunk, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            parts.append(model(value, ids, context))
    return torch.cat(parts)


def cached_full_backward(model, events, train, chunk=128, amp=True, verify_replay=False):
    """Exact dL/dtheta = direct center derivative + context-encoder derivative.

    Full-candidate loss yields derivatives for every final entity vector. Center
    replay accumulates both parameter gradients and the gradients of the current
    context leaf bank. Base replay then backpropagates all context gradients into
    the shared base encoder. The temporary leaf is a chain-rule boundary, never
    a frozen teacher or a stop-gradient training approximation.
    """
    device = next(model.parameters()).device

    def rng_state():
        return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)

    def restore(state):
        torch.set_rng_state(state[0])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state[1], device)

    def cache(function):
        parts, states = [], []
        for start in range(0, len(events), chunk):
            states.append(rng_state())
            value, ids = tensor_chunk(events, start, chunk, device)
            with torch.enable_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                    enabled=amp and device.type == "cuda"):
                output = function(value, ids)
            parts.append(output.detach())
            del output
        return parts, states

    context, base_parts, base_states = None, None, None
    if needs_context(model):
        base_parts, base_states = cache(model.encoder.base)
        context = torch.cat(base_parts).detach().requires_grad_(True)
    final_parts, final_states = cache(lambda value, ids: model(value, ids, context))
    outer_rng = rng_state()
    embedding = torch.cat(final_parts).detach().requires_grad_(True)
    loss = full_candidate_loss(embedding, train, model.n_source)
    loss.backward()
    gradients = embedding.grad.detach()
    errors = {"final": 0., "context": 0. if context is not None else None}

    def replay(function, parts, states, derivatives, name):
        for index, start in enumerate(range(0, len(events), chunk)):
            restore(states[index])
            value, ids = tensor_chunk(events, start, chunk, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                output = function(value, ids)
            if verify_replay:
                error = float((output.detach() - parts[index]).abs().max())
                errors[name] = max(errors[name], error)
                if error > 1e-6:
                    raise AssertionError(f"{name} cache/replay mismatch: {error}")
            output.backward(derivatives[ids])

    try:
        replay(lambda value, ids: model(value, ids, context), final_parts, final_states, gradients, "final")
        if context is not None:
            if context.grad is None:
                raise AssertionError("Missing context derivatives: a stop-gradient path was introduced")
            context_gradient = context.grad.detach()
            replay(model.encoder.base, base_parts, base_states, context_gradient, "context")
    finally:
        restore(outer_rng)
    return float(loss), errors
