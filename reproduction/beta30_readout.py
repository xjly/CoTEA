"""Evaluate the BETA-30% full model and the five paper decoder ablations."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .benchmark_retrieval import balance_inplace
from .beta_legacy_time_model import make_model
from .beta_seed30_train import configure, load_inputs
from .data import save_json
from .root_mask_training import encode_root_mask
from .train import average_ranks, metrics


def encode_with_context(model, roots, context_events, chunk=128):
    model.eval()
    device = next(model.parameters()).device

    def encode(events, indices, function):
        batches = []
        for start in range(0, len(events), chunk):
            values = torch.as_tensor(events[start:start + chunk], device=device)
            ids = torch.as_tensor(indices[start:start + chunk], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                batches.append(function(values, ids))
        return torch.cat(batches)

    bank, indices = model.encoder.context_inputs(context_events)
    context = encode(bank, indices, model.encoder.base)
    return encode(roots, np.arange(len(roots)), lambda values, ids: model(values, ids, context))


def year_view(model, events, parts):
    first_token = {}
    year_token = np.zeros(len(parts), dtype=np.int32)
    for token, component in enumerate(parts[1:], start=1):
        year_token[token] = first_token.setdefault(component[0], token)
    year_events = events.copy()
    year_events[:, :, 2] = year_token[events[:, :, 2]]
    year_events[:, :, 4] = year_token[events[:, :, 4]]

    components = model.encoder.times.components
    original = components.clone()
    try:
        components[1:, 1:3] = 0
        components[1:, 3] = 1
        return encode_with_context(model, year_events, year_events)
    finally:
        components.copy_(original)


def cosine_blocks(vectors, pairs):
    left = vectors[pairs[:, 0]].float()
    right = vectors[pairs[:, 1]].float()
    return lambda start, end: left[start:end] @ right.T


def raw_fusion(full, untimed, left_timed, right_timed):
    def scores(start, end):
        both = left_timed[start:end, None] & right_timed[None, :]
        struct = untimed(start, end)
        return torch.where(both, (full(start, end) + .1 * struct) / 1.1, struct)
    return scores


def csls_matrix(scores, count, device, score_block=512):
    k = 5
    rows = torch.empty(count, device=device, dtype=torch.float64)
    columns = torch.full((count, k), -torch.inf, device=device, dtype=torch.float32)
    for start in range(0, count, 512):
        end = min(start + 512, count)
        raw = scores(start, end)
        rows[start:end] = raw.topk(k, dim=1).values.double().mean(1)
        local = raw.topk(min(k, len(raw)), dim=0).values.T
        columns = torch.cat([columns, local], dim=1).topk(k, dim=1).values
    right_density = columns.double().mean(1)
    matrix = torch.empty((count, count), device=device, dtype=torch.float64)
    for start in range(0, count, score_block):
        end = min(start + score_block, count)
        matrix[start:end] = 2 * scores(start, end).double() - rows[start:end, None] - right_density
    return matrix


def sinkhorn_ranks(matrix):
    u, v = balance_inplace(matrix, temperature=.05, iterations=200, block=256)
    ranks = []
    for start in range(0, len(matrix), 64):
        end = min(start + 64, len(matrix))
        scores = matrix[start:end] + u[start:end, None] + v[None, :]
        gold = torch.arange(start, end, device=matrix.device)
        ranks.append(average_ranks(scores, gold).cpu().numpy())
    return np.concatenate(ranks).astype(np.float64)


def calibrated_gate(full, untimed, left_timed, right_timed, year=None, quantile=False):
    matrix = full.clone()
    for start in range(0, len(full), 256):
        end = min(start + 256, len(full))
        rows = left_timed[start:end, None]
        cols = right_timed[None, :]
        base = full[start:end]
        struct = untimed[start:end]
        if quantile:
            for source_timed, target_timed in ((rows[:, 0], ~cols[0]), (~rows[:, 0], cols[0])):
                source = torch.nonzero(source_timed, as_tuple=True)[0]
                target = torch.nonzero(target_timed, as_tuple=True)[0]
                index = (source[:, None], target[None, :])
                ordered = base[index].sort(dim=1).values
                guided = struct[index].argsort(dim=1, stable=True)
                matrix[start:end][index] = torch.empty_like(ordered).scatter_(1, guided, ordered)
        else:
            beta = .7
            matrix[start:end] = torch.where(rows ^ cols, (1.0 - beta) * base + beta * struct, base)
            if year is None:
                both = (base + .1 * struct) / 1.1
            else:
                both = (base + .5 * year[start:end] + .1 * struct) / 1.6
            matrix[start:end] = torch.where(rows & cols, both, matrix[start:end])
    return matrix


@torch.inference_mode()
def evaluate(prepared, checkpoint_path):
    cache, protocol, events, train, profiles, coordinates = load_inputs(prepared)
    with np.load(cache / 'splits.npz') as splits:
        pairs = splits['reference']
    configure(seed=2026)
    model = make_model(protocol, events, train, profiles, coordinates, 'no_period').cuda()
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved['step'] != 375 or saved['seed'] != 2026 or not saved['context_enabled']:
        raise ValueError('Expected the paper checkpoint from seed 2026 at step 375')
    model.load_state_dict(saved['model'], strict=True)
    model.encoder.use_neighbor = True
    del saved

    availability = ((events[:, :, 0] > 0) & ((events[:, :, 2] > 0) | (events[:, :, 4] > 0))).any(1)
    timed_sides = availability[pairs].sum(axis=1)
    groups = {'both_timed': timed_sides == 2,
              'only_one_timed': timed_sides == 1,
              'neither_timed': timed_sides == 0}
    device = torch.device('cuda')
    left_timed = torch.as_tensor(availability[pairs[:, 0]], device=device)
    right_timed = torch.as_tensor(availability[pairs[:, 1]], device=device)

    no_time = events.copy()
    no_time[:, :, 2] = 0
    no_time[:, :, 4] = 0
    full = cosine_blocks(encode_root_mask(model, events), pairs)
    global_untimed = cosine_blocks(encode_root_mask(model, no_time), pairs)
    root_untimed = cosine_blocks(encode_with_context(model, no_time, events), pairs)
    year = cosine_blocks(year_view(model, events, protocol['ymd_components']), pairs)
    del model

    def score_matrix(scores, block=512):
        return csls_matrix(scores, len(pairs), device, score_block=block)

    results = {}

    def record(name, matrix):
        ranks = sinkhorn_ranks(matrix)
        results[name] = {'overall': metrics(ranks),
                         'groups': {group: metrics(ranks[mask]) for group, mask in groups.items()}}
        print(f"{name}: MRR={100 * results[name]['overall']['mrr']:.2f}%", flush=True)

    record('global_untimed_raw_fusion', score_matrix(raw_fusion(full, global_untimed, left_timed, right_timed), 256))
    record('root_untimed_raw_fusion', score_matrix(raw_fusion(full, root_untimed, left_timed, right_timed), 256))
    record('single_full_view', score_matrix(full, 256))
    cf, cn = score_matrix(full), score_matrix(root_untimed)
    record('in_view_csls_quantile', calibrated_gate(cf, cn, left_timed, right_timed, quantile=True))
    record('in_view_csls_tri_state', calibrated_gate(cf, cn, left_timed, right_timed))
    cy = score_matrix(year)
    record('in_view_csls_year_tri_state', calibrated_gate(cf, cn, left_timed, right_timed, year=cy))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('BETA-30% evaluation requires CUDA')
    result = evaluate(args.prepared, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == '__main__':
    main()
