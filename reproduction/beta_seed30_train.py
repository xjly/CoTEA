"""Train the BETA-30% event model through the paper's step-375 checkpoint."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from .beta_legacy_time_model import make_model
from .chain_only import configure_attention
from .full_context_training import cached_complete_backward
from .priority_hybrid_train import checkpoint, mask_one_side_time


def configure(seed):
    torch.set_num_threads(4)
    configure_attention()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_inputs(prepared):
    cache = prepared / 'cache' / 'priority'
    protocol = json.loads((cache / 'protocol.json').read_text(encoding='utf-8'))
    events = np.load(cache / 'events.npz')['events']
    with np.load(cache / 'splits.npz') as splits:
        train = splits['train']
    profiles = np.load(prepared / 'profiles' / 'seed_relation_counts.npy')
    coordinates = np.load(cache / 'time_coordinates.npy')
    assert train.shape == (12109, 2) and events.shape == (84963, 64, 5)
    return cache, protocol, events, train, profiles, coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('BETA-30% training requires a CUDA GPU with BF16 support')

    _, protocol, events, train, profiles, coordinates = load_inputs(args.prepared)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)

    seed = 2026
    configure(seed)
    model = make_model(protocol, events, train, profiles, coordinates, 'no_period').cuda()
    context_parameters = model.encoder.extra_parameters()
    context_ids = {id(parameter) for parameter in context_parameters}
    backbone = [parameter for parameter in model.parameters()
                if parameter.requires_grad and id(parameter) not in context_ids]
    model.encoder.use_neighbor = False
    for parameter in context_parameters:
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(backbone, lr=1e-4, weight_decay=1e-4)

    for step in range(1, 376):
        if step == 201:
            model.encoder.use_neighbor = True
            for parameter in context_parameters:
                parameter.requires_grad_(True)
            optimizer.add_param_group({'params': context_parameters, 'lr': 5e-4})
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_events, _ = mask_one_side_time(events, train, .2, seed, step)
        loss, _, _ = cached_complete_backward(
            model, training_events, train, chunk=128,
            verify_replay=step in (1, 2, 200, 201, 202),
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(f'Step {step:3d}/375, loss={loss:.6f}', flush=True)

    checkpoint(args.output, model, optimizer, 375, seed, context=True)


if __name__ == '__main__':
    main()
