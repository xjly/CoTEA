"""Train and evaluate the CoTEA experimental configurations on WUXI-TEMP."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT_EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_EXP))

from reproduction.beta_legacy_time_model import make_model as make_legacy_model
from reproduction.benchmark_retrieval import balance_inplace
from reproduction.chain_only import configure_attention
from reproduction.data import save_json, sha256
from reproduction.full_context_training import cached_complete_backward
from reproduction.priority_hybrid_train import checkpoint, mask_one_side_time
from reproduction.root_mask_training import encode_root_mask
from reproduction.train import average_ranks


class CleanHierarchicalTime(nn.Module):
    """Hierarchical (year) + (year, month) + (year, month, day) frozen embeddings."""
    def __init__(self, calendar, w_y=0.5, w_ym=1.0, w_ymd=1.0, seed=732029):
        super().__init__()
        parts = calendar.components.detach().cpu().numpy()
        self.embedding_dim = calendar.embedding_dim
        self.num_embeddings = calendar.num_embeddings
        self.register_buffer('components', calendar.components.clone())
        self.register_buffer('year_index', calendar.components[:, 0].clone())
        ym = sorted({tuple(v[:2]) for v in parts if v[0] > 0 and v[1] > 0})
        ymd = sorted({tuple(v[:3]) for v in parts if v[0] > 0 and v[1] > 0 and v[2] > 0})
        ym_ids = {key: i + 1 for i, key in enumerate(ym)}
        ymd_ids = {key: i + 1 for i, key in enumerate(ymd)}
        self.register_buffer('ym_index', torch.tensor([ym_ids.get(tuple(v[:2]), 0) for v in parts], dtype=torch.long))
        self.register_buffer('ymd_index', torch.tensor([ymd_ids.get(tuple(v[:3]), 0) for v in parts], dtype=torch.long))
        self.w_y, self.w_ym, self.w_ymd = float(w_y), float(w_ym), float(w_ymd)
        self.drop_day = False
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.year_emb = nn.Embedding(int(parts[:, 0].max()) + 1, self.embedding_dim, padding_idx=0)
            self.ym_emb = nn.Embedding(len(ym) + 1, self.embedding_dim, padding_idx=0)
            self.ymd_emb = nn.Embedding(len(ymd) + 1, self.embedding_dim, padding_idx=0)
        self.requires_grad_(False)

    def forward(self, ids):
        yi = self.year_index[ids]
        ymi = self.ym_index[ids]
        ymdi = torch.zeros_like(ymi) if self.drop_day else self.ymd_index[ids]
        raw = self.w_y * self.year_emb(yi) + self.w_ym * self.ym_emb(ymi) + self.w_ymd * self.ymd_emb(ymdi)
        norm_sq = (
            (yi > 0).float() * (self.w_y ** 2)
            + (ymi > 0).float() * (self.w_ym ** 2)
            + (ymdi > 0).float() * (self.w_ymd ** 2)
        ).clamp_min(1.0)
        return raw / norm_sq.sqrt().to(raw.dtype)[..., None]


class OverflowHubObjects(nn.Module):
    """Leave-one-out untruncated seed-incidence representation for overflow hub objects (lengths > 64)."""
    def __init__(self, original_objects, relations_ref, obj_counts, lengths, dim=512, cap=64, seed=11732029):
        super().__init__()
        self.original = original_objects
        self.register_buffer('alias', original_objects.alias.clone())
        object.__setattr__(self, 'relations_ref', relations_ref)
        self.dim = dim
        self.entities = len(lengths)
        is_overflow = torch.tensor(np.asarray(lengths) > cap, dtype=torch.bool)
        self.register_buffer('is_overflow', is_overflow)
        coeff = torch.tensor(np.log1p(obj_counts), dtype=torch.float32)
        self.register_buffer('coefficient', coeff)
        self.register_buffer('support', (coeff > 0).sum((1, 2)))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            proj = torch.randn(2, coeff.shape[2], dim)
        self.register_buffer('projection', proj)
        self.register_buffer('totals', torch.einsum('oks,ksd->od', coeff, proj))
        self.requires_grad_(False)

    def forward(self, ids):
        base_out = self.original(ids)
        in_range = (ids >= 1) & (ids <= self.entities)
        if not in_range.any():
            return base_out
        obj_idx = (ids - 1).clamp(0, self.entities - 1)
        use_hub = in_range & (self.alias[ids] == 0) & self.is_overflow[obj_idx]
        if not use_hub.any():
            return base_out
        profile = self.totals[obj_idx]
        supp = self.support[obj_idx]
        excluded = self.relations_ref.excluded
        if excluded is not None:
            j = excluded.clamp_min(0)
            contrib = (self.coefficient[obj_idx, :, j][:, :, None] * self.projection[:, j].permute(1, 0, 2)).sum(1)
            has_ex = excluded >= 0
            profile = profile - contrib * has_ex[:, None]
            supp = supp - (self.coefficient[obj_idx, :, j] > 0).sum(1) * has_ex
        active = use_hub & (supp > 0)
        if not active.any():
            return base_out
        hub_vec = F.normalize(profile, dim=-1, eps=1e-6) * math.sqrt(self.dim)
        return torch.where(active[:, None], hub_vec.to(base_out.dtype), base_out)


def build_month_coarsened_events(events, protocol):
    """Map day-precision tokens YYYY.MMDD to their corresponding YYYY.MM00 token if present."""
    labels = protocol['time_labels']
    label_to_tok = {v: int(k) for k, v in labels.items()}
    tok_to_month_tok = np.arange(len(labels) + 1, dtype=np.int32)
    for k, v in labels.items():
        tok = int(k)
        if not v.endswith('00'):
            m_label = v[:7] + '00'
            if m_label in label_to_tok:
                tok_to_month_tok[tok] = label_to_tok[m_label]
    coarse = events.copy()
    coarse[:, :, 2] = tok_to_month_tok[coarse[:, :, 2]]
    coarse[:, :, 4] = tok_to_month_tok[coarse[:, :, 4]]
    return coarse, tok_to_month_tok


def mask_or_coarsen_time(events, train, mask_frac, coarse_frac, tok_to_month_tok, seed, step):
    view, sides = mask_one_side_time(events, train, mask_frac, seed, step)
    if coarse_frac <= 0:
        return view, sides
    rng = np.random.default_rng(seed + 100000 * step + 37)
    count = round(len(train) * coarse_frac)
    chosen = rng.choice(len(train), size=count, replace=False)
    side_choice = rng.integers(0, 2, size=count)
    for pair_idx, s in zip(chosen, side_choice):
        ent = int(train[pair_idx, s])
        view[ent, :, 2] = tok_to_month_tok[view[ent, :, 2]]
        view[ent, :, 4] = tok_to_month_tok[view[ent, :, 4]]
    return view, sides


@torch.inference_mode()
def encode_custom_root(model, base_events, root_events, chunk=128, drop_day=False):
    model.eval()
    device = next(model.parameters()).device

    def encode(data, ids, fn):
        parts = []
        for start in range(0, len(data), chunk):
            val = torch.as_tensor(data[start:start + chunk], device=device)
            cid = torch.as_tensor(ids[start:start + chunk], device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                parts.append(fn(val, cid))
        return torch.cat(parts)

    bank, ids = model.encoder.context_inputs(base_events)
    context = encode(bank, ids, model.encoder.base)
    if hasattr(model.encoder.times, 'drop_day'):
        model.encoder.times.drop_day = drop_day
    try:
        out = encode(root_events, np.arange(len(base_events)), lambda val, cid: model(val, cid, context))
    finally:
        if hasattr(model.encoder.times, 'drop_day'):
            model.encoder.times.drop_day = False
    return out.cpu().numpy()


@torch.inference_mode()
def csls_matrix(left, right, k=5, block=512):
    nr, nc = len(left), len(right)
    device = left.device
    rx = torch.empty(nr, device=device, dtype=torch.float64)
    cols = torch.full((nc, k), -torch.inf, device=device, dtype=torch.float32)
    for start in range(0, nr, block):
        raw = left[start:start + block].float() @ right.float().T
        rx[start:start + len(raw)] = raw.topk(k, dim=1).values.double().mean(1)
        local = raw.topk(min(k, len(raw)), dim=0).values.T
        cols = torch.cat([cols, local], 1).topk(k, dim=1).values
    ry = cols.double().mean(1)
    out = torch.empty((nr, nc), device=device, dtype=torch.float64)
    for start in range(0, nr, block):
        raw = left[start:start + block].float() @ right.float().T
        out[start:start + len(raw)] = 2.0 * raw.double() - rx[start:start + len(raw), None] - ry[None, :]
    return out


@torch.inference_mode()
def eval_pools(csls_n_by_m, reference, candidates_eval, candidates_all, n, total_e, groups, *, sinkhorn=True):
    def eval_one_pool(pool_rows, pool_cols):
        sub = csls_n_by_m[pool_rows][:, pool_cols - n].clone()
        if sinkhorn:
            u, v = balance_inplace(sub, temperature=0.05, iterations=200, block=256)
            sub.add_(u[:, None]).add_(v[None, :])
        rowmap = np.full(total_e, -1, dtype=np.int64)
        rowmap[pool_rows] = np.arange(len(pool_rows))
        colmap = np.full(total_e, -1, dtype=np.int64)
        colmap[pool_cols] = np.arange(len(pool_cols))
        q = torch.as_tensor(rowmap[reference[:, 0]], device='cuda')
        g = torch.as_tensor(colmap[reference[:, 1]], device='cuda')
        r = average_ranks(sub[q], g).cpu().numpy()

        def m_fn(arr):
            return dict(
                mrr=float((1.0 / arr).mean()),
                hits1=float((arr <= 1.0).mean()),
                hits10=float((arr <= 10.0).mean()),
                queries=int(len(arr)),
            )
        return dict(
            overall=m_fn(r),
            groups={k: m_fn(r[mask]) for k, mask in groups.items()},
        )

    return dict(
        eval_5483=eval_one_pool(reference[:, 0], candidates_eval),
        closed_3872=eval_one_pool(reference[:, 0], reference[:, 1]),
        full_6483=eval_one_pool(np.arange(n), candidates_all),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['no_period', 'hier_time', 'hier_time_hub'], required=True)
    parser.add_argument('--gpu', help='CUDA device identifier; defaults to the first visible GPU.')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be at least 1')
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    suite_dir = args.output_dir
    checkpoint_steps = sorted({step for step in (200, 300, 350, args.steps) if step <= args.steps})
    cache = suite_dir / 'cache/priority'
    profiles = suite_dir / 'profiles/seed_relation_counts.npy'
    obj_profiles = suite_dir / 'profiles/seed_object_counts.npy'
    out = suite_dir / f'runs/{args.arm}'
    out.mkdir(parents=True, exist_ok=True)

    protocol = json.loads((cache / 'protocol.json').read_text(encoding='utf-8'))
    n, m = protocol['entities']
    total_e = n + m
    with np.load(cache / 'events.npz') as ev_npz:
        events = ev_npz['events']
        lengths = ev_npz['lengths']
    with np.load(suite_dir / 'reference/splits.npz') as splits:
        train = splits['train']
        reference = splits['reference']
        candidates_eval = splits['candidates_eval']
        candidates_all = splits['candidates_all']
    groups = dict(np.load(suite_dir / 'wuxi_groups.npz'))
    coarse_events, tok_to_month_tok = build_month_coarsened_events(events, protocol)
    no_time_events = events.copy()
    no_time_events[:, :, 2] = 0
    no_time_events[:, :, 4] = 0

    torch.set_num_threads(4)
    configure_attention()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    model = make_legacy_model(
        protocol, events, train,
        np.load(profiles),
        np.load(cache / 'time_coordinates.npy'),
        'no_period',
    )
    if args.arm in ('hier_time', 'hier_time_hub'):
        model.encoder.times = CleanHierarchicalTime(model.encoder.times)
    if args.arm == 'hier_time_hub':
        model.encoder.objects = OverflowHubObjects(
            model.encoder.objects,
            model.encoder.relations,
            np.load(obj_profiles),
            lengths,
            dim=512,
            cap=64,
        )
    model = model.cuda()

    extra = model.encoder.extra_parameters()
    extra_ids = {id(v) for v in extra}
    backbone = [v for v in model.parameters() if v.requires_grad and id(v) not in extra_ids]
    model.encoder.use_neighbor = False
    for v in extra:
        v.requires_grad_(False)
    optimizer = torch.optim.AdamW(backbone, lr=1e-4, weight_decay=1e-4)

    started = time.perf_counter()
    context = False
    coarse_frac = 0.2 if args.arm in ('hier_time', 'hier_time_hub') else 0.0

    with (out / 'epochs.jsonl').open('w', buffering=1) as log:
        for step in range(1, args.steps + 1):
            tick = time.perf_counter()
            if step == 201 and not context:
                model.encoder.use_neighbor = True
                for v in extra:
                    v.requires_grad_(True)
                optimizer.add_param_group(dict(params=extra, lr=5e-4, weight_decay=1e-4))
                context = True

            model.train()
            optimizer.zero_grad(set_to_none=True)
            view, sides = mask_or_coarsen_time(events, train, 0.2, coarse_frac, tok_to_month_tok, 2026, step)
            loss, parts, _ = cached_complete_backward(model, view, train, chunk=128, verify_replay=(step in (1, 200, 201, args.steps)))
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            optimizer.step()

            row = dict(
                step=step,
                arm=args.arm,
                context_enabled=context,
                loss=float(loss),
                gradient_norm=grad_norm,
                update_seconds=time.perf_counter() - tick,
                elapsed_seconds=time.perf_counter() - started,
            )
            if step in checkpoint_steps:
                checkpoint(out / f'step{step}_training.pt', model, optimizer, step, 2026, context)
            if step % 25 == 0 or step in (1, 200, 201, args.steps):
                log.write(json.dumps(row) + '\n')
                save_json(out / 'status.json', dict(state='training', **row))
                print(json.dumps(row), flush=True)

    # Score the stage boundary, saved intermediate checkpoints, and final step.
    eval_summary = {}
    for ckpt_step in checkpoint_steps:
        saved = torch.load(out / f'step{ckpt_step}_training.pt', map_location='cuda', weights_only=True)
        model.load_state_dict(saved['model'], strict=True)
        model.encoder.use_neighbor = bool(saved['context_enabled'])
        del saved

        z_full = encode_root_mask(model, events, 128).cpu().numpy()
        z_notime = encode_custom_root(model, events, no_time_events, 128, drop_day=False)
        z_coarse = encode_custom_root(model, events, coarse_events, 128, drop_day=True)
        if ckpt_step == args.steps:
            np.save(out / 'final_full_vectors.npy', z_full)
            np.save(out / 'final_notime_vectors.npy', z_notime)
            np.save(out / 'final_coarse_vectors.npy', z_coarse)

        fl = torch.from_numpy(z_full[:n]).cuda()
        fr = torch.from_numpy(z_full[n:]).cuda()
        sl = torch.from_numpy(z_notime[:n]).cuda()
        sr = torch.from_numpy(z_notime[n:]).cuda()
        cl = torch.from_numpy(z_coarse[:n]).cuda()
        cr = torch.from_numpy(z_coarse[n:]).cuda()

        cf = csls_matrix(fl, fr, k=5)
        cs = csls_matrix(sl, sr, k=5)
        cc = csls_matrix(cl, cr, k=5)

        step_res = {
            'single_full': eval_pools(cf, reference, candidates_eval, candidates_all, n, total_e, groups),
            'dual_full_notime_015': eval_pools((cf + 0.15 * cs) / 1.15, reference, candidates_eval, candidates_all, n, total_e, groups),
            'tri_full_coarse_notime': eval_pools((cf + 0.35 * cc + 0.15 * cs) / 1.50, reference, candidates_eval, candidates_all, n, total_e, groups),
        }
        eval_summary[f'step_{ckpt_step}'] = step_res
        del fl, fr, sl, sr, cl, cr, cf, cs, cc
        gc.collect()
        torch.cuda.empty_cache()

    final_payload = dict(
        state='completed',
        arm=args.arm,
        total_seconds=time.perf_counter() - started,
        evaluations=eval_summary,
    )
    save_json(out / 'results_wuxi.json', final_payload)
    save_json(out / 'status.json', dict(state='completed', arm=args.arm, total_seconds=final_payload['total_seconds']))
    print('FINISHED ARM:', args.arm, json.dumps(eval_summary[f'step_{args.steps}']['tri_full_coarse_notime']['eval_5483']['overall']))


if __name__ == '__main__':
    main()
