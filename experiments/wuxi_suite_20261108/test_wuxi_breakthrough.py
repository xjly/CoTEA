"""Test the exact structural + multi-granularity temporal decomposition on WUXI-TEMP."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT_EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_EXP))

from reproduction.benchmark_retrieval import balance_inplace
from reproduction.data import read_vocab, save_json
from reproduction.train import average_ranks
from train_wuxi_suite import (
    CleanHierarchicalTime,
    build_month_coarsened_events,
    csls_matrix,
    encode_custom_root,
    eval_pools,
)
from reproduction.beta_legacy_time_model import make_model as make_legacy_model
from reproduction.chain_only import configure_attention
from reproduction.root_mask_training import encode_root_mask


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description='Evaluate the WUXI-TEMP event and topology streams.')
    parser.add_argument('--data-dir', type=Path, default=ROOT_EXP / 'datasets/WUXI-TEMP')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--checkpoint-step', type=int, default=350)
    parser.add_argument('--no-sinkhorn', action='store_true', help='Rank the fused CSLS scores directly.')
    args = parser.parse_args()
    raw_dir, suite_dir = args.data_dir, args.output_dir
    cache = suite_dir / 'cache/priority'
    profiles = suite_dir / 'profiles/seed_relation_counts.npy'
    obj_profiles = suite_dir / 'profiles/seed_object_counts.npy'

    protocol = json.loads((cache / 'protocol.json').read_text(encoding='utf-8'))
    n, m = protocol['entities']
    total_e = n + m
    total_r = protocol['shared_relations']

    with np.load(cache / 'events.npz') as ev_npz:
        events = ev_npz['events']
        lengths = ev_npz['lengths']
    with np.load(suite_dir / 'reference/splits.npz') as splits:
        train = splits['train']
        reference = splits['reference']
        candidates_eval = splits['candidates_eval']
        candidates_all = splits['candidates_all']
    groups = dict(np.load(suite_dir / 'wuxi_groups.npz'))

    # 1. Load the hierarchical calendar checkpoint and extract neural event vectors.
    configure_attention()
    torch.manual_seed(2026)
    model = make_legacy_model(
        protocol, events, train,
        np.load(profiles),
        np.load(cache / 'time_coordinates.npy'),
        'no_period',
    )
    model.encoder.times = CleanHierarchicalTime(model.encoder.times)
    model = model.cuda()
    ckpt_path = suite_dir / f'runs/hier_time/step{args.checkpoint_step}_training.pt'
    saved = torch.load(ckpt_path, map_location='cuda', weights_only=True)
    model.load_state_dict(saved['model'], strict=True)
    model.encoder.use_neighbor = bool(saved['context_enabled'])
    del saved

    coarse_events, _ = build_month_coarsened_events(events, protocol)
    no_time_events = events.copy()
    no_time_events[:, :, 2] = 0
    no_time_events[:, :, 4] = 0

    z_full = encode_root_mask(model, events, 128).cpu().numpy()
    z_notime = encode_custom_root(model, events, no_time_events, 128, drop_day=False)
    z_coarse = encode_custom_root(model, events, coarse_events, 128, drop_day=True)
    del model
    torch.cuda.empty_cache()

    fl = torch.from_numpy(z_full[:n]).cuda()
    fr = torch.from_numpy(z_full[n:]).cuda()
    sl = torch.from_numpy(z_notime[:n]).cuda()
    sr = torch.from_numpy(z_notime[n:]).cuda()
    cl = torch.from_numpy(z_coarse[:n]).cuda()
    cr = torch.from_numpy(z_coarse[n:]).cuda()

    csls_neural_full = csls_matrix(fl, fr, k=5)
    csls_neural_notime = csls_matrix(sl, sr, k=5)
    csls_neural_coarse = csls_matrix(cl, cr, k=5)
    csls_neural_tri = (csls_neural_full + 0.35 * csls_neural_coarse + 0.15 * csls_neural_notime) / 1.50

    # 2. Build Physical Topology Propagation (excluding overflow state-bucket hubs deg > 64)
    ev = [read_vocab(raw_dir / f'ent_ids_{i}') for i in (1, 2)]
    raw_ids = [sorted(v) for v in ev]
    indices = [{v: j + off for j, v in enumerate(ids)} for ids, off in zip(raw_ids, (0, n))]
    rv = [read_vocab(raw_dir / f'rel_ids_{i}') for i in (1, 2)]
    ri = {label: i for i, label in enumerate(protocol['relation_names'])}

    is_hub = lengths > 64
    topo_edges = Counter()
    all_idf_edges = Counter()
    node_deg = np.maximum(lengths.astype(np.float64), 1.0)

    # Also collect per-entity multi-granularity (relation/hub, time) event signatures
    rel_counts = np.load(profiles)  # (total_r, 2, 1000)
    obj_counts = np.load(obj_profiles)  # (total_e, 2, 1000)

    # Compute cross-graph aligned relation vectors and hub vectors from seed incidence alone
    gen = torch.Generator(device='cuda').manual_seed(11732026)
    rel_proj = torch.randn(2, len(train), 256, generator=gen, device='cuda')
    rel_coeff = torch.tensor(np.log1p(rel_counts), dtype=torch.float32, device='cuda')
    rel_vec = F.normalize(torch.einsum('rks,ksd->rd', rel_coeff, rel_proj), dim=-1)  # (total_r, 256)

    obj_coeff = torch.tensor(np.log1p(obj_counts), dtype=torch.float32, device='cuda')
    hub_vec = F.normalize(torch.einsum('oks,ksd->od', obj_coeff, rel_proj), dim=-1)  # (total_e, 256)

    # Build time vocabulary mappings for day, year-month, and year
    time_labels = protocol['time_labels']
    ym_vocab = {}
    y_vocab = {}
    tok_to_ym = np.zeros(len(time_labels) + 1, dtype=np.int64)
    tok_to_y = np.zeros(len(time_labels) + 1, dtype=np.int64)
    tok_is_day = np.zeros(len(time_labels) + 1, dtype=bool)
    for k_str, lbl in time_labels.items():
        tok = int(k_str)
        y_str = lbl[:4]
        if y_str not in y_vocab:
            y_vocab[y_str] = len(y_vocab) + 1
        tok_to_y[tok] = y_vocab[y_str]
        if lbl[5:7] != '00':
            ym_str = lbl[:7]
            if ym_str not in ym_vocab:
                ym_vocab[ym_str] = len(ym_vocab) + 1
            tok_to_ym[tok] = ym_vocab[ym_str]
        if not lbl.endswith('00'):
            tok_is_day[tok] = True

    ti = {lbl: int(k) for k, lbl in time_labels.items()}
    ti['~'] = 0

    for g in range(2):
        for line in (raw_dir / f'triples_{g + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h_raw, r_raw, t_raw, start, end = line.split('\t')
            h = indices[g][int(h_raw)]
            t = indices[g][int(t_raw)]
            if not is_hub[h] and not is_hub[t]:
                topo_edges[(h, t)] += 1.0
                topo_edges[(t, h)] += 1.0
            # IDF-weighted edge (downweights hubs by 1 / node_deg)
            all_idf_edges[(h, t)] += 1.0 / math.sqrt(node_deg[h] * node_deg[t])
            all_idf_edges[(t, h)] += 1.0 / math.sqrt(node_deg[h] * node_deg[t])

    def build_sparse(edges_cnt):
        rows, cols, vals = [], [], []
        rsum = defaultdict(float)
        for (u, v), w in edges_cnt.items():
            rows.append(u)
            cols.append(v)
            vals.append(w)
            rsum[u] += w
        vals = [w / max(rsum[u], 1e-12) for u, w in zip(rows, vals)]
        idx = torch.tensor([rows, cols], dtype=torch.long, device='cuda')
        vt = torch.tensor(vals, dtype=torch.float32, device='cuda')
        return torch.sparse_coo_tensor(idx, vt, size=(total_e, total_e), device='cuda').coalesce()

    A_topo = build_sparse(topo_edges)
    A_idf = build_sparse(all_idf_edges)

    seed_vecs = F.normalize(torch.randn(len(train), 512, generator=gen, device='cuda'), dim=-1)
    X0 = torch.zeros(total_e, 512, device='cuda')
    for idx_s, (u, v) in enumerate(train):
        X0[int(u)] = seed_vecs[idx_s]
        X0[int(v)] = seed_vecs[idx_s]

    # Multi-hop propagation on non-hub physical topology (3 hops)
    topo_hops = []
    curr = X0
    for _ in range(3):
        curr = torch.sparse.mm(A_topo, curr)
        topo_hops.append(F.normalize(curr, dim=-1))
    Z_topo = F.normalize(topo_hops[0] + 0.7 * topo_hops[1] + 0.5 * topo_hops[2], dim=-1)
    csls_topo = csls_matrix(Z_topo[:n], Z_topo[n:], k=5)

    # Multi-hop propagation on IDF-weighted full graph (3 hops)
    idf_hops = []
    curr_i = X0
    for _ in range(3):
        curr_i = torch.sparse.mm(A_idf, curr_i)
        idf_hops.append(F.normalize(curr_i, dim=-1))
    Z_idf = F.normalize(idf_hops[0] + 0.7 * idf_hops[1] + 0.5 * idf_hops[2], dim=-1)
    csls_idf = csls_matrix(Z_idf[:n], Z_idf[n:], k=5)

    # 3. Coupled Intra-Event (HubState + Relation + Multi-Granularity Time) Random Hypervector Readout
    # For each event (o, r_dir, start), bind [rel_vec(r) + hub_vec(o)] with time_vec(day / ym / y) via circular/elementwise binding!
    d_bind = 512
    rel_bind = F.normalize(torch.einsum('rks,ksd->rd', rel_coeff, torch.randn(2, len(train), d_bind, generator=gen, device='cuda')), dim=-1)
    hub_bind = F.normalize(torch.einsum('oks,ksd->od', obj_coeff, torch.randn(2, len(train), d_bind, generator=gen, device='cuda')), dim=-1)
    dir_bind = F.normalize(torch.randn(2, d_bind, generator=gen, device='cuda'), dim=-1)

    t_day_vecs = F.normalize(torch.randn(len(time_labels) + 1, d_bind, generator=gen, device='cuda'), dim=-1)
    t_day_vecs[0] = 0.0
    t_ym_vecs = F.normalize(torch.randn(len(ym_vocab) + 1, d_bind, generator=gen, device='cuda'), dim=-1)
    t_ym_vecs[0] = 0.0
    t_y_vecs = F.normalize(torch.randn(len(y_vocab) + 1, d_bind, generator=gen, device='cuda'), dim=-1)
    t_y_vecs[0] = 0.0

    ev_t = torch.as_tensor(events, dtype=torch.long, device='cuda')
    valid = ev_t[:, :, 0] > 0
    obj_id = (ev_t[:, :, 0] - 1).clamp_min(0)
    rel_id = ((ev_t[:, :, 1] - 1) // 2).clamp_min(0)
    rel_dir = (ev_t[:, :, 1] - 1).clamp_min(0) % 2
    start_tok = ev_t[:, :, 2]
    freq_w = torch.log1p(ev_t[:, :, 3].float()) * valid.float()

    # Bound semantic-state vector per event: relation + direction + hub-object
    sem_state = F.normalize(rel_bind[rel_id] + 0.5 * dir_bind[rel_dir] + hub_bind[obj_id], dim=-1)

    # Multi-granularity temporal binding: exact day, year-month, year, and time-invariant state
    ym_tok = torch.as_tensor(tok_to_ym, device='cuda')[start_tok]
    y_tok = torch.as_tensor(tok_to_y, device='cuda')[start_tok]
    is_day_mask = torch.as_tensor(tok_is_day, device='cuda')[start_tok] & valid

    bound_day = ((sem_state * t_day_vecs[start_tok]) * (freq_w * is_day_mask.float())[:, :, None]).sum(1)
    bound_ym = ((sem_state * t_ym_vecs[ym_tok]) * (freq_w * (ym_tok > 0).float())[:, :, None]).sum(1)
    bound_y = ((sem_state * t_y_vecs[y_tok]) * (freq_w * (y_tok > 0).float())[:, :, None]).sum(1)
    bound_notime = (sem_state * freq_w[:, :, None]).sum(1)

    Z_coupled = F.normalize(
        1.0 * F.normalize(bound_day, dim=-1)
        + 0.7 * F.normalize(bound_ym, dim=-1)
        + 0.3 * F.normalize(bound_y, dim=-1)
        + 0.4 * F.normalize(bound_notime, dim=-1),
        dim=-1,
    )
    csls_coupled = csls_matrix(Z_coupled[:n], Z_coupled[n:], k=5)

    # Also propagate Z_coupled 1 hop over physical topology A_topo!
    Z_coupled_prop = F.normalize(Z_coupled + 0.5 * torch.sparse.mm(A_topo, Z_coupled), dim=-1)
    csls_coupled_prop = csls_matrix(Z_coupled_prop[:n], Z_coupled_prop[n:], k=5)

    variants = {
        f'1_neural_hier_step{args.checkpoint_step}_tri': csls_neural_tri,
        '2_topo_only_nonhub': csls_topo,
        '3_coupled_event_multigran': csls_coupled,
        '4_coupled_event_plus_topo1hop': csls_coupled_prop,
        '5_msea_full_synergy': csls_neural_tri + 0.8 * csls_coupled_prop + 0.6 * csls_topo + 0.4 * csls_idf,
    }

    summary = {}
    for name, mat in variants.items():
        res = eval_pools(mat, reference, candidates_eval, candidates_all, n, total_e, groups, sinkhorn=not args.no_sinkhorn)
        summary[name] = res
        ov = res['eval_5483']['overall']
        cl_ov = res['closed_3872']['overall']
        nd = res['eval_5483']['groups']['no_exact_day_overlap']
        nh = res['eval_5483']['groups']['at_least_one_no_1hop_seed']
        bh = res['eval_5483']['groups']['both_have_1hop_seed']
        print(
            f"{name:32s} | eval_5483 MRR={ov['mrr']*100:.2f}% H@1={ov['hits1']*100:.2f}% H@10={ov['hits10']*100:.2f}% | "
            f"closed MRR={cl_ov['mrr']*100:.2f}% | "
            f"1hop_seed={bh['mrr']*100:.2f}% | no_1hop_seed={nh['mrr']*100:.2f}% | no_day_ov={nd['mrr']*100:.2f}%"
        )

    if args.checkpoint_step == 350 and not args.no_sinkhorn:
        filename = 'synergy_results.json'
    else:
        suffix = '_nosinkhorn' if args.no_sinkhorn else ''
        filename = f'synergy_step{args.checkpoint_step}{suffix}.json'
    save_json(suite_dir / filename, summary)


if __name__ == '__main__':
    main()
