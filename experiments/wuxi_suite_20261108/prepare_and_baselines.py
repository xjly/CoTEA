"""Prepare WUXI-TEMP caches and evaluate LightTEA baselines under exact manifest protocol."""
import argparse
import calendar
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT_EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_EXP))

from reproduction.benchmark_retrieval import balance_inplace
from reproduction.data import read_vocab, load_pairs, save_json, sha256
from reproduction.prepare_beta_calendar import parse_time
from reproduction.priority_hybrid_data import prepare as prepare_priority
from reproduction.seed_relation_context import prepare_profiles
from reproduction.train import average_ranks


INPUT_SHA256 = {
    'ent_ids_1': '13d998ace00930b0ae2e6c2ace05c68e601994d82b51e3df6521ee04dc00a49e',
    'ent_ids_2': '58bc9eb55d40e755ed8e98623f08ee87de9eb75861c3b9e1af95e9acabaab685',
    'rel_ids_1': 'f21e7d9b8c732b6a0633a4ca0095ba404f360261031491814eb0beafca6cd299',
    'rel_ids_2': '488db7b34bf5ed40339b5e8ce651558c44fbe74c2e8bf4e900a7a58b5afdd524',
    'sup_pairs': '4e97db8e19aeb3ab8d39484ae54994740206b6c104d3a0227af591665df3b0f7',
    'ref_pairs': 'a5b6edbf40bde9bb2aa38b5ee213f91a35638af7f79115b6d44d8b220a67217f',
    'triples_1': '4e7481a974a36857929ba6e8e2931b92cc9f46c2e253d9be210ac5f986329840',
    'triples_2': '9acf9e926576c68a399f74c5d2f6ddfa5bcdb659d18f9d19e704b8939536aca4',
}


def build_wuxi_cache(raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    ref_dir = out_dir / 'reference'
    for name, digest in INPUT_SHA256.items():
        if sha256(raw_dir / name) != digest:
            raise ValueError(f'WUXI-TEMP input checksum mismatch: {name}')
    if (out_dir / 'prepared.json').exists():
        return json.loads((out_dir / 'prepared.json').read_text(encoding='utf-8'))
    ref_dir.mkdir(parents=True, exist_ok=True)

    ev = [read_vocab(raw_dir / f'ent_ids_{i}') for i in (1, 2)]
    counts = list(map(len, ev))
    assert counts == [7537, 6483]
    raw_ids = [sorted(v) for v in ev]
    indices = [
        {value: j + offset for j, value in enumerate(ids)}
        for ids, offset in zip(raw_ids, (0, counts[0]))
    ]
    rv = [read_vocab(raw_dir / f'rel_ids_{i}') for i in (1, 2)]
    relation_names = sorted(set(rv[0].values()) | set(rv[1].values()))
    ri = {label: i for i, label in enumerate(relation_names)}
    assert not (set(rv[0].values()) & set(rv[1].values()))

    rows_by_graph, labels = [], set()
    date_stats = []
    for graph in range(2):
        rows, stats = [], Counter()
        for line in (raw_dir / f'triples_{graph + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = line.split('\t')
            assert len(row) == 5
            h, r, t = map(int, row[:3])
            start, end = row[3:]
            for label in (start, end):
                val = parse_time(label)
                if val is None:
                    stats['missing_endpoints'] += 1
                else:
                    labels.add(label)
                    y, m, d = val
                    stats['day_endpoints' if d else 'month_endpoints' if m else 'year_endpoints'] += 1
            stats['facts'] += 1
            rows.append((indices[graph][h], ri[rv[graph][r]], indices[graph][t], start, end))
        rows_by_graph.append(rows)
        date_stats.append(dict(stats))
    assert [len(x) for x in rows_by_graph] == [212573, 138605]

    known = sorted(labels, key=lambda label: (parse_time(label), label))
    ti = {label: i + 1 for i, label in enumerate(known)}
    ti['~'] = 0
    years = sorted({parse_time(label)[0] for label in known})
    yi = {year: i + 1 for i, year in enumerate(years)}
    parts = [[0, 0, 0, 0]]
    for label in known:
        y, m, d = parse_time(label)
        parts.append([yi[y], m, d, 1 + int(m > 0) + int(d > 0)])

    chains = [Counter() for _ in range(sum(counts))]
    for rows in rows_by_graph:
        for h, r, t, start, end in rows:
            s, e = ti[start], ti[end]
            chains[h][(t + 1, 2 * r + 1, s, e)] += 1
            chains[t][(h + 1, 2 * r + 2, s, e)] += 1

    events = np.zeros((len(chains), 64, 5), dtype=np.int32)
    lengths = np.array([len(c) for c in chains], dtype=np.int32)
    for entity, chain in enumerate(chains):
        rows = sorted(
            chain.items(),
            key=lambda x: (x[0][2] == 0, x[0][2], x[0][3] == 0, x[0][3], x[0][1], x[0][0]),
        )
        if len(rows) > 64:
            rows = [rows[j] for j in np.linspace(0, len(rows) - 1, 64, dtype=np.int64)]
        for j, ((obj, rel, start, end), frequency) in enumerate(rows):
            events[entity, j] = [obj, rel, start, frequency, end]
    np.savez_compressed(ref_dir / 'events.npz', events=events, lengths=lengths)

    train = load_pairs(raw_dir / 'sup_pairs', *indices)
    reference = load_pairs(raw_dir / 'ref_pairs', *indices)
    assert len(train) == 1000 and len(reference) == 3872
    for col in (0, 1):
        assert not (set(train[:, col]) & set(reference[:, col]))

    n, m = counts
    candidates_all = np.arange(n, n + m, dtype=np.int64)
    train_targets = set(train[:, 1].tolist())
    candidates_eval = np.array([c for c in candidates_all if int(c) not in train_targets], dtype=np.int64)
    assert len(candidates_eval) == 5483

    np.savez(
        ref_dir / 'splits.npz',
        train=train,
        reference=reference,
        candidates_all=candidates_all,
        candidates_eval=candidates_eval,
        source_raw_ids=np.array(raw_ids[0]),
        target_raw_ids=np.array(raw_ids[1]),
    )

    protocol = {
        'dataset': 'WUXI-TEMP-v2',
        'version': 'wuxi-coverage-gran-v2-20261024',
        'entities': counts,
        'facts': [len(r) for r in rows_by_graph],
        'relations': list(map(len, rv)),
        'shared_relations': len(relation_names),
        'relation_names': relation_names,
        'time_tokens': len(known),
        'time_labels': {str(i + 1): v for i, v in enumerate(known)},
        'ymd_components': parts,
        'year_values': years,
        'intervals': True,
        'event_columns': ['object', 'relation', 'start', 'frequency', 'end'],
        'max_length': 64,
        'training_seeds': 1000,
        'split': {'train': 1000, 'reference': 3872, 'candidates_eval': 5483, 'candidates_all': 6483},
        'endpoint_statistics_by_graph': date_stats,
        'chain_lengths': {
            'mean': float(lengths.mean()),
            'median': float(np.median(lengths)),
            'max': int(lengths.max()),
            'empty': int((lengths == 0).sum()),
            'truncated': int((lengths > 64).sum()),
            'cached_events': int((events[:, :, 0] > 0).sum()),
        },
        'source_sha256': INPUT_SHA256,
    }
    save_json(ref_dir / 'protocol.json', protocol)
    prepare_priority(raw_dir, ref_dir, out_dir / 'cache')
    prepare_profiles(raw_dir, out_dir / 'cache/priority', out_dir / 'profiles')

    # Also build untruncated object-seed incidence counts (entities, 2, 1000) for leave-one-out object profiles
    seed_index = {int(e): j for j, pair in enumerate(train) for e in pair}
    obj_counts = np.zeros((sum(counts), 2, len(train)), dtype=np.float32)
    for rows in rows_by_graph:
        for h, r, t, start, end in rows:
            if t in seed_index:
                obj_counts[h, 0, seed_index[t]] += 1.0
            if h in seed_index:
                obj_counts[t, 1, seed_index[h]] += 1.0
    np.save(out_dir / 'profiles/seed_object_counts.npy', obj_counts)

    # Build diagnostic subgroup masks on the 3,872 reference queries
    ev_prio = np.load(out_dir / 'cache/priority/events.npz')['events']
    seed_tokens = set((train.ravel() + 1).tolist())
    has_1hop_u = np.array([any(int(x) in seed_tokens for x in ev_prio[u, :, 0] if x > 0) for u, _ in reference])
    has_1hop_v = np.array([any(int(x) in seed_tokens for x in ev_prio[v, :, 0] if x > 0) for _, v in reference])

    day_sets = [set() for _ in range(sum(counts))]
    tok_sets = [set() for _ in range(sum(counts))]
    for rows in rows_by_graph:
        for h, r, t, start, end in rows:
            if start != '~':
                tok_sets[h].add(start)
                tok_sets[t].add(start)
                if not start.endswith('00'):
                    day_sets[h].add(start)
                    day_sets[t].add(start)
    exact_day_ov = np.array([len(day_sets[u] & day_sets[v]) for u, v in reference])
    exact_tok_ov = np.array([len(tok_sets[u] & tok_sets[v]) for u, v in reference])

    groups = {
        'no_exact_day_overlap': exact_day_ov == 0,
        'exact_day_1or2': (exact_day_ov >= 1) & (exact_day_ov <= 2),
        'exact_day_gt2': exact_day_ov > 2,
        'no_exact_token_overlap': exact_tok_ov == 0,
        'both_have_1hop_seed': has_1hop_u & has_1hop_v,
        'at_least_one_no_1hop_seed': ~(has_1hop_u & has_1hop_v),
        'both_no_1hop_seed': (~has_1hop_u) & (~has_1hop_v),
    }
    np.savez(out_dir / 'wuxi_groups.npz', **groups)

    summary = {
        'state': 'ready',
        'entities': counts,
        'facts': [len(r) for r in rows_by_graph],
        'train_pairs': 1000,
        'reference_queries': 3872,
        'candidates_eval': 5483,
        'candidates_all': 6483,
        'group_counts': {k: int(v.sum()) for k, v in groups.items()},
        'priority_audit': json.loads((out_dir / 'cache/audit.json').read_text(encoding='utf-8')),
    }
    save_json(out_dir / 'prepared.json', summary)
    return summary


@torch.inference_mode()
def run_lighttea(raw_dir, out_dir, seed=2026):
    """Exact implementation of LightTEA (Two-aspect Three-view LP + Dice Time Sim + Sinkhorn)."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    cache = out_dir / 'cache/priority'
    protocol = json.loads((cache / 'protocol.json').read_text(encoding='utf-8'))
    n, m = protocol['entities']
    total_e = n + m
    total_r = protocol['shared_relations']
    total_t = protocol['time_tokens']

    with np.load(out_dir / 'reference/splits.npz') as splits:
        train = splits['train']
        reference = splits['reference']
        candidates_eval = splits['candidates_eval']
        candidates_all = splits['candidates_all']
    groups = dict(np.load(out_dir / 'wuxi_groups.npz'))

    ev = [read_vocab(raw_dir / f'ent_ids_{i}') for i in (1, 2)]
    raw_ids = [sorted(v) for v in ev]
    indices = [{v: j + off for j, v in enumerate(ids)} for ids, off in zip(raw_ids, (0, n))]
    rv = [read_vocab(raw_dir / f'rel_ids_{i}') for i in (1, 2)]
    ri = {label: i for i, label in enumerate(protocol['relation_names'])}
    ti = {label: int(k) - 1 for k, label in protocol['time_labels'].items()}

    # Build sparse normalized adjacency matrices A_EE, A_ER, A_RE, A_ET, A_TE
    ee_edges = Counter()
    er_edges = Counter()
    re_edges = Counter()
    et_edges = Counter()
    te_edges = Counter()
    entity_time_multisets = [Counter() for _ in range(total_e)]
    entity_time_sets = [set() for _ in range(total_e)]

    for g in range(2):
        for line in (raw_dir / f'triples_{g + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h_raw, r_raw, t_raw, start, end = line.split('\t')
            h = indices[g][int(h_raw)]
            t = indices[g][int(t_raw)]
            r = ri[rv[g][int(r_raw)]]
            ee_edges[(h, t)] += 1.0
            ee_edges[(t, h)] += 1.0
            er_edges[(h, 2 * r)] += 1.0
            er_edges[(t, 2 * r + 1)] += 1.0
            re_edges[(2 * r, h)] += 1.0
            re_edges[(2 * r + 1, t)] += 1.0
            if start != '~':
                tid = ti[start]
                et_edges[(h, tid)] += 1.0
                et_edges[(t, tid)] += 1.0
                te_edges[(tid, h)] += 1.0
                te_edges[(tid, t)] += 1.0
                entity_time_multisets[h][tid] += 1
                entity_time_multisets[t][tid] += 1
                entity_time_sets[h].add(tid)
                entity_time_sets[t].add(tid)

    def make_sparse_row_norm(edge_counter, shape):
        rows, cols, vals = [], [], []
        row_sums = defaultdict(float)
        for (r, c), v in edge_counter.items():
            w = math.log1p(v)
            rows.append(r)
            cols.append(c)
            vals.append(w)
            row_sums[r] += w
        vals = [v / max(row_sums[r], 1e-12) for r, v in zip(rows, vals)]
        idx = torch.tensor([rows, cols], dtype=torch.long, device='cuda')
        val_t = torch.tensor(vals, dtype=torch.float32, device='cuda')
        return torch.sparse_coo_tensor(idx, val_t, size=shape, device='cuda').coalesce()

    A_EE = make_sparse_row_norm(ee_edges, (total_e, total_e))
    A_ER = make_sparse_row_norm(er_edges, (total_e, 2 * total_r))
    A_RE = make_sparse_row_norm(re_edges, (2 * total_r, total_e))
    A_ET = make_sparse_row_norm(et_edges, (total_e, total_t))
    A_TE = make_sparse_row_norm(te_edges, (total_t, total_e))

    # Initialize seed hyper-sphere vectors (dim=300 per step, 3 steps -> 900 dim as in LightTEA)
    gen = torch.Generator(device='cuda').manual_seed(seed)
    d = 300
    seed_vecs = F.normalize(torch.randn(len(train), d, generator=gen, device='cuda'), dim=-1)
    L_e0 = torch.zeros(total_e, d, device='cuda')
    for idx_s, (u, v) in enumerate(train):
        L_e0[int(u)] = seed_vecs[idx_s]
        L_e0[int(v)] = seed_vecs[idx_s]

    # Relational-aspect 3-view LP (2 steps)
    L_e_rel = [L_e0]
    L_r = torch.zeros(2 * total_r, d, device='cuda')
    curr_e = L_e0
    for _ in range(2):
        next_e = F.normalize(torch.sparse.mm(A_EE, curr_e) + torch.sparse.mm(A_ER, L_r), dim=-1)
        L_r = F.normalize(torch.sparse.mm(A_RE, curr_e), dim=-1)
        curr_e = next_e
        L_e_rel.append(curr_e)
    H_rel = F.normalize(torch.cat(L_e_rel, dim=-1), dim=-1)

    # Temporal-aspect 3-view LP (2 steps)
    L_e_time = [L_e0]
    L_t = torch.zeros(total_t, d, device='cuda')
    curr_et = L_e0
    for _ in range(2):
        next_et = F.normalize(torch.sparse.mm(A_EE, curr_et) + torch.sparse.mm(A_ET, L_t), dim=-1)
        L_t = F.normalize(torch.sparse.mm(A_TE, curr_et), dim=-1)
        curr_et = next_et
        L_e_time.append(curr_et)
    H_time = F.normalize(torch.cat(L_e_time, dim=-1), dim=-1)

    # Compute explicit Dice Time Similarity matrix S_t(e_i, e_j) = 2 * |T_i \cap T_j| / (|T_i| + |T_j|)
    inc = torch.zeros(total_e, total_t, dtype=torch.float32, device='cuda')
    for eid, tset in enumerate(entity_time_sets):
        if tset:
            inc[eid, list(tset)] = 1.0
    inc_l = inc[:n]
    inc_r = inc[n:]
    inter = inc_l @ inc_r.T
    denom = inc_l.sum(1, keepdim=True) + inc_r.sum(1)[None, :]
    S_time_dice = torch.where(denom > 0, 2.0 * inter / denom.clamp_min(1.0), torch.zeros_like(inter))

    def eval_matrix(sim_full_n_by_m, pool_rows, pool_cols, use_csls=False):
        raw_sub = sim_full_n_by_m[pool_rows][:, pool_cols - n]
        if use_csls:
            rx = raw_sub.topk(5, dim=1).values.double().mean(1, keepdim=True)
            ry = raw_sub.topk(5, dim=0).values.double().mean(0, keepdim=True)
            sub = 2.0 * raw_sub.double() - rx - ry
        else:
            sub = raw_sub.double().clone()
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

    results = {}
    for alpha, beta, tag in [
        (0.3, 0.3, 'lighttea_a03_b03_default'),
        (0.5, 0.2, 'lighttea_a05_b02'),
        (0.0, 0.0, 'lightea_rel_only'),
    ]:
        H = F.normalize((1.0 - alpha) * H_rel + alpha * H_time, dim=-1)
        S_label = H[:n] @ H[n:].T
        S_total = (1.0 - beta) * S_label + beta * S_time_dice
        results[tag] = {
            'eval_5483': eval_matrix(S_total, reference[:, 0], candidates_eval),
            'closed_3872': eval_matrix(S_total, reference[:, 0], reference[:, 1]),
            'full_6483': eval_matrix(S_total, np.arange(n), candidates_all),
        }

    save_json(out_dir / 'lighttea_results.json', results)
    return results


def main():
    parser = argparse.ArgumentParser(description='Build the WUXI-TEMP experiment inputs.')
    parser.add_argument('--data-dir', type=Path, default=ROOT_EXP / 'datasets/WUXI-TEMP')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--lighttea', action='store_true', help='Also run the LightEA/LightTEA baselines on CUDA.')
    args = parser.parse_args()
    raw, out = args.data_dir, args.output_dir
    prep = build_wuxi_cache(raw, out)
    print('PREPARED:', json.dumps(prep, indent=2))
    if args.lighttea:
        for name, result in run_lighttea(raw, out).items():
            print(name, 'eval_5483:', result['eval_5483']['overall'], 'closed_3872:', result['closed_3872']['overall'])


if __name__ == '__main__':
    main()
