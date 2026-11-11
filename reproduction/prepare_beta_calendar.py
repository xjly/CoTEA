"""BETA raw interval cache, exact published precision and a fixed10percent split."""
import argparse
import calendar
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from .data import read_vocab, load_pairs, partition_pairs, save_json, sha256


BETA_SOURCE_HASHES = {
    'ent_ids_1': '61630f395ff13fd1c6da41e5ecad24e38053d5e3a80c0f3ef3fed47988e6ff2e',
    'ent_ids_2': 'e79c73166980ca45c9430c04de6c26900e0906ab5d33fc65cc74feebeefa6b6e',
    'rel_ids_1': 'a215dccec6e7530a1acf84c123c98230ebb5aa44b20b4d90f2c0663a44038d01',
    'rel_ids_2': '9c2946f32fdbcc24eb0ded7f212ffbfc6cee2332a38a38432d2b5f347770410f',
    'triples_1': '0a2436037965588955a132f5820e0f24621f8bd7b86e832756d5c000e43c228a',
    'triples_2': 'a5f136e97ddf603842383cd70a6dbc040f95ec9c62620e7cf896788fc5d16b08',
    'sup_pairs': 'ccbfacbf391a1281b0230745cbe17771ba911951596677d5ed7b77b740906671',
    'ref_pairs': 'b920f46ec33556442b713a490bbbc6b8b9094c7fa0ba16523f42da7aa9a88d28',
}


def parse_time(label):
    if label == '~':
        return None
    match = re.fullmatch(r'(-?\d+)\.([0-9]{2})([0-9]{2})', label)
    if match is None:
        raise ValueError(f'Unexpected upstream date: {label!r}')
    y, m, d = map(int, match.groups())
    if not 0 <= m <= 12 or not 0 <= d <= 31 or (d and not m):
        raise ValueError(f'Invalid component encoding: {label!r}')
    return y, m, d


def prepare(raw, out, seeds=None, split_seed=2026):
    raw, out = Path(raw), Path(out)
    assert not out.exists() or not any(out.iterdir()), out
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in BETA_SOURCE_HASHES.items():
        if sha256(raw / name) != expected:
            raise ValueError(f'{raw / name} differs from the published BETA input')
    ev = [read_vocab(raw / f'ent_ids_{i}') for i in (1, 2)]
    counts = list(map(len, ev))
    assert counts == [42666, 42297]
    raw_ids = [sorted(v) for v in ev]
    indices = [{value: j + offset for j, value in enumerate(ids)}
               for ids, offset in zip(raw_ids, (0, counts[0]))]
    rv = [read_vocab(raw / f'rel_ids_{i}') for i in (1, 2)]
    relation_names = sorted(set(rv[0].values()) | set(rv[1].values()))
    ri = {label: i for i, label in enumerate(relation_names)}
    assert not set(rv[0].values()) & set(rv[1].values())
    rows_by_graph, labels = [], set()
    date_stats, invalid = [], Counter()
    for graph in range(2):
        rows, stats = [], Counter()
        # The *c files are year-collapsed derivatives and are deliberately not used.
        for line in (raw / f'triples_{graph + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = line.split()
            assert len(row) == 5
            h, r, t = map(int, row[:3])
            start, end = row[3:]
            for label in (start, end):
                value = parse_time(label)
                if value is None:
                    stats['missing_endpoints'] += 1
                else:
                    labels.add(label)
                    y, m, d = value
                    stats['day_endpoints' if d else 'month_endpoints' if m else 'year_endpoints'] += 1
                    if d and d > calendar.monthrange(y, m)[1]:
                        invalid[label] += 1
            stats['facts'] += 1
            stats['both_missing'] += start == end == '~'
            stats['open_start'] += start == '~' and end != '~'
            stats['open_end'] += end == '~' and start != '~'
            rows.append((indices[graph][h], ri[rv[graph][r]], indices[graph][t], start, end))
        rows_by_graph.append(rows)
        date_stats.append(dict(stats))
    assert [len(x) for x in rows_by_graph] == [199879, 162320]
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
    reversed_boundaries = 0
    for rows in rows_by_graph:
        for h, r, t, start, end in rows:
            s, e = ti[start], ti[end]
            reversed_boundaries += bool(s and e and parse_time(start) > parse_time(end))
            chains[h][(t + 1, 2 * r + 1, s, e)] += 1
            chains[t][(h + 1, 2 * r + 2, s, e)] += 1
    events = np.zeros((len(chains), 64, 5), dtype=np.int32)
    lengths = np.array([len(c) for c in chains], dtype=np.int32)
    for entity, chain in enumerate(chains):
        rows = sorted(chain.items(), key=lambda x: (x[0][2] == 0, x[0][2],
                      x[0][3] == 0, x[0][3], x[0][1], x[0][0]))
        if len(rows) > 64:
            rows = [rows[j] for j in np.linspace(0, len(rows) - 1, 64, dtype=np.int64)]
        for j, ((obj, rel, start, end), frequency) in enumerate(rows):
            events[entity, j] = [obj, rel, start, frequency, end]
    np.savez_compressed(out / 'events.npz', events=events, lengths=lengths)
    # Alignment labels are used only after graph/time features are finalized.
    pairs = np.concatenate([load_pairs(raw / name, *indices) for name in ('sup_pairs', 'ref_pairs')])
    assert len(pairs) == 40364
    if seeds is None:
        seeds = int(.1 * len(pairs))
    train, dev, test = partition_pairs(pairs, seeds, split_seed)
    for col in (0, 1):
        for a, b in ((train, dev), (train, test), (dev, test)):
            assert not set(a[:, col]) & set(b[:, col])
    candidates = np.arange(counts[0], sum(counts), dtype=np.int64)
    np.savez(out / 'splits.npz', train=train, dev=dev, test=test,
             candidates_all=candidates, source_raw_ids=np.array(raw_ids[0]), target_raw_ids=np.array(raw_ids[1]))
    protocol = {
        'dataset': 'BETA-masked-YMD-full-interval', 'entities': counts,
        'facts': [len(rows) for rows in rows_by_graph], 'relations': list(map(len, rv)),
        'shared_relations': len(relation_names), 'relation_names': relation_names,
        'relation_features': 'unchanged baseline exact-label shared namespace; unmatched relations zero; no rels_same files used',
        'object_features': 'unchanged shared frozen identities for original supervised seeds; unknown zero followed by learned root-masked context',
        'time_tokens': len(known), 'time_labels': {str(i + 1): v for i, v in enumerate(known)},
        'ymd_components': parts, 'year_values': years,
        'time_encoding': 'shared frozen E_year + masked E_month + masked E_day + E_precision; divided by sqrt(number of present components including precision)',
        'time_initialization_seed': 732028,
        'time_precision_policy': 'preserve literal upstream components; 00 means absent; never impute dates; never expand an event into days',
        'time_source': 'raw triples_1/triples_2 strings; not year-collapsed triples_1c/triples_2c',
        'missing_time': 'both missing preserves the fact and masks both temporal slots; missing endpoints token0',
        'intervals': True, 'event_columns': ['object', 'relation', 'start', 'frequency', 'end'],
        'pooling': 'unchanged interval baseline: unordered exact(start,end) equality groups',
        'max_length': 64, 'truncation': 'evenly spaced over start/end ordered aggregated full facts',
        'training_seeds': seeds, 'training_fraction': seeds / len(pairs), 'split_seed': split_seed,
        'split': {'train': len(train), 'dev': len(dev), 'test': len(test)},
        'split_policy': 'first floor(0.1*40364)=4036 published sup_pairs; seeded10percent of remaining published sup+ref pairs for dev; rest reserved',
        'candidate_policy': 'all42297 target entities, including unaligned distractors and training entities',
        'endpoint_statistics_by_graph': date_stats,
        'upstream_calendar_invalid_endpoints': dict(invalid),
        'calendar_invalid_policy': 'preserve literal components; no calendar arithmetic or date correction',
        'reversed_boundary_facts': reversed_boundaries,
        'reversed_boundary_policy': 'preserve upstream start/end roles; do not reorder',
        'chain_lengths': {'mean': float(lengths.mean()), 'median': float(np.median(lengths)),
                         'max': int(lengths.max()), 'empty': int((lengths == 0).sum()),
                         'truncated': int((lengths > 64).sum()), 'cached_events': int((events[:, :, 0] > 0).sum())},
        'source_sha256': BETA_SOURCE_HASHES,
        'test_evaluated': False,
    }
    save_json(out / 'protocol.json', protocol)
    save_json(out / 'CACHE_READY.json', {
        'cache_sha256': {name: sha256(out / name) for name in ('events.npz', 'splits.npz', 'protocol.json')},
        'test_evaluated': False, 'feature_construction_uses_held_out_alignment_labels': False,
        'relation_mapping_files_used': False,
    })
    print(json.dumps({key: protocol[key] for key in ('dataset', 'entities', 'facts', 'time_tokens', 'split',
                     'chain_lengths', 'endpoint_statistics_by_graph', 'upstream_calendar_invalid_endpoints')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    prepare(args.raw, args.output)
