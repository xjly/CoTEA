"""Rebuild point/interval caches using training anchors and numeric-time coverage."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from .data import read_vocab, save_json, sha256
from .prepare_beta_calendar import parse_time


def spread(indices, times, count):
    """Distinct nearest observations to equally spaced numeric-time targets."""
    indices = np.asarray(indices, dtype=np.int64)
    if count >= len(indices):
        return indices.tolist()
    if count == 0:
        return []
    values = np.asarray(times)[indices]
    known = np.isfinite(values)
    if not known.any() or np.nanmax(values[known]) == np.nanmin(values[known]):
        return indices[np.linspace(0, len(indices)-1, count, dtype=int)].tolist()
    present, missing = indices[known], indices[~known]
    missing_count = min(len(missing), round(count * len(missing) / len(indices)))
    count -= missing_count
    distance_values = np.asarray(times)[present].copy()
    chosen = []
    for target in np.linspace(distance_values.min(), distance_values.max(), count):
        distance = np.abs(distance_values-target)
        # Break equal-time ties across the original row range, deterministically.
        best = int(np.argmin(distance))
        chosen.append(int(present[best]))
        distance_values[best] = np.inf
    if missing_count:
        chosen += missing[np.linspace(0, len(missing)-1, missing_count, dtype=int)].tolist()
    assert len(chosen) == len(set(chosen))
    return chosen


def select(rows, numeric_times, seed_objects, shared_relations, cap=64):
    """Two priority buckets; object anchors break ties inside the high bucket."""
    if len(rows) <= cap:
        return list(range(len(rows)))
    anchor = [i for i, row in enumerate(rows) if row[0] in seed_objects]
    shared = [i for i, row in enumerate(rows)
              if row[0] not in seed_objects and row[1] in shared_relations]
    ordinary = [i for i, row in enumerate(rows)
                if row[0] not in seed_objects and row[1] not in shared_relations]
    result = []
    for bucket in (anchor, shared, ordinary):
        result += spread(bucket, numeric_times, min(cap-len(result), len(bucket)))
    assert len(result) == cap and len(set(result)) == cap
    return sorted(result)


def coordinates(protocol, raw, calendar=None):
    """Cached token -> actual days or years, never a reindexed timestamp ID."""
    raw = Path(raw)
    labels = protocol['time_labels']
    values = np.zeros(protocol['time_tokens']+1, dtype=np.float64)
    if 'ymd_components' in protocol:
        for token, label in labels.items():
            y,m,d = parse_time(label)
            # Lower boundary on a fractional-year axis for sampling/phase only.
            # The original precision mask is preserved in the discrete time slot.
            values[int(token)] = y + (max(m,1)-1)/12 + (max(d,1)-1)/372
        return values, {'unit':'year', 'source':'literal BETA year/month/day; lower-bound fractional coordinate',
                        'missing_components_masked_in_discrete_slot':True}
    if (raw/'time_id').exists():
        values[1:] = [float(v) for v in labels.values()]
        return values, {'unit':'year', 'origin':0, 'source':'upstream time_id numeric labels'}
    if calendar is None:
        raise ValueError('DICEWS requires an independently verified calendar mapping')
    calendar = Path(calendar)
    audit = json.loads((calendar/'calendar_audit.json').read_text(encoding='utf-8'))
    assert audit['verified'] and sha256(calendar/'calendar_mapping.csv') == audit['mapping_sha256']
    for name, digest in audit['dicews_input_sha256'].items():
        assert sha256(raw/name) == digest, name
    mapping = {int(r['raw_time_id']): float(r['days_since_2005_01_01'])
               for r in csv.DictReader((calendar/'calendar_mapping.csv').open(encoding='utf-8'))}
    values[1:] = [mapping[int(k)] for k in labels]
    return values, {'unit':'day', 'origin':audit['origin'],
                    'calendar_sha256':audit['mapping_sha256'], 'alignment_labels_used_for_calendar':False}


def prepare(raw, reference, output, calendar=None):
    raw, reference, output = map(Path, (raw, reference, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((reference/'protocol.json').read_text(encoding='utf-8'))
    for name, digest in protocol['source_sha256'].items():
        assert sha256(raw/name) == digest, name
    old = np.load(reference/'events.npz')['events']
    interval = old.shape[2] == 5
    assert old.shape[1] == 64 and old.shape[2] in (4, 5)
    train = np.load(reference/'splits.npz')['train']  # No held-out pair labels.
    seed_objects = set((train.ravel()+1).tolist())
    values, time_meta = coordinates(protocol, raw, calendar)
    np.save(output/'time_coordinates.npy', values)
    ev = [read_vocab(raw/f'ent_ids_{g}') for g in (1, 2)]
    index = [{k:j+offset for j,k in enumerate(sorted(v))}
             for v,offset in zip(ev, (0, protocol['entities'][0]))]
    rv = [read_vocab(raw/f'rel_ids_{g}') for g in (1, 2)]
    ri = {name:i for i,name in enumerate(protocol['relation_names'])}
    shared_names = set(rv[0].values()) & set(rv[1].values())
    shared = {2*ri[name]+direction for name in shared_names for direction in (1, 2)}
    beta = 'ymd_components' in protocol
    assert not beta or interval
    ti = ({label:int(token) for token,label in protocol['time_labels'].items()} if beta else
          {int(raw_id):token for token,raw_id in enumerate(protocol['time_labels'], 1)})
    missing = set()
    if beta:
        missing = {'~'}
    elif (raw/'time_id').exists():
        missing = {k for k,v in read_vocab(raw/'time_id').items()
                   if v.lower() in ('-inf','inf','+inf','null','none')}
    chains = [Counter() for _ in range(len(old))]
    for graph in range(2):
        with (raw/f'triples_{graph+1}').open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                fields = line.split()
                h,r,t = map(int,fields[:3])
                raw_times = fields[3:] if beta else list(map(int,fields[3:]))
                assert all(x in ti or x in missing for x in raw_times)
                h,t = index[graph][h],index[graph][t]
                relation = ri[rv[graph][r]]
                if interval:
                    start,end = ti.get(raw_times[0],0),ti.get(raw_times[-1],0)
                    chains[h][(t+1,2*relation+1,start,end)] += 1
                    chains[t][(h+1,2*relation+2,start,end)] += 1
                else:
                    for time in dict.fromkeys(ti.get(x,0) for x in raw_times):
                        chains[h][(t+1,2*relation+1,time)] += 1
                        chains[t][(h+1,2*relation+2,time)] += 1
    uniform = np.zeros_like(old)
    priority = np.zeros_like(old)
    lengths = np.asarray([len(c) for c in chains],dtype=np.int32)
    audit = Counter()
    for entity, chain in enumerate(chains):
        if interval:
            facts = sorted(chain.items(),key=lambda x:(x[0][2]==0,x[0][2],x[0][3]==0,x[0][3],x[0][1],x[0][0]))
            rows = [[o,r,s,f,e] for (o,r,s,e),f in facts]
        else:
            rows = sorted(([o,r,t,f] for (o,r,t),f in chain.items()),key=lambda x:(x[2]==0,x[2],x[1],x[0]))
        if not rows:
            continue
        chronological = [values[row[2]] if row[2] else
                         (values[row[4]] if interval and row[4] else np.nan) for row in rows]
        original_ids = np.linspace(0,len(rows)-1,64,dtype=int).tolist() if len(rows)>64 else list(range(len(rows)))
        priority_ids = select(rows,chronological,seed_objects,shared)
        uniform[entity,:len(original_ids)] = np.asarray(rows)[original_ids]
        priority[entity,:len(priority_ids)] = np.asarray(rows)[priority_ids]
        audit['changed_entities'] += original_ids != priority_ids
        anchors = {i for i,row in enumerate(rows) if row[0] in seed_objects}
        high = {i for i,row in enumerate(rows) if row[0] in seed_objects or row[1] in shared}
        for key, group in [('anchor',anchors),('high_priority',high)]:
            audit[f'{key}_total'] += len(group)
            audit[f'{key}_kept_uniform'] += len(group.intersection(original_ids))
            audit[f'{key}_kept_priority'] += len(group.intersection(priority_ids))
            audit[f'{key}_overflow_entities'] += len(group)>64
    # Matched control must recover the original point/full-interval inputs exactly.
    np.testing.assert_array_equal(uniform,old)
    alias = np.zeros(2*protocol['shared_relations']+1,dtype=np.int64)
    n = protocol['entities'][0]
    original_common = set(old[:n,:,1][old[:n,:,0]>0]) & set(old[n:,:,1][old[n:,:,0]>0])
    for token in original_common:
        alias[token] = token
    for arm, events in [('control',uniform),('priority',priority)]:
        dest = output/arm
        dest.mkdir()
        np.savez_compressed(dest/'events.npz',events=events,lengths=lengths)
        np.save(dest/'time_coordinates.npy',values)
        np.save(dest/'relation_alias.npy',alias)
        shutil.copyfile(reference/'splits.npz',dest/'splits.npz')
        save_json(dest/'protocol.json',{**protocol,'sampling_arm':arm,'time_coordinate_metadata':time_meta,
            'relation_mapping':'frozen original control mapping, identical across arms',
            'truncation': 'original uniform row-index sampling' if arm=='control' else
                'seed-object OR shared-relation high bucket, object anchors break overflow ties; numeric-time coverage in each bucket',
            'reference_cache_sha256':{f:sha256(reference/f) for f in ('events.npz','splits.npz','protocol.json')}})
        assert sha256(dest/'splits.npz') == sha256(reference/'splits.npz')
    save_json(output/'audit.json',{'reference_events_recovered_exactly':True,'split_bytes_identical':True,
        'train_pairs':len(train),'test_labels_read':False,'shared_relation_labels':len(shared_names),
        'all_relation_labels':len(ri),'time':time_meta,'events':dict(audit),
        'retained_events':int((priority[:,:,0]>0).sum()),'length_mean':float(lengths.mean()),
        'length_gt64':int((lengths>64).sum()),'length_max':int(lengths.max())})
    print((output/'audit.json').read_text(encoding='utf-8'),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);p.add_argument('--calendar')
    a=p.parse_args();prepare(a.raw,a.reference,a.output,a.calendar)
