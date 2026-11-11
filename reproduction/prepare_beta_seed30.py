"""Rebuild only seed-dependent BETA inputs for the published 30% supervision."""
import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from .data import read_vocab,load_pairs,save_json,sha256
from .prepare_beta_calendar import prepare as prepare_uniform
from .priority_hybrid_data import prepare as prepare_priority


def prepare(raw,reference,output):
    from .seed_relation_context import prepare_profiles

    raw,reference,output=map(Path,(raw,reference,output))
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'reference').exists() and not (output/'cache').exists()
    original=json.loads((reference/'protocol.json').read_text())
    for name,digest in original['source_sha256'].items():assert sha256(raw/name)==digest,name
    n,m=original['entities'];assert (n,m)==(42666,42297)
    source_ids=np.array(sorted(read_vocab(raw/'ent_ids_1')),dtype=np.int64)
    target_ids=np.array(sorted(read_vocab(raw/'ent_ids_2')),dtype=np.int64)
    source_index={int(v):i for i,v in enumerate(source_ids)}
    target_index={int(v):i+n for i,v in enumerate(target_ids)}
    sup=load_pairs(raw/'sup_pairs',source_index,target_index)
    ref=load_pairs(raw/'ref_pairs',source_index,target_index)
    assert len(sup)==12109 and len(ref)==28255
    pairs=np.concatenate([sup,ref]);assert len(pairs)==40364
    train=sup
    np.testing.assert_array_equal(train,sup)
    assert (len(train),len(ref))==(12109,28255)
    with np.load(reference/'splits.npz') as old:
        np.testing.assert_array_equal(train[:4036],old['train'])
    for col in (0,1):
        assert not set(train[:,col])&set(ref[:,col])
    dest=output/'reference';dest.mkdir()
    # Original uniform events/calendar vocabulary are independent of supervision.
    # Their bytes are kept identical; the priority sampler below is rebuilt using
    # only the new training pool, rather than reusing the 10% priority cache.
    shutil.copyfile(reference/'events.npz',dest/'events.npz')
    np.savez(dest/'splits.npz',train=train,reference=ref,
        candidates_all=np.arange(n,n+m,dtype=np.int64),
        source_raw_ids=source_ids,target_raw_ids=target_ids)
    updated={**original,'dataset':'BETA-masked-YMD-full-interval-seed30',
        'training_seeds':12109,'training_fraction':12109/40364,'split_seed':None,
        'split':dict(train=12109,reference=28255),
        'split_policy':'all12109 published sup_pairs train; all28255 published ref_pairs for final closed-reference evaluation; no validation split',
        'relation_features':'frozen shared directions and training-seed-induced profiles; no supplied relation mappings',
        'object_features':'shared frozen identities for12109 training pairs; unknown objects use original root-masked Transformer context',
        'seed_independent_reference_sha256':{name:sha256(reference/name) for name in ('events.npz','protocol.json')},
        'test_evaluated':False}
    save_json(dest/'protocol.json',updated)
    prepare_priority(raw,dest,output/'cache')
    prepare_profiles(raw,output/'cache/priority',output/'profiles')
    cache=output/'cache/priority'
    with np.load(cache/'splits.npz') as splits:
        np.testing.assert_array_equal(splits['train'],train)
        np.testing.assert_array_equal(splits['reference'],ref)
    profiles=np.load(output/'profiles/seed_relation_counts.npy')
    assert profiles.shape==(updated['shared_relations'],2,12109)
    np.testing.assert_array_equal(np.load(output/'cache/control/events.npz')['events'],np.load(reference/'events.npz')['events'])
    result=dict(state='ready',training_pairs=12109,reference_pairs=28255,validation_pairs=0,
        source_entities=n,training_target_candidates=m,evaluation_candidates=28255,
        source_raw=str(raw.resolve()),
        official_sup_matches_training_exactly=True,training_prefix_matches_old4036=True,
        train_reference_endpoint_disjoint=True,official_ref_matches_evaluation_exactly=True,
        reference_labels_used_for_checkpoint_selection=False,
        original_uniform_events_recovered_exactly=True,
        seed_priority_cache_rebuilt=True,relation_profiles_rebuilt=True,
        relation_profile_shape=list(profiles.shape),
        feature_construction_uses_held_out_alignment_labels=False,test_evaluated=False,
        source_sha256=updated['source_sha256'],
        cache_sha256={name:sha256(cache/name) for name in ('events.npz','splits.npz','protocol.json','time_coordinates.npy','relation_alias.npy')},
        relation_profiles_sha256=sha256(output/'profiles/seed_relation_counts.npy'),
        priority_audit=json.loads((output/'cache/audit.json').read_text()))
    save_json(output/'prepared.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with TemporaryDirectory() as temporary:
        reference=Path(temporary)
        prepare_uniform(args.raw,reference)
        prepare(args.raw,reference,args.output)
