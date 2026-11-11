"""Matched single-seed priority/hybrid experiments with no sparse query loss."""
import argparse,json,os,random,time,shutil,sys
from pathlib import Path
import numpy as np
import torch
from .data import save_json,sha256
from .chain_only import configure_attention
from .priority_hybrid_model import make_model,time_diagnostics
from .full_context_training import cached_complete_backward
from .root_mask_training import encode_root_mask
from .benchmark_retrieval import ranks
from .train import metrics

def checkpoint(path,model,optimizer,step,seed,context):
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'seed':seed,'context_enabled':context,
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
    temporary=path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(path)


def mask_one_side_time(events, train, fraction, seed, step):
    """Build one training view for query, candidate and context encoding."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    pairs = rng.choice(len(train), size=round(len(train) * fraction), replace=False)
    sides = rng.integers(0, 2, size=len(pairs))
    masked_entities = train[pairs, sides]
    training_events = events.copy()
    training_events[masked_entities, :, 2] = 0
    if events.shape[2] == 5:
        training_events[masked_entities, :, 4] = 0
    return training_events, np.bincount(sides, minlength=2).tolist()


def common_day_mask(events, pairs):
    left = events[pairs[:, 0], :, 2]
    right = events[pairs[:, 1], :, 2]
    shared = (left[:, :, None] == right[:, None, :]) & (left[:, :, None] > 0)
    return shared.any(axis=(1, 2))


def interval_time_groups(events, pairs):
    has_time = ((events[:, :, 2] > 0) | (events[:, :, 4] > 0)).any(axis=1)
    timed_sides = has_time[pairs].sum(axis=1)
    return {'both_timed': timed_sides == 2,
            'only_one_timed': timed_sides == 1,
            'neither_timed': timed_sides == 0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--hybrid',action='store_true');p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--memory-fraction',type=float,default=.15);p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--base-steps',type=int,default=200);p.add_argument('--context-steps',type=int,default=200)
    p.add_argument('--eval-every',type=int,default=5);p.add_argument('--resume',action='store_true')
    p.add_argument('--relation-arm',choices=['zero','direction','induced']);p.add_argument('--relation-profiles')
    p.add_argument('--sinkhorn-every',type=int,default=0)
    p.add_argument('--time-mask-fraction', type=float, default=0.,
                   help='Fraction of training pairs with all time fields masked on one side per update')
    args=p.parse_args()
    if not 0. <= args.time_mask_fraction <= 1.:
        p.error('--time-mask-fraction must be between 0 and 1')
    if args.sinkhorn_every: assert args.sinkhorn_every%args.eval_every==0 and args.sinkhorn_every>0
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.gpu
    assert args.base_steps==args.context_steps==200
    cache=Path(args.cache);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and not args.resume:raise FileExistsError(out)
    torch.set_num_threads(4);configure_attention();torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction);torch.cuda.reset_peak_memory_stats()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    protocol=json.loads((cache/'protocol.json').read_text());events=np.load(cache/'events.npz')['events']
    # Held-out test labels are never indexed here.
    splits=np.load(cache/'splits.npz');train=splits['train'];dev=splits['dev'];candidates=splits['candidates_all'];n=protocol['entities'][0]
    beta='ymd_components' in protocol
    assert not beta or (args.hybrid and args.relation_arm=='induced')
    assert len(train)==(4036 if beta else 1000) and events.shape[:2]==(sum(protocol['entities']),64) and events.shape[2] in (4,5)
    np.testing.assert_array_equal(candidates,np.arange(n,len(events)))
    for col in (0,1):assert not set(train[:,col])&set(dev[:,col])
    coordinates=np.load(cache/'time_coordinates.npy');relation_alias=np.load(cache/'relation_alias.npy')
    if args.relation_arm:
        from .seed_relation_context import SeedRelationModel
        assert args.hybrid and args.relation_profiles
        model=SeedRelationModel(protocol,events,train,np.load(args.relation_profiles),coordinates,args.relation_arm).cuda()
    else:
        model=make_model(protocol,events,train,coordinates,relation_alias,args.hybrid).cuda()
    extra=model.encoder.extra_parameters();extra_ids={id(x) for x in extra}
    backbone=[p for p in model.parameters() if p.requires_grad and id(p) not in extra_ids]
    model.encoder.use_neighbor=False
    for parameter in extra:parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(backbone,lr=1e-4,weight_decay=1e-4)
    first=0;best=-1.;beststep=None;context=False;previous={}
    if args.resume:
        previous_config = json.loads((out/'config.json').read_text())
        if previous_config.get('time_mask_fraction', 0.) != args.time_mask_fraction:
            raise ValueError('Resume requires the original time-mask fraction')
        saved=torch.load(out/'latest_training.pt',map_location='cuda',weights_only=True)
        assert saved['seed']==args.seed
        first=saved['step'];context=saved['context_enabled'];model.load_state_dict(saved['model'])
        if context:
            model.encoder.use_neighbor=True
            for parameter in extra:parameter.requires_grad_(True)
            optimizer.add_param_group({'params':extra,'lr':5e-4,'weight_decay':1e-4})
        optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        previous=json.loads((out/'status.json').read_text());best=previous.get('best_mrr',-1.);beststep=previous.get('best_step')
    else:
        sources={f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')}
        config={**vars(args),'dataset':protocol['dataset'],'entities':protocol['entities'],'validation_queries':len(dev),'target_candidates':len(candidates),
            'architecture':'shared2layer512heads8; unordered same-time/pair pooling; 64 events; one root-masked object context round; split endpoints for interval data',
            'event_slots': ['object','relation','time','frequency'] if events.shape[2]==4 else ['object','relation','start','end','frequency'],
            'time_encoding':(('frozen masked YMD/precision' if beta else 'frozen discrete') + ' + frozen sinusoid alpha2 / sqrt5') if args.hybrid else 'original frozen discrete lookup',
            'time_diagnostics':time_diagnostics(model.encoder,coordinates),
            'hidden_dim':512,'heads':8,'intra_layers':2,'dropout':0,'batch_pairs':len(train),'backbone_lr':1e-4,'context_lr':5e-4,'weight_decay':1e-4,'temperature':.07,'clip':1,
            'sparse_sizes':[],'sparse_weight':0.,'context_bank_rows':len(events)+len(model.encoder._pair_roots),
            'input_frozen':True,'pseudo_labels':False,'fixed_feature_branch':False,'source_sha256':sources,
            'cache_sha256':{f:sha256(cache/f) for f in ['events.npz','splits.npz','protocol.json','time_coordinates.npy','relation_alias.npy']},
            'selection':'context-phase best single-model CSLS5 validation MRR; no test-label access','scorer':'single seed CSLS5 for checkpoint selection, fixed200row/column normalization0.05 after freezing; no ensemble in this screening',
            'environment':{'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)},
            'full_candidate_gradients':True,'exact_context_replay':True,'test_labels_read':False}
        save_json(out/'config.json',config)
    if args.sinkhorn_every:
        from .frozen_sinkhorn_validation import balanced_ranks
        from .benchmark_retrieval import density
        configuration=json.loads((out/'config.json').read_text())
        configuration.update(selection='best context checkpoint by validation Sinkhorn MRR',
            scorer='CSLS5 monitoring; full-pool Sinkhorn validation checkpoint selection; no ensemble',
            sinkhorn_checkpoint_steps=list(range(200+args.sinkhorn_every,401,args.sinkhorn_every)),
            sinkhorn_temperature=.05,sinkhorn_iterations=200,normalization_pool='all source x all target entities',
            relation_policy='frozen shared direction; induced arm adds log1p untruncated head/tail training-seed incidence with shared frozen projection; query root removed in direct and context rows',
            relation_namespace='identical labels share a base relation and pooled seed counts; different labels use separate counts and a shared projection',
            relation_feature_time_added=False,relation_feature_random_seed=11732026,
            relation_profiles_sha256=sha256(args.relation_profiles) if args.relation_profiles else None)
        save_json(out/'config.json',configuration)
    groups=validation_groups(cache,train,dev)
    if args.time_mask_fraction:
        if events.shape[2] == 4:
            has_common_day = common_day_mask(events, dev)
            groups.update(common_day=has_common_day, no_common_day=~has_common_day)
        else:
            groups.update(interval_time_groups(events, dev))
    frozen={k:v.cpu().clone() for k,v in model.state_dict().items() if k not in {k for k,p in model.named_parameters() if p.requires_grad or id(p) in extra_ids}}
    started=time.perf_counter();row=previous
    with (out/'epochs.jsonl').open('a' if args.resume else 'w',buffering=1) as logfile:
        for step in range(first,401):
            if step==first and args.resume:continue
            tick=time.perf_counter()
            if step>200 and not context:
                assert all(float(v['step'])==200 for v in optimizer.state.values())
                model.encoder.use_neighbor=True
                for parameter in extra:parameter.requires_grad_(True)
                optimizer.add_param_group({'params':extra,'lr':5e-4,'weight_decay':1e-4});context=True
            row={'step':step,'context_enabled':context}
            if step>0:
                model.train();optimizer.zero_grad(set_to_none=True)
                verify=step in [1,21,200,201,300,400]
                training_events = events
                if args.time_mask_fraction:
                    training_events, masked_by_side = mask_one_side_time(
                        events, train, args.time_mask_fraction, args.seed, step)
                    row['time_masked_train_entities_by_side'] = masked_by_side
                loss,parts,error=cached_complete_backward(model,training_events,train,chunk=args.chunk,verify_replay=verify)
                assert np.isfinite(loss)
                gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));optimizer.step()
                row.update(loss=loss,loss_parts=parts,gradient_norm=gradient,replay_errors=error if verify else None)
            row['update_seconds']=time.perf_counter()-tick
            if step in (0,1,200,201,400) or step%args.eval_every==0:
                tick=time.perf_counter();embedding=encode_root_mask(model,events,args.chunk)
                rank=ranks(embedding,dev,n);measure=metrics(rank)
                selection_measure=measure;eligible_selection=context and not args.sinkhorn_every
                balanced=None
                if context and args.sinkhorn_every and step%args.sinkhorn_every==0:
                    rx,ry=density(embedding[:n],embedding[n:])
                    balanced,diagnostics=balanced_ranks(embedding,dev,n,rx,ry)
                    selection_measure=metrics(balanced);eligible_selection=True
                    row.update(sinkhorn_validation=selection_measure,sinkhorn_diagnostics=diagnostics,
                               sinkhorn_groups=group_metrics(balanced,dev,groups))
                del embedding
                row.update(validation=measure,validation_groups=group_metrics(rank,dev,groups),validation_seconds=time.perf_counter()-tick)
                if eligible_selection and selection_measure['mrr']>best:
                    best=selection_measure['mrr'];beststep=step
                    if balanced is not None: np.save(out/'best_sinkhorn_validation_ranks.npy',balanced)
                    checkpoint(out/'best_training.pt',model,optimizer,step,args.seed,context)
                    np.save(out/'best_validation_ranks.npy',rank)
                if step>0:checkpoint(out/'latest_training.pt',model,optimizer,step,args.seed,context)
                if step==200:checkpoint(out/'stage0_training.pt',model,optimizer,step,args.seed,context)
            row.update(elapsed_seconds=time.perf_counter()-started,peak_memory_gb=torch.cuda.max_memory_allocated()/2**30)
            logfile.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            save_json(out/'status.json',{'state':'training','seed':args.seed,'gpu':args.gpu,**row,'best_mrr':best,'best_step':beststep,'test_labels_read':False})
    for name,value in frozen.items():assert torch.equal(value,model.state_dict()[name].cpu()),name
    saved=torch.load(out/'best_training.pt',map_location='cuda',weights_only=True);model.load_state_dict(saved['model'])
    vec=encode_root_mask(model,events,args.chunk);recovered=ranks(vec,dev,n)
    np.testing.assert_array_equal(recovered,np.load(out/'best_validation_ranks.npy'))
    np.save(out/'best_vectors.npy',vec.cpu().numpy())
    selected_extra={}
    if args.sinkhorn_every:
        rx,ry=density(vec[:n],vec[n:]);br,diag=balanced_ranks(vec,dev,n,rx,ry)
        np.testing.assert_array_equal(br,np.load(out/'best_sinkhorn_validation_ranks.npy'))
        selected_extra={'selected_csls5':metrics(recovered),'selected_sinkhorn':metrics(br),
                        'selected_sinkhorn_groups':group_metrics(br,dev,groups),'sinkhorn_diagnostics':diag}
    result={'state':'completed_validation','seed':args.seed,'best_mrr':best,'best_step':beststep,'final_validation':row['validation'],
        **selected_extra,'validation_groups':group_metrics(recovered,dev,groups),'sparse_weight':0.,'time_diagnostics':time_diagnostics(model.encoder,coordinates),'frozen_inputs_unchanged':True,'selected_checkpoint_rank_recovery_exact':True,'total_seconds':time.perf_counter()-started,
        'checkpoint_sha256':sha256(out/'best_training.pt'),'vector_sha256':sha256(out/'best_vectors.npy'),'test_labels_read':False}
    save_json(out/'results_validation.json',result);save_json(out/'status.json',result);print(json.dumps(result),flush=True)

def validation_groups(cache,train,dev):
    # Define diagnostic groups from original inputs, identically in both arms.
    control=np.load(cache.parent/'control/events.npz')
    lengths=control['lengths'];events=control['events']
    minimum=lengths[dev].min(1)
    seeds=train.ravel()+1
    anchored=np.isin(events[:,:,0],seeds).any(1)
    return {'min_events_1':minimum==1,'min_events_2':minimum==2,
            'min_events_1or2':(minimum>=1)&(minimum<=2),
            'min_events_gt2':minimum>2,
            'at_least_one_no_seed_object':(~anchored[dev]).any(1),
            'both_no_seed_object':(~anchored[dev]).all(1)}


def group_metrics(rank,dev,groups):
    return {name:metrics(rank[mask]) if mask.any() else {'queries':0}
            for name,mask in groups.items()}


if __name__=='__main__':main()
