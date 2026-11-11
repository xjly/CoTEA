"""Frozen direction and seed-induced relation slots, with root leave-one-out."""
import math
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import read_vocab, save_json, sha256
from .interval_context import IntervalEncoder
from .root_mask_context import RootMaskedEncoder
from .set_model import SetModel
from .priority_hybrid_model import HybridTime


def prepare_profiles(raw, cache, output):
    raw, cache, output = map(Path, (raw, cache, output))
    protocol = json.loads((cache/'protocol.json').read_text(encoding='utf-8'))
    for name, digest in protocol['source_sha256'].items():
        if (raw/name).exists(): assert sha256(raw/name) == digest, name
    train = np.load(cache/'splits.npz')['train']
    n = protocol['entities'][0]
    seed_index = {int(e): j for j, pair in enumerate(train) for e in pair}
    counts = np.zeros((protocol['shared_relations'], 2, len(train)), np.float64)
    relations = {v:i for i,v in enumerate(protocol['relation_names'])}
    facts = 0
    for graph, offset in ((1,0),(2,n)):
        entities = read_vocab(raw/f'ent_ids_{graph}')
        index = {v:i+offset for i,v in enumerate(sorted(entities))}
        rv = read_vocab(raw/f'rel_ids_{graph}')
        with (raw/f'triples_{graph}').open(encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                h,r,t = map(int,line.split()[:3])
                r = relations[rv[r]]
                for role, e in enumerate((index[h], index[t])):
                    if e in seed_index: counts[r,role,seed_index[e]] += 1
                facts += 1
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/'seed_relation_counts.npy',counts)
    save_json(output/'profiles.json',{'facts':facts,'counts_shape':list(counts.shape),
        'train_pairs':len(train),'untruncated':True,'time_features_added':False,
        'test_labels_read':False,'counts_sha256':sha256(output/'seed_relation_counts.npy'),
        'splits_sha256':sha256(cache/'splits.npz')})
    return counts


class SeedRelations(nn.Module):
    def __init__(self, counts, train, entities, dim, arm):
        super().__init__()
        self.arm, self.dim = arm, dim
        gen = torch.Generator().manual_seed(11732026)
        orthogonal = torch.linalg.qr(torch.randn(dim,2,generator=gen)).Q.T * math.sqrt(dim)
        self.register_buffer('directions',orthogonal)
        # Independent fixed head/tail projections; common seed axes across graphs.
        projection = torch.randn(2, len(train), dim, generator=gen)
        coefficient = torch.tensor(np.log1p(counts),dtype=torch.float32)
        self.register_buffer('coefficient',coefficient)
        self.register_buffer('support',(coefficient>0).sum((1,2)))
        self.register_buffer('projection',projection)
        self.register_buffer('totals',torch.einsum('rks,ksd->rd',coefficient,projection))
        seed = torch.full((entities,),-1,dtype=torch.long)
        for j,(a,b) in enumerate(train): seed[int(a)]=seed[int(b)]=j
        self.register_buffer('seed_index',seed)
        self.excluded = None

    def forward(self, ids):
        if self.arm == 'zero': return self.totals.new_zeros((*ids.shape,self.dim))
        valid = ids > 0
        base = ((ids-1)//2).clamp_min(0)
        result = self.directions[(ids-1).clamp_min(0)%2]
        if self.arm == 'induced':
            profile = self.totals[base]
            support = self.support[base]
            if self.excluded is not None:
                j = self.excluded.clamp_min(0)
                contribution = (self.coefficient[base,:,j][:,:,None] * self.projection[:,j].permute(1,0,2)).sum(1)
                profile = profile - contribution * (self.excluded>=0)[:,None]
                support = support - (self.coefficient[base,:,j]>0).sum(1)*(self.excluded>=0)
            profile = profile * (support>0)[:,None]
            profile = F.normalize(profile,dim=-1,eps=1e-6)*math.sqrt(self.dim)
            result = (result + profile)/torch.sqrt(1.+(support>0).float())[:,None]
        return result * valid[:,None]


class _SeedRelationInputs:
    def install_seed_relation_inputs(self, protocol, train, counts, coordinates, arm, dim):
        if counts.shape != (protocol['shared_relations'], 2, len(train)):
            raise ValueError('Relation profile dimensions do not match the cache and training split')
        self.relations = SeedRelations(counts,train,self.entities,dim,arm)
        if 'ymd_components' in protocol:
            from .beta_calendar import MaskedYMDTime
            self.times = MaskedYMDTime(protocol['ymd_components'],dim)
        self.times = HybridTime(self.times,coordinates)
        self.register_buffer('pair_roots',torch.tensor(self._pair_roots),persistent=False)
        self.register_buffer('pair_objects',torch.tensor(self._pair_objects),persistent=False)

    def context_inputs(self, events):
        bank, ids = super().context_inputs(events)
        # Encoded IDs retain the excluded root through chunked context replay.
        ids[self.entities:] = self.entities + np.arange(len(ids)-self.entities)
        return bank, ids

    def event_vectors(self, events, central_ids, context=None):
        extra = central_ids >= self.entities
        actual, roots = central_ids.clone(), central_ids.clone()
        if extra.any():
            index = central_ids[extra]-self.entities
            actual[extra]=self.pair_objects[index]
            roots[extra]=self.pair_roots[index]
        valid = events[:,:,0]>0
        excluded = self.relations.seed_index[roots][:,None].expand_as(valid)[valid]
        self.relations.excluded=excluded
        try: return super().event_vectors(events,actual,context)
        finally: self.relations.excluded=None


class SeedRelationEncoder(_SeedRelationInputs, IntervalEncoder):
    def __init__(self, protocol, events, train, counts, coordinates, arm, dim=512, heads=8):
        super().__init__(protocol,events,train,dim,heads,time_mode='split')
        self.install_seed_relation_inputs(protocol,train,counts,coordinates,arm,dim)


class SeedRelationPointEncoder(_SeedRelationInputs, RootMaskedEncoder):
    def __init__(self, protocol, events, train, counts, coordinates, arm, dim=512, heads=8):
        super().__init__(protocol,events,train,dim,heads)
        self.install_seed_relation_inputs(protocol,train,counts,coordinates,arm,dim)


class SeedRelationModel(SetModel):
    def __init__(self,protocol,events,train,counts,coordinates,arm,dim=512,heads=8):
        nn.Module.__init__(self)
        self.n_source=protocol['entities'][0]
        encoder=SeedRelationEncoder if events.shape[-1]==5 else SeedRelationPointEncoder
        self.encoder=encoder(protocol,events,train,counts,coordinates,arm,dim,heads)
