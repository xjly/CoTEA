"""Object contexts that cannot return the querying gold root's own seed code."""
import numpy as np
import torch
from torch import nn
from .learned_object_context import LearnedObjectEncoder
from .set_model import SetModel


class _RoutedBank:
    """Adapt the existing object-token lookup to query-conditioned context rows."""
    def __init__(self,bank,route,centers):self.bank,self.route,self.centers=bank,route,centers
    def __getitem__(self,objects):return self.bank[self.route(self.centers,objects)]


class RootMaskedEncoder(LearnedObjectEncoder):
    def __init__(self,protocol,events,train,dim=512,heads=8):
        super().__init__(protocol,events,train,dim=dim,heads=heads)
        alias=self.objects.alias.cpu().numpy();pairs=[]
        for root in sorted(set(train.flatten().tolist())):
            for token in np.unique(events[root,:,0]):
                obj=int(token)-1
                if token>0 and obj!=root and alias[int(token)]==0:pairs.append((root,obj))
        pairs.sort();self._pair_roots=np.array([r for r,o in pairs],dtype=np.int64)
        self._pair_objects=np.array([o for r,o in pairs],dtype=np.int64)
        keys=self._pair_roots*self.entities+self._pair_objects
        self.register_buffer('_root_pair_keys',torch.from_numpy(keys),persistent=False)

    def context_inputs(self,events):
        """Same fixed graph, with extra copies whose returned root token is zeroed.

        Sentinel entities+1 is the existing valid-row zero-object sentinel; it
        preserves relation, time and frequency, and other anchors remain visible.
        Object IDs, not root IDs, are the central IDs of these copied context rows.
        """
        if len(events)!=self.entities:raise ValueError('Expected the complete original fixed graph cache')
        if not len(self._pair_roots):return events,np.arange(self.entities)
        extra=events[self._pair_objects].copy()
        extra[:,:,0]=np.where(extra[:,:,0]==self._pair_roots[:,None]+1,self.entities+1,extra[:,:,0])
        return np.concatenate([events,extra]),np.concatenate([np.arange(self.entities),self._pair_objects])

    def route_context(self,centers,objects):
        if not self._root_pair_keys.numel():return objects
        keys=centers*self.entities+objects
        positions=torch.searchsorted(self._root_pair_keys,keys)
        found=(positions<len(self._root_pair_keys))&(self._root_pair_keys[positions.clamp_max(len(self._root_pair_keys)-1)]==keys)
        return torch.where(found,self.entities+positions,objects)

    def event_vectors(self,events,central_ids,context=None):
        if context is not None:
            if self.intervention=='shuffle_context':raise ValueError('Unspecified root-conditioned shuffle intervention')
            valid=events[:,:,0]>0;centers=central_ids[:,None].expand_as(valid)[valid]
            context=_RoutedBank(context,self.route_context,centers)
        return super().event_vectors(events,central_ids,context)


class RootMaskedModel(SetModel):
    def __init__(self,protocol,events,train,dim=512,heads=8):
        nn.Module.__init__(self);self.n_source=protocol['entities'][0]
        self.encoder=RootMaskedEncoder(protocol,events,train,dim=dim,heads=heads)
