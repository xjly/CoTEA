"""Shared Transformer context for unknown, nonself object slots only."""
import math
import torch
from torch import nn
from .controlled import mask_self_objects
from .set_model import SetEncoder,SetModel


class LearnedObjectEncoder(SetEncoder):
    def __init__(self,protocol,events,train,dim=512,heads=8):
        # Same tensors as base; the formerly frozen zero context map is enabled.
        super().__init__(protocol,events,train,kind='neighbor',dim=dim,heads=heads)

    def event_vectors(self,events,central_ids,context=None):
        value=mask_self_objects(events,central_ids,self.entities)
        valid=events[:,:,0]>0;real=value[valid]
        if not len(real):return self.empty.new_zeros(*valid.shape,self.dim),valid,value
        objects=self.objects(real[:,0]);relations=self.relations(real[:,1]);times=self.times(real[:,2])
        if context is not None:
            raw=events[:,:,0][valid]-1
            centers=central_ids[:,None].expand_as(valid)[valid]
            eligible=(self.objects.alias[real[:,0]]==0)&(raw!=centers)
            routed=self.diagnostic_context_permutation[raw] if self.intervention=='shuffle_context' else raw
            message=self.context_project(context[routed]*math.sqrt(self.dim))
            gate=torch.sigmoid(self.context_gate(torch.cat([relations,times],-1)))
            # Known gold anchors retain their original shared random vectors.
            # Raw IDs route within the observed graph; they are not ID embeddings.
            objects=objects+message*gate*eligible[:,None]
        slots=torch.stack([objects,relations,times,self.frequency(torch.log1p(real[:,3].float())[:,None])],1)+self.slot_position
        missing=torch.zeros(len(real),4,dtype=torch.bool,device=events.device);missing[:,2]=real[:,2]==0
        for layer in self.intra:slots=layer(slots,src_key_padding_mask=missing)
        slots=self.intra_norm(slots)
        vector=(slots*(~missing)[:,:,None]).sum(1)/(~missing).sum(1)[:,None]
        sequence=vector.new_zeros(*valid.shape,self.dim);sequence[valid]=vector
        return sequence,valid,value


class LearnedObjectModel(SetModel):
    def __init__(self,protocol,events,train,dim=512,heads=8):
        nn.Module.__init__(self);self.n_source=protocol['entities'][0]
        self.encoder=LearnedObjectEncoder(protocol,events,train,dim=dim,heads=heads)
