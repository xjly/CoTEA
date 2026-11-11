"""Full-interval events: object, relation, start, frequency, end; paired time readout."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .root_mask_context import RootMaskedEncoder
from .set_model import SetModel,day_membership
from .controlled import mask_self_objects

class IntervalEncoder(RootMaskedEncoder):
    def __init__(self,protocol,events,train,dim=512,heads=8,time_mode='split'):
        if time_mode not in ('fused','split'):raise ValueError(time_mode)
        super().__init__(protocol,events,train,dim,heads)
        self.time_mode=time_mode;self.time_base=protocol['time_tokens']+1
        generator=torch.Generator().manual_seed(732027)
        self.register_buffer('end_role',torch.randn(dim,generator=generator)*.02)

    def event_vectors(self,events,central_ids,context=None):
        if events.shape[-1]!=5:raise ValueError('Full intervals require5columns')
        value=mask_self_objects(events,central_ids,self.entities);valid=events[:,:,0]>0;real=value[valid]
        if not len(real):return self.empty.new_zeros(*valid.shape,self.dim),valid,value
        objects=self.objects(real[:,0]);relations=self.relations(real[:,1])
        start,end=self.times(real[:,2]),self.times(real[:,4])
        start_ok,end_ok=real[:,2]>0,real[:,4]>0
        time_mean=(start*start_ok[:,None]+end*end_ok[:,None])/(start_ok.long()+end_ok.long()).clamp_min(1)[:,None]
        if context is not None:
            raw=events[:,:,0][valid]-1;centers=central_ids[:,None].expand_as(valid)[valid]
            eligible=(self.objects.alias[real[:,0]]==0)&(raw!=centers)
            routed=self.route_context(centers,raw)
            message=self.context_project(context[routed]*math.sqrt(self.dim))
            gate=torch.sigmoid(self.context_gate(torch.cat([relations,time_mean],-1)))
            objects=objects+message*gate*eligible[:,None]
        frequency=self.frequency(torch.log1p(real[:,3].float())[:,None]);p=self.slot_position
        if self.time_mode=='split':
            slots=torch.stack([objects+p[0],relations+p[1],start+p[2],end+self.end_role,frequency+p[3]],1)
            missing=torch.zeros(len(real),5,dtype=torch.bool,device=events.device)
            missing[:,2]=~start_ok;missing[:,3]=~end_ok
        else:
            slots=torch.stack([objects+p[0],relations+p[1],time_mean+p[2],frequency+p[3]],1)
            missing=torch.zeros(len(real),4,dtype=torch.bool,device=events.device);missing[:,2]=~(start_ok|end_ok)
        for layer in self.intra:slots=layer(slots,src_key_padding_mask=missing)
        slots=self.intra_norm(slots);vector=(slots*(~missing)[:,:,None]).sum(1)/(~missing).sum(1)[:,None]
        sequence=vector.new_zeros(*valid.shape,self.dim);sequence[valid]=vector
        return sequence,valid,value

    def pool(self,sequence,events,valid,conditioned=False):
        assert not conditioned,'No extra relation pooling branch in this experiment'
        key=events[:,:,2]*self.time_base+events[:,:,4]
        membership,present=day_membership(key,valid)
        with torch.autocast(sequence.device.type,enabled=False):
            group=membership.float();count=group.sum(-1)
            means=torch.bmm(group,sequence.float())/count.clamp_min(1)[:,:,None]
        means=self.pool_norm(means)
        pooled=(means*present[:,:,None]).sum(1)/present.sum(1).clamp_min(1)[:,None]
        pooled=torch.where(present.any(1)[:,None],pooled,self.empty)
        return F.normalize(pooled.float(),dim=-1,eps=1e-8)

class IntervalModel(SetModel):
    def __init__(self,protocol,events,train,dim=512,heads=8,time_mode='split'):
        nn.Module.__init__(self);self.n_source=protocol['entities'][0]
        self.encoder=IntervalEncoder(protocol,events,train,dim,heads,time_mode)
