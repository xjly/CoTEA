"""Exact derivatives through an enlarged, root-masked Transformer context bank."""
import numpy as np
import torch
from .chain_only import full_candidate_loss
from .sparse_views import sparse_loss
from .set_training import needs_context


def cached_root_mask_backward(model,events,train,views=None,query_ids=None,weight=.1,chunk=128,amp=True,verify_replay=False):
    if not needs_context(model):raise ValueError('An enabled learned context model is required')
    if weight<=0:raise ValueError('Root-masked screen uses the active sparse124 phase only')
    expected=np.concatenate([train[:,0],train[:,1]])
    if views is None or query_ids is None or not np.array_equal(query_ids,expected):raise ValueError('Sparse views must use original training entities only')
    device=next(model.parameters()).device
    def rng():return torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    def restore(state):
        torch.set_rng_state(state[0])
        if device.type=='cuda':torch.cuda.set_rng_state(state[1],device)
    def inputs(data,ids,start):
        return torch.as_tensor(data[start:start+chunk],dtype=torch.long,device=device),torch.as_tensor(ids[start:start+chunk],dtype=torch.long,device=device)
    def cache(function,data,ids):
        parts,states=[],[]
        for start in range(0,len(data),chunk):
            states.append(rng());value,central=inputs(data,ids,start)
            with torch.enable_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
                output=function(value,central)
            parts.append(output.detach());del output
        return parts,states
    full_ids=np.arange(len(events))
    bank_events,bank_ids=model.encoder.context_inputs(events)
    base_parts,base_states=cache(model.encoder.base,bank_events,bank_ids)
    context=torch.cat(base_parts).detach().requires_grad_(True)
    forward=lambda value,ids:model(value,ids,context)
    full_parts,full_states=cache(forward,events,full_ids)
    sparse_parts,sparse_states=cache(forward,views,query_ids)
    outer_rng=rng();full=torch.cat(full_parts).detach().requires_grad_(True);queries=torch.cat(sparse_parts).detach().requires_grad_(True)
    complete=full_candidate_loss(full,train,model.n_source);auxiliary=sparse_loss(queries,full,train,model.n_source)
    loss=complete+weight*auxiliary;loss.backward()
    errors={'full':0.,'sparse':0.,'context':0.}
    def replay(function,data,ids,parts,states,derivative,name):
        for index,start in enumerate(range(0,len(data),chunk)):
            restore(states[index]);value,central=inputs(data,ids,start)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):output=function(value,central)
            if verify_replay:
                error=float((output.detach()-parts[index]).abs().max());errors[name]=max(errors[name],error)
                if error>1e-6:raise AssertionError(f'{name} replay mismatch: {error}')
            output.backward(derivative[start:start+chunk])
    try:
        replay(forward,events,full_ids,full_parts,full_states,full.grad.detach(),'full')
        replay(forward,views,query_ids,sparse_parts,sparse_states,queries.grad.detach(),'sparse')
        if context.grad is None:raise AssertionError('Missing learned-object context derivatives')
        context_derivative=context.grad.detach()
        # Both center and query passes contribute to the shared context encoder.
        replay(model.encoder.base,bank_events,bank_ids,base_parts,base_states,context_derivative,'context')
    finally:restore(outer_rng)
    return float(loss),{'full':float(complete),'sparse':float(auxiliary),'weight':weight,'context_gradient_norm':float(context_derivative.norm())},errors


@torch.inference_mode()
def encode_root_mask(model,events,chunk=128,amp=True):
    model.eval();device=next(model.parameters()).device;context=None
    def encode(data,ids,function):
        values=[]
        for start in range(0,len(data),chunk):
            value=torch.as_tensor(data[start:start+chunk],dtype=torch.long,device=device)
            central=torch.as_tensor(ids[start:start+chunk],dtype=torch.long,device=device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):values.append(function(value,central))
        return torch.cat(values)
    if needs_context(model):
        bank,ids=model.encoder.context_inputs(events);context=encode(bank,ids,model.encoder.base)
    return encode(events,np.arange(len(events)),lambda value,ids:model(value,ids,context))
