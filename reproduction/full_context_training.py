"""Exact root-masked context derivatives for the complete alignment loss only."""
import numpy as np
import torch
from .chain_only import cached_full_backward, full_candidate_loss
from .set_training import needs_context


def cached_complete_backward(model,events,train,chunk=128,amp=True,verify_replay=False,
                             loss_fn=full_candidate_loss):
    if not needs_context(model):
        if loss_fn is not full_candidate_loss:
            raise ValueError('A custom loss requires the context replay path')
        loss,error=cached_full_backward(model,events,train,chunk=chunk,amp=amp,verify_replay=verify_replay)
        return loss,{'full':loss,'sparse_weight':0.},{'full':error}
    device=next(model.parameters()).device
    def rng():
        return torch.get_rng_state(),torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    def restore(state):
        torch.set_rng_state(state[0])
        if device.type=='cuda':torch.cuda.set_rng_state(state[1],device)
    def inputs(data,ids,start):
        return (torch.as_tensor(data[start:start+chunk],dtype=torch.long,device=device),
                torch.as_tensor(ids[start:start+chunk],dtype=torch.long,device=device))
    def cache(function,data,ids):
        parts,states=[],[]
        for start in range(0,len(data),chunk):
            states.append(rng());value,central=inputs(data,ids,start)
            with torch.enable_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
                output=function(value,central)
            parts.append(output.detach());del output
        return parts,states
    bank,bank_ids=model.encoder.context_inputs(events)
    base_parts,base_states=cache(model.encoder.base,bank,bank_ids)
    context=torch.cat(base_parts).detach().requires_grad_(True)
    forward=lambda value,ids:model(value,ids,context)
    ids=np.arange(len(events))
    parts,states=cache(forward,events,ids)
    outer_rng=rng()
    full=torch.cat(parts).detach().requires_grad_(True)
    loss=loss_fn(full,train,model.n_source)
    loss.backward()
    errors={'full':0.,'context':0.}
    def replay(function,data,central_ids,cached,recorded,derivative,name):
        for i,start in enumerate(range(0,len(data),chunk)):
            restore(recorded[i]);value,central=inputs(data,central_ids,start)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
                output=function(value,central)
            if verify_replay:
                error=float((output.detach()-cached[i]).abs().max())
                errors[name]=max(errors[name],error)
                if error>1e-6:raise AssertionError(f'{name} replay mismatch: {error}')
            output.backward(derivative[start:start+chunk])
    try:
        replay(forward,events,ids,parts,states,full.grad.detach(),'full')
        if context.grad is None:raise AssertionError('Missing context derivative')
        context_grad=context.grad.detach()
        replay(model.encoder.base,bank,bank_ids,base_parts,base_states,context_grad,'context')
    finally:
        restore(outer_rng)
    return float(loss),{'full':float(loss),'sparse_weight':0.,'context_gradient_norm':float(context_grad.norm())},errors
