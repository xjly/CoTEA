"""Bounded-memory full-pool CSLS; same cosine/FP64-density definition."""
import math
import numpy as np
import torch
from .train import average_ranks

@torch.inference_mode()
def density(left,right,k=5,block=512):
    assert 0<k<=min(len(left),len(right))
    rx=torch.empty(len(left),device=left.device,dtype=torch.float64)
    columns=torch.full((len(right),k),-torch.inf,device=left.device,dtype=torch.float32)
    for start in range(0,len(left),block):
        with torch.autocast(left.device.type,enabled=False):
            scores=left[start:start+block].float()@right.float().T
        assert torch.isfinite(scores).all()
        rx[start:start+len(scores)]=scores.topk(k,dim=1).values.double().mean(1)
        local=scores.topk(min(k,len(scores)),dim=0).values.T
        columns=torch.cat([columns,local],1).topk(k,dim=1).values
    return rx,columns.double().mean(1)

@torch.inference_mode()
def ranks(embedding,pairs,n_source,densities=None,block=64):
    left,right=embedding[:n_source],embedding[n_source:]
    rx,ry=densities if densities is not None else density(left,right)
    result=[]
    for start in range(0,len(pairs),block):
        p=pairs[start:start+block];q=torch.as_tensor(p[:,0],device=embedding.device)
        with torch.autocast(embedding.device.type,enabled=False):raw=left[q].float()@right.float().T
        score=2*raw.double()-rx[q,None]-ry[None,:]
        gold=torch.as_tensor(p[:,1]-n_source,device=embedding.device)
        result.append(average_ranks(score,gold).cpu().numpy())
    return np.concatenate(result).astype(np.float64)

@torch.inference_mode()
def balance_inplace(matrix,temperature=.05,iterations=200,block=1024,callback=None):
    """Input already contains CSLS; scale in place, with bounded temporaries."""
    assert matrix.dtype==torch.float64
    n,m=matrix.shape;matrix.div_(temperature)
    u=torch.zeros(n,device=matrix.device,dtype=matrix.dtype);v=torch.zeros(m,device=matrix.device,dtype=matrix.dtype)
    for iteration in range(iterations):
        for start in range(0,n,block):
            u[start:start+block]=-math.log(n)-torch.logsumexp(matrix[start:start+block]+v[None,:],1)
        column_lse=torch.full((m,),-torch.inf,device=matrix.device,dtype=matrix.dtype)
        for start in range(0,n,block):
            part=torch.logsumexp(matrix[start:start+block]+u[start:start+block,None],0)
            column_lse=torch.logaddexp(column_lse,part)
        v=-math.log(m)-column_lse
        if callback and (iteration==0 or (iteration+1)%20==0):callback(iteration+1)
    return u,v
