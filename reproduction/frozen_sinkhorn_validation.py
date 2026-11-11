"""Full-pool log Sinkhorn validation, caching the fixed matrix on GPU."""
import math
import torch
import numpy as np
from .benchmark_retrieval import balance_inplace
from .train import average_ranks


@torch.inference_mode()
def balanced_ranks(embedding, pairs, n, rx, ry, block=256):
    if embedding.is_cuda:
        torch.cuda.empty_cache()
        free,_=torch.cuda.mem_get_info(embedding.device)
        if free >= 8*n*(len(embedding)-n)+2*2**30:
            try:
                return _dense_ranks(embedding,pairs,n,rx,ry,block)
            except torch.cuda.OutOfMemoryError:
                pass
        torch.cuda.empty_cache()
        return _streamed_ranks(embedding,pairs,n,rx,ry,block)
    return _dense_ranks(embedding,pairs,n,rx,ry,block)


@torch.inference_mode()
def _streamed_ranks(embedding,pairs,n,rx,ry,block):
    from .priority_hybrid_score import streamed_balance
    left,right=embedding[:n],embedding[n:]
    u,v=streamed_balance(left,right,rx,ry,block=block)
    values=[]
    for start in range(0,len(pairs),64):
        pair=torch.as_tensor(pairs[start:start+64],device=embedding.device)
        q=pair[:,0]
        raw=left[q].float()@right.float().T
        score=(2*raw.double()-rx[q,None]-ry[None,:])/.05+u[q,None]+v[None,:]
        values.append(average_ranks(score,pair[:,1]-n).cpu().numpy())
    row_error=0.;column=torch.full_like(v,-torch.inf)
    for start in range(0,n,block):
        raw=left[start:start+block].float()@right.float().T
        end=start+len(raw)
        logp=(2*raw.double()-rx[start:end,None]-ry[None,:])/.05+u[start:end,None]+v[None,:]
        row_error=max(row_error,float(torch.expm1(torch.logsumexp(logp,1)+math.log(n)).abs().max()))
        column=torch.logaddexp(column,torch.logsumexp(logp,0))
    return np.concatenate(values),{'iterations':200,'temperature':.05,'storage':'streamed',
        'row_relative_error_max':row_error,
        'column_relative_error_max':float(torch.expm1(column+math.log(len(right))).abs().max())}


@torch.inference_mode()
def _dense_ranks(embedding, pairs, n, rx, ry, block):
    left,right=embedding[:n],embedding[n:]
    matrix=torch.empty((len(left),len(right)),dtype=torch.float64,device=embedding.device)
    for start in range(0,len(left),block):
        raw=left[start:start+block].float()@right.float().T
        matrix[start:start+len(raw)]=2*raw.double()-rx[start:start+len(raw),None]-ry[None,:]
    u,v=balance_inplace(matrix,temperature=.05,iterations=200,block=block)
    values=[]
    for start in range(0,len(pairs),64):
        pair=torch.as_tensor(pairs[start:start+64],device=embedding.device)
        score=matrix[pair[:,0]]+u[pair[:,0],None]+v[None,:]
        values.append(average_ranks(score,pair[:,1]-n).cpu().numpy())
    # Relative marginal errors distinguish convergence from merely 200 iterations.
    row_error=0.;column=torch.full_like(v,-torch.inf)
    for start in range(0,len(left),block):
        logp=matrix[start:start+block]+u[start:start+block,None]+v[None,:]
        row_error=max(row_error,float(torch.expm1(torch.logsumexp(logp,1)+math.log(len(left))).abs().max()))
        column=torch.logaddexp(column,torch.logsumexp(logp,0))
    diagnostics={'iterations':200,'temperature':.05,'storage':'dense','row_relative_error_max':row_error,
                 'column_relative_error_max':float(torch.expm1(column+math.log(len(right))).abs().max())}
    del matrix
    torch.cuda.empty_cache() if embedding.is_cuda else None
    return np.concatenate(values),diagnostics
