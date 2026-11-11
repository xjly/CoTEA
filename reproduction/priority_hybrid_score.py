"""Frozen single-model validation with memory-bounded global balancing."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .benchmark_retrieval import density, ranks
from .data import save_json,sha256
from .priority_hybrid_train import validation_groups,group_metrics
from .train import metrics,average_ranks


@torch.inference_mode()
def streamed_balance(left,right,rx,ry,iterations=200,temperature=.05,block=256,callback=None):
    """Same log-domain row/column normalization, recomputing fixed score blocks."""
    n,m=len(left),len(right)
    u=torch.zeros(n,device=left.device,dtype=torch.float64)
    v=torch.zeros(m,device=left.device,dtype=torch.float64)
    def logits(start):
        raw=left[start:start+block].float()@right.float().T
        return (2*raw.double()-rx[start:start+len(raw),None]-ry[None,:])/temperature
    for iteration in range(iterations):
        for start in range(0,n,block):
            value=logits(start)
            u[start:start+len(value)]=-math.log(n)-torch.logsumexp(value+v[None,:],1)
        total=torch.full((m,),-torch.inf,device=left.device,dtype=torch.float64)
        for start in range(0,n,block):
            value=logits(start)
            total=torch.logaddexp(total,torch.logsumexp(value+u[start:start+len(value),None],0))
        v=-math.log(m)-total
        if callback and (iteration==0 or (iteration+1)%20==0):callback(iteration+1)
    return u,v


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--run',required=True)
    p.add_argument('--output',required=True);p.add_argument('--block',type=int,default=256)
    a=p.parse_args();cache,run,out=map(Path,(a.cache,a.run,a.output));out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.11)
    selection=json.loads((run/'results_validation.json').read_text())
    assert selection['selected_checkpoint_rank_recovery_exact'] and not selection['test_labels_read']
    protocol=json.loads((cache/'protocol.json').read_text());n=protocol['entities'][0]
    split=np.load(cache/'splits.npz');dev,train=split['dev'],split['train']
    started=time.perf_counter()
    with torch.inference_mode():
        embedding=torch.from_numpy(np.load(run/'best_vectors.npy')).cuda()
        left,right=embedding[:n],embedding[n:]
        rx,ry=density(left,right)
        native=ranks(embedding,dev,n,(rx,ry))
        np.testing.assert_array_equal(native,np.load(run/'best_validation_ranks.npy'))
        def progress(iteration):
            row={'state':'normalizing','iteration':iteration,'elapsed_seconds':time.perf_counter()-started}
            save_json(out/'status.json',row);print(json.dumps(row),flush=True)
        u,v=streamed_balance(left,right,rx,ry,block=a.block,callback=progress)
        values=[]
        for start in range(0,len(dev),64):
            pairs=dev[start:start+64];q=torch.as_tensor(pairs[:,0],device='cuda')
            raw=left[q].float()@right.float().T
            score=(2*raw.double()-rx[q,None]-ry[None,:])/.05+u[q,None]+v[None,:]
            gold=torch.as_tensor(pairs[:,1]-n,device='cuda')
            values.append(average_ranks(score,gold).cpu().numpy())
        balanced=np.concatenate(values)
        np.savez(out/'balance_offsets.npz',u=u.cpu().numpy(),v=v.cpu().numpy())
    np.save(out/'csls5_validation_ranks.npy',native);np.save(out/'balanced_validation_ranks.npy',balanced)
    groups=validation_groups(cache,train,dev)
    result={'state':'completed_validation','best_step':selection['best_step'],'seed':2026,
        'csls5':metrics(native),'balanced200':metrics(balanced),
        'csls5_groups':group_metrics(native,dev,groups),'balanced200_groups':group_metrics(balanced,dev,groups),
        'temperature':.05,'iterations':200,'ensemble':False,'test_labels_read':False,
        'checkpoint_sha256':sha256(run/'best_training.pt'),'vector_sha256':sha256(run/'best_vectors.npy'),
        'total_seconds':time.perf_counter()-started,'score_storage':'streamed fixed full-pool CSLS, FP64 log-domain balancing'}
    save_json(out/'results_validation.json',result);save_json(out/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
