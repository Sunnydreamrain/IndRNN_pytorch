import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    #if args.cuda:
    data = data.cuda()
    return data

from __main__ import parser,args
def batchify_drop(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    import numpy as np
    n = int(len(data0)*args.dropout_in)
    indices = sorted(np.random.choice(len(data0)-1,len(data0)-n,replace=False))
    #print(indices[:100])
    #assert 2==3
    data=data0[indices]
    indices1=np.array(indices)+1
    indices1=indices1.tolist()
    target=data0[indices1]
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    #if args.cuda:
    train_data=data.cuda()
    target=target.narrow(0, 0, nbatch * bsz)
    target=target.view(bsz, -1).t().contiguous()
    target=target.cuda()
    return train_data,target
def get_batch_drop(source, target, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target=target[i:i+seq_len].view(-1)
    #target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
