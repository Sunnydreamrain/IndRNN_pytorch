import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import batchify, get_batch


import opts
parser = argparse.ArgumentParser(description='pytorch cPTB')
opts.train_opts(parser)

parser.add_argument('--val', action='store_true',
                    help='set for validation error, test by default')
parser.add_argument('--lamb', type=float, default=0.002,
                    help='decay parameter lambda')
parser.add_argument('--epsilon', type=float, default=0.00002,
                    help='stabilization parameter epsilon')
parser.add_argument('--oldhyper', action='store_true',
                    help='Transforms hyperparameters, equivalent to running old version of code')
parser.add_argument('--grid', action='store_true',
                    help='grid search for best hyperparams')
parser.add_argument('--gridfast', action='store_true',
                    help='grid search with partial validation set')
parser.add_argument('--data_len', type=int, default=20,
                    help='decay parameter lambda')

args = parser.parse_args()


print('loading')

import denseIndrnn_word_network_dropafter_FC as language_model
gradientclip_value = args.gradclipvalue
U_bound = language_model.U_bound


###############################################################################
# Load data
###############################################################################
import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 1
test_batch_size = 1



def gradstat():
    total_loss = 0
    ntokens = len(corpus.dictionary)
    batch, i = 0, 0

    for param in model.parameters():
        param.MS = 0*param.data  

    start_len=0
    hidden=None 
    while i < train_data.size(0) - 1 - 1:
        seq_len = args.data_len
        model.eval()

        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        model.zero_grad()
        output, hidden = model(data, hidden, start_len)

        start_len=start_len+seq_len
        loss = criterion(output, targets)
        loss.backward()
        clip_gradient(model, gradientclip_value)#_usemaxgrad  
        

        for param in model.parameters():
            param.MS = param.MS + param.grad.data*param.grad.data
        clip_weight(model, U_bound)  # different part
        #hidden = hidden.detach()   
        for key in hidden.keys():
            hidden[key].detach_() 


        total_loss += loss.data
        batch += 1
        i += seq_len

    gsum = 0
    count = 0

    for param in model.parameters():
        param.MS = torch.sqrt(param.MS/batch)
        gsum+=torch.mean(param.MS)
        count+=1
    gsum/=count
    if args.oldhyper:
        args.lamb /=count
        args.lr /=math.sqrt(batch)
        args.epsilon /=math.sqrt(batch)
        print("transformed lambda: " + str(args.lamb))
        print("transformed lr: " + str(args.lr))
        print("transformed epsilon: " + str(args.epsilon))


    for param in model.parameters():
        param.decrate = param.MS/gsum
        param.data0 = 1*param.data

def evaluate():
    #clips decay rates at 1/lamb
    #otherwise scaled decay rates can be greater than 1
    #would cause decay updates to overshoot
    for param in model.parameters():
        decratenp = param.decrate.cpu().numpy()
        ind = np.nonzero(decratenp>(1/lamb))
        decratenp[ind] = (1/lamb)
        param.decrate = torch.from_numpy(decratenp).type(torch.cuda.FloatTensor)

    total_loss = 0
    ntokens = len(corpus.dictionary)
    batch, i = 0, 0
    last = False
    seq_len= args.data_len
    seq_len0 = seq_len
    hidden=None      
    start_len=0 
    #loops through data
    while i < eval_data.size(0) - 1 - 1:

        model.eval()
        #gets last chunk of seqlence if seqlen doesn't divide full sequence cleanly
        if (i+seq_len)>=eval_data.size(0):
            if last:
                break
            seq_len = eval_data.size(0)-i-1
            last = True

        data, targets = get_batch(eval_data,i, args, seq_len=seq_len)
        model.zero_grad()

        output, hidden = model(data, hidden, start_len)
        start_len=start_len+seq_len
        loss = criterion(output, targets)

        #compute gradient on sequence segment loss
        loss.backward()
        clip_gradient(model, gradientclip_value)#_usemaxgrad

        #update rule
        for param in model.parameters():
            dW = lamb*param.decrate*(param.data0-param.data)-lr*param.grad.data/(param.MS+epsilon)
            param.data+=dW
        clip_weight(model, U_bound)
        for key in hidden.keys():
            hidden[key].detach_()             
        #seq_len/seq_len0 will be 1 except for last sequence
        #for last sequence, we downweight if sequence is shorter
        total_loss += (seq_len/seq_len0)*loss.data
        batch += (seq_len/seq_len0)

        i += seq_len

    #since entropy of first token was never measured
    #can conservatively measure with uniform distribution
    #makes very little difference, usually < 0.01 perplexity point
    #total_loss += (1/seq_len0)*torch.log(torch.from_numpy(np.array([ntokens])).type(torch.cuda.FloatTensor))
    #batch+=(1/seq_len0)

    perp = torch.exp(total_loss/batch)
    return perp.cpu().numpy()

def clip_gradient_usemaxgrad(model, clip):
    tempmax_all = 0
    for name, p in model.named_parameters():
        #print(name)
        tempmax_all = max(tempmax_all, p.grad.data.abs().max())
    clip_coef = clip / (tempmax_all + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return clip_coef
def clip_gradient(model, clip):
    for p in model.parameters():
        p.grad.data.clamp_(-clip,clip)

def clip_weight(RNNmodel, clip):
    for name, param in RNNmodel.named_parameters():
        if 'weight_hh' in name:
            param.data.clamp_(-clip, clip)
#load model
with open('indrnn_model', 'rb') as f:
    model = torch.load(f)

ntokens = len(corpus.dictionary)
criterion = nn.CrossEntropyLoss()

val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

if args.val== True:
    eval_data= val_data
else:
    eval_data=test_data
train_data = batchify(corpus.train, args.batch_size, args)

print('collecting gradient statistics')
#collect gradient statistics on training data
gradstat()

lr = args.lr
lamb = args.lamb
epsilon = args.epsilon

#change batch size to 1 for dynamic eval
args.batch_size=1
if not(args.grid or args.gridfast):
    print('running dynamic evaluation')
    #apply dynamic evaluation
    loss = evaluate()
    print('perplexity loss: ' + str(loss))#[0]
else:
    vbest = 99999999
    lambbest = lamb
    lrbest = lr

    if args.gridfast:
        eval_data = val_data[:30000]
    else:
        eval_data = val_data
    print('tuning hyperparameters')

    #hyperparameter values to be searched
    lrlist = [3e-7,4e-7,5e-7,6e-7,7e-7,8e-7,1e-6,2e-6,3e-6,4e-6] #[0.00003,0.00004,0.00005,0.00006,0.00007,0.0001]
    lamblist = [0.0005,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008]

    #rescale values if sequence segment length is changed
    lrlist = [x*(args.data_len/5.0) for x in lrlist]
    lamblist = [x*(args.data_len/5.0) for x in lamblist]

    for i in range(0,len(lamblist)):
        for j in range(0,len(lrlist)):
            lamb = lamblist[i]
            lr = lrlist[j]
            loss = evaluate()
            loss = loss#[0]
            print(loss)
            if loss<vbest:
                lambbest = lamb
                lrbest = lr
                vbest = loss
            for param in model.parameters():
                param.data = 1*param.data0
    print('best hyperparams: lr = ' + str(lrbest) + ' lamb = '+ str(lambbest))
    print('getting validation and test error')
    eval_data = val_data
    lamb = lambbest
    lr = lrbest
    vloss = evaluate()
    for param in model.parameters():
        param.data = 1*param.data0

    eval_data = test_data
    lamb = lambbest
    lr = lrbest
    tloss = evaluate()
    print('validation perplexity loss: ' + str(vloss))#[0]
    print('test perplexity loss: ' + str(tloss))#[0]
