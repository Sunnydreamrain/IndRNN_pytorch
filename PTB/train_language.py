from __future__ import print_function
import sys
import argparse
import os
import math

import time
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import data

# Set the random seed manually for reproducibility.
seed = 100
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
else:
    print("WARNING: CUDA not available")

import opts

parser = argparse.ArgumentParser(description='pytorch cPTB')
opts.train_opts(parser)
args = parser.parse_args()
print(args)

from language_utils import batchify, get_batch


batch_size = args.batch_size
gradientclip_value = args.gradclipvalue
if args.U_bound==0:
  U_bound=np.power(10,(np.log10(args.MAG)/args.seq_len))   
else:
  U_bound=args.U_bound


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
    if 'text8' in args.data:
        import readtext8
        corpus = readtext8.Corpus(args.data)
    else:
        corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)




###############################################################################
# Build the model
###############################################################################
ntokens = len(corpus.dictionary)
print ('output classes: ',ntokens)

  
if args.model=='plainIndRNN':
  import Indrnn_plainnet_returnstates as Indrnn_network
  model = Indrnn_network.stackedIndRNN_encoder(ntokens)  
elif args.model=='residualIndRNN':
  import Indrnn_residualnet_preact_returnstates as Indrnn_network
  model = Indrnn_network.ResidualNet(ntokens)  
elif args.model=='denseIndRNN':
  import Indrnn_densenet_returnstates as Indrnn_network
  from ast import literal_eval
  block_config = literal_eval(args.block_config)
  model = Indrnn_network.DenseNet(ntokens, growth_rate=args.growth_rate, block_config=block_config)
else:
  print('set the model type: plainIndRNN, residualIndRNN, denseIndRNN')
  assert 2==3
model.cuda()
criterion = nn.CrossEntropyLoss().cuda()

###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

learning_rate = args.lr  # np.float32(args.lr)
# Adam with lr 2e-4 works fine.
param_decay=[]
param_nodecay=[]
for name, param in model.named_parameters(): 
  if ('weight_hh' in name) or ('bias' in name):
    param_nodecay.append(param)      
    print('parameters no weight decay: ',name)    
  elif (not args.bn_decay) and ('norm' in name) and ('weight' in name):
    param_nodecay.append(param)      
    print('parameters no weight decay: ',name)        
  else:
    param_decay.append(param)      
    print('parameters with weight decay: ',name)     
optimizer = torch.optim.Adam([
        {'params': param_nodecay},
        {'params': param_decay, 'weight_decay': args.decayfactor}], #,        {'params': param_nodecay_hidini, 'lr': 1e-3}
        lr=learning_rate
    ) 

###############################################################################
# Training code
###############################################################################
def train(epoch):
    dropindex=np.random.randint(args.seq_len*args.batch_size*args.rand_drop_ini//2)
    train_data = batchify(corpus.train[dropindex:], args.batch_size, args)
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = None#hidden_ini#torch.zeros(args.num_layers, batch_size, args.hidden_size).cuda()

    batch, i = 0, 0
    start_len=0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.seq_len if np.random.random() < 0.95 else args.seq_len / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.seq_len + 10)
        
        if args.rand_drop_ini>0:
            if(np.random.randint(args.rand_drop_ini))==0:# and patience<100: # or (epoc<3) or (epoc<6 and batch%2==0)
                hidden=None      
                start_len=0     
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
       # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        
        optimizer.zero_grad()
        output, hidden = model(data, hidden, start_len)

        start_len=start_len+seq_len
        raw_loss = criterion(output, targets)

        loss = raw_loss
        loss.backward()
        clip_gradient(model, gradientclip_value)
        optimizer.step()
        clip_weight(model, U_bound)  # different part
        #hidden = hidden.detach()   
        for key in hidden.keys():
            hidden[key].detach_()         

        total_loss += raw_loss.data
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.seq_len, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(np.minimum(200,cur_loss)), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len



def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = None#hidden_ini#None
    hidden_last=None
    start_len=0
    for i in range(0, data_source.size(0) - 1, args.seq_len):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden, start_len)
        start_len=start_len+args.seq_len
        total_loss += len(data) * criterion(output, targets).data
        #hidden = hidden.detach()         
        for key in hidden.keys():
            hidden[key].detach_()   
        output = output.detach()  # to reduce the memory in case two graphs are generated due to the scoping rule of python
    return total_loss.item() / len(data_source)


def clip_gradient(model, clip):
    for p in model.parameters():
        p.grad.data.clamp_(-clip, clip)

def clip_weight(RNNmodel, clip):
    for name, param in RNNmodel.named_parameters():
        if 'weight_hh' in name:
            param.data.clamp_(-clip, clip)
def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr     

adjust_learning_rate(optimizer, learning_rate)
lastperp = 10000
patience = 0
reduced = 1
for epoch in range(1, 10000000):
    epoch_start_time = time.time()
    train(epoch)
    test_perp = evaluate(val_data, eval_batch_size)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
      epoch, (time.time() - epoch_start_time), test_perp, math.exp(np.minimum(200,test_perp)), test_perp / math.log(2)))
    print('-' * 89)
    if (test_perp < lastperp):
        model_clone = copy.deepcopy(model.state_dict())
        opti_clone = copy.deepcopy(optimizer.state_dict())
        lastperp = test_perp
        patience = 0
    elif patience > int(args.pThre/ reduced + 0.5):
        reduced = reduced * 2
        print('learning rate', learning_rate)
        model.load_state_dict(model_clone)
        optimizer.load_state_dict(opti_clone)
        patience = 0
        learning_rate = learning_rate * 0.2  # np.float32()        
        adjust_learning_rate(optimizer, learning_rate)   
        if learning_rate < 1e-6:
            break
        test_acc = evaluate(test_data, test_batch_size)
        print('=' * 89)
        print('| Reducing learning rate | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_acc, math.exp(np.minimum(200,test_acc)), test_acc / math.log(2)))
        print('=' * 89)
    else:
        patience += 1

test_acc = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_acc, math.exp(np.minimum(200,test_acc)), test_acc / math.log(2)))
print('=' * 89)

save_name = 'indrnn_model'
with open(save_name, 'wb') as f:
    torch.save(model.state_dict(), f)
