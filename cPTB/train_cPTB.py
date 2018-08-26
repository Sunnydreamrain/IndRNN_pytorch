from __future__ import print_function
import sys
import argparse
import os

import time
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Set the random seed manually for reproducibility.
seed=100
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

import language_model

batch_size = args.batch_size
seq_len=args.seq_len
outputclass=50

gradientclip_value=args.gradclipvalue
U_bound=language_model.U_bound


from reader import data_iterator, ptb_raw_data
name_dataset='ptb.char.'
def get_raw_data(dataset='ptb',data_path='data/'):
  raw_data = ptb_raw_data(data_path,filename=name_dataset)
  return raw_data
train_data, valid_data, test_data, _ = get_raw_data('ptb')
num_train_batches =((len(train_data) // args.batch_size) - 1) // args.seq_len
num_eval_batches =((len(valid_data) // args.batch_size) - 1) // args.seq_len
num_test_batches =((len(test_data) // args.batch_size) - 1) // args.seq_len

model = language_model.stackedIndRNN_encoder(outputclass)  


model.cuda()
criterion = nn.CrossEntropyLoss()
learning_rate=args.lr#np.float32(args.lr)

#Adam with lr 2e-4 works fine.
if args.use_weightdecay_nohiddenW:
  param_decay=[]
  param_nodecay=[]
  for name, param in model.named_parameters():
    if 'weight_hh' in name or 'bias' in name:
      param_nodecay.append(param)      
      #print('parameters no weight decay: ',name)          
    else:
      param_decay.append(param)      
      #print('parameters with weight decay: ',name)          

  if args.opti=='sgd':
    optimizer = torch.optim.SGD([
            {'params': param_nodecay},
            {'params': param_decay, 'weight_decay': args.decayfactor}
        ], lr=learning_rate,momentum=0.9,nesterov=True)   
  else:                
    optimizer = torch.optim.Adam([
            {'params': param_nodecay},
            {'params': param_decay, 'weight_decay': args.decayfactor}
        ], lr=learning_rate) 
else:  
  if args.opti=='sgd':   
    optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,nesterov=True)
  else:                      
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)


def train():
  model.train()
  tperp=0
  tbpc=0
  count=0
  start_time = time.time()
  hidden=Variable(torch.zeros(args.num_layers,batch_size,args.hidden_size).cuda(), requires_grad=False) 
  dropindex=np.random.randint(seq_len*5)  
  for batchi, (x, y) in enumerate(data_iterator(train_data[dropindex:], batch_size, seq_len)): 
    inputs=x
    targets=y
    targets=targets.transpose(1,0)
    targets=targets.reshape((-1))
    inputs=inputs.transpose(1,0)   
    inputs=Variable(torch.from_numpy(np.int64(inputs)).cuda(), requires_grad=False)
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda(), requires_grad=False)

    model.zero_grad()
    output,hidden=model(inputs,hidden)
    hidden=hidden.detach()
    loss = criterion(output, targets)   
    perp=torch.exp(loss)
    bpc = (loss/np.log(2.0)) 
          
    loss.backward()
    clip_gradient(model,gradientclip_value)
    optimizer.step()
    clip_weight(model,U_bound)
    
    tperp=tperp+perp.data.cpu().numpy()#accuracy
    tbpc=tbpc+bpc.data.cpu().numpy()
    count+=1
  elapsed = time.time() - start_time
  print ("train perp and bpc: ", tperp/(count+0.0), tbpc/(count+0.0)  )
  #print ('time per batch: ', elapsed/num_train_batches)
  

def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.train()    
def eval(data,Is_test=False,use_bn_trainstat=False):
  model.eval()
  if use_bn_trainstat:
    model.apply(set_bn_train)
  tperp=0
  tbpc=0
  count=0  
  start_time = time.time()
  hidden=Variable(torch.zeros(args.num_layers,batch_size,args.hidden_size).cuda(), requires_grad=False)
  for batchi, (x, y) in enumerate(data_iterator(data, args.batch_size, args.seq_len)):
    inputs=x
    targets=y
    targets=targets.transpose(1,0)
    targets=targets.reshape((-1))
    inputs=inputs.transpose(1,0)   
    inputs=Variable(torch.from_numpy(np.int64(inputs)).cuda(), requires_grad=False)
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda(), requires_grad=False)
        
    output,hidden=model(inputs,hidden)
    hidden = hidden.detach()
    output = output.detach() # to reduce the memory in case two graphs are generated due to the scoping rule of python
    loss = criterion(output, targets)
    perp=torch.exp(loss)
    bpc = (loss/np.log(2.0))   
    
    tperp=tperp+perp.data.cpu().numpy()#accuracy
    tbpc=tbpc+bpc.data.cpu().numpy()
    count+=1
  elapsed = time.time() - start_time
  if Is_test:
    print ("test perp and bpc: ", tperp/(count+0.0), tbpc/(count+0.0)  )
  else:
    print ("eval perp and bpc: ", tperp/(count+0.0), tbpc/(count+0.0)  )
  #print ('eval time per batch: ', elapsed/(count+0.0))
  return tperp/(count+0.0)


def clip_gradient(model, clip):
    for p in model.parameters():
        p.grad.data.clamp_(-clip,clip)

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr     

def clip_weight(RNNmodel, clip):
    for name, param in RNNmodel.named_parameters():
      if 'weight_hh' in name:
        param.data.clamp_(-clip,clip)

lastperp=10000
patience=0
reduced=1
for batchi in range(1,10000000):
  train()
  test_perp=eval(valid_data)
  if (test_perp < lastperp):
    model_clone = copy.deepcopy(model.state_dict())   
    opti_clone = copy.deepcopy(optimizer.state_dict())
    lastperp=test_perp
    patience=0
  elif patience>int(args.pThre/reduced+0.5):
    reduced=reduced*2
    print ('learning rate',learning_rate)
    model.load_state_dict(model_clone)
    optimizer.load_state_dict(opti_clone)
    patience=0
    learning_rate=learning_rate*0.2#np.float32()
    adjust_learning_rate(optimizer,learning_rate)      
    if learning_rate<1e-6:
      break   
    test_acc=eval(test_data,True)  
  else:
    patience+=1 
    

test_acc=eval(test_data,True) 
test_acc=eval(test_data,True,True) 
save_name='indrnn_cPTB_model' 
with open(save_name, 'wb') as f:
    torch.save(model, f)



 
