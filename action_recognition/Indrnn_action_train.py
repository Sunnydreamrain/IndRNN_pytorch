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
parser = argparse.ArgumentParser(description='pytorch action')
opts.train_opts(parser)
args = parser.parse_args()
print(args)


outputclass=60
indim=50*3
if args.geo_aug:
  indim=50*3+897*2
batch_size = args.batch_size
seq_len=args.seq_len
gradientclip_value=args.gradclipvalue
if args.U_bound==0:
  U_bound=np.power(10,(np.log10(args.MAG)/args.seq_len))   
else:
  U_bound=args.U_bound

if args.model=='plainIndRNN':
  import Indrnn_plainnet as Indrnn_network
  model = Indrnn_network.stackedIndRNN_encoder(indim, outputclass)  
elif args.model=='residualIndRNN':
  import Indrnn_residualnet_preact as Indrnn_network
  model = Indrnn_network.ResidualNet(indim, outputclass)  
elif args.model=='denseIndRNN':
  import Indrnn_densenet as Indrnn_network
  if args.time_diff:
    import Indrnn_densenet_FA as Indrnn_network
  from ast import literal_eval
  block_config = literal_eval(args.block_config)
  model = Indrnn_network.DenseNet(indim, outputclass, growth_rate=args.growth_rate, block_config=block_config,
                                        num_init_features=args.growth_rate * args.num_first)
else:
  print('set the model type: plainIndRNN, residualIndRNN, denseIndRNN')
  assert 2==3                                        
model.cuda()
criterion = nn.CrossEntropyLoss()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

#Adam with lr 2e-4 works fine.
learning_rate=args.lr
param_decay=[]
param_nodecay=[]
for name, param in model.named_parameters():
  if 'weight_hh' in name or 'bias' in name:
    param_nodecay.append(param)      
    #print('parameters no weight decay: ',name)    
  elif (not args.bn_decay) and ('norm' in name):
    param_nodecay.append(param)      
    #print('parameters no weight decay: ',name)             
  else:
    param_decay.append(param)      
    #print('parameters with weight decay: ',name)          
            
optimizer = torch.optim.Adam([
        {'params': param_nodecay},
        {'params': param_decay, 'weight_decay': args.decayfactor}
    ], lr=learning_rate) 


if args.test_CV:
  train_datasets='train_CV_ntus'
  test_dataset='test_CV_ntus'
else:
  train_datasets='train_ntus'
  test_dataset='test_ntus'
geo_aug=args.geo_aug 
data_randtime_aug=args.data_randtime_aug 
from data_reader import DataHandler
dh_train = DataHandler(batch_size,seq_len,train_or_eval='train')
dh_eval = DataHandler(batch_size,seq_len,train_or_eval='eval')
dh_test= DataHandler(batch_size,seq_len,train_or_eval='test')
num_train_batches=int(np.ceil(dh_train.GetDatasetSize()/(batch_size+0.0)))
num_eval_batches=int(np.ceil(dh_eval.GetDatasetSize()/(batch_size+0.0)))
num_test_batches=int(np.ceil(dh_test.GetDatasetSize()/(batch_size+0.0)))


def train(num_train_batches):
  model.train()
  tacc=0
  count=0
  start_time = time.time()
  for batchi in range(0,num_train_batches):
    inputs,targets=dh_train.GetBatch()
    inputs=inputs.transpose(1,0,2,3)    
    
    inputs=Variable(torch.from_numpy(inputs).cuda(), requires_grad=True)
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda(), requires_grad=False)
    seq_len, batch_size, joints_no,_=inputs.size()             
    inputs=inputs.view(seq_len,batch_size,3*joints_no)   

    model.zero_grad()
    if args.constrain_U:
      clip_weight(model,U_bound)
    output=model(inputs)
    loss = criterion(output, targets)

    pred = output.data.max(1)[1] # get the index of the max log-probability
    accuracy = pred.eq(targets.data).cpu().sum().numpy()/(0.0+targets.size(0))      
          
    loss.backward()
    clip_gradient(model,gradientclip_value)
    optimizer.step()
    
    tacc=tacc+accuracy#loss.data.cpu().numpy()#accuracy
    count+=1
  elapsed = time.time() - start_time
  print ("training accuracy: ", tacc/(count+0.0)  )
  #print ('time per batch: ', elapsed/num_train_batches)
  
def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.train()       
def eval(dh,num_batches,use_bn_trainstat=False):
  model.eval()
  if use_bn_trainstat:
    model.apply(set_bn_train)
  tacc=0
  count=0  
  start_time = time.time()
  while(1):  
    inputs,targets=dh.GetBatch()
    inputs=inputs.transpose(1,0,2,3)
    inputs=Variable(torch.from_numpy(inputs).cuda())
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda())
    seq_len, batch_size, joints_no,_=inputs.size()             
    inputs=inputs.view(seq_len,batch_size,3*joints_no)   
        
    output=model(inputs)
    pred = output.data.max(1)[1] # get the index of the max log-probability
    accuracy = pred.eq(targets.data).cpu().sum().numpy()        
    tacc+=accuracy
    count+=1
    if count==num_batches*args.eval_fold:
      break
  elapsed = time.time() - start_time
  print ("eval accuracy: ", tacc/(count*targets.data.size(0)+0.0)  )
  #print ('eval time per batch: ', elapsed/(count+0.0))
  return tacc/(count*targets.data.size(0)+0.0)


def test(dh,num_batches,use_bn_trainstat=False):
  model.eval()
  if use_bn_trainstat:
    model.apply(set_bn_train)
  tacc=0
  count=0  
  start_time = time.time()
  total_testdata=dh_test.GetDatasetSize()  
  total_ave_acc=np.zeros((total_testdata,outputclass))
  testlabels=np.zeros((total_testdata))
  while(1):  
    inputs,targets,index=dh.GetBatch()
    inputs=inputs.transpose(1,0,2,3)
    testlabels[index]=targets
    inputs=Variable(torch.from_numpy(inputs).cuda())
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda())
    seq_len, batch_size, joints_no,_=inputs.size()             
    inputs=inputs.view(seq_len,batch_size,3*joints_no)   
        
    output=model(inputs)
    pred = output.data.max(1)[1] # get the index of the max log-probability
    accuracy = pred.eq(targets.data).cpu().sum().numpy()    
    total_ave_acc[index]+=output.data.cpu().numpy()
    
    tacc+=accuracy
    count+=1
    if count==num_batches*args.test_no:
      break    
  top = np.argmax(total_ave_acc, axis=-1)
  eval_acc=np.mean(np.equal(top, testlabels))    
  elapsed = time.time() - start_time
  print ("test accuracy: ", tacc/(count*targets.data.size(0)+0.0), eval_acc  )
  #print ('test time per batch: ', elapsed/(count+0.0))
  return tacc/(count*targets.data.size(0)+0.0)#, eval_acc/(total_testdata+0.0)

def clip_gradient(model, clip):
    for p in model.parameters():
        p.grad.data.clamp_(-clip,clip)

def clip_weight(RNNmodel, clip):
    for name, param in RNNmodel.named_parameters():
      if 'weight_hh' in name:
        param.data.clamp_(-clip,clip)
def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
            
lastacc=0
dispFreq=20
patience=0
reduced=1
for batchi in range(1,10000000):
  for i in range(num_train_batches//dispFreq):
    train(dispFreq)
  test_acc=eval(dh_eval,num_eval_batches,args.use_bneval)

  if (test_acc >lastacc):
    model_clone = copy.deepcopy(model.state_dict())   
    opti_clone = copy.deepcopy(optimizer.state_dict()) 
    lastacc=test_acc
    patience=0
  elif patience>int(args.pThre/reduced+0.5):
    reduced=reduced*2
    print ('learning rate',learning_rate)
    model.load_state_dict(model_clone)
    optimizer.load_state_dict(opti_clone)
    patience=0
    learning_rate=learning_rate*0.1
    adjust_learning_rate(optimizer,learning_rate)     
    if learning_rate<args.end_rate:
      break  
    test_acc=test(dh_test,num_test_batches)     
 
  else:
    patience+=1 
    
test_acc=test(dh_test,num_test_batches)  
test_acc=test(dh_test,num_test_batches,True)     
save_name='indrnn_action_model' 
with open(save_name, 'wb') as f:
    torch.save(model.state_dict(), f)



 
