from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.init as weight_init
import torch.nn.functional as F
import numpy as np

from cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN
#from IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN
 
class Batch_norm_overtime(nn.Module):
    def __init__(self,  hidden_size,seq_len):
        super(Batch_norm_overtime, self).__init__()
        self.bn = nn.BatchNorm1d(hidden_size) 

    def forward(self, x):
        x=x.permute(1,2,0)
        x= self.bn(x.clone())
        x=x.permute(2,0,1)
        return x
class Dropout_overtime(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, p=0.5,training=False):
    output = input.clone()
    noise = input.data.new(input.size(-2),input.size(-1))  #torch.ones_like(input[0])
    if training:            
      noise.bernoulli_(1 - p).div_(1 - p)
      noise = noise.unsqueeze(0).expand_as(input)
      output.mul_(noise)
    ctx.save_for_backward(noise)
    ctx.training=training
    return output
  @staticmethod
  def backward(ctx, grad_output):
    noise,=ctx.saved_tensors
    if ctx.training:
      return grad_output.mul(noise),None,None
    else:
      return grad_output,None,None
dropout_overtime=Dropout_overtime.apply

import argparse
import opts     
parser = argparse.ArgumentParser(description='pytorch pixel mnist')
opts.train_opts(parser)
args = parser.parse_args()
MAG=args.MAG
U_bound=np.power(10,(np.log10(MAG)/args.seq_len))
U_lowbound=np.power(10,(np.log10(1.0/MAG)/args.seq_len))  
  
class stackedIndRNN_encoder(nn.Module):
    def __init__(self, input_size, outputclass):
        super(stackedIndRNN_encoder, self).__init__()        
        hidden_size=args.hidden_size
        
        self.DIs=nn.ModuleList()
        denseinput=nn.Linear(input_size, hidden_size, bias=True)
        self.DIs.append(denseinput)
        for x in range(args.num_layers - 1):
            denseinput = nn.Linear(hidden_size, hidden_size, bias=True)
            self.DIs.append(denseinput)                
        
        self.BNs = nn.ModuleList()
        for x in range(args.num_layers):
            bn = Batch_norm_overtime(hidden_size,args.seq_len)
            self.BNs.append(bn)                      
  
        self.RNNs = nn.ModuleList()
        rnn = IndRNN(hidden_size=hidden_size) #IndRNN
        self.RNNs.append(rnn)  
        for x in range(args.num_layers-1):
            rnn = IndRNN(hidden_size=hidden_size) #IndRNN
            self.RNNs.append(rnn)         
            
        self.lastfc = nn.Linear(hidden_size, outputclass, bias=True)
        self.init_weights()

    def init_weights(self):
      for name, param in self.named_parameters():
        if 'weight_hh' in name:
          param.data.uniform_(0,U_bound)          
        if 'RNNs.'+str(args.num_layers-1)+'.weight_hh' in name:
          param.data.uniform_(U_lowbound,U_bound)    
        if 'DIs' in name and 'weight' in name:
          param.data.uniform_(-args.ini_in2hid,args.ini_in2hid)               
        if 'bns' in name and 'weight' in name:
          param.data.fill_(1)     
        if 'bias' in name:
          param.data.fill_(0.0)                        
    def forward(self, input):
        all_output = []
        rnnoutputs={}
        hidden_x={}               
        seq_len, batch_size, indim=input.size()            
                 
        for x in range(1,len(self.RNNs)+1):
          hidden_x['hidden%d'%x]=Variable(torch.zeros(1,batch_size,args.hidden_size).cuda())
                            
        rnnoutputs['rnnlayer0']=input
        for x in range(1,len(self.RNNs)+1):
          rnnoutputs['rnnlayer%d'%(x-1)]=rnnoutputs['rnnlayer%d'%(x-1)].contiguous()
          rnnoutputs['rnnlayer%d'%(x-1)]=rnnoutputs['rnnlayer%d'%(x-1)].view(seq_len*batch_size,-1)
          rnnoutputs['rnnlayer%d'%(x-1)]=self.DIs[x-1](rnnoutputs['rnnlayer%d'%(x-1)])   
          rnnoutputs['rnnlayer%d'%(x-1)]=rnnoutputs['rnnlayer%d'%(x-1)].view(seq_len,batch_size,-1)  
          rnnoutputs['rnnlayer%d'%x]= self.RNNs[x-1](rnnoutputs['rnnlayer%d'%(x-1)], hidden_x['hidden%d'%x])        
          rnnoutputs['rnnlayer%d'%x]=self.BNs[x-1](rnnoutputs['rnnlayer%d'%x])     
          if args.dropout>0:
            rnnoutputs['rnnlayer%d'%x]= dropout_overtime(rnnoutputs['rnnlayer%d'%x],args.dropout,self.training)
        temp=rnnoutputs['rnnlayer%d'%len(self.RNNs)][-1]
        output = self.lastfc(temp)
        return output                
            
      
