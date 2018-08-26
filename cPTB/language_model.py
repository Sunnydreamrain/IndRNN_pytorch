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
#from IndRNN_onlyrecurrent import IndRNNCell_onlyrecurrent as IndRNN
import argparse
import opts     
parser = argparse.ArgumentParser(description='pytorch char level pentree')
opts.train_opts(parser)
args = parser.parse_args()
MAG=args.MAG
U_bound=1#np.power(10,(np.log10(MAG)/args.seq_len))
           
 
 
class Batch_norm_step(nn.Module):
    def __init__(self, hidden_size,seq_len):
      super(Batch_norm_step, self).__init__()      
      self.max_time_step=seq_len
      self.bns = nn.ModuleList()
      for x in range(self.max_time_step):
        bn = nn.BatchNorm1d(hidden_size)#.train(True)
        self.bns.append(bn)        
      for x in range(1,self.max_time_step):
        self.bns[x].weight=self.bns[0].weight    
        self.bns[x].bias=self.bns[0].bias    
    def forward(self, x):
      output=[]
      for t, x_t in enumerate(x.split(1)):
        x_t=x_t.squeeze()
        y= self.bns[t](x_t)#.clone()
        output.append(y)
      output=torch.stack(output)
      return output
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

class stackedIndRNN_encoder(nn.Module):

    def __init__(self,outputclass):
      super(stackedIndRNN_encoder, self).__init__()
      
      self.embed=torch.nn.Embedding(outputclass,args.hidden_size)
      
      self.DIs=nn.ModuleList()
      denseinput=nn.Linear(args.hidden_size,  args.hidden_size, bias=True)
      self.DIs.append(denseinput)
      for x in range(args.num_layers - 1):
        denseinput = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.DIs.append(denseinput)                
      
      self.BNs = nn.ModuleList()
      for x in range(args.num_layers):
        bn = Batch_norm_step(args.hidden_size,args.seq_len)
        self.BNs.append(bn)                      

      self.RNNs = nn.ModuleList()
      for x in range(args.num_layers):
        rnn = IndRNN(hidden_size=args.hidden_size) #IndRNN
        self.RNNs.append(rnn)          
  
      self.lastfc = nn.Linear(args.hidden_size, outputclass)
      self.init_weights()
      self.outputclass=outputclass  
    def init_weights(self):
      for name, param in self.named_parameters():
        if 'weight_hh' in name:
          param.data.uniform_(0,U_bound)              
        if 'DIs' in name and 'weight' in name:
          #torch.nn.init.kaiming_uniform(param, a=2, mode='fan_in')
          torch.nn.init.kaiming_uniform(param, a=2, mode='fan_in')             
        if 'bns' in name and 'weight' in name:
          param.data.fill_(1)      
        if 'embed' in name and 'weight' in name:
          param.data.uniform_(-0.04,0.04)           
        if 'lastfc' in name and 'weight' in name:
          param.data.uniform_(-0.04,0.04)                                 
        if 'bias' in name:
          param.data.fill_(0.0)                
    def forward(self, input,hidden):
      rnnoutputs={}
      
      seq_len, batch_size=input.size()
      for x in range(1,len(self.RNNs)+1):
        rnnoutputs['h0layer%d'%x]=hidden[x-1]                         

      output=[]
      hidden_new=[]

      input=input.view(seq_len*batch_size)
      input=self.embed(input)
      rnnoutputs['outlayer0']=input
      for x in range(1,len(self.RNNs)+1):
        rnnoutputs['DIlayer%d'%(x-1)]=self.DIs[x-1](rnnoutputs['outlayer%d'%(x-1)])  
        rnnoutputs['DIlayer%d'%(x-1)]=rnnoutputs['DIlayer%d'%(x-1)].view(seq_len,batch_size,args.hidden_size)            
        if args.use_residual and x>args.residual_layers and (x-1)%args.residual_layers==0:
          rnnoutputs['sum%d'%x]=rnnoutputs['DIlayer%d'%(x-1)]+rnnoutputs['sum%d'%(x-args.residual_layers)]
        else:
          rnnoutputs['sum%d'%x]=rnnoutputs['DIlayer%d'%(x-1)]                  
                      
        rnnoutputs['sum%d'%x]=self.BNs[x-1](rnnoutputs['sum%d'%x])                                   
        rnnoutputs['rnnlayer%d'%x]= self.RNNs[x-1](rnnoutputs['sum%d'%x], rnnoutputs['h0layer%d'%x])
        hidden_new.append(rnnoutputs['rnnlayer%d'%x][-1]) 
        #rnnoutputs['rnnlayer%d'%x],rnnoutputs['laststatelayer%d'%x]= self.RNNs[x-1](rnnoutputs['sum%d'%x], rnnoutputs['h0layer%d'%x])
        #hidden_new.append(rnnoutputs['laststatelayer%d'%x])              
        rnnoutputs['outlayer%d'%x]=rnnoutputs['rnnlayer%d'%x]
        
        if args.dropout>0:
          rnnoutputs['outlayer%d'%x]= dropout_overtime(rnnoutputs['outlayer%d'%x],args.dropout,self.training)    
        rnnoutputs['outlayer%d'%x]=rnnoutputs['outlayer%d'%x].view(seq_len*batch_size,args.hidden_size)                        
        
      output=self.lastfc(rnnoutputs['outlayer%d'%len(self.RNNs)])
      hidden_new = torch.stack(hidden_new)
      return output,hidden_new
      
