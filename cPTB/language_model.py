from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.init as weight_init
import torch.nn.functional as F
import numpy as np


from IndRNN_onlyrecurrent import IndRNNCell_onlyrecurrent as IndRNNCell

import argparse
import opts     
parser = argparse.ArgumentParser(description='pytorch char level pentree')
opts.train_opts(parser)
args = parser.parse_args()
MAG=args.MAG
U_bound=1#np.power(10,(np.log10(MAG)/args.seq_len))
           
 
 
class Batch_norm_step(nn.Module):
    def __init__(self,  hidden_size,seq_len):
        super(Batch_norm_step, self).__init__()
        self.hidden_size = hidden_size
        
        self.max_time_step=seq_len
        self.bns = nn.ModuleList()

        for x in range(self.max_time_step):
            bn = nn.BatchNorm1d(hidden_size)#.train(True)
            self.bns.append(bn)        
        for x in range(1,self.max_time_step):
            self.bns[x].weight=self.bns[0].weight    
            self.bns[x].bias=self.bns[0].bias    
    def forward(self, x, t):
        y= self.bns[t](x)#.clone()
        return y


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
            rnn = IndRNNCell(hidden_size=args.hidden_size) #IndRNN
            self.RNNs.append(rnn)          
    
        self.lastfc = nn.Linear(args.hidden_size, outputclass)
        self.init_weights()
        self.outputclass=outputclass
        self.noise_rnn={}
        for x in range(len(self.RNNs)):
          self.noise_rnn['rnnlayer%d'%x]=torch.FloatTensor(args.hidden_size).cuda()    
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
          rnnoutputs['rnnlayer%d'%x]=hidden[x-1]         
          
        if args.dropout>0:
          for x in range(len(self.RNNs)):
            self.noise_rnn['rnnlayer%d'%x]=torch.FloatTensor(args.batch_size,args.hidden_size).cuda()
            self.noise_rnn['rnnlayer%d'%x].bernoulli_(1 - args.dropout).div_(1 - args.dropout)
            #self.noise_rnn['rnnlayer%d'%x]=self.noise_rnn['rnnlayer%d'%x].unsqueeze(0).expand(args.batch_size, args.hidden_size)
            if not self.training:
              self.noise_rnn['rnnlayer%d'%x].fill_(1)    
            self.noise_rnn['rnnlayer%d'%x]=Variable(self.noise_rnn['rnnlayer%d'%x])         

        output=[]
        hidden_new=[]

        for i, input_t in enumerate(input.split(1)):
          input_t=input_t.view(batch_size)
          input_t=self.embed(input_t)
          rnnoutputs['outlayer0']=input_t
          for x in range(1,len(self.RNNs)+1):
              rnnoutputs['DIlayer%d'%(x-1)]=self.DIs[x-1](rnnoutputs['outlayer%d'%(x-1)])              
              if args.use_residual and x>args.residual_layers and (x-1)%args.residual_layers==0:
                rnnoutputs['sum%d'%x]=rnnoutputs['DIlayer%d'%(x-1)]+rnnoutputs['sum%d'%(x-self.residual_layers)]
              else:
                rnnoutputs['sum%d'%x]=rnnoutputs['DIlayer%d'%(x-1)]       
                            
              rnnoutputs['sum%d'%x]=self.BNs[x-1](rnnoutputs['sum%d'%x],i)                                   
              rnnoutputs['rnnlayer%d'%x]= self.RNNs[x-1](rnnoutputs['sum%d'%x], rnnoutputs['rnnlayer%d'%x])
              if i==seq_len-1:
                hidden_new.append(rnnoutputs['rnnlayer%d'%x])
                
              rnnoutputs['outlayer%d'%x]=rnnoutputs['rnnlayer%d'%x]
              
              if args.dropout>0:
                rnnoutputs['outlayer%d'%x]=rnnoutputs['outlayer%d'%x]*self.noise_rnn['rnnlayer%d'%(x-1)]                              

          output.append(rnnoutputs['outlayer%d'%x])
          
        output = torch.stack(output)
        output=output.view(-1,args.hidden_size)
        output=self.lastfc(output)
        hidden_new = torch.stack(hidden_new)
        return output,hidden_new
      
