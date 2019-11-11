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
#if no cuda, then use the following line
#from IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN 


from __main__ import parser,args,U_bound
MAG=args.MAG
#U_bound=np.power(10,(np.log10(MAG)/args.seq_len))
U_lowbound=np.power(10,(np.log10(1.0/MAG)/args.seq_len))  
from utils import Batch_norm_overtime,Linear_overtime_module,Dropout_overtime
BN=Batch_norm_overtime
Linear_overtime=Linear_overtime_module
dropout_overtime=Dropout_overtime.apply


class IndRNNwithBN(nn.Sequential):
    def __init__(self, hidden_size, seq_len,bn_location='bn_before'):
        super(IndRNNwithBN, self).__init__()  
        if bn_location=="bn_before":      
            self.add_module('norm1', BN(hidden_size, args.seq_len))
        self.add_module('indrnn1', IndRNN(hidden_size))        
        if bn_location=="bn_after":   
            self.add_module('norm1', BN(hidden_size, args.seq_len))
        if (bn_location!='bn_before') and (bn_location!='bn_after'):
            print('Please select a batch normalization mode.')
            assert 2==3

class stackedIndRNN_encoder(nn.Module):
    def __init__(self, input_size, outputclass):
        super(stackedIndRNN_encoder, self).__init__()        
        hidden_size=args.hidden_size
        
        self.DIs=nn.ModuleList()
        denseinput=Linear_overtime(input_size, hidden_size)
        self.DIs.append(denseinput)
        for x in range(args.num_layers - 1):
            denseinput = Linear_overtime(hidden_size, hidden_size)
            self.DIs.append(denseinput)   

        self.RNNs = nn.ModuleList()
        for x in range(args.num_layers):
            rnn = IndRNNwithBN(hidden_size=hidden_size, seq_len=args.seq_len,bn_location=args.bn_location) #IndRNN
            self.RNNs.append(rnn)         
            
        self.classifier = nn.Linear(hidden_size, outputclass, bias=True)
        self.init_weights()

    def init_weights(self):
      for name, param in self.named_parameters():
        if 'weight_hh' in name:
          param.data.uniform_(0,U_bound)          
        if args.u_lastlayer_ini and 'RNNs.'+str(args.num_layers-1)+'.weight_hh' in name:
          param.data.uniform_(U_lowbound,U_bound)    
        if ('fc' in name) and 'weight' in name:#'denselayer' in name and 
            nn.init.kaiming_uniform_(param, a=8, mode='fan_in')#
        if 'classifier' in name and 'weight' in name:
            nn.init.kaiming_normal_(param.data)
        if ('norm' in name or 'Norm' in name)  and 'weight' in name:
            param.data.fill_(1)
        if 'bias' in name:
            param.data.fill_(0.0)


    def forward(self, input):
        rnnoutputs={}    
        rnnoutputs['outlayer-1']=input
        for x in range(len(self.RNNs)):
          rnnoutputs['dilayer%d'%x]=self.DIs[x](rnnoutputs['outlayer%d'%(x-1)])    
          rnnoutputs['outlayer%d'%x]= self.RNNs[x](rnnoutputs['dilayer%d'%x])          
          if args.dropout>0:
            rnnoutputs['outlayer%d'%x]= dropout_overtime(rnnoutputs['outlayer%d'%x],args.dropout,self.training) 
        temp=rnnoutputs['outlayer%d'%(len(self.RNNs)-1)][-1]
        output = self.classifier(temp)
        return output                
            
      
