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
from utils import Linear_overtime_module,Dropout_overtime,embedded_dropout
from utils import MyBatchNorm_stepCompute  # a quick implementation of frame-wise batch normalization
BN=MyBatchNorm_stepCompute
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
        self.bn_location=bn_location
        if (bn_location!='bn_before') and (bn_location!='bn_after'):
            print('Please select a batch normalization mode.')
            assert 2==3

    def forward(self, input,BN_start=0,hidden_x=None):
        out_1=input 
        if self.bn_location=="bn_before":  
            out_1=self.norm1(out_1, BN_start)
        out_2=self.indrnn1(out_1,hidden_x)
        hidden_out=out_2[-1].clone()
        if self.bn_location=="bn_after":  
            out_2=self.norm1(out_2, BN_start)
        return out_2,hidden_out
        


class stackedIndRNN_encoder(nn.Module):
    def __init__(self, outputclass):
        super(stackedIndRNN_encoder, self).__init__()        
        hidden_size=args.hidden_size
        self.encoder = torch.nn.Embedding(outputclass, args.embed_size)
        self.DIs=nn.ModuleList()
        denseinput=Linear_overtime(args.embed_size, hidden_size)
        self.DIs.append(denseinput)
        for x in range(args.num_layers - 1):
            denseinput = Linear_overtime(hidden_size, hidden_size)
            self.DIs.append(denseinput)           
  
        self.RNNs = nn.ModuleList()
        for x in range(args.num_layers):
            rnn = IndRNNwithBN(hidden_size,args.seq_len,args.bn_location) #IndRNN
            self.RNNs.append(rnn)              

        self.last_fc = Linear_overtime(args.hidden_size, args.embed_size) 
        self.last_indrnn = IndRNNwithBN(args.embed_size,args.seq_len,args.bn_location)
        if args.bn_location=='bn_before':
            self.extra_bn = BN(args.embed_size, args.seq_len) 
        self.decoder = nn.Linear(args.embed_size, outputclass)
        if args.w_tying:
            self.decoder.weight = self.encoder.weight  # .data.transpose(0,1)

        self.init_weights()

    def init_weights(self):
      for name, param in self.named_parameters():
        if 'weight_hh' in name:
          param.data.uniform_(0,U_bound)          
        if ('fc' in name) and 'weight' in name:#'denselayer' in name and 
            nn.init.kaiming_uniform_(param, a=8, mode='fan_in')#
            if args.small_normini:
                nn.init.kaiming_uniform_(param, mode='fan_in')
        if ('encoder' or 'decoder') in name and 'weight' in name:
            param.data.uniform_(-0.01, 0.01)
        if ('norm' in name or 'Norm' in name)  and 'weight' in name:
            param.data.fill_(1)
            if args.small_normini:
                param.data.fill_(0.1)
        if 'bias' in name:
            param.data.fill_(0.0)

    def forward(self, input,hidden_input_all, BN_start=0):
        seq_len, batch_size = input.size()
        rnnoutputs={}
        hidden_x={}     
        if hidden_input_all is not None:
          hidden_x.update(hidden_input_all)   
        else:
          for x in range(len(self.RNNs)):
            hidden_x['hidden%d'%x]=Variable(torch.zeros(1,batch_size,args.hidden_size).cuda())
          hidden_x['hidden_lastindrnn']=Variable(torch.zeros(1,batch_size,args.embed_size).cuda())
        
        input = input.view(seq_len * batch_size)   
        input = embedded_dropout(self.encoder,input, dropout=args.dropout_embedding if self.training else 0)
        input = input.view(seq_len, batch_size, args.embed_size)
        if args.dropout_words > 0:            
            input = dropout_overtime(input, args.dropout_words, self.training)

        rnnoutputs['outlayer-1']=input
        for x in range(len(self.RNNs)):
          rnnoutputs['dilayer%d'%x]=self.DIs[x](rnnoutputs['outlayer%d'%(x-1)])      
          rnnoutputs['outlayer%d'%x],hidden_x['hidden%d'%x]= self.RNNs[x](rnnoutputs['dilayer%d'%x], BN_start, hidden_x['hidden%d'%x])   
          if args.dropout>0:
            dropout=args.dropout
            if x==len(self.RNNs)-1:
                dropout=args.dropout_last
            rnnoutputs['outlayer%d'%x]= dropout_overtime(rnnoutputs['outlayer%d'%x],dropout,self.training) 

        rnn_out=rnnoutputs['outlayer%d'%(len(self.RNNs)-1)]
        rnn_out=self.last_fc(rnn_out)    
        rnn_out,hidden_x['hidden_lastindrnn']=self.last_indrnn(rnn_out,BN_start,hidden_x['hidden_lastindrnn'])       
        if args.bn_location=='bn_before':
          rnn_out=self.extra_bn(rnn_out,BN_start)
        rnn_out = dropout_overtime(rnn_out, args.dropout_extrafc, self.training)
        rnn_out = rnn_out.view(seq_len * batch_size, -1)
        output = self.decoder(rnn_out)
        return output, hidden_x               
            
      