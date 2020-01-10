from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.init as weight_init
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.nn import Parameter
from cuda_IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN
#if no cuda, then use the following line
# from IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN



from __main__ import parser,args,U_bound
MAG=args.MAG
#U_bound=np.power(10,(np.log10(MAG)/args.seq_len))
U_lowbound=np.power(10,(np.log10(1.0/MAG)/args.seq_len))  
from utils import Linear_overtime_module,Dropout_overtime,embedded_dropout,Dropout_overtime_module
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

class _residualBlock_preact(nn.Sequential):
    def __init__(self, hidden_size, drop_rate):
        super(_residualBlock_preact, self).__init__()        
        self.add_module('IndRNNwithBN1', IndRNNwithBN(hidden_size, args.seq_len,bn_location=args.bn_location))        
        self.add_module('fc1', Linear_overtime(hidden_size, hidden_size))
        self.add_module('IndRNNwithBN2', IndRNNwithBN(hidden_size, args.seq_len,bn_location=args.bn_location))    
        self.add_module('fc2', Linear_overtime(hidden_size, hidden_size))
        self.drop_rate=drop_rate

    def forward(self, input, BN_start=0,hidden_out1=None,hidden_out2=None):
        x=input
        out_1,hidden_out1=self.IndRNNwithBN1(input, BN_start,hidden_out1)
        if self.drop_rate>0:
            out_1=dropout_overtime(out_1,self.drop_rate, self.training)
        out_2=self.fc1(out_1)
        out_3,hidden_out2=self.IndRNNwithBN2(out_2, BN_start,hidden_out2)      
        if self.drop_rate>0:
            out_3=dropout_overtime(out_3, self.drop_rate, self.training)
        out_4=self.fc2(out_3)            
        out_4=x + out_4
        return out_4,hidden_out1,hidden_out2
_residualBlock=_residualBlock_preact
class ResidualNet(nn.Module):  # DenseNet(nn.Module):
    def __init__(self, outputclass, drop_rate=args.dropout):#,drop_rate_first=args.dropout_first,drop_rate_last=args.dropout_last

        super(ResidualNet, self).__init__()
        self.encoder = torch.nn.Embedding(outputclass, args.embed_size)
        self.fc0=Linear_overtime(args.embed_size, args.hidden_size)
        self.resblocks = nn.ModuleList()
        # Each resblock
        for i in range(args.num_blocks):
            block = _residualBlock(args.hidden_size, drop_rate=drop_rate)
            self.resblocks.append(block)

        self.IndRNNwithBN_resfinal=IndRNNwithBN(args.hidden_size, args.seq_len,bn_location=args.bn_location)  
        self.last_fc=Linear_overtime(args.hidden_size, args.embed_size)              
        self.last_indrnn=IndRNNwithBN(args.embed_size, args.seq_len,bn_location=args.bn_location)
        if args.bn_location=='bn_before':
            self.extra_bn=BN(args.embed_size, args.seq_len)

        self.decoder = nn.Linear(args.embed_size, outputclass)
        if args.w_tying:# and args.embed_size == 0:
            self.decoder.weight = self.encoder.weight  # .data.transpose(0,1)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            #print (name,param)
            if 'weight_hh' in name:
                param.data.uniform_(0, U_bound)                
            elif 'fc' in name and 'weight' in name:
                nn.init.kaiming_uniform_(param, a=8, mode='fan_in')
                if args.small_normini:
                    nn.init.kaiming_uniform_(param, mode='fan_in')
            elif ('encoder' or 'decoder') in name and 'weight' in name:
                param.data.uniform_(-0.01, 0.01)                
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)  
                if args.small_normini:
                    param.data.fill_(0.1)
            elif 'bias' in name:
                param.data.fill_(0.0)

    def forward(self, input,hidden_input_all, BN_start=0): 
        seq_len, batch_size = input.size()
        rnnoutputs={}
        hidden_x={}     
        if hidden_input_all is not None:
            hidden_x.update(hidden_input_all)   
        else:
            for x in range(2*len(self.resblocks)):
                hidden_x['hidden%d'%x]=Variable(torch.zeros(1,batch_size,args.hidden_size).cuda())
            hidden_x['hidden_resfinal']=Variable(torch.zeros(1,batch_size,args.hidden_size).cuda())
            hidden_x['hidden_lastindrnn']=Variable(torch.zeros(1,batch_size,args.embed_size).cuda())
        
        input = input.view(seq_len * batch_size)
        input = embedded_dropout(self.encoder,input, dropout=args.dropout_embedding if self.training else 0)
        input = input.view(seq_len, batch_size, args.embed_size)
        if args.dropout_words > 0:            
            input = dropout_overtime(input, args.dropout_words, self.training)#  
        
        rnnoutputs['outlayer-1']=self.fc0(input)    
        for x in range(len(self.resblocks)):   
            rnnoutputs['outlayer%d'%x],hidden_x['hidden%d'%(2*x)],hidden_x['hidden%d'%(2*x+1)]= self.resblocks[x](rnnoutputs['outlayer%d'%(x-1)], BN_start, hidden_x['hidden%d'%(2*x)], hidden_x['hidden%d'%(2*x+1)])   

        rnn_out=rnnoutputs['outlayer%d'%(len(self.resblocks)-1)]
        rnn_out,hidden_x['hidden_resfinal']=self.IndRNNwithBN_resfinal(rnn_out, BN_start, hidden_x['hidden_resfinal'])  
        if args.dropout_last>0:
            rnn_out=dropout_overtime(rnn_out,args.dropout_last, self.training)

        rnn_out=self.last_fc(rnn_out)        
        rnn_out1, hidden_x['hidden_lastindrnn']=self.last_indrnn(rnn_out, BN_start, hidden_x['hidden_lastindrnn'])
        if args.bn_location=='bn_before':
            rnn_out1=self.extra_bn(rnn_out1, BN_start)
        if args.dropout_extrafc>0:
           rnn_out1=dropout_overtime(rnn_out1,args.dropout_extrafc, self.training)

        rnn_out1 = rnn_out1.view(seq_len * batch_size, args.embed_size)
        rnn_out1=self.decoder(rnn_out1)        
        return rnn_out1, hidden_x
