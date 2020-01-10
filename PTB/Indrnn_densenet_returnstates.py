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
from utils import Linear_overtime_module,Dropout_overtime,embedded_dropout
from utils import MyBatchNorm_stepCompute  # a quick implementation of frame-wise batch normalization
#from utils import Batch_norm_step as MyBatchNorm_stepCompute
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
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,drop_rate_2):
        super(_DenseLayer, self).__init__()
        self.add_module('fc1', Linear_overtime(num_input_features, bn_size * growth_rate))
        self.add_module('IndRNNwithBN1', IndRNNwithBN(bn_size * growth_rate, args.seq_len,args.bn_location))
        self.add_module('fc2', Linear_overtime(bn_size * growth_rate, growth_rate))
        self.add_module('IndRNNwithBN2', IndRNNwithBN( growth_rate, args.seq_len,args.bn_location))
        self.drop_rate_2=drop_rate_2
        self.drop_rate=drop_rate

    def forward(self, input, BN_start=0,hidden_out1=None,hidden_out2=None):
        x=input
        out_1=self.fc1(input)
        out_1,hidden_out1=self.IndRNNwithBN1(out_1, BN_start,hidden_out1)
        if self.drop_rate_2>0:
            out_1=dropout_overtime(out_1,self.drop_rate_2, self.training)
        out_2=self.fc2(out_1)
        out_2,hidden_out2=self.IndRNNwithBN2(out_2, BN_start,hidden_out2)
        if self.drop_rate>0:
            out_2=dropout_overtime(out_2,self.drop_rate, self.training)
        return torch.cat([x, out_2], 2),hidden_out1,hidden_out2

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate,drop_rate_2):
        super(_DenseBlock, self).__init__()
        self.num_layers=num_layers
        self.denselayers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate,drop_rate_2)
            self.denselayers.append(layer)

    def forward(self, input, BN_start=0,hidden_x={},hid_ind_layer=0):
        rnnoutputs={}
        if not ('hidden%d'%hid_ind_layer in hidden_x.keys()):
            for x in range(len(self.denselayers)):   
                hidden_x['hidden%d'%(hid_ind_layer+2*x)]=None
                hidden_x['hidden%d'%(hid_ind_layer+2*x+1)]=None
        rnnoutputs['outlayer-1']=input
        for x in range(len(self.denselayers)):   
            rnnoutputs['outlayer%d'%x],hidden_x['hidden%d'%(hid_ind_layer+2*x)],hidden_x['hidden%d'%(hid_ind_layer+2*x+1)]= \
            self.denselayers[x](rnnoutputs['outlayer%d'%(x-1)], BN_start, hidden_x['hidden%d'%(hid_ind_layer+2*x)], hidden_x['hidden%d'%(hid_ind_layer+2*x+1)]) 
        return rnnoutputs['outlayer%d'%(len(self.denselayers)-1)],hidden_x,hid_ind_layer+2*len(self.denselayers)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate, lasttrans=False):
        super(_Transition, self).__init__()
        self.add_module('fc', Linear_overtime(num_input_features, num_output_features))
        self.add_module('IndRNNwithBN1', IndRNNwithBN(num_output_features, args.seq_len,args.bn_location))
        self.drop_rate=drop_rate
        self.lasttrans=lasttrans
        if args.bn_location=='bn_before' and lasttrans:
            self.add_module('extra_bn', BN(num_output_features, args.seq_len))
    def forward(self, input, BN_start=0,hidden_x={},hid_ind_layer=0):
        if not ('hidden%d'%(hid_ind_layer) in hidden_x.keys()):
            hidden_x['hidden%d'%(hid_ind_layer)]=None
        out_1=self.fc(input)
        out_1,hidden_x['hidden%d'%(hid_ind_layer)]=self.IndRNNwithBN1(out_1, BN_start,hidden_x['hidden%d'%(hid_ind_layer)])
        if args.bn_location=='bn_before' and self.lasttrans:
            out_1=self.extra_bn(out_1, BN_start)
        if self.drop_rate>0:
            out_1=dropout_overtime(out_1,self.drop_rate, self.training)
        return out_1,hidden_x,hid_ind_layer+1

class DenseNet(nn.Module):  # DenseNet(nn.Module):
    def __init__(self, outputclass, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=args.dropout,drop_rate_2=args.dropout_sec,
                 drop_rate_trans=args.dropout_trans):#,drop_rate_first=args.dropout_first,drop_rate_last=args.dropout_last

        super(DenseNet, self).__init__()
        self.encoder = torch.nn.Embedding(outputclass, args.embed_size)
        num_features = args.embed_size  
        self.DenseBlocks = nn.ModuleList()  
        self.TransBlocks = nn.ModuleList()  
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, drop_rate_2=drop_rate_2)
            self.DenseBlocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                    drop_rate=drop_rate_trans)
                self.TransBlocks.append(trans)
                num_features = num_features // 2
        
        extra_fc = _Transition(num_input_features=num_features, num_output_features=args.embed_size,
                            drop_rate=args.dropout_extrafc,lasttrans=True)
        self.TransBlocks.append(extra_fc)
        self.num_blocks=len(self.DenseBlocks)

        self.decoder = nn.Linear(args.embed_size, outputclass)
        if args.w_tying:# and args.embed_size == 0:
            self.decoder.weight = self.encoder.weight  # .data.transpose(0,1)

        self.init_weights()

    def init_weights(self):
        paras0=self.named_parameters()
        init_param_values=list(paras0)
        i=0
        for name, param in self.named_parameters():
            #print (name,param)
            if 'weight_hh' in name:
                param.data.uniform_(0, U_bound)                    
            elif ('encoder' or 'decoder') in name and ('weight' in name):
                param.data.uniform_(-0.01, 0.01)                     
            elif ('fc' in name) and ('weight' in name):
                nn.init.kaiming_uniform_(param, a=8, mode='fan_in')    
                if args.small_normini:
                    nn.init.kaiming_uniform_(param, mode='fan_in')        
            elif ('norm' in name) and ('weight' in name):
                param.data.fill_(1.0)   
                if args.small_normini:
                    param.data.fill_(0.1)   
            elif 'bias' in name:
                param.data.fill_(0.0)
            else:
                print('wrong initialization')
            i=i+1

    def forward(self, input,hidden_input_all, BN_start=0): 
        seq_len, batch_size = input.size()
        rnnoutputs={}
        hidden_x={}  
        hid_ind_layer=0
        if hidden_input_all is not None:
            hidden_x.update(hidden_input_all)  
        input = input.view(seq_len * batch_size)
        input = embedded_dropout(self.encoder,input, dropout=args.dropout_embedding if self.training else 0)
        input = input.view(seq_len, batch_size, args.embed_size)
        if args.dropout_words > 0:            
            input = dropout_overtime(input, args.dropout_words, self.training)#

        rnnoutputs['trans_outlayer-1']=input
        for i in range(self.num_blocks):
            rnnoutputs['dense_outlayer%d'%i],hidden_x,hid_ind_layer=self.DenseBlocks[i](rnnoutputs['trans_outlayer%d'%(i-1)], BN_start,hidden_x,hid_ind_layer)
            if i != self.num_blocks - 1:
                rnnoutputs['trans_outlayer%d'%i],hidden_x,hid_ind_layer=self.TransBlocks[i](rnnoutputs['dense_outlayer%d'%i], BN_start,hidden_x,hid_ind_layer)
        rnnoutputs['trans_outlayer%d'%(self.num_blocks-1)],hidden_x,hid_ind_layer=self.TransBlocks[self.num_blocks-1](rnnoutputs['dense_outlayer%d'%(self.num_blocks-1)], BN_start,hidden_x,hid_ind_layer)

        rnn_out=rnnoutputs['trans_outlayer%d'%(self.num_blocks-1)]
        rnn_out = rnn_out.view(seq_len * batch_size, args.embed_size)
        rnn_out=self.decoder(rnn_out)
        return rnn_out, hidden_x
