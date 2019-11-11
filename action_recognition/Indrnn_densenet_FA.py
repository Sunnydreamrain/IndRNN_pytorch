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
#from IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN

from __main__ import parser,args,U_bound
MAG=args.MAG
#U_bound=np.power(10,(np.log10(MAG)/args.seq_len))
U_lowbound=np.power(10,(np.log10(1.0/MAG)/args.seq_len))  
from utils import Batch_norm_overtime,Linear_overtime_module,Dropout_overtime_module
BN=Batch_norm_overtime
Linear_overtime=Linear_overtime_module

if args.time_diff:
    from utils import FA_timediff_f
    FA_factor=2
    FA_timediff=FA_timediff_f
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
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,drop_rate_2):
        super(_DenseLayer, self).__init__()
        self.add_module('fc1', Linear_overtime(num_input_features, bn_size * growth_rate))
        self.add_module('IndRNNwithBN1', IndRNNwithBN(bn_size * growth_rate, args.seq_len, args.bn_location))
        if drop_rate_2>0:
            self.add_module('drop2', Dropout_overtime_module(drop_rate_2))
        if args.time_diff:
            self.add_module('FA1', FA_timediff())
            bn_size=bn_size*FA_factor

        self.add_module('fc2', Linear_overtime(bn_size * growth_rate, growth_rate))
        self.add_module('IndRNNwithBN2', IndRNNwithBN(growth_rate, args.seq_len, args.bn_location))
        if drop_rate > 0:
            self.add_module('drop1', Dropout_overtime_module(drop_rate))
        if args.time_diff:
            self.add_module('FA2', FA_timediff())

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 2)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate,drop_rate_2):
        super(_DenseBlock, self).__init__()
        feature_factor=1
        if args.time_diff:
            feature_factor=FA_factor
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + feature_factor * i * growth_rate, growth_rate, bn_size, drop_rate,drop_rate_2)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate, last_layer=False):
        super(_Transition, self).__init__()
        self.add_module('fc', Linear_overtime(num_input_features, num_output_features))
        self.add_module('IndRNNwithBN', IndRNNwithBN(num_output_features, args.seq_len, args.bn_location))
        if drop_rate>0:
            self.add_module('drop', Dropout_overtime_module(drop_rate))
        if args.time_diff:
            self.add_module('FA1', FA_timediff())
        #self.add_module('drop', Dropout_overtime_module(drop_rate))

class DenseNet(nn.Module):  # DenseNet(nn.Module):
    def __init__(self, input_size, num_classes, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=args.dropout,drop_rate_2=args.dropout_sec,
                 drop_rate_trans=args.dropout_trans,drop_rate_first=args.dropout_first,drop_rate_last=args.dropout_last):

        super(DenseNet, self).__init__()
        self.features = nn.Sequential()
        if args.time_diff:
            self.features.add_module('FA0', FA_timediff())
            input_size=input_size*2
        self.features.add_module('fc0', Linear_overtime(input_size, num_init_features))
        self.features.add_module('IndRNNwithBN0', IndRNNwithBN(num_init_features, args.seq_len, args.bn_location))

        num_features = num_init_features      
        if drop_rate_first>0:
            self.features.add_module('drop0', Dropout_overtime_module(drop_rate_first))   
        if args.time_diff:
            self.features.add_module('FA1', FA_timediff())         

        # Each denseblock
        #num_features = num_init_features
        num_features_input=num_features
        if args.time_diff:
            num_features_input=num_features*FA_factor
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features_input,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, drop_rate_2=drop_rate_2)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features_input = num_features_input + num_layers * growth_rate            
            num_features_output=num_features + num_layers * growth_rate
            if args.time_diff:
                num_features_input=num_features_input + num_layers * growth_rate*(FA_factor-1)#1344-768=576
            num_features=num_features_output
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features_input, num_output_features=num_features // 2,
                                    drop_rate=drop_rate_trans)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            num_features_input=num_features
            if args.time_diff:
                num_features_input=num_features*FA_factor

        if args.add_last_layer:      
            last_layer = _Transition(num_input_features=num_features_input, num_output_features=num_features // 2,
                                drop_rate=drop_rate_last,last_layer=True)
            self.features.add_module('lastlayer', last_layer)
            num_features = num_features // 2
            num_features_input=num_features
            if args.time_diff:
                num_features_input=num_features*2

        self.classifier = nn.Linear(num_features_input, num_classes)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_hh' in name:
                param.data.uniform_(0, U_bound)
                if args.u_lastlayer_ini and 'lastlayer' in name and 'indrnn' in name:
                    param.data.uniform_(U_lowbound, U_bound)
                    print('correct last layer ini')
            if ('fc' in name) and 'weight' in name:#'denselayer' in name and 
                nn.init.kaiming_uniform_(param, a=8, mode='fan_in')#
                if args.small_normini:
                    nn.init.kaiming_uniform_(param, mode='fan_in')
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if ('norm' in name or 'Norm' in name)  and 'weight' in name:
                param.data.fill_(1)
                if args.small_normini:
                    param.data.fill_(0.1)
            if 'bias' in name:
                param.data.fill_(0.0)

    def forward(self, x):
        features = self.features(x)
        out = features[-1]
        out = self.classifier(out)
        return out
