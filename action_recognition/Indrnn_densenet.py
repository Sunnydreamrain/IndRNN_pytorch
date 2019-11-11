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
from utils import Batch_norm_overtime,Linear_overtime_module,Dropout_overtime_module
BN=Batch_norm_overtime
Linear_overtime=Linear_overtime_module
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
        if (bn_location!='bn_before'):
            print('You are selecting the bn_before mode, where batch normalization is used before the recurrent connection.\
            It generally provides a stable but worse results than the bn_after mode. So use bn_after first.')
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,drop_rate_2):
        super(_DenseLayer, self).__init__()
        self.add_module('fc1', Linear_overtime(num_input_features, bn_size * growth_rate))
        self.add_module('IndRNNwithBN1', IndRNNwithBN(bn_size * growth_rate, args.seq_len, args.bn_location))
        if drop_rate_2>0:
            self.add_module('drop2', Dropout_overtime_module(drop_rate_2))

        self.add_module('fc2', Linear_overtime(bn_size * growth_rate, growth_rate))
        self.add_module('IndRNNwithBN2', IndRNNwithBN(growth_rate, args.seq_len, args.bn_location))
        if drop_rate > 0:
            self.add_module('drop1', Dropout_overtime_module(drop_rate))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 2)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate,drop_rate_2):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate,drop_rate_2)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate, last_layer=False):
        super(_Transition, self).__init__()
        self.add_module('fc', Linear_overtime(num_input_features, num_output_features))
        self.add_module('IndRNNwithBN', IndRNNwithBN(num_output_features, args.seq_len, args.bn_location))
        if drop_rate>0:
            self.add_module('drop', Dropout_overtime_module(drop_rate))

class DenseNet(nn.Module):  # DenseNet(nn.Module):
    def __init__(self, input_size, num_classes, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=args.dropout,drop_rate_2=args.dropout_sec,
                 drop_rate_trans=args.dropout_trans,drop_rate_first=args.dropout_first,drop_rate_last=args.dropout_last):

        super(DenseNet, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('fc0', Linear_overtime(input_size, num_init_features))
        self.features.add_module('IndRNNwithBN0', IndRNNwithBN(num_init_features, args.seq_len, args.bn_location))
        if drop_rate_first>0:
            self.features.add_module('drop0', Dropout_overtime_module(drop_rate_first))          

        # Each denseblock
        num_features = num_init_features  
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, drop_rate_2=drop_rate_2)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features=num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                    drop_rate=drop_rate_trans)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        #It seems adding one last transition layer shows little effect. But in the language modeling task where weight tying is used, it is necessary.
        #It also eases the weight_hh initialization if the last layer is specially initialized.  
        if args.add_last_layer:   
            last_layer = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                drop_rate=drop_rate_trans,last_layer=True)
            self.features.add_module('lastlayer', last_layer)
            num_features = num_features // 2

        self.classifier = nn.Linear(num_features, num_classes)
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
            if 'classifier' in name and 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if ('norm' in name or 'Norm' in name)  and 'weight' in name:
                param.data.fill_(1)
            if 'bias' in name:
                param.data.fill_(0.0)

    def forward(self, x):        
        features = self.features(x)
        out = features[-1]
        out = self.classifier(out)
        return out
