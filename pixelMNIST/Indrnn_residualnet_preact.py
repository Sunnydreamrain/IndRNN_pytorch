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
        if bn_location=='bn_before':      
            self.add_module('norm1', BN(hidden_size, args.seq_len))
        self.add_module('indrnn1', IndRNN(hidden_size))        
        if bn_location=='bn_after':   
            self.add_module('norm1', BN(hidden_size, args.seq_len))
        if (bn_location!='bn_before') and (bn_location!='bn_after'):
            print('Please select a batch normalization mode.')
            assert 2==3

class _residualBlock_ori(nn.Sequential):
    def __init__(self, hidden_size, drop_rate):
        super(_residualBlock_ori, self).__init__()
        self.add_module('fc1', Linear_overtime(hidden_size, hidden_size))
        self.add_module('IndRNNwithBN1', IndRNNwithBN(hidden_size, args.seq_len, args.bn_location))
        if drop_rate>0:
            self.add_module('drop1', Dropout_overtime_module(drop_rate))

        self.add_module('fc2', Linear_overtime(hidden_size, hidden_size))
        self.add_module('IndRNNwithBN2', IndRNNwithBN( hidden_size, args.seq_len, args.bn_location))
        if drop_rate > 0:
            self.add_module('drop2', Dropout_overtime_module(drop_rate))

    def forward(self, x):
        new_features = super(_residualBlock_ori, self).forward(x)
        new_features=x+new_features
        return new_features


class _residualBlock_preact(nn.Sequential):
    def __init__(self, hidden_size, drop_rate):
        super(_residualBlock_preact, self).__init__()
        self.add_module('IndRNNwithBN1', IndRNNwithBN(hidden_size, args.seq_len, args.bn_location))
        if drop_rate > 0:
            self.add_module('drop1', Dropout_overtime_module(drop_rate))
        self.add_module('fc1', Linear_overtime(hidden_size, hidden_size))
        self.add_module('IndRNNwithBN2', IndRNNwithBN(hidden_size, args.seq_len, args.bn_location))
        if drop_rate>0:
            self.add_module('drop2', Dropout_overtime_module(drop_rate))
        self.add_module('fc2', Linear_overtime(hidden_size, hidden_size))

    def forward(self, x):
        new_features = super(_residualBlock_preact, self).forward(x)
        new_features=x+new_features
        return new_features

_residualBlock=_residualBlock_preact
class ResidualNet(nn.Module):  # DenseNet(nn.Module):
    def __init__(self, input_size, num_classes, drop_rate=args.dropout):

        super(ResidualNet, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('fc0', Linear_overtime(input_size, args.hidden_size))               
        # Each resblock
        for i in range(args.num_blocks):
            block = _residualBlock(args.hidden_size, drop_rate=drop_rate)
            self.features.add_module('resblock%d' % (i + 1), block)

        self.features.add_module('IndRNNwithBN_last', IndRNNwithBN(args.hidden_size, args.seq_len, args.bn_location))        
        if drop_rate>0:
           self.features.add_module('droplast', Dropout_overtime_module(drop_rate))  
        self.classifier = nn.Linear(args.hidden_size, num_classes)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            #print (name)
            if 'weight_hh' in name:
                param.data.uniform_(0, U_bound)
                if args.u_lastlayer_ini and ('IndRNNwithBN_last' in name) and ('.weight_hh' in name):
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
