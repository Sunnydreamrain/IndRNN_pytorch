"""
This code is to implement the IndRNN (only the recurrent part). The code is based on the implementation from 
https://github.com/StefOe/indrnn-pytorch/blob/master/indrnn.py.
Since this only contains the recurrent part of IndRNN, fully connected layers or convolutional layers are needed before it.
Please cite the following paper if you find it useful.
Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN," 
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5457-5466. 2018.
@inproceedings{li2018independently,
  title={Independently recurrent neural network (indrnn): Building A longer and deeper RNN},
  author={Li, Shuai and Li, Wanqing and Cook, Chris and Zhu, Ce and Gao, Yanbo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5457--5466},
  year={2018}
}
"""


import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class IndRNNCell_onlyrecurrent(nn.Module):
    r"""An IndRNN cell with ReLU non-linearity. This is only the recurrent part where the input is already processed with w_{ih} * x + b_{ih}.

    .. math::
        input=w_{ih} * x + b_{ih}
        h' = \relu(input +  w_{hh} (*) h)
    With (*) being element-wise vector multiplication.

    Args:
        hidden_size: The number of features in the hidden state h

    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
    """

    def __init__(self, hidden_size, 
                 hidden_max_abs=None, recurrent_init=None):
        super(IndRNNCell_onlyrecurrent, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent_init = recurrent_init
        self.weight_hh = Parameter(torch.Tensor(hidden_size))            
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "weight_hh" in name:
                if self.recurrent_init is None:
                    nn.init.uniform(weight, a=0, b=1)
                else:
                    self.recurrent_init(weight)

    def forward(self, input, hx):
        return F.relu(input + hx * self.weight_hh.unsqueeze(0).expand(hx.size(0), len(self.weight_hh)))


class IndRNN_onlyrecurrent(nn.Module):
    r"""Applies an IndRNN with `ReLU` non-linearity to an input sequence. 
    This is only the recurrent part where the input is already processed with w_{ih} * x + b_{ih}.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

        h_t = \relu(input_t +  w_{hh} (*) h_{(t-1)})

    where :math:`h_t` is the hidden state at time `t`, and :math:`input_t`
    is the input at time `t`. (*) is element-wise multiplication.

    Args:
        hidden_size: The number of features in the hidden state `h`        

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          or :func:`torch.nn.utils.rnn.pack_sequence`
          for details.
        - **h_0** of shape `( batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: output  
        - **output** of shape `(seq_len, batch, hidden_size)`
    """

    def __init__(self, hidden_size,recurrent_init=None, **kwargs):
        super(IndRNN_onlyrecurrent, self).__init__()
        self.hidden_size = hidden_size
        self.indrnn_cell=IndRNNCell_onlyrecurrent(hidden_size, **kwargs)

        if recurrent_init is not None:
            kwargs["recurrent_init"] = recurrent_init
        self.recurrent_init=recurrent_init
        # h0 = torch.zeros(hidden_size * num_directions)
        # self.register_buffer('h0', torch.autograd.Variable(h0))
        self.reset_parameters()

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "weight_hh" in name:
                if self.recurrent_init is None:
                    nn.init.uniform(weight, a=0, b=1)
                else:
                    self.recurrent_init(weight)

    def forward(self, input, h0=None):
        assert input.dim() == 2 or input.dim() == 3        
        if h0 is None:
            h0 = input.data.new(input.size(-2),input.size(-1)).zero_().contiguous()
        elif (h0.size(-1)!=input.size(-1)) or (h0.size(-2)!=input.size(-2)):
            raise RuntimeError(
                'The initial hidden size must be equal to input_size. Expected {}, got {}'.format(
                    h0.size(), input.size()))
        outputs=[]
        hx_cell=h0
        for input_t in input:
            hx_cell = self.indrnn_cell(input_t, hx_cell)
            outputs.append(hx_cell)
        out_put = torch.stack(outputs, 0)
        return out_put
