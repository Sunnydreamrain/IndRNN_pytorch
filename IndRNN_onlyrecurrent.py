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
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)`

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          or :func:`torch.nn.utils.rnn.pack_sequence`
          for details.
        - **h_0** of shape `(num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, hidden_size * num_directions)`: tensor
          containing the output features (`h_k`) from the last layer of the RNN,
          for each `k`.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_directions, batch, hidden_size): tensor
          containing the hidden state for `k = seq_len`.
    """

    def __init__(self, hidden_size, 
                 batch_first=False, bidirectional=False, recurrent_inits=None,
                 **kwargs):
        super(IndRNN_onlyrecurrent, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        num_directions = 2 if self.bidirectional else 1

        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        cells = []
        directions = []
        if recurrent_inits is not None:
            kwargs["recurrent_init"] = recurrent_inits
        for dir in range(num_directions):
            directions.append(IndRNNCell_onlyrecurrent(hidden_size, **kwargs))
        self.cells = nn.ModuleList(directions)

        h0 = torch.zeros(hidden_size * num_directions)
        self.register_buffer('h0', torch.autograd.Variable(h0))

    def forward(self, x, hidden=None):
        time_index = self.time_index
        batch_index = self.batch_index
        num_directions = 2 if self.bidirectional else 1
        hidden_init = self.h0.unsqueeze(0).expand(
            x.size(batch_index),
            self.hidden_size * num_directions).contiguous()

        x_n = []
        for dir, cell in enumerate(self.cells):
            hx_cell = hidden_init[
                :, self.hidden_size * dir: self.hidden_size * (dir + 1)]
            outputs = []
            hiddens = []
            x_T = torch.unbind(x, time_index)
            if dir == 1:
                x_T = reversed(x_T)
            for x_t in x_T:
                hx_cell = cell(x_t, hx_cell)
                outputs.append(hx_cell)
            if dir == 1:
                outputs = outputs[::-1]
            x_cell = torch.stack(outputs, time_index)
            x_n.append(x_cell)
            hiddens.append(hx_cell)
        x = torch.cat(x_n, -1)
        return x.squeeze(2)#, torch.cat(hiddens, -1)
