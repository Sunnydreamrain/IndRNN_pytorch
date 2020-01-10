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

# from IndRNN_onlyrecurrent import IndRNN_onlyrecurrent as IndRNN

class Batch_norm_overtime(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super(Batch_norm_overtime, self).__init__()
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.bn(x.clone())
        x = x.permute(2, 0, 1)
        return x

class Batch_norm_step_module(nn.Module):
    def __init__(self,  hidden_size,seq_len):
        super(Batch_norm_step_module, self).__init__()
        self.hidden_size = hidden_size
        
        self.max_time_step=seq_len+50
        self.bns = nn.ModuleList()

        for x in range(self.max_time_step):
            bn = nn.BatchNorm1d(hidden_size)
            self.bns.append(bn)       
        for x in range(1,self.max_time_step):
            self.bns[x].weight=self.bns[0].weight    
            self.bns[x].bias=self.bns[0].bias                    

    def forward(self, x):
        output=[]
        for t, input_t in enumerate(x.split(1)):
          input_t=input_t.squeeze(dim=0)
          input_tbn= self.bns[t](input_t)
          output.append(input_tbn)
        output = torch.stack(output)
        return output

class Linear_overtime_module(nn.Module):
    def __init__(self, input_size, hidden_size,bias=True):
        super(Linear_overtime_module, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size, bias=bias)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        y = x.contiguous().view(-1, self.input_size)
        y = self.fc(y)
        y = y.view(x.size()[0], x.size()[1], self.hidden_size)
        return y


class FA_timediff(nn.Module):
    def __init__(self):
        super(FA_timediff, self).__init__()

    def forward(self, x):
        new_x_b=x.clone()
        new_x_f=x.clone()
        new_x_f[1::]=x[1::]-x[0:-1]
        new_x_b[0:-1]=x[0:-1]-x[1::]
        y=torch.cat([x,new_x_f,new_x_b],dim=2)
        #print (y.size())
        return y

class FA_timediff_f(nn.Module):
    def __init__(self):
        super(FA_timediff_f, self).__init__()

    def forward(self, x):
        new_x_f=x.clone()
        new_x_f[1::]=x[1::]-x[0:-1]
        y=torch.cat([x,new_x_f],dim=2)
        #print (y.size())
        return y

class Dropout_overtime(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p=0.5, training=False):
        output = input.clone()
        noise = input.data.new(input.size(-2), input.size(-1))  # torch.ones_like(input[0])
        if training:
            noise.bernoulli_(1 - p).div_(1 - p)
            noise = noise.unsqueeze(0).expand_as(input)
            output.mul_(noise)
        ctx.save_for_backward(noise)
        ctx.training = training
        return output

    @staticmethod
    def backward(ctx, grad_output):
        noise, = ctx.saved_tensors
        if ctx.training:
            return grad_output.mul(noise), None, None
        else:
            return grad_output, None, None


dropout_overtime = Dropout_overtime.apply


class Dropout_overtime_module(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout_overtime_module, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        output = input.clone()
        if self.training and self.p > 0:
            noise = output.data.new(output.size(-2), output.size(-1))  # torch.ones_like(input[0])
            noise.bernoulli_(1 - self.p).div_(1 - self.p)
            noise = noise.unsqueeze(0).expand_as(output)
            output.mul_(noise)
        return output


def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight
  else:
    masked_embed_weight = embed.weight
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
      padding_idx = -1

  X = torch.nn.functional.embedding(words, masked_embed_weight,
    padding_idx, embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
  )
  return X

class Batch_norm_step(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super(Batch_norm_step, self).__init__()
        self.max_time_step = seq_len*3+1#00
        self.bns = nn.ModuleList()
        for x in range(self.max_time_step):
            bn = nn.BatchNorm1d(hidden_size)  # .train(True)
            self.bns.append(bn)
        for x in range(1, self.max_time_step):
            self.bns[x].weight = self.bns[0].weight
            self.bns[x].bias = self.bns[0].bias

    def forward(self, x, BN_start):
        output = []
        for t, x_t in enumerate(x.split(1)):
            x_t = x_t.squeeze(dim=0)
            t_step=t+BN_start
            if t+BN_start>=self.max_time_step:
                t_step=self.max_time_step-1
            y = self.bns[t_step](x_t)  # .clone()
            output.append(y)
        output = torch.stack(output)
        return output


class Batch_norm_step_maxseq(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super(Batch_norm_step_maxseq, self).__init__()
        self.max_time_step = seq_len*4+1#00
        self.max_start = seq_len*2
        self.bns = nn.ModuleList()
        for x in range(self.max_time_step):
            bn = nn.BatchNorm1d(hidden_size)  # .train(True)
            self.bns.append(bn)
        for x in range(1, self.max_time_step):
            self.bns[x].weight = self.bns[0].weight
            self.bns[x].bias = self.bns[0].bias

    def forward(self, x, BN_start0):
        output = []
        if BN_start0>self.max_start:
            BN_start=self.max_start
        else:
            BN_start=BN_start0
        for t, x_t in enumerate(x.split(1)):
            x_t = x_t.squeeze(dim=0)
            t_step=t+BN_start
            y = self.bns[t_step](x_t)  # .clone()
            output.append(y)
        output = torch.stack(output)
        return output

class MyBatchNorm_stepCompute(nn.Module):#quick framewise bn, the first few batches are only updated by themselves, but the later ones are shared
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, hidden_size, seq_len, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):#num_features
        super(MyBatchNorm_stepCompute, self).__init__()
        time_d=seq_len*4+1
        self.max_time_step = seq_len*2
        channel_d=hidden_size
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.axes=(1,)
        if self.affine:
            self.weight = Parameter(torch.Tensor(channel_d))#time_d
            self.bias = Parameter(torch.Tensor(channel_d))
            self.register_parameter('weight', self.weight)
            self.register_parameter('bias', self.bias)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(time_d,channel_d))
            self.register_buffer('running_var', torch.ones(time_d,channel_d))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()
        # for i in range(1,time_d):
        #     self.weight[i]=self.weight[0]
        #     self.bias[i]=self.bias[0]

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            #nn.init.uniform_(self.weight)
            #nn.init.zeros_(self.bias)
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))


    def forward(self, input, BN_start0):
        self._check_input_dim(input)
        BN_start=BN_start0
        if BN_start0>self.max_time_step:
            BN_start=self.max_time_step

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        input_t,_,_=input.size()
        #input_t=len(input)
        # calculate running estimates
        if self.training:
            mean = input.mean(1)#torch.cumsum(a, dim=0)  cumsummdr100 / np.arange(1,51)
            # use biased var in train
            var = input.var(1, unbiased=False)
            n = input.size()[1]#input.numel() / input.size(1)#* n / (n - 1)
            
            #part_mean=self.running_mean[:input_t]
            with torch.no_grad():
                self.running_mean[BN_start:input_t+BN_start] = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean[BN_start:input_t+BN_start]
                # update running_var with unbiased var
                self.running_var[BN_start:input_t+BN_start] = exponential_average_factor * var* n / (n - 1) \
                    + (1 - exponential_average_factor) * self.running_var[BN_start:input_t+BN_start]
        else:
            mean = self.running_mean[BN_start:input_t+BN_start]
            var = self.running_var[BN_start:input_t+BN_start]

        input1 = (input - mean[:, None, :]) / (torch.sqrt(var[:, None, :] + self.eps))
        if self.affine:
            input1 = input1 * self.weight[None, None, :] + self.bias[None, None, :]#[None, :, None, None]

        return input1


# class MyBatchNorm_stepCompute(nn.Module):#quick framewise bn, the first few batches are only updated by themselves, but the later ones are shared
#     _version = 2
#     __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
#                      'running_mean', 'running_var', 'num_batches_tracked']

#     def __init__(self, hidden_size, seq_len, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=True):#num_features
#         super(MyBatchNorm_stepCompute, self).__init__()
#         time_d=seq_len*4+1
#         self.max_time_step = seq_len*2
#         channel_d=hidden_size
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         self.axes=(1,)
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(channel_d))#time_d
#             self.bias = Parameter(torch.Tensor(channel_d))
#             self.register_parameter('weight', self.weight)
#             self.register_parameter('bias', self.bias)
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(time_d,channel_d))
#             self.register_buffer('running_var', torch.ones(time_d,channel_d))
#             self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#         else:
#             self.register_parameter('running_mean', None)
#             self.register_parameter('running_var', None)
#             self.register_parameter('num_batches_tracked', None)
#         self.reset_parameters()
#         # for i in range(1,time_d):
#         #     self.weight[i]=self.weight[0]
#         #     self.bias[i]=self.bias[0]

#     def reset_running_stats(self):
#         if self.track_running_stats:
#             self.running_mean.zero_()
#             self.running_var.fill_(1)
#             self.num_batches_tracked.zero_()

#     def reset_parameters(self):
#         self.reset_running_stats()
#         if self.affine:
#             #nn.init.uniform_(self.weight)
#             #nn.init.zeros_(self.bias)
#             self.weight.data.fill_(1.0)
#             self.bias.data.fill_(0.0)

#     def _check_input_dim(self, input):
#         if input.dim() != 3:
#             raise ValueError('expected 3D input (got {}D input)'
#                              .format(input.dim()))


#     def forward(self, input, BN_start0):
#         self._check_input_dim(input)
#         BN_start=BN_start0
#         if BN_start0>self.max_time_step:
#             BN_start=self.max_time_step

#         exponential_average_factor = 0.0

#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum

#         input_t,_,_=input.size()
#         #input_t=len(input)
#         # calculate running estimates
#         if self.training:
#             mean = input.mean(1)#torch.cumsum(a, dim=0)  cumsummdr100 / np.arange(1,51)
#             # use biased var in train
#             var = input.var(1, unbiased=False)
#             var=torch.sqrt(var + self.eps)  #var actually represents the inv_std. This is useful using the running variance
#             n = input.size()[1]#input.numel() / input.size(1)#* n / (n - 1)
            
#             with torch.no_grad():##constrain the mean to be not too far away from the global average
#                 stable_mean=mean
#                 stable_var=var
#                 # if self.num_batches_tracked>1000:
#                 #     stable_mean.data = torch.min(stable_mean.data, 10*self.running_mean[BN_start:input_t+BN_start].data)
#                 #     stable_var.data = torch.min(stable_var.data, 10*self.running_var[BN_start:input_t+BN_start].data)
                    
#                 self.running_mean[BN_start:input_t+BN_start] = exponential_average_factor * stable_mean\
#                     + (1 - exponential_average_factor) * self.running_mean[BN_start:input_t+BN_start]
#                 # update running_var with unbiased var #* n / (n - 1)
#                 self.running_var[BN_start:input_t+BN_start] = exponential_average_factor * stable_var \
#                     + (1 - exponential_average_factor) * self.running_var[BN_start:input_t+BN_start]
#         else:
#             mean = self.running_mean[BN_start:input_t+BN_start]
#             var = self.running_var[BN_start:input_t+BN_start]

#         input1 = (input - mean[:, None, :]) / var[:, None, :]
#         if self.affine:
#             input1 = input1 * self.weight[None, None, :] + self.bias[None, None, :]#[None, :, None, None]

#         return input1