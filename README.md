# Independently Recurrent Neural Networks
This code is to implement the [IndRNN](https://arxiv.org/abs/1803.04831) and the [Deep IndRNN](https://arxiv.org/abs/1910.06251). It is based on Pytorch.

`cuda_IndRNN_onlyrecurrent` is the CUDA version. It is much faster than the simple pytorch implementation. For the sequential MNIST example (length 784), it runs over `31` times faster.     

Please cite the following paper if you find it useful.  
[Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN." CVPR 2018.](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Independently_Recurrent_Neural_CVPR_2018_paper.html)  
[Shuai Li, Wanqing Li, Chris Cook, Yanbo Gao, and Ce Zhu. "Deep Independently Recurrent Neural Network (IndRNN)." arXiv preprint arXiv:1910.06251, 2019.](https://arxiv.org/abs/1910.06251)
@inproceedings{li2018independently,
  title={Independently recurrent neural network (indrnn): Building a longer and deeper rnn},
  author={Li, Shuai and Li, Wanqing and Cook, Chris and Zhu, Ce and Gao, Yanbo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5457--5466},
  year={2018}
}
@article{li2019deep,
  title={Deep Independently Recurrent Neural Network (IndRNN)},
  author={Li, Shuai and Li, Wanqing and Cook, Chris and Gao, Yanbo and Zhu, Ce},
  journal={arXiv preprint arXiv:1910.06251},
  year={2019}
}

# Summary of advantages
- Able to process longer sequences (over 5000 steps): gradient vanishing and exploding problem is solved.  
- Able to construct deeper networks (over 20layer, much deeper if GPU memory supports): techniques from CNN such as Batch normalization and non-saturated functions such as ReLU can be efficiently used.  
- Able to be robustly trained with ReLU  
- Able to interpret the behaviour of IndRNN neurons independently without the effect from the others  
- Reduced complexity (10x faster than cuDNN LSTM)

# Usage  
`IndRNN_onlyrecurrent.py` provides only the `recurrent+activation of the IndRNN function`. Therefore, processing of the input with dense connection or convolution operation is needed. This is usedful for adding batch normalization (BN) between the processing of input and activation function. Just consider it as an Relu function with recurrent connections. I believe this is more flexible since you can add all different processings to the inputs.   
`cuda_IndRNN_onlyrecurrent` is the CUDA version. It is much faster than the simple pytorch implementation. For the sequential MNIST example (length 784), it runs over 31 times faster.   
It is much more flexible by treating it as an activitation function such as ReLU where BN or other techniqeus can be inserted between the fully connected layers and the Recurrent part.  

### Requirements  
- Pytorch  

For the CUDA version
- CuPy  
- pynvrtc  

### Running  
Please refer to the tasks.  

# Considerations in implementation  
### 1, Initialization of the recurrent weights
For relu, `Uniform(0,1)` is used to make different neurons keep different kinds of memory. But for problems that only use the output of the last time step such as the adding problem, MNIST classification problem, and action recognition problem, the recurrent weights for the last IndRNN layer (caution: only the last one not all) can be initialized to be all `1` or a proper range `(1-epsilon, 1+epsilon)` where `epsilon` is a small number, since only long-term memory is needed for the output of this layer. Examples are shown in [Indrnn_action_network.py](https://github.com/Sunnydreamrain/IndRNN_pytorch/blob/master/action_recognition/Indrnn_action_network.py#L72).  

### 2, Constraint of the recurrent weights  
For relu, generally it can be set to `[-U_bound, U_bound]` where `U_bound=pow(args.MAG, 1.0 / seq_len)` and `MAG` can be 2 or 10 or others. If the sequence is very long, it can be `[-1, 1]` since it is very close to 1 and the precision of GPU is limited. If the sequence is short such as 20, no constraint is needed. Example of the constraint is shown at [Indrnn_action_train.py](https://github.com/Sunnydreamrain/IndRNN_pytorch/blob/master/action_recognition/Indrnn_action_train.py#L107). By the way, this constraint can also be implemented as a weight decay of ||max(0,|U|-U_bound)||.  
For simplicity, the constraint can always set to `[-1, 1]` as it can keep long-term memory already and the difference in performance is small.

### 3, Usage of batch normalization (BN)  
Generally, over 3 layers, BN can help accelerate the training. BN can be used before the activation function or after it. In our experiments, we find it converges faster by putting BN after the activation function. However, for tasks such as PTB_c where the output of one batch is further used as the initialization of the next batch, it is better to put BN before activation as mentioned at the above example.  

### 4, Learning rate  
In our experiments, ADAM with a learning rate of 2e-4 works well.  

### 5, Weight decay  
If weight decay is used, no need to add decay on the recurrent weights.  

### 6, Usage of dropout  
Dropout (if used) is applied with the same mask over time.  

### Note  
The above considerations are just suggestions. I did not explore lots of training techniques such as training methods, initialization techniques. So better results may be achieved with other options.  

# Other implementations
Theano and Lasagne:  
[https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne](https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne)  
Tensorflow:  
[https://github.com/batzner/indrnn](https://github.com/batzner/indrnn)  
Keras:  
[https://github.com/titu1994/Keras-IndRNN](https://github.com/titu1994/Keras-IndRNN)  
Pytorch:  
[https://github.com/StefOe/indrnn-pytorch](https://github.com/StefOe/indrnn-pytorch)  
[https://github.com/theSage21/IndRNN](https://github.com/theSage21/IndRNN)  
[https://github.com/zhangxu0307/Ind-RNN](https://github.com/zhangxu0307/Ind-RNN)  
Chainer:  
[https://github.com/0shimax/chainer-IndRNN](https://github.com/0shimax/chainer-IndRNN)  
