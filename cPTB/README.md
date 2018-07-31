## The language modeling example using character-level Penn Treebank (PTB-c)  
### Usage
1, First, download the data and add it to the `data` folder.  
>> The PTB dataset used comes from Tomas Mikolov's webpage:  
>> http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz  

2, Run the code using the generally stacked IndRNN or the residual IndRNN.  

>> Stacked IndRNN: `python -u train_cPTB.py --data_aug --hidden_units 2000 --num_layers 6 --dropout 0.25 --seq_len 150 --use_weightdecay_nohiddenW`  
 
>> Residual IndRNN: `python -u train_cPTB.py --data_aug --hidden_units 2000 --use_residual --num_layers 11 --dropout 0.3 --seq_len 150 --use_weightdecay_nohiddenW`  
>> The example code provides the very basic implementation of residual IndRNN where the number of units in all the IndRNN layers are the same and the left branch is fixed to be 1 without further using weight processing. Other network architectures can be explored which may provide better results.

>> For this task, output is provided at each time step and can only use the information before the current time step. Therefore, the statistics (mean and variance) of the batch normalization (BN) are obtained for each time step. It is used before the activation which is more robust than putting it after the activation. The main reason is that the outputs of all the IndRNN layers at the last time step is further used as initialization of the next batch. By putting BN before the activation (which is also before the recurrent accumulation), the statistics of BN is more stable than putting BN after the activation.  

>> `data_aug` here only provides different start for each training epoch to provide stable statistics for BN.  

3, Longer sequence performs better in our experiments, showing that longer dependency helps.   

![alt text](https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne/blob/master/cPTB/results/cPTB.PNG)

3, Please refer to https://github.com/Sunnydreamrain/IndRNN_Theano_Lasagne for some considerations in the implementation.   
