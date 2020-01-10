## The language modeling example using character-level Penn Treebank (PTB-c) and word-level Penn Treebank (PTB-word)  
### Usage
1, First, download the data and add it to the `data` folder. Or use the command `./getdata.sh`    
>> The PTB dataset used comes from Tomas Mikolov's webpage:  
>> http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz  

2, Run the code using the generally stacked IndRNN, the residual IndRNN, or the dense IndRNN. Currently, the dense IndRNN shows the best performance.  

>> Dense IndRNN for PTB-word: `python -u train_language.py --bn_location 'bn_before' --model 'denseIndRNN' --U_bound 0.99 --block_config '(8,6,4)' --growth_rate 256 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.4 --batch_size 128 --data=data/ptb. --lr 2e-4 --w_tying --dropout_words 0.65 --dropout_extrafc 0.65 --dropout_embedding 0.2 --embed_size 600 --seq_len 50 --pThre 200 --small_normini --rand_drop_ini 10`  
>> Dense IndRNN for PTB-c: `python -u train_language.py --bn_location 'bn_before' --model 'denseIndRNN' --U_bound 0.99 --block_config '(8,6,4)' --growth_rate 256 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.4 --batch_size 128 --data=data/ptb.char. --lr 2e-4 --w_tying --dropout_words 0.0 --dropout_extrafc 0.2 --dropout_embedding 0.0 --embed_size 600 --seq_len 50 --pThre 200 --small_normini --rand_drop_ini 10`  

>> The example code provides the very basic implementation of dense IndRNN. Other network architectures can be explored which may provide better results.

>> For this task, output is provided at each time step and can only use the information before the current time step. Therefore, the statistics (mean and variance) of the batch normalization (BN) are obtained for each time step. It is used before the activation which is more robust than putting it after the activation. The main reason is that the outputs of all the IndRNN layers at the last time step is further used as initialization of the next batch. By putting BN before the activation (which is also before the recurrent accumulation), the statistics of BN is more stable than putting BN after the activation. `U_bound` is set to 0.99 since the sequence (continued) is too long and a small precision error may result in some differences.   

>> Post-processing such as the dynamic evaluation can also be used. 
>> `python -u dynamiceval.py --grid --lr 2e-6 --lamb 0.012  --bn_location 'bn_before' --model 'denseIndRNN' --block_config '(8,6,4)' --growth_rate 256 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.4  --data=data/ptb. --w_tying --dropout_words 0.65 --dropout_extrafc 0.65 --dropout_embedding 0.2 --embed_size 600 --seq_len 50`  

3, Please refer to https://github.com/Sunnydreamrain/IndRNN_pytorch for some considerations in the implementation.   
