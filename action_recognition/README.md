## The skeleton-based Action Recognition example  
### Usage  
1, First, ready the data. Two ways.  
  (1) Use your own data reader. Change the code at [Indrnn_action_train.py](https://github.com/Sunnydreamrain/IndRNN_pytorch/blob/master/action_recognition/Indrnn_action_train.py#L80)   
  (2) Use the provided data reader. Generate the data ndarray. Download the NTU RGB+D dataset, change the skeleton into a ndarray, and keep the length and label of each data entry.  
2, Run the code. 
   The command to run the code has included in the run_x.sh fiels for different networks (plain IndRNN, residual IndRNN and dense IndRNN). Following are two examples.  
   For Plain IndRNN:
   `python -u Indrnn_action_train.py --model 'plainIndRNN' --bn_location 'bn_after' --u_lastlayer_ini --constrain_U --num_layers 6 --hidden_size 512 --dropout 0.5 --batch_size 128 --pThre 100`   
   For Dense IndRNN:  
   `python -u Indrnn_action_train.py --bn_location 'bn_after' --constrain_U --model 'denseIndRNN' --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 100` 
   
### Considerations
1, Usually sequence length of 20 is used for this dataset. It is short, so no need to impose the constraint on the recurrent weight (Similar results using it).  
