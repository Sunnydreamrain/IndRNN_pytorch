## The skeleton-based Action Recognition example  
### Usage  
1, First, ready the data. Two ways.  
  (1) Use your own data reader. Change the code at [Indrnn_action_train.py](https://github.com/Sunnydreamrain/IndRNN_pytorch/blob/master/action_recognition/Indrnn_action_train.py#L80)   
  (2) Use the provided data reader. Generate the data ndarray. Download the NTU RGB+D dataset, change the skeleton into a ndarray, and keep the length and label of each data entry.  
2, Run the code. 
   `python -u Indrnn_action_train.py --dropout 0.25 --use_weightdecay_nohiddenW`   
   If use the CV test setting, add `--test_CV`. For example:  
   `python -u Indrnn_action_train.py --test_CV --dropout 0.1 --use_weightdecay_nohiddenW` 
   
### Considerations
1, Usually sequence length of 20 is used for this dataset. It is short, so no need to impose the constraint on the recurrent weight (Similar results using it).  
