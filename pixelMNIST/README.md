## The Sequential MNIST example  
### Usage
Run the code with the CUDA version. It is much faster than the pytorch implementation of IndRNN.  
The command to run the code has included in the run_x.sh fiels for different networks (plain IndRNN, residual IndRNN and dense IndRNN). Following are two examples. 
   For Plain IndRNN:
   `python -u Indrnn_mnist_train.py --u_lastlayer_ini --constrain_U --model 'plainIndRNN' --bn_location 'bn_after' --num_layers 6 --hidden_size 128 --dropout 0.1 --batch_size 32 --pThre 100`   
   For Dense IndRNN:  
   `python -u Indrnn_mnist_train.py --constrain_U --model 'denseIndRNN' --bn_location 'bn_after' --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 16 --dropout 0.2 --dropout_sec 0.1 --dropout_trans 0.1 --dropout_first 0.2 --dropout_last 0.1 --batch_size 32 --pThre 100` 


 
