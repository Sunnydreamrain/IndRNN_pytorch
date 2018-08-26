## The Sequential MNIST example  
### Usage
Run the code with the CUDA version.  

>> CUDA_VISIBLE_DEVICES=0 python -u Indrnn_mnist_train.py --use_permute --lr 2e-4 --num_layers 6 --use_weightdecay_nohiddenW --decayfactor 1e-4 --hidden_size 128 --batch_size 64 --dropout 0.1 --pThre 100 --constrain_U  
 
