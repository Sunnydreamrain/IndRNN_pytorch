CUDA_VISIBLE_DEVICES=4 python -u Indrnn_mnist_train.py --use_permute --u_lastlayer_ini --constrain_U --model 'residualIndRNN' --bn_location 'bn_after' --num_blocks 5 --hidden_size 128 --dropout 0.1 --batch_size 32 --pThre 100 2>&1 | tee permute_reslogindrnn_5blocks_128_drop1_bs32_consU_pThre100_preact_inilasth.log &


CUDA_VISIBLE_DEVICES=5 python -u Indrnn_mnist_train.py --u_lastlayer_ini --constrain_U --model 'residualIndRNN' --bn_location 'bn_after' --num_blocks 5 --hidden_size 128 --dropout 0.1 --batch_size 32 --pThre 100 2>&1 | tee reslogindrnn_5blocks_128_drop1_bs32_consU_pThre100_preact_inilasth.log &
