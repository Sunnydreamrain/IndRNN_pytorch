CUDA_VISIBLE_DEVICES=0 python -u Indrnn_action_train.py --test_CV --u_lastlayer_ini --constrain_U --model 'residualIndRNN' --bn_location 'bn_after' --num_blocks 5 --hidden_size 512 --dropout 0.5 --batch_size 128 --pThre 100 2>&1 | tee res_CVlogindrnn_5blocks_512_drop5_bs128_consU_pThre100_prenorm.log &

CUDA_VISIBLE_DEVICES=1 python -u Indrnn_action_train.py --u_lastlayer_ini --constrain_U --model 'residualIndRNN' --bn_location 'bn_after' --num_blocks 5 --hidden_size 512 --dropout 0.5 --batch_size 128 --pThre 100 2>&1 | tee res_logindrnn_5blocks_512_drop5_bs128_consU_pThre100_prenorm.log &












