


CUDA_VISIBLE_DEVICES=0 python -u Indrnn_action_train.py --bn_location 'bn_after' --constrain_U --model 'denseIndRNN' --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 100 2>&1 | tee dense_logindrnn_864_first6_grow48_drop53531_bs128_consU_inilast_pThre100.log &
CUDA_VISIBLE_DEVICES=1 python -u Indrnn_action_train.py --bn_location 'bn_after' --test_CV --constrain_U --model 'denseIndRNN' --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 100 2>&1 | tee dense_CVlogindrnn_864_first6_grow48_drop53531_bs128_consU_inilast_pThre100.log &















