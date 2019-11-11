CUDA_VISIBLE_DEVICES=0 python -u Indrnn_action_train.py --test_CV --u_lastlayer_ini --constrain_U --model 'plainIndRNN' --bn_location 'bn_after' --num_layers 4 --hidden_size 512 --dropout 0.5 --batch_size 128 --pThre 100 2>&1 | tee plain_CVlogindrnn_4layers_512_drop5_bs128_consU_pThre100_inilasth.log &


CUDA_VISIBLE_DEVICES=1 python -u Indrnn_action_train.py --u_lastlayer_ini --constrain_U --model 'plainIndRNN' --bn_location 'bn_after' --num_layers 4 --hidden_size 512 --dropout 0.5 --batch_size 128 --pThre 100 2>&1 | tee plain_logindrnn_4layers_512_drop5_bs128_consU_pThre100_inilasth.log &



CUDA_VISIBLE_DEVICES=0 python -u Indrnn_action_train.py --test_CV --u_lastlayer_ini --constrain_U --model 'plainIndRNN' --bn_location 'bn_after' --num_layers 6 --hidden_size 512 --dropout 0.5 --batch_size 128 --pThre 100 2>&1 | tee plain_CVlogindrnn_6layers_512_drop5_bs128_consU_pThre100_inilasth.log &


CUDA_VISIBLE_DEVICES=1 python -u Indrnn_action_train.py --u_lastlayer_ini --constrain_U --model 'plainIndRNN' --bn_location 'bn_after' --num_layers 6 --hidden_size 512 --dropout 0.5 --batch_size 128 --pThre 100 2>&1 | tee plain_logindrnn_6layers_512_drop5_bs128_consU_pThre100_inilasth.log &











