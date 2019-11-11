CUDA_VISIBLE_DEVICES=2 python -u Indrnn_action_train.py --test_no 40 --data_randtime_aug --constrain_U --model 'denseIndRNN' --bn_location 'bn_after' --geo_aug --time_diff --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 100 2>&1 | tee logindrnn_864_first6_grow48_drop53531_bs128_consU_inilast_pThre100_geoaug_timediff_datarandtime_testno40.log &

CUDA_VISIBLE_DEVICES=3 python -u Indrnn_action_train.py --test_no 40 --test_CV --data_randtime_aug --constrain_U --model 'denseIndRNN' --bn_location 'bn_after' --geo_aug --time_diff --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 100 2>&1 | tee CVlogindrnn_864_first6_grow48_drop53531_bs128_consU_inilast_pThre100_geoaug_timediff_datarandtime_testno40.log &



#CUDA_VISIBLE_DEVICES=3 python -u Indrnn_action_train.py --data_randtime_aug --constrain_U --model 'denseIndRNN' --bn_location 'bn_after' --geo_aug --time_diff --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 150 2>&1 | tee logindrnn_864_first6_grow48_drop53531_bs128_consU_inilast_pThre150_geoaug_timediff_datarandtime.log &

#CUDA_VISIBLE_DEVICES=3 python -u Indrnn_action_train.py --test_CV --data_randtime_aug --constrain_U --model 'denseIndRNN' --bn_location 'bn_after' --geo_aug --time_diff --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 150 2>&1 | tee CVlogindrnn_864_first6_grow48_drop53531_bs128_consU_inilast_pThre150_geoaug_timediff_datarandtime.log &


#CUDA_VISIBLE_DEVICES=0 python -u Indrnn_action_train.py --data_randtime_aug --constrain_U --model 'denseIndRNN' --geo_aug --time_diff --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 150 2>&1 | tee logindrnn_864_first6_grow48_drop53531_bs128_consU_inilast_pThre150_geoaug_timediff_datarandtime.log &
#CUDA_VISIBLE_DEVICES=1 python -u Indrnn_action_train.py --data_randtime_aug --geo_aug --time_diff --small_normini --constrain_U --model 'denseIndRNN' --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 150 2>&1 | tee logindrnn_864_first6_grow48_drop53531_bs128_consU_inilast_inismallnorm_pThre150_geoaug_timediff_datarandtime.log &





#CUDA_VISIBLE_DEVICES=0 python -u Indrnn_action_train.py --constrain_U --model 'denseIndRNN' --geo_aug --time_diff --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 150 2>&1 | tee logindrnn_864_first6_grow48_drop53531_bs128_consU_inilast_pThre150_geoaug_timediff.log &
#CUDA_VISIBLE_DEVICES=1 python -u Indrnn_action_train.py --geo_aug --time_diff --small_normini --constrain_U --model 'denseIndRNN' --num_first 6 --add_last_layer --u_lastlayer_ini --block_config '(8,6,4)' --growth_rate 48 --dropout 0.5 --dropout_sec 0.1 --dropout_trans 0.3 --dropout_first 0.5 --dropout_last 0.3 --batch_size 128 --pThre 150 2>&1 | tee logindrnn_864_first6_grow48_drop53531_bs128_consU_inilast_inismallnorm_pThre150_geoaug_timediff.log &













