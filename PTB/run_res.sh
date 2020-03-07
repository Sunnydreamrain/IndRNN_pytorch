



CUDA_VISIBLE_DEVICES=0 python -u train_language.py --bn_location 'bn_before' --model 'residualIndRNN' --constrain_U --U_bound 0.99 --num_blocks 5 --hidden_size 2000 --dropout 0.45 --dropout_last 0.5 --rand_drop_ini 10 --batch_size 128 --data=data/ptb. --lr 2e-4 --w_tying --dropout_words 0.65 --dropout_extrafc 0.65 --dropout_embedding 0.2 --embed_size 600 --seq_len 50 --pThre 200 2>&1 | tee res_logwordptb_wtying_lr2e4_5blocks_2000_drop45_last5_65652_RecHuniform_pThre200_NOsmallnormini_bnbefore.log &













