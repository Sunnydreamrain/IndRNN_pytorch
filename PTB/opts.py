import argparse
import numpy as np

def train_opts(parser):
    # parser = argparse.ArgumentParser(description='IndRNN for the char level PennTreeBank Language Model')
    parser.add_argument('--data', type=str, default='data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='plain')
    parser.add_argument('--lr', type=float, default=2e-4, help='lr')
    parser.add_argument('--hidden_size', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--seq_len', type=int, default=50, help='seq_len')
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=6, help='num_layers')
    parser.add_argument('--gradclipvalue', type=float, default=10, help='gradclipvalue')
    parser.add_argument('--constrain_U', action='store_true', default=False)
    parser.add_argument('--MAG', type=int, default=2)
    parser.add_argument('--U_bound', type=float, default=0.0)
    parser.add_argument('--opti', type=str, default='adam')
    parser.add_argument('--w_tying', action='store_true', default=False)
    parser.add_argument('--embed_size', type=int, default=600)
    parser.add_argument('--pThre', type=int, default=100)


    # weight decay
    #parser.add_argument('--use_weightdecay_nohiddenW', action='store_true', default=False)
    parser.add_argument('--decayfactor', type=float, default=1e-4, help='decayfactor')

    # initialization
    parser.add_argument('--small_normini', action='store_true', default=False) 	
    parser.add_argument('--add_lastbn', action='store_true', default=False)	
    parser.add_argument('--add_lastindrnn', action='store_true', default=False)	
	
    parser.add_argument('--rand_drop_ini', type=int, default=0)	

    # drop
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--dropout_embedding', type=float, default=0.0)
    parser.add_argument('--dropout_words', type=float, default=0.0)
    parser.add_argument('--dropout_extrafc', type=float, default=0.0)
    
    parser.add_argument('--dropout_sec', type=float, default=0.0)
    parser.add_argument('--dropout_trans', type=float, default=0.0)
    parser.add_argument('--dropout_last', type=float, default=0.0)
    parser.add_argument('--dropout_first', type=float, default=0.0)


    parser.add_argument('--bn_location', type=str, default='bn_before')

    ###residual IndRNN opts
    parser.add_argument('--num_blocks', type=int, default=6)
    
    ###dense IndRNN opts
    parser.add_argument('--u_lastlayer_ini', action='store_true', default=False)
    parser.add_argument('--growth_rate', type=int, default=12)
    parser.add_argument('--num_first', type=int, default=4)
    parser.add_argument('--block_config', type=str, default='(8,6,4)')
    parser.add_argument('--bn_decay', action='store_true', default=False) 
    parser.add_argument('--mos', action='store_true', default=False) 