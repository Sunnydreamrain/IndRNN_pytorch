import argparse
import numpy as np


def train_opts(parser):
  parser.add_argument('--model', type=str, default='plain')
  parser.add_argument('--lr', type=float, default=2e-4,help='lr')
  parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
  parser.add_argument('--seq_len', type=int, default=784)
  parser.add_argument('--num_layers', type=int, default=6,help='num_layers')
  parser.add_argument('--hidden_size', type=int, default=512)
  #parser.add_argument('--use_weightdecay_nohiddenW', action='store_true', default=False)
  parser.add_argument('--decayfactor', type=float, default=1e-4,help='lr')
  parser.add_argument('--opti', type=str, default='adam')
  parser.add_argument('--pThre', type=int, default=50)
  parser.add_argument('--gradclipvalue', type=float, default=10, help='gradclipvalue')
  parser.add_argument('--use_permute', action='store_true', default=False)

  #parser.add_argument('--ini_in2hid', type=float, default=0.002)

  parser.add_argument('--constrain_U', action='store_true', default=False)
  parser.add_argument('--MAG', type=float, default=5.0)
  parser.add_argument('--U_bound', type=float, default=0.0)

  parser.add_argument('--use_bneval', action='store_true', default=False)
  parser.add_argument('--ini_b', type=float, default=0.0)
  parser.add_argument('--end_rate', type=float, default=1e-6)
  
  parser.add_argument('--dropout', type=float, default=0.1)
  parser.add_argument('--small_normini', action='store_true', default=False) 
  
  parser.add_argument('--bn_location', type=str, default='bn_after')
  ###residual IndRNN opts
  parser.add_argument('--num_blocks', type=int, default=6)

  ###dense IndRNN opts
  parser.add_argument('--growth_rate', type=int, default=16)
  parser.add_argument('--num_first', type=int, default=4)
  parser.add_argument('--block_config', type=str, default='(8,6,4)')
  parser.add_argument('--add_last_layer', action='store_true', default=False) 
  parser.add_argument('--u_lastlayer_ini', action='store_true', default=False)
  parser.add_argument('--dropout_sec', type=float, default=0.0)
  parser.add_argument('--dropout_trans', type=float, default=0.0)
  parser.add_argument('--dropout_last', type=float, default=0.0)
  parser.add_argument('--dropout_first', type=float, default=0.0)