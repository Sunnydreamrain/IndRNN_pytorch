import argparse
import numpy as np


def train_opts(parser):
  parser.add_argument('--lr', type=float, default=2e-4,help='lr')
  parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
  parser.add_argument('--seq_len', type=int, default=20)
  parser.add_argument('--num_layers', type=int, default=6,help='num_layers')
  parser.add_argument('--hidden_units', type=int, default=512)
  parser.add_argument('--test_CV', action='store_true', default=False,help='use the CS test setting. If True, then use CV test setting.')
  parser.add_argument('--use_weightdecay_nohiddenW', action='store_true', default=False)
  parser.add_argument('--decayrate', type=np.float32, default=1e-4,help='lr')
  parser.add_argument('--opti', type=str, default='adam')
  parser.add_argument('--pThre', type=int, default=20)
  parser.add_argument('--test_no', type=int, default=20)

  parser.add_argument('--ini_in2hid', type=np.float32, default=0.002)

  parser.add_argument('--constrain_U', action='store_true', default=False)
  parser.add_argument('--MAG', type=np.float32, default=5.0)

  parser.add_argument('--eval_fold', type=int, default=5)
  parser.add_argument('--use_bneval', action='store_true', default=False)
  parser.add_argument('--ini_b', type=np.float32, default=0.0)
  parser.add_argument('--end_rate', type=np.float32, default=1e-6)
  
  parser.add_argument('--dropout', type=np.float32, default=0.1,help='lr')
  
  
  
  
  
