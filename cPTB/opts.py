import argparse
import numpy as np


def train_opts(parser):
  #parser = argparse.ArgumentParser(description='IndRNN for the char level PennTreeBank Language Model')
  parser.add_argument('--hidden_size', type=int, default=2000)
  parser.add_argument('--batch_size', type=int, default=128,help='batch_size')
  parser.add_argument('--seq_len', type=int, default=50,help='seq_len')
  parser.add_argument('--num_layers', type=int, default=6,help='num_layers')
  parser.add_argument('--lr', type=float, default=2e-4, help='lr')
  parser.add_argument('--act', type=str, default='relu', help='act')
  #parser.add_argument('--data_aug', action='store_true', default=False)
  parser.add_argument('--gradclipvalue', type=np.float32, default=10,  help='gradclipvalue')
  parser.add_argument('--MAG', type=int, default=2)
  parser.add_argument('--opti', type=str, default='adam')
  
  #drop
  parser.add_argument('--dropout', type=np.float32, default=0.25, help='lr')
  
  #residual
  parser.add_argument('--use_residual', action='store_true', default=False)
  parser.add_argument('--residual_layers', type=int, default=2)
  parser.add_argument('--residual_block', type=int, default=3)
  parser.add_argument('--unit_factor', type=np.float32, default=1, help='lr')
  
  #weight decay
  parser.add_argument('--use_weightdecay_nohiddenW', action='store_true', default=False)
  parser.add_argument('--decayfactor', type=float, default=1e-4, help='decayfactor')
  
  #initialization
  parser.add_argument('--pThre', type=int, default=50)
  parser.add_argument('--ini_in2hid', type=np.float32, default=0.005, help='ini_in2hid')
  parser.add_argument('--ini_b', type=np.float32, default=0.0, help='ini_in2hid')
  
  
