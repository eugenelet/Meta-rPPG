import argparse
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
import numpy as np
import random

import pdb

class TrainOptions():
   def __init__(self):
      self.parser = argparse.ArgumentParser(
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      self.parser.add_argument('--name', type=str, default='meta_rPPG')
      self.parser.add_argument('--network', type=str, default='MAML')
      self.parser.add_argument('--continue_train', action="store_true")
      self.parser.add_argument('--load_file', type=str, default='smallest')
      self.parser.add_argument("--delay", type=int, default=48)
      self.parser.add_argument('--fewshots', type=int, default=1)
      self.parser.add_argument('--lr_ratio', type=float, default=0.1)



      self.parser.add_argument('--per_iter_task', type=int, default=3)
      self.parser.add_argument('--lstm_num_layers', type=int, default=2)
      self.parser.add_argument('--valid_ratio', type=float, default=0.75)

      self.parser.add_argument('--batch_size', type=int, default=3)
      self.parser.add_argument('--lr', type=float, default=1e-3)
      self.parser.add_argument('--train_epoch', type=int, default=1)
      self.parser.add_argument('--gpu_ids', type=str, default='0')
      self.parser.add_argument('--print_net', action="store_true")
      self.parser.add_argument('--epoch_count', type=int, default=1)
      # self.parser.add_argument('--lr_policy', type=str, default='cosine')
      # self.parser.add_argument('--lr_decay_iters', type=int, default=1)
      # self.parser.add_argument('--lr_update_iter', type=int, default=5000)

      self.parser.add_argument('--print_freq', type=int, default=10)
      self.parser.add_argument('--save_latest_freq', type=int, default=100)
      self.parser.add_argument('--save_epoch_freq', type=int, default=50)
      self.parser.add_argument('--save_by_iter', action="store_true")


      self.parser.add_argument('--display_id', type=int, default=1)
      self.parser.add_argument(
         '--display_server', type=str, default="http://localhost")
      self.parser.add_argument('--display_env', type=str, default='main')
      self.parser.add_argument('--display_port', type=int, default=8800)
      self.parser.add_argument('--display_winsize', type=int, default=256)
      self.parser.add_argument('--verbose', type=bool, default=True)
      self.parser.add_argument('--no_html', type=bool, default=True)
      self.parser.add_argument(
         '--checkpoints_dir', type=str, default='checkpoints')
      self.parser.add_argument('--save_dir', type=str, default='save')
      self.parser.add_argument('--max_dataset_size',type=int, default=float("inf"))

      self.parser.add_argument('--num_threads', type=int, default=4)
      self.parser.add_argument('--phase', type=str, default='train')

      self.parser.add_argument('--load_iter', type=int, default='0')
      self.parser.add_argument('--epoch', type=str, default='latest')
      self.parser.add_argument('--win_size', type=int, default=60)
      self.parser.add_argument('--adapt_position', type=str, default="extractor")

   def get_options(self):
      return self.parser.parse_args()
   
   def get_parser(self):
      return self.parser



class custom_scheduler():
   def __init__(self, optimizer, Tmax):
      self.optimizer = optimizer
      self.Tmax = Tmax
      self.Max = optimizer.param_groups[0]['lr']
      self.Min = self.Max*0.01
      self.Tcur = 1

   def step(self):
      pi = torch.Tensor([np.pi])
      for param_group in self.optimizer.param_groups:
         param_group['lr'] = float(self.Min + 0.5*(self.Max - self.Min)*(1 + torch.cos(pi*self.Tcur/self.Tmax)))

      if self.Tcur == 10 or self.Tcur == 30 or self.Tcur == 50 or self.Tcur == 70 or self.Tcur == 90:
         self.Max = 10*self.optimizer.param_groups[0]['lr']
      elif self.Tcur == 20 or self.Tcur == 40 or self.Tcur == 60 or self.Tcur == 80 or self.Tcur == 100:
         self.Min = 0.01*self.optimizer.param_groups[0]['lr']
      self.Tcur += 1






