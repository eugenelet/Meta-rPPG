import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import time
import os
import random
import matplotlib.pyplot as plt
from data import SlideWindowDataLoader, testing
from model import meta_rPPG
from settings import TrainOptions

import pdb


opt = TrainOptions().get_options()
iter_num = opt.batch_size

model = meta_rPPG(opt, isTrain=True, continue_train=opt.continue_train)
model.setup(opt)

dataset = SlideWindowDataLoader(opt, isTrain=True)
testset = SlideWindowDataLoader(opt, isTrain=False)

per_idx = opt.per_iter_task
dataset_size = dataset.num_tasks * (dataset.task_len[0] - (opt.win_size))
task_len = (dataset.task_len[0] - per_idx*opt.win_size)


total_iters = 0

print("Data Size: %d ||||| Batch Size: %d ||||| initial lr: %f" %
      (dataset_size, opt.batch_size, opt.lr))
# pdb.set_trace()

task_list = random.sample(range(5), opt.batch_size)
model.dataset = dataset
data = dataset[task_list, 0]
# pdb.set_trace()
model.set_input(data)
model.update_prototype()
min_mae = [10, 10]
min_rmse = [10, 10]
min_merate = [10, 10]
saving = 1


for epoch in range(opt.epoch_count, opt.train_epoch + 1):
   epoch_start_time = time.time()
   epoch_iter = 0
   i = 0
   
   

   for data_idx in range(0, task_len, 1):
      task_list = random.sample(range(5), opt.batch_size)
      model.B_net.feed_hc([model.h, model.c])

      model.progress = epoch + float(data_idx)/float(task_len)


      for i in range(per_idx):
         # pdb.set_trace()
         data = dataset[task_list, data_idx + i*opt.win_size]
         iter_start_time = time.time()
         total_iters += opt.win_size
         model.set_input(data)
         if i == 0:
            model.new_theta_update(epoch) # Adaptation phase
         else:
            model.new_psi_phi_update(epoch) # Learning phase
      # pdb.set_trace()
      loss, test_loss = testing(opt, model, testset, data_idx, epoch)
      
      epoch_iter += 1
      data = dataset[task_list, np.random.randint(task_len)]
      model.set_input(data)
      model.update_prototype()


   model.save_networks('latest')
   model.save_networks(epoch)

   # pdb.set_trace()
   new_lr = model.update_learning_rate(epoch)
   print('Epoch %d/%d ||||| Time: %d sec ||||| Lr: %.7f ||||| Loss: %.3f/%.3f' %
         (epoch, opt.train_epoch, time.time() - epoch_start_time, new_lr,
          loss, test_loss))
