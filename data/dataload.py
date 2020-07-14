import torch
from data.pre_dataload import BaselineDataset
# from Visualize.visualizer import Visualizer
import random

from scipy import signal
import numpy as np
import pdb
# pdb.set_trace()

class SlideWindowDataLoader():
   """Wrapper class of Dataset class that performs multi-threaded data loading.
      The class is only a container of the dataset.

      There are two ways to get a data out of the Loader. 

         1) feed in a list of videos: input = dataset[[0,3,5,10], 2020]. This gets the data starting at 2020 frame from 0, 3, 5, 10th video.
         2) feed a single value of videos:  input = dataset[0, 2020]. This gets a batch of data starting at 2020 from the 0th video.
   """

   def __init__(self, opt, isTrain):
      """Initialize this class
      """
      # self.visualizer = Visualizer(opt, isTrain=True)
      # self.visualizer.reset()
      self.opt = opt
      self.isTrain = isTrain

      self.dataset = BaselineDataset(opt, isTrain)
      if self.isTrain:
         print("dataset [%s-%s] was created" % ('rPPGDataset', 'train'))
      else:
         print("dataset [%s-%s] was created" % ('rPPGDataset', 'test'))
      self.length = int(len(self.dataset))

      self.num_tasks = self.dataset.num_tasks
      self.task_len = self.dataset.task_len

   def load_data(self):
      return self

   def __len__(self):
      """Return the number of data in the dataset"""
      return self.length

   def __getitem__(self, items):
      """Return a batch of data
         items -- [task_num, index of data for specified task]
      """

      inputs = []
      ppg = []
      frame = []
      mask = []

      if self.isTrain:
         batch = self.opt.batch_size
      else:
         batch = self.opt.batch_size + self.opt.fewshots

      if not isinstance(items[0], list):
         for i in range(batch):
            dat = self.dataset[items[0], items[1]+60*i]
            inputs.append(dat['input'])
            ppg.append(dat['PPG'])
      else:
         for idx in items[0]:
            dat = self.dataset[idx, items[1]]
            inputs.append(dat['input'])
            ppg.append(dat['PPG'])

         # pdb.set_trace()

      inputs = torch.stack(inputs)
      ppg = torch.stack(ppg)
      return {'input': inputs, 'rPPG': ppg}


   def quantify(self,rppg):
      quantified = torch.empty(rppg.shape[0], dtype=torch.long)
      binary = torch.ones(rppg.shape[0], dtype=torch.long)
      tmax = rppg.max()
      tmin = rppg.min()
      interval = (tmax - tmin)/39
      for i in range(len(quantified)):
         quantified[i] = ((rppg[i] - tmin)/interval).round().long()
      return quantified

   def __call__(self):
      output_list = []
      for idx in range(self.num_tasks):
         tmp = self.dataset(idx)
         tmp['rPPG'] = tmp.pop('PPG')
         output_list.append(tmp)
      return output_list
      # pdb.set_trace()

