from __future__ import print_function
import torch
import os
# import pickle
import numpy as np
import sys

from sklearn.preprocessing import normalize
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from data.data_utils import butter_bandpass_filter
import pdb



class BaselineDataset():
   """Preprocessing class of Dataset class that performs multi-threaded data loading

   """
   def __init__(self, opt, isTrain):
      """Initialize this dataset class.

      Parameters:
         opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

         The self.dataset is a list of facial data, the length of the list is 18, and each element is a torch tensor of shape [2852, 3, 64, 64]
         The self.maskset is the corresponding mask data, constructed of 0 and 255, so it determines the landmarks we're using in self.dataset  

      """
      # get the image directory

      self.isTrain = isTrain
      self.opt = opt

      temp_data = torch.load('data/example.pth')
      if self.isTrain:
         self.maskset = temp_data['mask'][:5]
         self.dataset = temp_data['image'][:5]
         self.ppg_dataset = temp_data['ppg'][:5]
         self.num_tasks = len(self.dataset)
         self.task_len = [self.dataset[i].shape[0]
                          for i in range(len(self.dataset))]
      else:
         self.maskset = temp_data['mask'][-1:]
         self.dataset = temp_data['image'][-1:]
         self.ppg_dataset = temp_data['ppg'][-1:]
         self.num_tasks = 1
         self.task_len = self.dataset[0].shape[0]
         # pdb.set_trace()

      self.length = 0
      for i in range(len(self.ppg_dataset)):
         self.length += self.ppg_dataset[i].shape[0] - self.opt.win_size


   def __getitem__(self, items):
      """Return a data point and its metadata information.

      Parameters:
         items -- [task_number, index of data for specified task]
         items[0] -- a integer in range 0 to 4 in train mode, only 0 available in test mode
         items[1] -- determined by the length of the video

      Returns a dictionary that contains input, PPG, diff and orig
         input - - a set of frames from the pickle file (60 x 3 x 64 x 64)
         PPG - - the corresponding signal (60)
      """

      inputs = []
      masks = []
      if not self.isTrain:
         # pdb.set_trace()
         for i in range(items[1], items[1] + self.opt.win_size):
            frame = self.dataset[items[0]][i].clone()
            mask = self.maskset[items[0]][i].clone()
            inputs.append(frame)
            masks.append(mask)
         ppg = self.ppg_dataset[items[0]][items[1]: items[1] + self.opt.win_size].clone()
      else:
         for i in range(items[1], items[1] + self.opt.win_size):
            frame = self.dataset[items[0]][i].clone()
            mask = self.maskset[items[0]][i].clone()
            inputs.append(frame)
            masks.append(mask)
         ppg = self.ppg_dataset[items[0]][items[1]
            : items[1] + self.opt.win_size].clone()


     

      inputs = np.stack(inputs)
      inputs = torch.from_numpy(inputs)
      masks = np.stack(masks)
      masks = torch.from_numpy(masks)

      self.baseline_procress(inputs, masks.clone())
      ppg = self.quantify(ppg)

      return {'input': inputs, 'PPG': ppg}

   def __len__(self):
      """Return the total number of images in the dataset."""

      return self.length

   def quantify(self, rppg):
      quantified = torch.empty(rppg.shape[0], dtype=torch.long)

      tmax = rppg.max()
      tmin = rppg.min()
      interval = (tmax - tmin)/39
      for i in range(len(quantified)):
         quantified[i] = ((rppg[i] - tmin)/interval).round().long()

      return quantified
   
   def baseline_procress(self, data, mask):

      mask /= 255
      mask = mask.float()

      # pdb.set_trace()
      input_mean = data.sum(dim=(0, 2, 3), keepdim=False) / \
          mask.sum(dim=(0, 2, 3), keepdim=False)  # mean of W H T
      for i in range(data.shape[1]):
         data[:, i, :, :] = data[:, i, :, :] - input_mean[i]  # minus the total mean
      data = data*mask
      
      x_hat = data.sum(dim=(2, 3), keepdim=False)/ \
               mask.sum(dim=(2, 3), keepdim=False)  # mean of H T
      G_x = np.empty(x_hat.size())  # filtered x_hat

      for i in range(data.shape[1]):  # shape 1 is RGB channels
         # pdb.set_trace()
         G_x[:, i] = butter_bandpass_filter(x_hat[:, i], 1, 8, 30, order=3)
         for j in range(data.shape[0]):
            data[j, i, :, :] = data[j, i, :, :] - \
                  (x_hat[j, i] - G_x[j, i])
      data = data*mask
      # pdb.set_trace()
      return data

   def __call__(self, idx):
      inputs = []
      masks = []
      items = [idx, 0]

      if not self.isTrain:
         # pdb.set_trace()
         decision = 0
         new_index = items[1] % (
             self.task_len - (self.opt.batch_size + self.opt.fewshots)*self.opt.win_size)
         for i in range(new_index, new_index + 15*self.opt.win_size):
            frame = self.dataset[items[0]][i].clone()
            mask = self.maskset[items[0]][i].clone()
            inputs.append(frame)
            masks.append(mask)
         ppg = self.ppg_dataset[items[0]
                                ][new_index: new_index + 15*self.opt.win_size].clone()
         orig = self.original[items[0]
                              ][new_index: new_index + 15*self.opt.win_size].clone()
      else:
         for i in range(items[1], items[1] + 15*self.opt.win_size):
            frame = self.dataset[items[0]][i].clone()
            mask = self.maskset[items[0]][i].clone()
            inputs.append(frame)
            masks.append(mask)
         ppg = self.ppg_dataset[items[0]][items[1]: items[1] + 15*self.opt.win_size].clone()
         orig = self.original[items[0]][items[1]: items[1] + 15*self.opt.win_size].clone()

      inputs = np.stack(inputs)
      inputs = torch.from_numpy(inputs)
      masks = np.stack(masks)
      masks = torch.from_numpy(masks)

      self.baseline_procress(inputs, masks.clone())
      ppg = self.quantify(ppg)

      return {'input': inputs, 'PPG': ppg}
