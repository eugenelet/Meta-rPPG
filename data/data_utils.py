from __future__ import print_function
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import itertools
import torch
from scipy import signal
from scipy.signal import butter, lfilter

class FunctionSet():
   def __init__(self, sample_rate=30.0, display_port=8093):
      self.fps = sample_rate

   def CHROM_method(self, data):
      '''CHROM matrix'''
      project_matrix = np.array([[3, -2, 0], [1.5, 1, -1.5]]) 
      frames = data['frame'].copy()
      mask = data['mask'].copy()
      mask /= 255
      mask = mask.astype(float)

      rgb_mean = self.spatial_mean(frames, mask)
      rgb_mean = rgb_mean.transpose()
      rgb_mean = rgb_mean[[2, 1, 0], :]
      
      win_size = rgb_mean.shape[1]
      C_norm = np.zeros([3, win_size])
      for i in range(win_size):
         C_norm[:, i] = rgb_mean[:, i] / np.mean(rgb_mean, axis=1)
      S = np.matmul(project_matrix, C_norm)
      S1 = S[0,:]
      S2 = S[1,:]
      alpha = np.std(S1)/np.std(S2)
      h = S1 + alpha*S2  # POS
      h = butter_bandpass_filter(h, 0.4, 5, self.fps, order=6)

      return h - np.mean(h)



   def POS_method(self, data):
      '''POS matrix'''
      project_matrix = np.array([[0, 1, -1], [-2, 1, 1]]) 
      frames = data['frame'].copy()
      mask = data['mask'].copy()
      mask /= 255
      mask = mask.astype(float)

      rgb_mean = self.spatial_mean(frames, mask)
      rgb_mean = rgb_mean.transpose()
      rgb_mean = rgb_mean[[2, 1, 0], :]
      
      win_size = rgb_mean.shape[1]
      C_norm = np.zeros([3, win_size])
      for i in range(win_size):
         C_norm[:, i] = rgb_mean[:, i] / np.mean(rgb_mean, axis=1)
      S = np.matmul(project_matrix, C_norm)
      S1 = S[0,:]
      S2 = S[1,:]
      alpha = np.std(S1)/np.std(S2)
      h = S1 + alpha*S2  # POS
      h = butter_bandpass_filter(h, 0.4, 5, self.fps, order=6)
      return h - np.mean(h)
   
   def spatial_mean(self, frames, mask):
      t0 = np.sum(frames, axis=(0, 2, 3))
      t1 = np.sum(mask, axis=(0,2,3))

      mean = t0/t1
      return mean
      # pdb.set_trace()


def butter_bandpass(lowcut, highcut, fs, order=5):
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
   b, a = butter_bandpass(lowcut, highcut, fs, order=order)
   # y = lfilter(b, a, data)
   y = signal.filtfilt(b, a, data, method="pad")
   return y


def normed(a):
   amin, amax = np.min(a), np.max(a)
   t = a.copy()
   # pdb.set_trace()
   for i in range(a.shape[0]):
      t[i] = (a[i]-amin) / (amax-amin)
   return t


def testing(opt, model, testset, data_idx, epoch):
   results, true_rPPG = model.get_current_results(0)
   loss = model.get_current_losses(0)
   test_data = testset[0, 0]

   # model.eval() rnn can't be adapted in eval mode
   model.set_input(test_data)
   model.fewshot_test(epoch)

   t_results, t_true_rPPG = model.get_current_results(1)
   test_loss = model.get_current_losses(1)

   model.train()

   return loss[2], test_loss


def amp_equalize(sig):
   # sig = Sig.clone()
   mean = sig.mean()
   min = sig.min()
   max = sig.max()
   ans = (sig - mean)/(max-min)*10
   yhat = torch.from_numpy(signal.savgol_filter(ans, 11, 5))
   # pdb.set_trace()
   return yhat

         
def get_bpm(Sig, rate= 30.0):
   sig = Sig.copy()
   n = len(sig)
   # print(n)
   fps = rate

   win = signal.hann(sig.size)
   sig = sig - np.expand_dims(np.mean(sig, -1), -1)
   sig = sig * win

   filtered_sig = butter_bandpass_filter(sig, 0.4, 4, fps, order=3)

   f, Pxx_den = signal.welch(sig, fps, nperseg=n)
   index = np.argmax(Pxx_den)
   HR_estimate = round(f[index]*60.0)

   return HR_estimate
