import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import os
from torch.autograd import Variable
from torch.nn.functional import conv1d

from scipy import signal
import torch.nn.functional as F
import pdb


class ordLoss(nn.Module):
   """
   Ordinal loss is defined as the average of pixelwise ordinal loss F(h, w, X, O)
   over the entire image domain:
   """

   def __init__(self):
      super(ordLoss, self).__init__()
      self.loss = 0.0

   def forward(self, orig_ord_labels, orig_target):
      """
      :param ord_labels: ordinal labels for each position of Image I.
      :param target:     the ground_truth discreted using SID strategy.
      :return: ordinal loss
      """
      device = orig_ord_labels.device
      ord_labels = orig_ord_labels.clone()
      # ord_labels = ord_labels.unsqueeze(0)
      ord_labels = torch.transpose(ord_labels, 1, 2)

      N, C, W = ord_labels.size()
      ord_num = C 

      self.loss = 0.0

      # faster version
      if torch.cuda.is_available():
         K = torch.zeros((N, C, W), dtype=torch.int).to(device)
         for i in range(ord_num):
               K[:, i, :] = K[:, i, :] + i * \
                  torch.ones((N, W), dtype=torch.int).to(device)
      else:
         K = torch.zeros((N, C, W), dtype=torch.int)
         for i in range(ord_num):
               K[:, i, :] = K[:, i, :] + i * \
                  torch.ones((N, W), dtype=torch.int)
      # pdb.set_trace()

      # target = orig_target.clone().type(torch.DoubleTensor)
      if device == torch.device('cpu'):
         target = orig_target.clone().type(torch.IntTensor)
      else:
         target = orig_target.clone().type(torch.cuda.IntTensor)

      mask_0 = torch.zeros((N, C, W), dtype=torch.bool)
      mask_1 = torch.zeros((N, C, W), dtype=torch.bool)
      for i in range(N):
         mask_0[i] = (K[i] <= target[i]).detach()
         mask_1[i] = (K[i] > target[i]).detach()


      one = torch.ones(ord_labels[mask_1].size())
      if torch.cuda.is_available():
         one = one.to(device)

      self.loss += torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-8, max=1e8))) \
         + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-8, max=1e8)))

      N = N * W
      self.loss /= (-N)  # negative
      # pdb.set_trace()
      return self.loss

class customLoss(nn.Module):
   """
   This customize loss is contained of Ordloss and MSELoss of the frequency magnitude
   """
   def __init__(self, device):
      super(customLoss, self).__init__()
      self.loss = 0.0
      self.ord = ordLoss()

      self.vis = Visdom(port=8093, env='main')

      # self.cross = torch.nn.CrossEntropyLoss()
      # self.cross = torch.nn.NLLLoss()
      # self.cross = torch.nn.MSELoss()

      self.reg = regressLoss()
      # self.weight = torch.autograd.Variable(torch.tensor(1.0), requires_grad=True).to(device)
      self.weight = nn.Linear(2,1).to(device)
      with torch.no_grad():
         self.weight.weight.copy_(torch.tensor([1.0,1.0]))
      pdb.set_trace()
      self.t = torch.tensor([2.0,2.0]).to(device)
      self.device = device

   def forward(self, predict, true_rPPG):

      self.loss1 = self.ord(predict[0], true_rPPG)
      self.true_fft = self.torch_style_fft(true_rPPG)  # (batch size x 60)
      self.predict_fft = self.torch_style_fft(predict[1])  # (batch size x 60)

      self.loss2 = self.reg(self.predict_fft, self.true_fft)
      if torch.isnan(self.loss2):
         pdb.set_trace()

      # self.loss = self.loss1 + self.weight * self.loss2
      # pdb.set_trace()
      self.t1 = self.weight(self.t)
      self.loss = self.weight(torch.stack([self.loss1, self.loss2]))
      pdb.set_trace()

      return self.loss
      # pdb.set_trace()

   def torch_style_fft(self, sig):
      # pdb.set_trace()
      S, _ = torch_welch(sig, fps = 30)

      return S


class regressLoss(nn.Module):
    def __init__(self):
        super(regressLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
      #   self.weight = weight

    def forward(self, outputs, targets):

      preoutput = outputs.clone()
      if torch.isnan(preoutput.cpu().detach()).any():
         pdb.set_trace()
      # small_number = torch.tensor(1e-45).to(targets.get_device())
      targets = self.softmax(targets)
      outputs = self.softmax(outputs)
      if torch.isnan(outputs.cpu().detach()).any():
         pdb.set_trace()
      # outputs = outputs + small_number

      loss = -targets.float() * torch.log(outputs) 
      # if np.isnan(torch.mean(loss).cpu().detach().numpy()):
      #    pdb.set_trace()
      return torch.mean(loss)


class KLDivLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(KLDivLoss, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction=reduction)
      #   self.weight = weight

    def forward(self, outputs, targets):
      out = outputs.clone()
      tar = targets.clone()
      out.uniform_(0, 1)
      tar.uniform_(0, 1)
      # loss = self.criterion(F.log_softmax(out, -1), tar)
      loss = self.criterion(F.log_softmax(outputs, dim=1), F.softmax(targets, dim=1))

      return loss


def torch_welch(sig, fps):
   nperseg = sig.size(1)
   nfft = sig.size(1)
   noverlap = nperseg//2
   # pdb.set_trace()

   sig = sig.type(torch.cuda.FloatTensor)
   win = torch.from_numpy(signal.hann(sig.size(1))).to(sig.get_device()).type(torch.cuda.FloatTensor)
   sig = sig.unsqueeze(1)
   # pdb.set_trace()

   '''detrend'''
   sig = sig - torch.from_numpy(np.expand_dims(np.mean(sig.detach().cpu().numpy(), -1), -1)).to(sig.get_device())
   sig = sig * win
   S = torch.rfft(sig, 1, normalized=True, onesided=True)
   S = torch.sqrt(S[..., 0]**2 + S[..., 1]**2)   
   freqs = torch.from_numpy(np.fft.rfftfreq(nfft, 1/float(fps)))

   S = S.squeeze(1)

   return S, freqs

