import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import os
import math
# from model.sub_models import ResNet, BasicBlock
# from model.sub_models import OrdinalRegressionLayer
import itertools
from collections import OrderedDict
import torch.nn.functional as F

import pdb



class Synthetic_Gradient_Generator(nn.Module):
   def __init__(self, input_channel, isTrain, device):
      super(Synthetic_Gradient_Generator, self).__init__()
      self.layer1 = nn.Sequential(
          nn.Conv1d(60, 40, kernel_size=3, padding=1),
          nn.BatchNorm1d(40),
          nn.ReLU()
      )
      self.layer2 = nn.Sequential(
          nn.Conv1d(40, 20, kernel_size=3, padding=1),
          nn.BatchNorm1d(20),
          nn.ReLU()
      )
      self.layer3 = nn.Sequential(
          nn.ConvTranspose1d(20, 40, kernel_size=3, padding=1),
          nn.BatchNorm1d(40),
          nn.ReLU()
      )
      self.layer4 = nn.Sequential(
          nn.ConvTranspose1d(40, 60, kernel_size=3, padding=1)
      )

   def forward(self, x): 
      # x's shape = [6, 60, 120]
      res_x1 = self.layer1(x)  # res_x1's shape = [6, 40, 120]
      res_x2 = self.layer2(res_x1)  # res_x2's shape = [6, 20, 120]
      res_x3 = self.layer3(res_x2) + res_x1 # res_x3's shape = [6, 40, 120]
      out = self.layer4(res_x3) # out's shape = [6, 60, 120]
      # pdb.set_trace()

      return out


class Convolutional_Encoder(nn.Module):
   def __init__(self, input_channel, isTrain, device):
      super(Convolutional_Encoder, self).__init__()
      self.conv = nn.Conv3d
      self.conv1 = self.conv(input_channel, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
      self.conv2 = self.conv(32, 48, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
      self.conv3 = self.conv(48, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
      self.conv4 = self.conv(64, 80, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
      self.conv5 = self.conv(80, 120, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

      self.bn1 = nn.BatchNorm3d(32)
      self.bn2 = nn.BatchNorm3d(48)
      self.bn3 = nn.BatchNorm3d(64)
      self.bn4 = nn.BatchNorm3d(80)
      self.bn5 = nn.BatchNorm3d(120)

      self.cnn = {'c1': self.conv1, 'c2': self.conv2, 'c3': self.conv3, 'c4': self.conv4,
                  'c5': self.conv5, 'b1': self.bn1, 'b2': self.bn2, 'b3': self.bn3, 
                  'b4': self.bn4, 'b5': self.bn5}

      self.relu = nn.ReLU(inplace=True)
   
   def forward(self, x):
      win_size = x.shape[1]
      x = x.permute(0, 2, 1, 3, 4)
      
      x = self.conv1(x)
      # pdb.set_trace()
      x = self.bn1(x)
      x = F.avg_pool3d(x,(1,2,2))
      x = self.relu(x)
      x = self.conv2(x)
      x = self.bn2(x)
      x = F.avg_pool3d(x,(1,2,2))
      x = self.relu(x)
      x = self.conv3(x)
      x = self.bn3(x)
      x = F.avg_pool3d(x,(1,2,2))
      x = self.relu(x)
      x = self.conv4(x)
      x = self.bn4(x)
      x = F.avg_pool3d(x,(1,2,2))
      x = self.relu(x)
      x = self.conv5(x)
      x = self.bn5(x)
      x = F.avg_pool3d(x,(1,2,2))
      x = self.relu(x)

      x = F.adaptive_avg_pool3d(x, (win_size, 1, 1))
      x = x.permute(0, 2, 1, 3, 4)
      x = x.reshape(x.size(0), x.size(1),  - 1)

      return x
   
   def return_grad(self):
      # pdb.set_trace()
      c1 = self.conv1.weight.grad.data.clone()
      c2 = self.conv2.weight.grad.data.clone()
      c3 = self.conv3.weight.grad.data.clone()
      c4 = self.conv4.weight.grad.data.clone()
      c5 = self.conv5.weight.grad.data.clone()
      b1 = self.bn1.weight.grad.data.clone()
      b2 = self.bn2.weight.grad.data.clone()
      b3 = self.bn3.weight.grad.data.clone()
      b4 = self.bn4.weight.grad.data.clone()
      b5 = self.bn5.weight.grad.data.clone()

      return {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5,
              'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5}
   

class rPPG_Estimator(nn.Module):
   def __init__(self, input_channel, num_layers, isTrain, device, num_classes=40, h=None, c=None):
      super(rPPG_Estimator, self).__init__()
      self.lstm = nn.LSTM(input_size=120, hidden_size=60,
                          num_layers=num_layers, batch_first=True, bidirectional=True)
      self.fc = nn.Linear(120, 80)
      self.h, self.c = h, c
      self.orl = OrdinalRegressionLayer()
   
   def forward(self, x):
      self.lstm.flatten_parameters()
      # pdb.set_trace()
      if self.h is not None:
         x, (self.h, self.c) = self.lstm(x, (self.h.data, self.c.data))
      else:
         x, _ = self.lstm(x)
      # pdb.set_trace()

      x = self.fc(x)
      decision, prob = self.orl(x)
      decision = decision.squeeze(2)
      # pdb.set_trace()

      return decision, prob
   def feed_hc(self, data):
      # pdb.set_trace()
      self.h = data[0].data
      self.c = data[1].data
      # pdb.set_trace()

   def return_grad(self):
      fc_grad = self.fc.weight.grad.data.clone()
      lstm_list = self.lstm._all_weights
      lstm_dict = {}
      for sublist in lstm_list:
         for name in sublist:
            # pdb.set_trace()
            lstm_dict[name] = self.lstm._parameters[name].grad.data.clone()
      return {'fc': fc_grad, 'lstm': lstm_dict}



class OrdinalRegressionLayer(nn.Module):
   def __init__(self):
      super(OrdinalRegressionLayer, self).__init__()

   def forward(self, x):
      """
      :param x: N X H X W X C, N is batch_size, C is channels of features
      :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
               decode_label is the ordinal labels for each position of Image I
      """
      # pdb.set_trace()
      x = x.permute(0, 2, 1)
      N, C, W = x.size()
      # N, W, C = x.size()
      ord_num = C // 2

      """
      replace iter with matrix operation
      fast speed methods
      """
      A = x[:, ::2, :].clone()
      B = x[:, 1::2, :].clone()
      # pdb.set_trace()
      A = A.view(N, 1, ord_num * W)
      B = B.view(N, 1, ord_num * W)
      # pdb.set_trace()
      C = torch.cat((A, B), dim=1)
      C = torch.clamp(C, min=1e-8, max=1e8)  # prevent nans
      # pdb.set_trace()
      ord_c = nn.functional.softmax(C, dim=1)

      # pdb.set_trace()
      ord_c1 = ord_c[:, 1, :].clone()
      ord_c1 = ord_c1.view(-1, ord_num, W)
      decode_c = torch.sum((ord_c1 > 0.5), dim=1).view(-1, 1, W)
      ord_c1 = ord_c1.permute(0, 2, 1)
      decode_c = decode_c.permute(0, 2, 1)
      # pdb.set_trace()
      return decode_c, ord_c1

   
   


