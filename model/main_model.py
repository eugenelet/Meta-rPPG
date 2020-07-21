import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import os
import itertools
from model.sub_model import rPPG_Estimator, Convolutional_Encoder, Synthetic_Gradient_Generator
from model.loss import ordLoss, KLDivLoss
from scipy import signal
import pickle
from data.data_utils import butter_bandpass_filter
import time
import pdb


class meta_rPPG(nn.Module):
   """
   You can name your own checkpoint directory (opt.checkpoints_dir).

   A_net refers to Conv_Encoder, B_net refers to rPPG_Estimator, Grad_net refers to Synth_Grad_Gen.
   The loading directory can be changed to opt.checkpoints_dir if some other checkpoints are in need.

   """

   def __init__(self, opt, isTrain, continue_train=False, norm_layer=nn.BatchNorm2d):
      """
      Attention_ResNet -- using EfficientNet with LSTM
      AttentionNet -- using a attention strcture without a LSTM
      """
      super(meta_rPPG, self).__init__()
      self.save_dir = os.path.join(os.getcwd(), opt.checkpoints_dir)
      self.load_dir = os.path.join(os.getcwd(), opt.checkpoints_dir)
      if os.path.exists(self.save_dir) == False:
         os.makedirs(self.save_dir)
      self.isTrain = isTrain
      self.opt = opt
      self.gpu_ids = opt.gpu_ids
      self.thres = 0.5
      self.continue_train = continue_train
      self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 
      # self.device = torch.device('cpu') 

      self.prototype = torch.zeros(120)
      self.h = torch.zeros(2*opt.lstm_num_layers, opt.batch_size, 60).to(self.device)
      self.c = torch.zeros(2*opt.lstm_num_layers, opt.batch_size, 60).to(self.device)

      self.A_net = Convolutional_Encoder(input_channel=3, isTrain=self.isTrain, device=self.device)

      self.B_net = rPPG_Estimator(input_channel=120, num_layers=opt.lstm_num_layers, 
            isTrain=self.isTrain, device=self.device, h=self.h, c=self.c)

      self.Grad_net = Synthetic_Gradient_Generator(input_channel=120, isTrain=self.isTrain, device=self.device)
      
      
      self.A_net.to(self.device)
      self.B_net.to(self.device)
      self.Grad_net.to(self.device)
      self.model = [self.A_net, self.B_net, self.Grad_net]
      self.fewloss = 0.0
      self.ordloss = 0.0
      self.gradloss = 0.0

      self.criterion1 = torch.nn.MSELoss()
      self.criterion2 = ordLoss()
      self.criterion3 = torch.nn.MSELoss()

      self.optimizerA = torch.optim.SGD(self.A_net.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
      self.optimizerB = torch.optim.SGD(self.B_net.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
      self.optimizerGrad = torch.optim.SGD(self.Grad_net.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
      if self.opt.adapt_position == "extractor":
         self.optimizerPsi = torch.optim.SGD(self.A_net.parameters(), opt.lr*1e-2, momentum=0.9, weight_decay=5e-4)
      elif self.opt.adapt_position == "estimator":
         self.optimizerPsi = torch.optim.SGD(self.B_net.parameters(), opt.lr*1e-2, momentum=0.9, weight_decay=5e-4)
      elif self.opt.adapt_position == "both":
         self.optimizerPsi = torch.optim.SGD(itertools.chain(self.A_net.parameters(),
                           self.B_net.parameters()), opt.lr*1e-2, momentum=0.9, weight_decay=5e-4)

      self.schedulerA = optim.lr_scheduler.CosineAnnealingLR(self.optimizerA, T_max=5, eta_min=0.1*opt.lr)
      self.schedulerB = optim.lr_scheduler.CosineAnnealingLR(self.optimizerB, T_max=5, eta_min=0.1*opt.lr)
      self.schedulerGrad = optim.lr_scheduler.CosineAnnealingLR(self.optimizerGrad, T_max=5, eta_min=0.1*opt.lr)
      self.schedulerPsi = optim.lr_scheduler.CosineAnnealingLR(self.optimizerPsi, T_max=5, eta_min=0.1*1e-2*opt.lr)



      # pdb.set_trace()
   def print_networks(self, print_net):
      """Print the total number of parameters in the network and (if verbose) network architecture

      Parameters:
      verbose (bool) -- if verbose: print the network architecture
      """
      print('----------- Networks initialized -------------')
      num_params = 0
      for param in self.A_net.parameters():
         num_params += param.numel()
      for param in self.B_net.parameters():
         num_params += param.numel()
      for param in self.Grad_net.parameters():
         num_params += param.numel()
      if print_net:
         print(self.model)
      print('Total number of parameters : %.3f M' %
            (num_params / 1e6))
      # pdb.set_trace()
      print('---------------------end----------------------')

   def set_input(self, input):

      self.input = input['input']
      self.true_rPPG = input['rPPG']
      if 'center' in input:
         self.center = input['center']

   def set_input_for_test(self, input):
      self.input = input.to(self.device)
      # if self.opt.lstm_hc_usage:
      self.B_net.feed_hc([self.h, self.c])

   def forward(self, x):
      """Run forward pass; called by both functions <optimize_parameters> and <test>."""
      # if not self.opt.branch:
      self.inter = self.A_net(x)
      self.decision, self.predict = self.B_net(self.inter)
      if self.opt.adapt_position == "extractor":
         self.gradient = self.Grad_net(self.inter.detach())
      elif self.opt.adapt_position == "estimator":
         self.gradient = self.Grad_net(self.predict.detach())
      elif self.opt.adapt_position == "both":
         self.gradient1 = self.Grad_net(self.inter.detach())
         self.gradient2 = self.Grad_net(self.predict.detach())
   
   def new_theta_update(self, epoch):
      inter = self.A_net(self.input.to(self.device))
      decision, predict = self.B_net(inter)

      fewloss = self.criterion1(self.prototype.expand(self.opt.batch_size,60,120), inter)
      ordloss = self.criterion2(predict, self.true_rPPG.to(self.device))

      self.optimizerA.zero_grad()
      loss = fewloss + ordloss
      loss.backward()
      self.optimizerA.step()

      if self.opt.adapt_position == "extractor":
         for i in range(self.opt.fewshots):
            inter = self.A_net(self.input.to(self.device))
            decision, predict = self.B_net(inter)
            inter_grad = self.Grad_net(inter.detach())
            # self.optimizerA.zero_grad()
            self.optimizerPsi.zero_grad()
            grad = torch.autograd.grad(outputs=inter, inputs=self.A_net.parameters(),
                                       grad_outputs=inter_grad, create_graph=False, retain_graph=False)
            torch.autograd.backward(self.A_net.parameters(), grad_tensors=grad, retain_graph=False, create_graph=False)

            self.optimizerPsi.step()
         self.gradient = inter_grad.detach().clone()
      elif self.opt.adapt_position == "estimator":
         for i in range(self.opt.fewshots):
            inter = self.A_net(self.input.to(self.device))
            decision, predict = self.B_net(inter)
            predict_grad = self.Grad_net(predict.detach())
            # self.optimizerA.zero_grad()
            self.optimizerPsi.zero_grad()
            grad = torch.autograd.grad(outputs=predict, inputs=self.B_net.parameters(),
                                       grad_outputs=predict_grad, create_graph=False, retain_graph=False)
            torch.autograd.backward(self.B_net.parameters(), grad_tensors=grad, retain_graph=False, create_graph=False)
            self.optimizerPsi.step()
         self.gradient = predict_grad.detach().clone()

      elif self.opt.adapt_position == "both":
         for i in range(self.opt.fewshots):
            inter = self.A_net(self.input.to(self.device))
            decision, predict = self.B_net(inter)
            inter_grad = self.Grad_net(inter.detach())
            predict_grad = self.Grad_net(predict.detach())

            self.optimizerPsi.zero_grad()
            grad = torch.autograd.grad(outputs=inter, inputs=self.A_net.parameters(),
                                       grad_outputs=inter_grad, create_graph=False, retain_graph=False)
            torch.autograd.backward(self.A_net.parameters(), grad_tensors=grad, retain_graph=False, create_graph=False)

            grad = torch.autograd.grad(outputs=predict, inputs=self.B_net.parameters(),
                                       grad_outputs=predict_grad, create_graph=False, retain_graph=False)
            torch.autograd.backward(self.B_net.parameters(), grad_tensors=grad, retain_graph=False, create_graph=False)

            self.optimizerPsi.step()
         self.gradient = predict_grad.detach().clone()

      '''release the retained graph, free all the variables'''
      self.fewloss = fewloss.detach().clone()
      self.ordloss = ordloss.detach().clone()
      self.inter = inter.detach().clone()

   def new_psi_phi_update(self, epoch):
      if self.opt.adapt_position == "extractor":
         inter = self.A_net(self.input.to(self.device))
         decision, predict = self.B_net(inter)
         inter_grad = self.Grad_net(inter.detach())

         inter.retain_grad()
         ordloss = self.criterion2(predict, self.true_rPPG.to(self.device))
         fewloss = self.criterion1(self.prototype.expand(self.opt.batch_size,60,120), inter)
         loss = ordloss + fewloss

         self.optimizerB.zero_grad()
         self.optimizerA.zero_grad()
         loss.backward()
         self.optimizerA.step()
         self.optimizerB.step()

         # pdb.set_trace()
         gradloss = self.criterion3(inter_grad, inter.grad)
         self.optimizerGrad.zero_grad()
         gradloss.backward()
         self.optimizerGrad.step()
         self.gradloss = gradloss.detach().clone()

      elif self.opt.adapt_position == "estimator":
         inter = self.A_net(self.input.to(self.device))
         decision, predict = self.B_net(inter)
         predict_grad = self.Grad_net(predict.detach())

         predict.retain_grad()
         ordloss = self.criterion2(predict, self.true_rPPG.to(self.device))
         fewloss = self.criterion1(self.prototype.expand(
             self.opt.batch_size, 60, 120), inter)
         loss = ordloss + fewloss

         self.optimizerB.zero_grad()
         self.optimizerA.zero_grad()
         loss.backward()
         self.optimizerA.step()
         self.optimizerB.step()

         gradloss = self.criterion3(predict_grad, predict.grad)
         self.optimizerGrad.zero_grad()
         gradloss.backward()
         self.optimizerGrad.step()
         self.gradloss = gradloss.detach().clone()
      
      elif self.opt.adapt_position == "both":
         inter = self.A_net(self.input.to(self.device))
         decision, predict = self.B_net(inter)
         predict_grad = self.Grad_net(predict.detach())
         inter_grad = self.Grad_net(inter.detach())

         predict.retain_grad()
         inter.retain_grad()
         ordloss = self.criterion2(predict, self.true_rPPG.to(self.device))
         fewloss = self.criterion1(self.prototype.expand(
             self.opt.batch_size, 60, 120), inter)
         loss = ordloss + fewloss

         self.optimizerB.zero_grad()
         self.optimizerA.zero_grad()
         loss.backward()
         self.optimizerA.step()
         self.optimizerB.step()

         gradloss = self.criterion3(
               predict_grad, predict.grad) + self.criterion3(inter_grad, inter.grad)
         self.optimizerGrad.zero_grad()
         gradloss.backward()
         self.optimizerGrad.step()
         self.gradloss = gradloss.detach().clone()

      self.decision = decision.detach().clone()
      self.predict = predict.detach().clone()
      self.ordloss = ordloss.detach().clone()


   def update_prototype(self):
      proto_tmp = torch.zeros(120).to(self.device)
      h_tmp = torch.zeros(2*self.opt.lstm_num_layers, self.opt.batch_size, 60).to(self.device)
      c_tmp = torch.zeros(2*self.opt.lstm_num_layers, self.opt.batch_size, 60).to(self.device)
      self.B_net.feed_hc([self.h, self.c])
      # pdb.set_trace()

      self.forward(self.input.to(self.device))
      # pdb.set_trace()
      proto_tmp += self.inter.data.mean(axis=[0,1])
      h_tmp += self.B_net.h.data
      c_tmp += self.B_net.c.data

      if torch.sum(self.prototype) == 0: # first update
         self.prototype = proto_tmp
         (self.h, self.c) = (h_tmp, c_tmp)
      else:
         self.prototype = 0.8*self.prototype + 0.2*proto_tmp
         (self.h, self.c) = (0.8*self.h + 0.2*h_tmp, 0.8*self.c + 0.2*c_tmp)



   def setup(self, opt):
      self.init_weights(self.A_net, self.B_net)
      # pdb.set_trace()
      if self.continue_train:
         self.load_networks(opt.load_file)
         self.thres = 0.01
      if not self.isTrain:
         # load_suffix = 'latest'
         # load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
         self.load_networks(opt.load_file)
         # self.progress = 1.45
      # pdb.set_trace()
      self.print_networks(opt.print_net)

   def init_weights(net1, net2, init_type='normal', init_gain=0.02):
      net1.apply(init_func)
      net2.apply(init_func)

   def save_networks(self, suffix):
      """Save all the networks to the disk.

      Parameters:
         epoch (int) -- current epoch; used in the file name '%s_%s.pth' % (epoch, name)
      """
      save_filename1 = '%s_%s.pth' % (suffix, self.opt.name)
      save_path1 = os.path.join(self.save_dir, save_filename1)
      # pdb.set_trace()
      torch.save({'A': self.A_net.state_dict(), 
                  'B': self.B_net.state_dict(),
                  'Grad': self.Grad_net.state_dict(),
                  'proto': self.prototype.cpu(),
                  'h': self.h.data.cpu(), 
                  'c': self.c.data.cpu()},
                   save_path1)


   def get_current_losses(self, istest):
      # return [self.criterion.loss1.clone(), self.criterion.loss2.clone()]
      if istest:
         return self.t_ordloss
      else:
         return [self.fewloss, self.gradloss, self.ordloss]

   def eval(self):
      """Make models eval mode during test time"""
      self.A_net.eval()
      self.B_net.eval()
      self.Grad_net.eval()
      # self.attention.eval()
      # self.lstm.eval()
      # self.fc.eval()

   def train(self):
      """Make models train mode after test time"""
      self.A_net.train()
      self.B_net.train()
      self.Grad_net.train()
      # self.attention.train()
      # self.lstm.train()
      # self.fc.train()

   def test(self):
      """Forward function used in test time. """
      with torch.no_grad():
         self.forward(self.input[len(self.input)-1].unsqueeze(0).to(self.device))

      self.t_ordloss = self.criterion2(self.predict, self.true_rPPG[len(self.true_rPPG)-1].unsqueeze(0).to(self.device))

   def fewshot_test(self, epoch):
      A = pickle.loads(pickle.dumps(self.A_net))
      optim = torch.optim.SGD(A.parameters(), self.opt.lr*1e-2, momentum=0.9, weight_decay=5e-4)
      
      for i in range(self.opt.fewshots):
         optim.zero_grad()
         inter = A(self.input[i].unsqueeze(0).to(self.device))
         inter_grad = self.Grad_net(inter)
         grad = torch.autograd.grad(outputs=inter, inputs=A.parameters(),
                     grad_outputs=inter_grad, create_graph=False, retain_graph=False)
         torch.autograd.backward(A.parameters(), grad_tensors=grad, retain_graph=False, create_graph=False)
         optim.step()

      for i in range(self.opt.fewshots):
         optim.zero_grad()
         inter = A(self.input[i].unsqueeze(0).to(self.device))
         loss = self.criterion1(inter, self.prototype.expand(1, 60, 120))
         loss.backward()
         optim.step()

      with torch.no_grad():
         tmp_h = self.B_net.h
         tmp_c = self.B_net.c
         # if self.opt.lstm_hc_usage:
         self.B_net.feed_hc([self.h, self.c])

         data = self.input[self.opt.fewshots:]
         inter = A(data.to(self.device))
         self.decision, self.predict = self.B_net(inter)
         self.B_net.feed_hc([tmp_h, tmp_c])

      self.t_ordloss = self.criterion2(self.predict[0].unsqueeze(0), self.true_rPPG[0].unsqueeze(0).to(self.device))


   def get_current_results(self, istest):
      if istest:
         return self.decision[-1].cpu().clone(), self.true_rPPG[-1].cpu().clone()
      else:
         return self.decision[-1].cpu().clone(), self.true_rPPG[-1].cpu().clone()
         # return self.decision[0].cpu().clone(), self.true_rPPG[len(self.input)-1][0].cpu().clone()
         
   # def get_freq_results(self):
   #    return self.criterion.true_fft[0].cpu().clone(), self.criterion.predict_fft[0].detach().cpu().clone()

   def get_current_results_of_test(self):
      # pdb.set_trace()
      return self.decision[0].cpu().clone()

   def load_networks(self, suffix):
      """Load all the networks from the disk.

      Parameters:
         suffix (str) -- current epoch; used in the file name '%s_%s.pth' % (suffix, name)
      """

      load_filename1 = '%s_%s.pth' % (suffix, self.opt.name)
      load_path1 = os.path.join(self.load_dir, load_filename1)

      print('loading model from %s' % load_path1)
      model_dict = torch.load(load_path1)
      self.A_net.load_state_dict(model_dict['A'])
      self.B_net.load_state_dict(model_dict['B'])
      self.Grad_net.load_state_dict(model_dict['Grad'])

      self.prototype = model_dict['proto'].to(self.device)
      self.h = model_dict['h'].to(self.device)
      self.c = model_dict['c'].to(self.device)

      # self.A_net.eval()
      # self.B_net.eval()
      # self.Grad_net.eval()

      

      


   def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
      """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
      key = keys[i]
      if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
         if module.__class__.__name__.startswith('InstanceNorm') and \
                 (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
               state_dict.pop('.'.join(keys))
         if module.__class__.__name__.startswith('InstanceNorm') and \
            (key == 'num_batches_tracked'):
               state_dict.pop('.'.join(keys))
      else:
         self.__patch_instance_norm_state_dict(
             state_dict, getattr(module, key), keys, i + 1)

   def get_param(self):
      return [self.A_net.get_param(), self.B_net.get_param()]

   def update_learning_rate(self, epoch):
      """Update learning rates for all the networks; called at the end of every epoch"""

      self.schedulerA.step()
      self.schedulerB.step()
      self.schedulerGrad.step()
      self.schedulerPsi.step()
      

      # pdb.set_trace()
      lr = self.optimizerB.param_groups[0]['lr']
      return lr
      # print('\nlearning rate = %.7f' % lr)

def init_func(m):  # define the initialization function
   classname = m.__class__.__name__
   if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
      init.normal_(m.weight.data, 0.0, 0.02)
   # if hasattr(m, 'bias') and m.bias is not None:
   #    init.constant_(m.bias.data, 0.0)
   # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
   elif classname.find('BatchNorm2d') != -1:
      init.normal_(m.weight.data, 1.0, 0.02)
      init.constant_(m.bias.data, 0.0)

