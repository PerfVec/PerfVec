import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InsEmModel(nn.Module):
  def __init__(self, embed_model, main_model):
    super(InsEmModel, self).__init__()
    self.embed_model = embed_model
    self.main_model = main_model

  def extract_representation(self, x):
    x = self.embed_model(x)
    x = self.main_model.extract_representation(x)
    return x

  def forward(self, x):
    x = self.extract_representation(x)
    x = self.main_model.linear(x)
    return x


class InsEm(nn.Module):
  def __init__(self, cfg, nop=1, nmem=0, nctrl=0, nr=0):
    super(InsEm, self).__init__()

    assert nop > 0
    self.op_embed = nn.Embedding(50, nop)
    self.nin = cfg.input_length - 1 + nop
    self.nmem = nmem
    if nmem > 0:
      self.mem_embed = nn.Embedding(256, nmem)
      self.nin += nmem - 7
    self.nctrl = nctrl
    if nctrl > 0:
      self.ctrl_embed = nn.Embedding(512, nctrl)
      self.nin += nctrl - 9
    self.nr = nr
    if nr > 0:
      self.reg_embed = nn.Embedding(1040, nr)
      self.nin += (nr - 2) * 14
    self.seq_length = cfg.seq_length

    # Load normalization factors.
    stats = np.load(cfg.data_set_dir + "stats.npz")
    mean = stats['mean']
    std = stats['std']
    std[std == 0.0] = 1.0
    mean = torch.from_numpy(mean.astype('f'))
    std = torch.from_numpy(std.astype('f'))
    self.mean = nn.Parameter(mean)
    self.std = nn.Parameter(std)
    self.mean.requires_grad = False
    self.std.requires_grad = False
    print("InsEm output size:", self.nin)

  def forward(self, x):
    ori_x = x * self.std + self.mean
    ori_x = ori_x.round().long()
    embeddings = self.op_embed(ori_x[:, :, 1])
    if self.nmem > 0:
      mem_idx = ori_x[:, :, 0]
      mem_idx = torch.bitwise_left_shift(mem_idx, 1) + ori_x[:, :, 2]
      mem_idx = torch.bitwise_left_shift(mem_idx, 1) + ori_x[:, :, 3]
      mem_idx = torch.bitwise_left_shift(mem_idx, 2) + ori_x[:, :, 11]
      mem_idx = torch.bitwise_left_shift(mem_idx, 1) + ori_x[:, :, 12]
      mem_idx = torch.bitwise_left_shift(mem_idx, 1) + ori_x[:, :, 13]
      mem_idx = torch.bitwise_left_shift(mem_idx, 1) + ori_x[:, :, 19]
      memem = self.mem_embed(mem_idx)
      embeddings = torch.cat((embeddings, memem), 2)
    if self.nctrl > 0:
      ctrl_idx = ori_x[:, :, 4]
      ctrl_idx = torch.bitwise_left_shift(ctrl_idx, 1) + ori_x[:, :, 5]
      ctrl_idx = torch.bitwise_left_shift(ctrl_idx, 1) + ori_x[:, :, 6]
      ctrl_idx = torch.bitwise_left_shift(ctrl_idx, 1) + ori_x[:, :, 7]
      ctrl_idx = torch.bitwise_left_shift(ctrl_idx, 1) + ori_x[:, :, 8]
      ctrl_idx = torch.bitwise_left_shift(ctrl_idx, 1) + ori_x[:, :, 9]
      ctrl_idx = torch.bitwise_left_shift(ctrl_idx, 1) + ori_x[:, :, 10]
      ctrl_idx = torch.bitwise_left_shift(ctrl_idx, 1) + ori_x[:, :, 14]
      ctrl_idx = torch.bitwise_left_shift(ctrl_idx, 1) + ori_x[:, :, 15]
      ctrlem = self.ctrl_embed(ctrl_idx)
      embeddings = torch.cat((embeddings, ctrlem), 2)
    if self.nr > 0:
      regs = ori_x[:, :, 23:].reshape(-1, self.seq_length, 14, 2)
      reg_idx = 50 * regs[:, :, :, 0] + regs[:, :, :, 1]
      #check = torch.logical_and(regs[:, :, :, 1] >= 50, regs[:, :, :, 0] < 7)
      #if torch.any(check):
      #  idxx = check.nonzero()
      #  print("ch", check.nonzero())
      #  print("ha", ori_x[idxx[0, 0].item(), idxx[0, 1].item()])
      #assert not torch.any(check)
      #if torch.any(torch.gt(reg_idx, 1024)):
      #  #print("p", reg_idx)
      #  #print("q", regs)
      #  idxx = torch.gt(reg_idx, 1024).nonzero()
      #  print("hi", torch.gt(reg_idx, 1024).nonzero())
      #  print("oo", idxx[0, 0].item(), idxx[0, 1].item(), idxx[0, 2].item())
      #  print("ho", reg_idx[idxx[0, 0].item(), idxx[0, 1].item(), idxx[0, 2].item()], regs[idxx[0, 0].item(), idxx[0, 1].item(), idxx[0, 2].item()])
      #  print("q", regs[idxx[0, 0].item(), idxx[0, 1].item()])
      #  print("all", ori_x[idxx[0, 0].item(), idxx[0, 1].item()])
      #  print("ha", torch.masked_select(reg_idx, torch.gt(reg_idx, 1024)))
      regem = self.reg_embed(reg_idx).view(-1, self.seq_length, 14 * self.nr)
      embeddings = torch.cat((embeddings, regem), 2)

    # Combine the rest input.
    if self.nmem > 0 and self.nctrl > 0:
      rest = torch.cat((x[:, :, 16:19], x[:, :, 20:23]), 2)
    elif self.nmem > 0:
      rest = torch.cat((x[:, :, 4:11], x[:, :, 14:19], x[:, :, 20:23]), 2)
    elif self.nctrl > 0:
      rest = torch.cat((x[:, :, 0:1], x[:, :, 2:4], x[:, :, 11:14], x[:, :, 16:23]), 2)
    else:
      rest = torch.cat((x[:, :, 0:1], x[:, :, 2:23]), 2)
    if self.nr == 0:
      rest = torch.cat((rest, x[:, :, 23:]), 2)

    x = torch.cat((embeddings, rest), 2)
    return x
