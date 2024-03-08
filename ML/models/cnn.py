import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
  def __init__(self, cfg, l, h, *args, narchs=1):
    super(CNN, self).__init__()
    self.conv = nn.ModuleList()
    self.input_length = cfg.input_length
    self.seq_length = cfg.seq_length
    ic = cfg.input_length
    num = cfg.seq_length
    assert l > 0
    assert len(args) == 4 * l
    for i in range(l):
      idx = i * 4
      ks = args[idx]
      oc = args[idx+1]
      stride = args[idx+2]
      pad = args[idx+3]
      self.conv.append(nn.Conv1d(in_channels=ic, out_channels=oc, kernel_size=ks, stride=stride, padding=pad)) 
      ic = oc
      num = math.floor((num + 2 * pad - ks) / stride + 1)
      print(i, num, oc)
    self.fc_in = int(num * ic)
    self.fc1 = nn.Linear(self.fc_in, h)
    self.linear = nn.Linear(h, narchs * cfg.tgt_length, bias=False)

  def extract_representation(self, x):
    x = x.view(-1, self.seq_length, self.input_length).transpose(2,1)
    for cv in self.conv:
      x = F.relu(cv(x))
    x = x.view(-1, self.fc_in)
    x = F.relu(self.fc1(x))
    return x

  def forward(self, x):
    x = self.extract_representation(x)
    x = self.linear(x)
    return x


class MLP(nn.Module):
  def __init__(self, cfg, l, h, *args, narchs=1):
    super(MLP, self).__init__()
    self.fc = nn.ModuleList()
    self.il = int(cfg.input_length * cfg.seq_length)
    assert l > 0
    assert len(args) == l - 1
    fc_in = self.il
    for i in range(l):
      if i == l - 1:
        fc_out = h
      else:
        fc_out = args[i]
      self.fc.append(nn.Linear(fc_in, fc_out))
      fc_in = fc_out
    self.linear = nn.Linear(h, narchs * cfg.tgt_length, bias=False)

  def extract_representation(self, x):
    x = x.view(-1, self.il)
    for fc in self.fc:
      x = F.relu(fc(x))
    return x

  def forward(self, x):
    x = self.extract_representation(x)
    x = self.linear(x)
    return x
