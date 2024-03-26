import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from CFG import seq_length, input_length, tgt_length, data_set_dir


class SeqLSTM(nn.Module):
  def __init__(self, nhidden, nlayers, narchs=1, nembed=0, gru=False, bi=False, norm=False, bias=True):
    super(SeqLSTM, self).__init__()

    if nembed != 0:
      self.embed = True
      self.inst_embed = nn.Linear(input_length, nembed)
      nin = nembed
    else:
      self.embed = False
      nin = input_length
    self.bi = bi
    self.norm = norm
    if norm:
      #self.inst_norm = nn.LayerNorm(seq_length)
      self.inst_norm = nn.LayerNorm([seq_length, nin])
    if gru:
      self.lstm = nn.GRU(nin, nhidden, nlayers, batch_first=True, bidirectional=bi)
    else:
      self.lstm = nn.LSTM(nin, nhidden, nlayers, batch_first=True, bidirectional=bi)
    if bi:
      nhidden *= 2
    self.linear = nn.Linear(nhidden, narchs * tgt_length, bias=bias)

  #def init_hidden(self):
  #  # type: () -> Tuple[nn.Parameter, nn.Parameter]
  #  return (
  #    nn.Parameter(torch.zeros(1, 1, nhidden, requires_grad=True)),
  #    nn.Parameter(torch.zeros(1, 1, nhidden, requires_grad=True)),
  #  )

  def extract_representation(self, x):
    if self.embed:
      x = self.inst_embed(x)
    if self.norm:
      #x = self.inst_norm(x.transpose(1, 2)).transpose(1, 2)
      x = self.inst_norm(x)
    x, _ = self.lstm(x)
    x = F.relu(x)
    #x = torch.sigmoid(x)
    return x

  def forward(self, x):
    x = self.extract_representation(x)
    x = self.linear(x)
    return x


class InsLSTM(SeqLSTM):
  def __init__(self, nhidden, nlayers, narchs=1, nembed=0, gru=False, bi=False, norm=False, bias=True):
    super().__init__(nhidden, nlayers, narchs, nembed, gru, bi, norm, bias)

  def extract_representation(self, x):
    x = super().extract_representation(x)
    return x[:, -1, :]


class InsLSTMDSE(InsLSTM):
  def __init__(self, nhidden, nlayers, narchs=1, nembed=0, gru=False, bi=False, norm=False, bias=True):
    assert not bias
    super().__init__(nhidden, nlayers, narchs, nembed, gru, bi, norm, bias)
    self.nhidden = nhidden
    if bi:
      self.nhidden *= 2

  def init_paras(self, nparas=2, nparahidden=16):
    assert nparas == 2
    self.uarch_net = nn.Sequential(
      nn.Linear(nparas, nparahidden),
      nn.ReLU(),
      nn.Linear(nparahidden, self.nhidden * tgt_length),
    )
    narchs = 18
    uarch_paras = torch.tensor([[1, 1], [1, 2], [1, 3],
                                [6, 4], [6, 5], [6, 6],
                                [3, 3], [3, 4], [3, 2],
                                [4, 3], [4, 4], [4, 2],
                                [5, 3], [5, 4], [5, 5],
                                [2, 3], [2, 4], [2, 2]],
                                dtype=torch.float)
    self.uarch_paras = nn.Parameter(uarch_paras)
    self.uarch_paras.requires_grad = False
    self.output_num = narchs * tgt_length
    print("Paras:", self.uarch_paras)

  def setup_test(self):
    narchs = 36
    uarch_paras = torch.tensor([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
                                [6, 4], [6, 5], [6, 6], [6, 1], [6, 2], [6, 3],
                                [3, 3], [3, 4], [3, 2], [3, 1], [3, 5], [3, 6],
                                [4, 3], [4, 4], [4, 2], [4, 1], [4, 5], [4, 6],
                                [5, 3], [5, 4], [5, 5], [5, 1], [5, 2], [5, 6],
                                [2, 3], [2, 4], [2, 2], [2, 1], [2, 5], [2, 6]],
                                dtype=torch.float)
    self.uarch_paras = nn.Parameter(uarch_paras)
    self.uarch_paras.requires_grad = False
    self.output_num = narchs * tgt_length
    print("Test paras:", self.uarch_paras)

  def forward(self, x):
    rep = super().extract_representation(x)
    uarch_rep = self.uarch_net(self.uarch_paras).view(self.output_num, -1)
    x = F.linear(rep, uarch_rep)
    return x


class SeqEmLSTM(nn.Module):
  def __init__(self, nhidden, nlayers, narchs=1, nop=1, nmem=0, nctrl=0, nr=0, gru=False, bi=False, bias=True):
    super(SeqEmLSTM, self).__init__()

    assert nop > 0
    self.op_embed = nn.Embedding(50, nop)
    self.nin = input_length - 1 + nop
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
      self.reg_embed = nn.Embedding(1024, nr)
      self.nin += (nr - 2) * 14
    self.bi = bi
    if gru:
      self.lstm = nn.GRU(self.nin, nhidden, nlayers, batch_first=True, bidirectional=bi)
    else:
      self.lstm = nn.LSTM(self.nin, nhidden, nlayers, batch_first=True, bidirectional=bi)
    if bi:
      nhidden *= 2
    self.linear = nn.Linear(nhidden, narchs * tgt_length, bias=bias)

    # Load normalization factors.
    stats = np.load(data_set_dir + "stats.npz")
    mean = stats['mean']
    std = stats['std']
    std[std == 0.0] = 1.0
    mean = torch.from_numpy(mean.astype('f'))
    std = torch.from_numpy(std.astype('f'))
    self.mean = nn.Parameter(mean)
    self.std = nn.Parameter(std)
    self.mean.requires_grad = False
    self.std.requires_grad = False

  def get_embeddings(self, x):
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
      regs = ori_x[:, :, 23:].view(-1, seq_length, 14, 2)
      reg_idx = 50 * regs[:, :, :, 0] + regs[:, :, :, 1]
      regem = self.reg_embed(reg_idx).view(-1, seq_length, 14 * self.nr)
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

  def extract_representation(self, x):
    x = self.get_embeddings(x)
    x, _ = self.lstm(x)
    x = F.relu(x)
    #x = torch.sigmoid(x)
    return x

  def forward(self, x):
    x = self.extract_representation(x)
    x = self.linear(x)
    return x


class InsEmLSTM(SeqEmLSTM):
  def __init__(self, nhidden, nlayers, narchs=1, nop=1, nmem=0, nctrl=0, nr=0, gru=False, bi=False, bias=True):
    super().__init__(nhidden, nlayers, narchs, nop, nmem, nctrl, nr, gru, bi, bias)

  def extract_representation(self, x):
    x = super().extract_representation(x)
    return x[:, -1, :]
