import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from CFG import seq_length, input_length, tgt_length, data_set_dir


class Predictor(nn.Module):
  def __init__(self, cfg, bias=True, global_bias=False, test_bias=True):
    super(Predictor, self).__init__()
    assert not bias or not global_bias
    self.linear = nn.Linear(cfg.input_length, cfg.cfg_num * cfg.tgt_length, bias=bias)
    self.global_bias = global_bias
    #FIXME: need multiple global biases when predicting multiple targets.
    if self.global_bias:
      self.gb = nn.Parameter(torch.randn(1))
    self.test_bias = test_bias

  def setup_test(self):
    if not self.test_bias:
      print("Remove bias %f in testing." % self.linear.bias)
      self.linear.bias.requires_grad = False
      self.linear.bias.fill_(0.)

  def forward(self, x):
    #FIXME: subtract x with a non-negative vector that represents the rest program.
    x = self.linear(x)
    if self.global_bias:
      x += self.gb
    return x


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

  def forward(self, x):
    x = super().forward(x)
    return x[:, -1, :]


class InsLSTMRep(SeqLSTM):
  def __init__(self, nhidden, nlayers, narchs=1, nembed=0, gru=False, bi=False, norm=False, bias=True):
    super().__init__(nhidden, nlayers, narchs, nembed, gru, bi, norm, bias)

  def forward(self, x):
    rep = super().extract_representation(x)
    rep = rep[:, -1, :]
    x = self.linear(rep)
    return x, rep


class InsLSTMDSE(SeqLSTM):
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
    rep = rep[:, -1, :]
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
    ori_x = ori_x.long()
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

  def forward(self, x):
    x = super().forward(x)
    return x[:, -1, :]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        if d_model % 2 == 1:
            dd = d_model + 1
        else:
            dd = d_model
        div_term = torch.exp(torch.arange(0, dd, 2) * (-math.log(10000.0)/dd))
        pe = torch.zeros(1, max_len, dd)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = pe[:, :seq_length, :d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self,
                 nhead, # number of 'heads'
                 nhid,  # dimension of the feedforward network in nn.TransformerEncoder
                 nlayers, #number of encoder layers
                 narchs=1,
                 nembed = 0,
                 dropout=0.1):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        if nembed != 0:
          self.embed = True
          self.inst_embed = nn.Linear(input_length, nembed)
          self.nfeatures = nembed
        else:
          self.embed = False
          self.nfeatures = input_length
        self.pos_encoder = PositionalEncoding(self.nfeatures, dropout, max_len=seq_length)
        encoder_layers = TransformerEncoderLayer(d_model=self.nfeatures,
                                                 nhead=nhead,
                                                 dim_feedforward=nhid,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)

        self.linear = nn.Linear(self.nfeatures, narchs * tgt_length)
        #src_mask = self.generate_square_subsequent_mask(seq_length)
        #self.register_buffer('src_mask', src_mask)
        self.init_weights()

    #def generate_square_subsequent_mask(self, sz):
    #    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #    return mask
    #    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def init_weights(self):
        initrange = 0.1
        if self.embed:
          self.inst_embed.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def extract_representation(self, x):
        if self.embed:
          x = self.inst_embed(x)
        x = self.pos_encoder(x)
        #x = self.encoder(x, self.src_mask)
        x = self.encoder(x)
        return x

    def forward(self, x):
        x = self.extract_representation(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


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
