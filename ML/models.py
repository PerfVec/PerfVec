import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from CFG import seq_length, input_length, tgt_length


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


class SeqEmLSTM(SeqLSTM):
  def __init__(self, nhidden, nlayers, narchs, nembed=0, gru=False, bi=False, norm=False):
    super().__init__(nhidden, nlayers, 1, nembed, gru, bi, norm)
    self.arch_embed = nn.Embedding(narchs, (nhidden + 1) * tgt_length)

  def forward(self, x, a):
    if self.embed:
      x = self.inst_embed(x)
    if self.norm:
      #x = self.inst_norm(x.transpose(1, 2)).transpose(1, 2)
      x = self.inst_norm(x)
    x, _ = self.lstm(x)
    arch = nn.arch_embed(a).view(tgt_length, nhidden + 1)
    x = torch.nn.functional.linear(x, arch[:, 0:nhidden], bias=arch[:, nhidden])
    return x


class InsEmLSTM(SeqEmLSTM):
  def __init__(self, nhidden, nlayers, narchs, nembed=0, gru=False, bi=False, norm=False):
    super().__init__(nhidden, nlayers, narchs, nembed, gru, bi, norm)

  def forward(self, x, a):
    x = super().forward(x, a)
    return x[:, -1, :]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :seq_length, :]
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
        self.pos_encoder = PositionalEncoding(self.nfeatures, dropout)
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

    def forward(self, x):
        if self.embed:
          x = self.inst_embed(x)
        x = self.pos_encoder(x)
        #x = self.encoder(x, self.src_mask)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


class CNN(nn.Module):
    def __init__(self, l, h, narchs=1, *args):
        super(CNN, self).__init__()
        self.conv = nn.ModuleList()
        lc = input_length
        num = seq_length
        assert l > 0
        assert len(args) == 4 * l
        for i in range(l):
          idx = i * 4
          ks = args[idx]
          oc = args[idx+1]
          stride = args[idx+2]
          pad = args[idx+3]
          self.conv.append(nn.Conv1d(in_channels=lc, out_channels=oc, kernel_size=ks, stride=stride, padding=pad)) 
          lc = oc
          num = math.floor((num + 2 * pad - ks) / stride + 1)
          print(i, num)
        self.fc_in = int(num * lc)
        self.fc1 = nn.Linear(self.fc_in, h)
        self.linear = nn.Linear(h, narchs * tgt_length)

    def forward(self, x):
        x = x.view(-1, seq_length, input_length).transpose(2,1)
        for cv in self.conv:
          x = F.relu(cv(x))
        x = x.view(-1, self.fc_in)
        x = F.relu(self.fc1(x))
        x = self.linear(x)
        return x
