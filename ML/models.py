import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from cfg import seq_length, input_length, tgt_length


class SeqLSTM(nn.Module):
  def __init__(self, nhidden, nlayers, nembed = 0):
    super(SeqLSTM, self).__init__()

    if nembed != 0:
      self.embed = True
      self.inst_embed = nn.Linear(input_length, nembed)
      self.inst_norm = nn.BatchNorm1d(nembed)
      #self.inst_norm = nn.LayerNorm([seq_length, nembed])
      self.lstm = nn.LSTM(nembed, nhidden, nlayers, batch_first=True)
    else:
      self.embed = False
      self.lstm = nn.LSTM(input_length, nhidden, nlayers, batch_first=True)
    self.linear = nn.Linear(nhidden, tgt_length)

  def init_hidden(self):
    # type: () -> Tuple[nn.Parameter, nn.Parameter]
    return (
      nn.Parameter(torch.zeros(1, 1, nhidden, requires_grad=True)),
      nn.Parameter(torch.zeros(1, 1, nhidden, requires_grad=True)),
    )

  def forward(self, x):
    if self.embed:
      x = self.inst_embed(x)
      x = F.relu(self.inst_norm(x.transpose(1, 2)).transpose(1, 2))
      #x = F.relu(self.inst_norm(x))
    x, _ = self.lstm(x)
    x = self.linear(x)
    return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:seq_length]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self,
                 nhead, # number of 'heads'
                 nhid,  # dimension of the feedforward network in nn.TransformerEncoder
                 nlayers, #number of encoder layers
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
                                                 dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)

        self.decoder = nn.Linear(self.nfeatures, tgt_length)
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
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.transpose(0, 1)
        if self.embed:
          src = F.relu(self.inst_embed(src))
        src = self.pos_encoder(src)
        #output = self.encoder(src, self.src_mask)
        output = self.encoder(src)
        output = self.decoder(output)
        output = output.transpose(0, 1)
        return output
