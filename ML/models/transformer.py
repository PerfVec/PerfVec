import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from CFG import seq_length, input_length, tgt_length, data_set_dir


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
        return x[:, -1, :]

    def forward(self, x):
        x = self.extract_representation(x)
        x = self.linear(x)
        return x
