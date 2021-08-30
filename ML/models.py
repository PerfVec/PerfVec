import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import seq_length, input_length, tgt_length


class SeqLSTM(nn.Module):
  def __init__(self, nhidden, nlayers, nembed = 0):
    super(SeqLSTM, self).__init__()

    if nembed != 0:
      self.inst_embed = nn.Linear(input_length, nembed)
      #self.inst_embed = nn.Sequential(
      #    nn.Linear(input_length, nembed)
      #    F.relu(),
      #)
      self.lstm = nn.LSTM(nembed, nhidden, nlayers)
    else:
      self.inst_embed = nn.Identity()
      self.lstm = nn.LSTM(input_length, nhidden, nlayers)
    self.linear = nn.Linear(nhidden, tgt_length)

  def init_hidden(self):
    # type: () -> Tuple[nn.Parameter, nn.Parameter]
    return (
      nn.Parameter(torch.zeros(1, 1, nhidden, requires_grad=True)),
      nn.Parameter(torch.zeros(1, 1, nhidden, requires_grad=True)),
    )

  def forward(self, src):
    src = src.view(-1, seq_length, input_length)
    x = self.inst_embed(src)
    x = x.transpose(0, 1)
    x, _ = self.lstm(x)
    x = x.transpose(0, 1)
    x = self.linear(x)
    return x
