import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import seq_length, inst_length


class SeqLSTM(nn.Module):
  def __init__(self, nout, nhidden, nlayers, nembed = 0):
    super(SeqLSTM, self).__init__()

    if nembed != 0:
      self.inst_embed = nn.Linear(inst_length, nembed)
      #self.inst_embed = nn.Sequential(
      #    nn.Linear(inst_length, nembed)
      #    F.relu(),
      #)
    else:
      self.inst_embed = nn.Identity()
    self.lstm = nn.LSTM(nembed, nhidden, nlayers)
    self.linear = nn.Linear(nhidden, nout)

  def init_hidden(self):
    # type: () -> Tuple[nn.Parameter, nn.Parameter]
    return (
      nn.Parameter(torch.zeros(1, 1, nhidden, requires_grad=True)),
      nn.Parameter(torch.zeros(1, 1, nhidden, requires_grad=True)),
    )

  def forward(self, src):
    src = src.view(-1, context_length, inst_length)
    x = self.inst_embed(src)
    x = x.transpose(0, 1)
    x, _ = self.lstm(x)
    x = self.linear(x)
    return x
