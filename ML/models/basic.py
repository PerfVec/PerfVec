import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RepExtractor(nn.Module):
  def __init__(self, model):
    super(RepExtractor, self).__init__()
    self.model = model

  def forward(self, x):
    rep = self.model.extract_representation(x)
    return rep


class MultiModel(nn.Module):
  def __init__(self, *args):
    super(MultiModel, self).__init__()
    self.models = nn.ModuleList()
    for model in args:
      self.models.append(model)

  def forward(self, x):
    for model in self.models:
      x = model(x)
    return x
