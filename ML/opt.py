import argparse
import os
import sys
import time
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .models import *
from .custom_data import *


class DSE(nn.Module):
  def __init__(self, model, prog_rep):
    super(DSE, self).__init__()
    self.uarch_net = model.uarch_net
    for param in self.uarch_net.parameters():
      param.requires_grad = False
    self.prog_rep = nn.Parameter(prog_rep)
    self.prog_rep.requires_grad = False

  def forward(self, x):
    # Legal check.
    x = torch.clamp(x, min=1, max=6)
    uarch_rep = self.uarch_net(x)
    x = F.linear(self.prog_rep, uarch_rep)
    return x


def opt_int(tgt):
  for i in range(tgt.shape[0]):
    if tgt.grad[i] > 0 and tgt.data[i] > 1:
      tgt.data[i] -= 1
    elif tgt.grad[i] < 0 and tgt.data[i] < 6:
      tgt.data[i] += 1
  return tgt


def zero_grad_int(model, tgt):
  model.zero_grad()
  if tgt.grad is not None:
    for i in range(tgt.shape[0]):
      tgt.grad[i] = 0


def load_checkpoint(name, model, training=False, optimizer=None):
    assert 'checkpoints/' in name
    cp = torch.load(name, map_location=torch.device('cpu'))
    model.load_state_dict(cp['model_state_dict'])
    if training:
        assert optimizer is not None
        optimizer.load_state_dict(cp['optimizer_state_dict'])
    print("Loaded checkpoint", name)


def main():
    # Settings
    parser = argparse.ArgumentParser(description='PerfVec Optimization')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA')
    parser.add_argument('--int-opt', action='store_true', default=False,
                        help='Integer optimizer')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--prog-rep', required=True)
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    assert len(args.models) == 1
    model = eval(args.models[0])
    model.init_paras()
    load_checkpoint(args.checkpoints, model)
    prog_reps = torch.load(args.prog_rep, map_location=torch.device('cpu'))
    for ben in range(prog_reps.shape[0]):
        prog_rep = prog_reps[ben]
        dse_model = DSE(model, prog_rep)
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda and torch.cuda.device_count() > 1:
            print ('Available devices', torch.cuda.device_count())
            print ('Current cuda device', torch.cuda.current_device())
            dse_model = nn.DataParallel(model)
        dse_model.to(device)
        paras = nn.Parameter(torch.tensor([3, 3], dtype=torch.float), requires_grad = True).to(device)
        if not args.int_opt:
            opt = torch.optim.Adam([paras], lr=0.1)

        for epoch in range(1, args.epochs + 1):
            if args.int_opt:
                zero_grad_int(dse_model, paras)
            else:
                opt.zero_grad()
            perf = dse_model(paras)
            sizes = torch.clamp(paras, min=1, max=6)
            sizes = torch.pow(2, sizes) * torch.FloatTensor([2, 128])
            obj = perf * (1000 + 10 * sizes[0] + sizes[1]) / 100000000000
            obj.backward()
            if args.int_opt:
                paras = opt_int(paras)
            else:
                opt.step()
            print(epoch, perf.item(), obj.item(), paras.grad, paras.data)

        print("Benchmark", ben, "optimization results:", paras.data)


if __name__ == '__main__':
    main()
