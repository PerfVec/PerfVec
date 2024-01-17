import numpy as np
import torch
from datetime import datetime
from ptflops import get_model_complexity_info
from .models import *


def print_arr(arr):
    print(', '.join('{:0.5f}'.format(i) for i in arr))


def generate_model_name(name, epoch=None):
    name = name.replace(".pt", "")
    name = name.replace(" ", "_")
    name = name.replace(",", "_")
    name = name.replace(".", "_")
    name = name.replace("'", "_")
    name = name.replace("\"", "_")
    name = name.replace("(", "_")
    name = name.replace(")", "_")
    name = name.replace("[", "_")
    name = name.replace("]", "_")
    name = name.replace("-", "_")
    name = name.replace("=", "_")
    name = name.replace("True", "1")
    name = name.replace("False", "0")
    name = name.replace("__", "_")
    if len(name) > 40:
        name = name[0:40]
    if epoch is not None:
        name += "_e" + str(epoch)
    name += "_" + datetime.now().strftime("%m%d%y")
    return name


def profile_model(cfg, model, para=False):
    print("Model info:")
    if para:
        def input_constructor(res):
            x = torch.ones(()).new_empty((1, *res))
            para = torch.ones(()).new_empty((1, 1))
            return {'x': x, 'para': para}
        constructor = input_constructor
    else:
        #constructor = None
        def input_constructor(res):
            x = torch.zeros(1, *res)
            return {'x': x}
        constructor = input_constructor
    macs, params = get_model_complexity_info(model, (cfg.seq_length, cfg.input_length), as_strings=True,
                                             input_constructor=constructor, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def get_representation_dim(cfg, model):
    rep = model.extract_representation(torch.zeros(1, cfg.seq_length, cfg.input_length))
    print("Representation dimensionality is", rep.shape[2])
    return rep.shape[2]
