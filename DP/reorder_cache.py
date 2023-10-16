import sys
import time
import os
import numpy as np
import argparse
import torch
from CFG.com_cache_test_1122 import data_set_dir, sim_datasets


parser = argparse.ArgumentParser(description="Reorder cache data")
parser.add_argument('--cfg', required=True, help='config file')
parser.add_argument('--checkpoints', required=True)
args = parser.parse_args()

for i in range(len(sim_datasets)):
  name = sim_datasets[i][0].replace(data_set_dir, '').replace(".in.mmap.norm", '')
  file_name = args.checkpoints.replace("checkpoints/", "res/sim_%s_%s_" % (args.cfg, name))
  old_data = torch.load(file_name, map_location=torch.device('cpu'))
  new_data = torch.zeros(old_data.shape)
  print(name, new_data.shape, flush=True)
  new_data[0:6] = old_data[0:6]

  new_data[6] = old_data[33]
  new_data[7] = old_data[32]
  new_data[8] = old_data[30]
  new_data[9] = old_data[31]
  new_data[10] = old_data[34]
  new_data[11] = old_data[35]

  new_data[12] = old_data[15]
  new_data[13] = old_data[14]
  new_data[14] = old_data[12]
  new_data[15] = old_data[13]
  new_data[16] = old_data[16]
  new_data[17] = old_data[17]

  new_data[18] = old_data[21]
  new_data[19] = old_data[20]
  new_data[20] = old_data[18]
  new_data[21] = old_data[19]
  new_data[22] = old_data[22]
  new_data[23] = old_data[23]

  new_data[24] = old_data[27]
  new_data[25] = old_data[28]
  new_data[26] = old_data[24]
  new_data[27] = old_data[25]
  new_data[28] = old_data[26]
  new_data[29] = old_data[29]

  new_data[30:33] = old_data[9:12]
  new_data[33:36] = old_data[6:9]
  for j in range(new_data.shape[0]):
    print(new_data[j].item())
