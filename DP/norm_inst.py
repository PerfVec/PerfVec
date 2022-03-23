import sys
import time
import os
import numpy as np
import argparse
#from CFG import data_item_format, inst_length, input_start, input_length, data_set_dir, data_set_idx, datasets
from CFG import feature_format, input_length, data_set_dir, data_set_idx, datasets

parser = argparse.ArgumentParser(description="Compute normalization factors")
args = parser.parse_args()

t1 = time.time()

all_sum = np.zeros(input_length)
all_mean = np.zeros(input_length)
all_std = np.zeros(input_length)

for i in range(input_length):
  cur_data = np.empty(0)
  for j in range(data_set_idx):
    fname = datasets[j][0]
    insts = datasets[j][1]
    #data = np.memmap(fname, dtype=data_item_format, mode='r',
    #                 shape=(insts, inst_length))
    data = np.memmap(fname, dtype=feature_format, mode='r',
                     shape=(insts, input_length))
    #cur_data = np.array([cur_data, data[:, input_start + i]])
    #cur_data = np.concatenate((cur_data, data[:, input_start + i]))
    cur_data = np.concatenate((cur_data, data[:, i]))
  print("Calculate", i, cur_data.shape, flush=True)
  all_mean[i] = np.mean(cur_data)
  all_std[i] = np.std(cur_data)
  print(all_mean[i], all_std[i], flush=True)

print("Global mean is %s" % str(all_mean))
print("Global std is %s" % str(all_std))
print("Took %f to compute" % (time.time() - t1))

np.savez("%s/stats" % data_set_dir, mean=all_mean, std=all_std)
np.savetxt("%s/mean.txt" % data_set_dir, all_mean)
np.savetxt("%s/std.txt" % data_set_dir, all_std)
