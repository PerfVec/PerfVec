import sys
import time
import os
import numpy as np
import argparse
from ML.cfg import data_item_format, seq_length, inst_length, input_start, input_length, data_set_dir, data_set_idx, datasets

parser = argparse.ArgumentParser(description="Compute normalization factors")
args = parser.parse_args()

t1 = time.time()

all_sum = np.zeros(input_length)
all_mean = np.zeros(input_length)
all_std = np.zeros(input_length)

fname = datasets[data_set_idx][0]
seqs = datasets[data_set_idx][1]
print("Open", fname)
data = np.memmap(fname, dtype=data_item_format, mode='r',
                 shape=(seqs, seq_length, inst_length))
for i in range(input_length):
  all_mean[i] = np.mean(data[:, :, input_start + i])
  all_std[i] = np.std(data[:, :, input_start + i])
  print(all_mean[i], all_std[i])

print("Global mean is %s" % str(all_mean))
print("Global std is %s" % (str(all_std), np.linalg.norm(all_std), np.sum(all_std)))
print("Took %f to compute" % (time.time() - t1))

np.savez("%s/stats" % data_set_dir, mean=all_mean, std=all_std)
np.savetxt("%s/mean.txt" % data_set_dir, all_mean)
np.savetxt("%s/std.txt" % data_set_dir, all_std)
