import sys
import os
import argparse
import numpy as np
from CFG import data_item_format, seq_length, input_start, inst_length, data_set_dir, datasets, data_set_idx


parser = argparse.ArgumentParser(description="Normalize memmap datasets")
args = parser.parse_args()

total_size = datasets[data_set_idx][1]
shp = (total_size, seq_length, inst_length)
output = datasets[data_set_idx][0] + '.norm'
all_data = np.memmap(datasets[data_set_idx][0], dtype=data_item_format, mode='r', shape=shp)
norm_data = np.memmap(output, dtype=np.float32, mode='w+', shape=shp)
print("Normalize memmap dataset", datasets[data_set_idx][0], "shape is", shp, flush=True)

stats = np.load(data_set_dir + "stats.npz")
mean = stats['mean']
std = stats['std']
std[std == 0.0] = 1.0
norm_data[:, :, :] = all_data[:, :, :]
norm_data[:, :, input_start:inst_length] -= mean
norm_data[:, :, input_start:inst_length] /= std

norm_data.flush()
print(norm_data[0, 0])
print('Done.')
