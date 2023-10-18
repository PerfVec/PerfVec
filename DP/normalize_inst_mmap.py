import sys
import os
import argparse
import numpy as np
#from CFG import data_item_format, input_start, inst_length, data_set_dir, datasets, data_set_idx
#from CFG import feature_format, input_length, data_set_dir, data_set_idx, datasets
from CFG import feature_format, input_length, sim_datasets as datasets


def norm_inmmap(filename, length, width, fileformat, mean, std):
  shp = (length, width)
  output = filename + '.norm'
  all_data = np.memmap(filename, dtype=fileformat, mode='r', shape=shp)
  norm_data = np.memmap(output, dtype=fileformat, mode='w+', shape=shp)
  print("Normalize memmap dataset", filename, "shape is", shp, flush=True)

  norm_data[:, :] = all_data[:, :]
  norm_mmap(norm_data, mean, std)

  print(norm_data[0])
  print('Done.')


def norm_mmap(data, mean, std):
  #data[:, input_start:inst_length] -= mean
  #data[:, input_start:inst_length] /= std
  data[:, :] -= mean
  data[:, :] /= std
  data.flush()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Normalize memmap datasets")
  args = parser.parse_args()

  #stats = np.load(data_set_dir + "stats.npz")
  stats = np.load("Data/stats.npz")
  mean = stats['mean']
  std = stats['std']
  std[std == 0.0] = 1.0
  for j in range(len(datasets)):
    norm_inmmap(datasets[j][0], datasets[j][1], input_length, feature_format, mean, std)
