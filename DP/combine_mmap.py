import sys
import os
import argparse
import numpy as np
from ML.cfg import data_item_format, seq_length, inst_length, data_set_dir, datasets


parser = argparse.ArgumentParser(description="Combine memmap datasets")
parser.add_argument('-n', '--file-num', type=int, default=0)
parser.add_argument('-c', '--chunk-size', type=int, default=16384)
parser.add_argument('--norm', action='store_true', default=False)
args = parser.parse_args()
file_num = args.file_num
chunk_size = args.chunk_size

nlines = 0
nfilled = 0
nseqs = 0
ninsts = 0
bad_lines = 0

total_size = 0
mm_sets = []
for i in range(file_num):
  total_size += datasets[i][1]
  arr = np.memmap(datasets[i][0], dtype=data_item_format, mode='r',
                  shape=(datasets[i][1], seq_length, inst_length))
  mm_sets.append(arr)
output = os.path.join(data_set_dir, "all.mmap")
shp = (total_size, seq_length, inst_length)
chunk_num = total_size // chunk_size
print("Make memmap dataset", output, " shape is", shp, chunk_num, 'chunks',flush=True)
all_data = np.memmap(output, dtype=data_item_format, mode='w+', shape=shp)

chunks = []
bounds = [0]
cum_size = 0
for i in range(file_num):
  if i != file_num - 1:
    frac = round(datasets[i][1] / total_size * chunk_size)
    if frac > datasets[i][1] / chunk_num:
      frac = datasets[i][1] / chunk_num
  else:
    frac = chunk_size - cum_size
  print(datasets[i][0], 'has %d entries in a chunk of %d.' % (frac, chunk_size))
  assert frac * chunk_num <= datasets[i][1]
  cum_size += frac
  chunks.append(frac)
  bounds.append(cum_size)

for c in range(chunk_num):
  for i in range(file_num):
    all_data[c*chunk_size + bounds[i]:c*chunk_size + bounds[i+1]] = mm_sets[i][c*chunks[i]:(c+1)*chunks[i]]
  print('.', end='', flush=True)
print(flush=True)

last_start = chunk_num*chunk_size
for i in range(file_num):
  last_size = datasets[i][1] - chunk_num*chunks[i]
  print('Lastly fill', last_size, 'from', datasets[i][0])
  all_data[last_start:last_start + last_size] = mm_sets[i][chunk_num*chunks[i]:datasets[i][1]]
  last_start += last_size
assert last_start == total_size

if args.norm:
  print('Normalize data.')
  stats = np.load(data_set_dir + "stats.npz")
  mean = stats['mean']
  std = stats['std']
  std[std == 0.0] = 1.0
  all_data[:, :, input_start:inst_length] -= mean
  all_data[:, :, input_start:inst_length] /= std

all_data.flush()
print('Done.')
