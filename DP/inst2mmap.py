import sys
import os
import argparse
import numpy as np
from CFG import *

parser = argparse.ArgumentParser(description="Transform from text to memmap dataset")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=0)
parser.add_argument('-l', '--length', type=int, default=0)
parser.add_argument('-f', '--feature', action='store_true', default=False)
parser.add_argument('-t', '--target', action='store_true', default=False)
parser.add_argument('fname', nargs='*')
args = parser.parse_args()

if args.feature:
  inst_length = input_length
  data_item_format = feature_format
elif args.target:
  inst_length = cfg_num * ori_tgt_length
  data_item_format = target_format
else:
  inst_length = input_length + tgt_length
  data_item_format = data_item_format
start = args.start
end = args.end
output = args.fname[0]
if len(args.fname) > 1:
  output = os.path.join(os.path.dirname(args.fname[0]), "all")
output += ".mmap"
shp = (args.length, inst_length)

print("Make memmap dataset", output, "start from", start, " end with", end, " shape is ", shp, flush=True)

nlines = 0
ninsts = 0
bad_lines = 0
all_feats = np.memmap(output, dtype=data_item_format, mode='w+', shape=shp)

for i in range(len(args.fname)):
  fname = args.fname[i]
  print("read", fname, flush=True)
  with open(fname) as f:
    for line in f:
      if ninsts >= args.length:
        print("Find more instructions in the input.")
        break
      if nlines < start:
        nlines += 1
        continue

      try:
        #vals = [int(val) for val in line.split()]
        vals = [float(val) for val in line.split()]
        assert len(vals) == inst_length
      except:
        print("Bad line:", len(vals), vals, flush=True)
        bad_lines += 1
        continue

      all_feats[ninsts] = np.array(vals)

      if ninsts == 0:
        print("First sample:", all_feats[ninsts].shape, len(vals))
        print(all_feats[ninsts], flush=True)
      nlines += 1
      ninsts += 1
      if ninsts % 1000000 == 0:
        all_feats.flush()
        print('.', flush=True, end='')
      if end != 0 and nlines == end:
        break


all_feats.flush()
print()
print("Finished with", ninsts, "instructions,", bad_lines, "bad lines.")
