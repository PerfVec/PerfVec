import sys
import os
import argparse
import numpy as np
from CFG import *


def inst2mmap(fnames, num, output, start, end, is_feature, is_target, printout=sys.stdout):
  if is_feature:
    inst_length = input_length
    data_item_format = feature_format
  elif is_target:
    inst_length = cfg_num * ori_tgt_length
    data_item_format = target_format
  else:
    inst_length = input_length + tgt_length
    data_item_format = data_item_format
  shp = (num, inst_length)

  print("Make memmap dataset", output, "start from", start, " end with", end, " shape is ", shp, flush=True, file=printout)

  nlines = 0
  ninsts = 0
  bad_lines = 0
  all_feats = np.memmap(output, dtype=data_item_format, mode='w+', shape=shp)

  for i in range(len(fnames)):
    fname = fnames[i]
    print("read", fname, flush=True, file=printout)
    with open(fname) as f:
      for line in f:
        if ninsts >= num:
          print("Find more instructions in the input.", file=printout)
          break
        if nlines < start:
          nlines += 1
          continue

        try:
          #vals = [int(val) for val in line.split()]
          vals = [float(val) for val in line.split()]
          assert len(vals) == inst_length
        except:
          print("Bad line:", len(vals), vals, flush=True, file=printout)
          bad_lines += 1
          continue

        all_feats[ninsts] = np.array(vals)

        if ninsts == 0:
          print("First sample:", all_feats[ninsts].shape, len(vals), file=printout)
          print(all_feats[ninsts], flush=True, file=printout)
        nlines += 1
        ninsts += 1
        if ninsts % 1000000 == 0:
          all_feats.flush()
          print('.', flush=True, end='', file=printout)
        if ninsts % 100000000 == 0:
          print('', flush=True, file=printout)
        if end != 0 and nlines == end:
          break

  all_feats.flush()
  print(file=printout)
  print("Finished with", ninsts, "instructions,", bad_lines, "bad lines.", file=printout)
  return all_feats, ninsts, bad_lines


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Transform from text to memmap dataset")
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--end', type=int, default=0)
  parser.add_argument('-l', '--length', type=int, default=0)
  parser.add_argument('-f', '--feature', action='store_true', default=False)
  parser.add_argument('-t', '--target', action='store_true', default=False)
  parser.add_argument('fname', nargs='*')
  args = parser.parse_args()

  output = args.fname[0]
  if len(args.fname) > 1:
    print("Warning: more than one input files.")
    output = os.path.join(os.path.dirname(args.fname[0]), "all")
  output += ".mmap"

  inst2mmap(args.fname, args.length, output, args.start, args.end, args.feature, args.target)
