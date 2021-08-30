import sys
import os
import argparse
import numpy as np
from data_format import seq_length, inst_length

parser = argparse.ArgumentParser(description="Transform from text to memmap dataset")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=0)
parser.add_argument('-s', '--total-seqs', type=int, default=0)
parser.add_argument('fname', nargs='*')
args = parser.parse_args()

start = args.start
end = args.end
output = args.fname[0]
if len(args.fname) > 1:
  output = os.path.join(os.path.dirname(args.fname[0]), "all")
output += ".mmap"
shp = (args.total_seqs, seq_length, inst_length)

print("Make memmap dataset", output, "start from", start, " end with", end, " shape is ", shp, flush=True)

nlines = 0
nfilled = 0
nseqs = 0
ninsts = 0
bad_lines = 0
all_feats = np.memmap(output, dtype=np.uint16, mode='w+', shape=shp)

for i in range(len(args.fname)):
  fname = args.fname[i]
  print("read", fname, flush=True)
  with open(fname) as f:
    for line in f:
      if nseqs >= args.total_seqs:
        print("Find more sequences in the input.")
        break
      if nlines < start:
        nlines += 1
        continue

      try:
        vals = [int(val) for val in line.split()]
        assert len(vals) == inst_length
      except:
        print("Bad line:", len(vals), vals, flush=True)
        bad_lines += 1
        continue

      all_feats[nseqs, ninsts] = np.array(vals)

      if nfilled == 0:
        print("First sample:", all_feats[nseqs, ninsts].shape, len(vals))
        print(all_feats[nseqs, ninsts], flush=True)
      nfilled += 1
      nlines += 1
      ninsts += 1
      if ninsts == seq_length:
        ninsts = 0
        nseqs += 1
        if nseqs % 10000 == 0:
          all_feats.flush()
          print("Processed %d seqences." % nseqs, flush=True)
      if end != 0 and nlines == end:
        assert ninsts == 0
        break
  assert ninsts == 0


all_feats.flush()
print("Finished with", nseqs, "seqences,", nfilled, "lines,", bad_lines, "bad lines.")
