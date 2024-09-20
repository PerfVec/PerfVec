import sys
import os
import timeit
import subprocess
import numpy as np
from DP.inst2mmap import inst2mmap
from DP.normalize_inst_mmap import norm_mmap


def trace2nmmap(filename, is_norm, mean, std):
  start_time = timeit.default_timer()
  log_name = filename.replace('.txt', '.t2n.log')
  with open(log_name, 'w') as log_file:
    try:
      cmd = ['./DP/buildInstFeature', filename]
      log_file.write(str(cmd) + '\n')
      proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
      output = proc.stderr
      log_file.write(output)
      output = output.splitlines()[-3:]
      length = int(output[0].split()[2])
    except subprocess.CalledProcessError as e:
      print("Error when extracting features of ", filename)
      log_file.write(str(repr(e.stderr)) + '\n')
      return 1
    in_file = filename.replace('.txt', '.in')
    if is_norm:
      out_file = in_file + ".nmmap"
    else:
      out_file = in_file + ".mmap"
    array, ninsts, bad_lines = inst2mmap([in_file], length, out_file, 0, 0, True, False, log_file)
    if ninsts != length or bad_lines != 0:
      print("Error when converting", in_file, "to mmap:", output)
      return 1
    if is_norm:
      norm_mmap(array, mean, std)
      print(array[0], file=log_file)
      print('Normalization done.', file=log_file)
  print("Convert", filename, "with", ninsts, "instructions using", str(timeit.default_timer()-start_time), "s", flush=True)
  return 0


if __name__ == '__main__':
  assert len(sys.argv) > 1
  if len(sys.argv) > 2:
    start = int(sys.argv[2])
  else:
    start = 0
  stats = np.load("Tutorials/stats.npz")
  mean = stats['mean']
  std = stats['std']
  std[std == 0.0] = 1.0
  nerrs = 0
  name = sys.argv[1]
  nerrs += trace2nmmap(name, True, mean, std)
  #for i in range(start, int(sys.argv[1])):
  #  name = "/mnt/md0/t2v/trace_sear_arm/t" + str(i) + ".txt"
  #  nerrs += trace2nmmap(name, True, mean, std)
  print("There were %d errors in total." % nerrs)
