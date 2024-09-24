import argparse
import os
import subprocess
from datetime import datetime

bmk_list = [
  "trace_sample_"
]
num = 77
leading_zero = 4

parser = argparse.ArgumentParser(description="Combine output of multiple identical trace.")
args = parser.parse_args()
trace_dir = "/data1/lld/trace_arm_test/mm/"

log_name = './comout.log'
with open(log_name, 'w') as log_file:
  for bmk in bmk_list:
    cmd = ["./DP/buildComOut"]
    for n in range(num):
      path = trace_dir + '/' + bmk + str(n).zfill(leading_zero) + ".txt"
      cmd.append(path)
    print(cmd)
    log_file.write(str(cmd))
    log_file.flush()
    process = subprocess.call(cmd, stdout=log_file, stderr=log_file, universal_newlines=True)
