import argparse
import os
import subprocess
from datetime import datetime

bmk_list = [
  "500.perlbench_r",
  "502.gcc_r",
  "507.cactuBSSN_r",
  "508.namd_r"
]

cfg_list = [
  "trace_set_arm3",
  "trace_set_ex5",
  "trace_set_pk",
  "trace_set_o3",
  "trace_set_minor2",
  "trace_set_hpi",
  "trace_set_ex5l"
]

parser = argparse.ArgumentParser(description="Combine output of multiple configurations")
args = parser.parse_args()
trace_dir = "/lfs1/work/lli/"

log_name = os.getcwd() + '/log/comout_' + datetime.now().strftime("%m%d%y") + '.log'
with open(log_name, 'w') as log_file:
  for bmk in bmk_list:
    cmd = ["./DP/buildComOut"]
    for cfg in cfg_list:
      path = trace_dir + cfg + '/' + bmk + ".txt"
      cmd.append(path)
    print(cmd)
    log_file.write(str(cmd))
    log_file.flush()
    process = subprocess.call(cmd, stdout=log_file, stderr=log_file, universal_newlines=True)
