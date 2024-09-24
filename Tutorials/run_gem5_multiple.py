import os
import random
import argparse

# 4-digits serial number for each sample by default, i.e. max <# of samples> = 9999
# change it when needed.
leading_zero = 4

def run(args):
  """ 
    The driver function to run gem5 simulation and get <args.samples> of traces on sampled uarchs. 
    Program to be simulated should be specified, although this script takes default.
  """
  num_samples = args.num_samples
  out_dir = args.out_dir 
  gem5_opt_file = args.gem5_opt_file
  gem5_config_file = args.gem5_config_file
  gem5_cpu_params = args.gem5_cpu_params
  prog_cmd = args.prog_cmd

  # params for gem5 simulation
  
  for i in range(num_samples):
    # run gem5 to generate trace
    cmd = gem5_opt_file + " " + gem5_config_file + " " + gem5_cpu_params + " \"" + prog_cmd + "\""
    os.system(cmd)

    # file rename w.r.t i
    cmd = "mv "+os.path.join(".", "sq.trace.txt")+" "+os.path.join(out_dir, "trace_sample_"+str(i).zfill(leading_zero)+".sq.txt")
    os.system(cmd)
    print("output file by: ", cmd)
    cmd = "mv "+os.path.join(".", "trace.txt")+" "+os.path.join(out_dir, "trace_sample_"+str(i).zfill(leading_zero)+".txt")
    os.system(cmd)
    print("output file by: ", cmd)


if __name__ == '__main__':
  # NOTE: change accordingly
  home_dir    = os.path.expanduser( '~' )
  gem5_dir    = os.path.join(home_dir, "playground", "gem5")
  perfvec_dir  = os.path.join(home_dir, "PerfVec")
  #prog_cmd      = os.path.join(perfvec_dir, "Tutorials", "mm") + " 8 8 8"      
  prog_cmd      = os.path.join(gem5_dir, "tests", "test-progs", "hello", "bin", "arm", "linux", "hello")       
  
  gem5_opt_file = os.path.join(gem5_dir, "build", "ARM", "gem5.opt")
  gem5_config_file   = os.path.join(gem5_dir, "configs", "example", "arm", "starter_se.py")

  s = 2024
  random.seed(s)
  r = random.randint(-2147483648, 2147483647)
  gem5_cpu_params    = "--cpu=timing -r="+str(r)
  
  parser = argparse.ArgumentParser(description="trace generation driver, specify number of samples on random uarch. configurations.")
  parser.add_argument('--num_samples', type=int, required=True)
  parser.add_argument('--gem5_opt_file', type=str, default=gem5_opt_file)
  parser.add_argument('--gem5_config_file', type=str, default=gem5_config_file)
  parser.add_argument('--gem5_cpu_params', type=str, default=gem5_cpu_params)
  parser.add_argument('--out_dir', type=str, default="./") # user should make sure the output file exists
  parser.add_argument('--prog_cmd', type=str, default=prog_cmd)
  args = parser.parse_args()

  run(args)	
