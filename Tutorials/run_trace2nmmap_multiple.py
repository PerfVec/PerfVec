import os
import glob
import argparse

def run(args):
  in_dir  = args.in_dir
  out_dir = args.out_dir
  for trace_path in sorted(glob.glob(os.path.join(in_dir, '*'))):
    cmd = "python -m DP.trace2nmmap " + trace_path
    print(cmd)
    os.system(cmd)
    if out_dir != "":
      file_name = os.path.basename(trace_path)
      cmd = "mv " + trace_path + " " + os.path.join(out_dir, file_name)
      os.system(cmd)
      print(cmd)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="run trace2nmmap on all traces under in_dir at once")
  parser.add_argument('--in_dir',  type=str, required=True)
  parser.add_argument('--out_dir', type=str, default="")
  args = parser.parse_args()
  run(args)
  
