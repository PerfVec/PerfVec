import sys
import argparse


def parse_testlog(logname):
  stage = 0
  with open(logname) as f:
    for line in f:
      # Find out loss.
      if stage == 0 and line.startswith("Test set: Loss:"):
        loss = float(line.split()[3])
        print("Loss:", loss)
        stage += 1
      # Parse simulation results.
      elif stage == 1:
        if line.startswith("Run"):
          print("Avg abs err: %f \tavg err: %f" % (err_abs_avg, err_avg))
          stage += 1
        elif line.find("Error abs avg, norm abs avg,") != -1:
          line = line.split()
          err_abs_avg = float(line[12])
          err_avg = float(line[13])
      # Parse simulation results.
      elif stage == 2:
        if line.startswith("Use the maximum"):
          ben_name = pre_line.strip()
        elif line.startswith("Mean error"):
          mean_err = float(line.split()[2].split('[')[1].split(']')[0])
        elif line.startswith("Mean normalized error"):
          mean_norm_err = float(line.split()[3].split('[')[1].split(']')[0])
          print("%s, \t%.4f, \t%.4f" % (ben_name, mean_err, mean_norm_err))
        pre_line = line


if __name__ == '__main__':
  # Settings
  parser = argparse.ArgumentParser(description='PerfVec Test Log Parser')
  parser.add_argument('log', nargs='*')
  args = parser.parse_args()
  assert len(args.log) == 1
  parse_testlog(args.log[0])
