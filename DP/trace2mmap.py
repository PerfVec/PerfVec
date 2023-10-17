import sys
import os
import timeit
import subprocess


def trace2mmap(filename):
  start_time = timeit.default_timer()
  log_name = filename.replace('.txt', '.t2m.log')
  with open(log_name, 'w') as log_file:
    try:
      cmd = ['./DP/buildInstFeature', filename]
      log_file.write(str(cmd) + '\n')
      proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
      output = proc.stderr.decode()
      log_file.write(output)
      output = output.splitlines()[-3:]
      length = output[0].split()[2]
    except subprocess.CalledProcessError:
      print("Error when extracting features of ", filename)
      return 1
    try:
      in_file = filename.replace('.txt', '.in')
      cmd = ['python', '-m', 'DP.inst2mmap', '-f', '-l=' + length, in_file]
      log_file.write(str(cmd) + '\n')
      proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
      output = proc.stdout.decode()
      log_file.write(output)
    except subprocess.CalledProcessError:
      print("Error when converting", in_file, "to mmap")
      return 1
  print("Convert", filename, "using", str(timeit.default_timer()-start_time), "s", flush=True)
  output = output.splitlines()[-1].split()
  if len(output) != 7 or output[2] != length or output[4] != str(0):
    print("Error when converting", in_file, "to mmap:", output)
    return 1
  return 0


assert len(sys.argv) > 1
if len(sys.argv) > 2:
  start = int(sys.argv[2])
else:
  start = 0
nerrs = 0
for i in range(start, int(sys.argv[1])):
  name = "/mnt/md0/t2v/trace_sear_arm/t" + str(i) + ".txt"
  nerrs += trace2mmap(name)
print("There were %d errors in total." % nerrs)
