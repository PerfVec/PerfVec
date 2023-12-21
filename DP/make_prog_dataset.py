import sys
import os
import torch


def make_prog_dataset(prep_file, perf_file, name, min_time=0.001, std_percent=0.1):
  preps = torch.load(prep_file, map_location=torch.device('cpu'))
  perf = torch.load(perf_file, map_location=torch.device('cpu'))
  print(preps.shape, perf.shape)
  print("There are %d initial samples." % preps.shape[0])
  assert preps.shape[0] == perf.shape[0]
  mean = torch.mean(perf, dim=2)
  std = torch.std(perf, dim=2)
  arr = torch.cat((preps, mean[:, 0:1]), 1)
  arr = arr[mean[:, 0] >= min_time]
  std = std[mean[:, 0] >= min_time]
  mean = mean[mean[:, 0] >= min_time]
  #print(mean[:, 0] >= min_time)
  print("There are %d samples whose running time >= %f." % (arr.shape[0], min_time))
  norm_std = std / mean
  arr = arr[norm_std[:, 0] <= std_percent]
  print("There are %d samples whose normalized std <= %f." % (arr.shape[0], std_percent))
  #print("Make a dataset with %d samples" % num)
  print(arr.shape, arr)
  torch.save(arr, name + ".pt")


if __name__ == '__main__':
  assert len(sys.argv) == 4
  make_prog_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
