import sys
import os
import torch


def make_prog_dataset(prep_file, perf_file, name, min_time=0.01, std_percent=0.1):
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


benchmark_list = [
  #('503.bwaves_r', '../run_base_test_mytest-64.0000/bwaves_r_base.mytest-64 bwaves_1 < bwaves_1.in'),
  ('507.cactuBSSN_r', '../run_base_test_mytest-64.0000/cactusBSSN_r_base.mytest-64 spec_test.par'),
  ('508.namd_r', '../run_base_test_mytest-64.0000/namd_r_base.mytest-64 --input apoa1.input --iterations 1 --output apoa1.test.output'),
  ('519.lbm_r', '../run_base_test_mytest-64.0000/lbm_r_base.mytest-64 20 reference.dat 0 1 100_100_130_cf_a.of'),
  ('521.wrf_r', '../run_base_test_mytest-64.0000/wrf_r_base.mytest-64'),
  #('526.blender_r', '../run_base_test_mytest-64.0000/blender_r_base.mytest-64 cube.blend --render-output cube_ --threads 1 -b -F RAWTGA -s 1 -e 1 -a'),
  #('527.cam4_r', '../run_base_test_mytest-64.0000/cam4_r_base.mytest-64'),
  ('538.imagick_r', '../run_base_test_mytest-64.0000/imagick_r_base.mytest-64 -limit disk 0 test_input.tga -shear 25 -resize 640x480 -negate -alpha Off test_output.tga'),
  ('544.nab_r', '../run_base_test_mytest-64.0000/nab_r_base.mytest-64 hkrdenq 1930344093 1000'),
  ('549.fotonik3d_r', '../run_base_test_mytest-64.0000/fotonik3d_r_base.mytest-64'),
  #('554.roms_r', '../run_base_test_mytest-64.0000/roms_r_base.mytest-64 < ocean_benchmark0.in.x'),
  ('997.specrand_fr', '../run_base_test_mytest-64.0000/specrand_fr_base.mytest-64 324342 24239'),
  ('500.perlbench_r', '../run_base_test_mytest-64.0000/perlbench_r_base.mytest-64 -I. -I./lib makerand.pl'),
  ('502.gcc_r', '../run_base_test_mytest-64.0000/cpugcc_r_base.mytest-64 t1.c -O3 -finline-limit=50000 -o t1.opts-O3_-finline-limit_50000.s'),
  ('505.mcf_r', '../run_base_test_mytest-64.0000/mcf_r_base.mytest-64 inp.in'),
  ('523.xalancbmk_r', '../run_base_test_mytest-64.0000/cpuxalan_r_base.mytest-64 -v test.xml xalanc.xsl'),
  #('525.x264_r', '../run_base_test_mytest-64.0000/x264_r_base.mytest-64 --dumpyuv 50 --frames 156 -o BuckBunny_New.264 BuckBunny.yuv 1280x720'),
  ('531.deepsjeng_r', '../run_base_test_mytest-64.0000/deepsjeng_r_base.mytest-64 test.txt'),
  ('548.exchange2_r', '../run_base_test_mytest-64.0000/exchange2_r_base.mytest-64 0'),
  ('557.xz_r', '../run_base_test_mytest-64.0000/xz_r_base.mytest-64 cpu2006docs.tar.xz 4 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1548636 1555348 0'),
  ('999.specrand_ir', '../run_base_test_mytest-64.0000/specrand_ir_base.mytest-64 324342 24239')
]


def make_spec_dataset(prep_pre, prep_suf, perf_file, out_name, min_time=0.001, std_percent=0.1):
  perf = torch.load(perf_file, map_location=torch.device('cpu'))
  print(perf.shape)
  print("There are %d initial samples." % perf.shape[0])
  mean = torch.mean(perf, dim=1)
  std = torch.std(perf, dim=1)
  norm_std = std / mean
  idx = -1
  arr = None
  for name, cmd in benchmark_list:
    prep_file = prep_pre + '_' + name + '_' + prep_suf
    idx += 1
    if not os.path.isfile(prep_file):
      print("Skip %s due to the missing representation file." % name)
      continue
    print("Process %s." % name)
    preps = torch.load(prep_file, map_location=torch.device('cpu'))
    if mean[idx] < min_time:
      print("Skip %s due to too low runtime." % name)
      continue
    if norm_std[idx] > std_percent:
      print("%s has a large normalized std." % name)
      maxone = torch.max(perf[idx])
      minone = torch.min(perf[idx])
      if maxone - mean[idx] > mean[idx] - minone:
        print("Remove max %f." % maxone)
        removed_idx = torch.argmax(perf[idx])
      else:
        print("Remove min %f." % minone)
        removed_idx = torch.argmin(perf[idx])
      reduced_perf = torch.cat((perf[idx, :removed_idx], perf[idx, removed_idx+1:]), 0)
      new_std = torch.std(reduced_perf)
      new_mean = torch.mean(reduced_perf)
      new_norm_std = new_std / new_mean
      if new_norm_std > std_percent:
        print("Still has a large normalized std after removing, give up.")
        continue
      mean[idx] = new_mean
    entry = torch.cat((preps, mean[idx:idx+1]), 0)
    if arr is None:
      arr = entry.reshape(1, -1)
    else:
      arr = torch.cat((arr, entry.reshape(1, -1)), 0)
  assert idx == perf.shape[0] - 1
  print(arr.shape, arr)
  torch.save(arr, out_name + ".pt")


if __name__ == '__main__':
  if len(sys.argv) == 4:
    make_prog_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
  elif len(sys.argv) == 5:
    make_spec_dataset(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
