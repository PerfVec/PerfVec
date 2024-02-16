import numpy as np

# Data set configuration.
data_set_dir = 'Data4/'
#data_set_idx = 8
data_set_idx = 3
datasets = [
  (data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 1035397081, 15326847),
  (data_set_dir + '508.namd_r.in.mmap.norm', 1105247403, 111803697),
  (data_set_dir + '500.perlbench_r.in.mmap.norm', 1184866587, 794193),
  (data_set_dir + '502.gcc_r.in.mmap.norm', 16220981, 8606664),
  (data_set_dir + '519.lbm_r.in.mmap.norm', 1104042304, 131263455),
  (data_set_dir + '521.wrf_r.in.mmap.norm', 1084282512, 329410),
  (data_set_dir + '505.mcf_r.in.mmap.norm', 1045214900, 109655167),
  (data_set_dir + '523.xalancbmk_r.in.mmap.norm', 385894391, 1000130)
]

sim_datasets = datasets

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.mmap").replace(data_set_dir, data_set_dir + "cache/")

feature_format = np.float32
target_format = np.uint16
if data_set_idx == 8:
  # total batch number is 378,779,563 / 4096 = 92,475.48
  # training set size = 81000 * 4096 = 331776000
  testbatchnum = 85500
  testbatchsize = 1024
  validbatchnum = 81000
  validbatchsize = 4500
elif data_set_idx == 3:
  datasets = datasets[1:2] + datasets[4:5] + datasets[6:7]
  # total batch number is 352,722,319 / 4096 = 86,113.85
  # training set size = 80000 * 4096 = 327680000
  testbatchnum = 83000
  testbatchsize = 1024
  validbatchnum = 80000
  validbatchsize = 3000

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 51
#tgt_length = 16
tgt_length = 1
cfg_num = 19
seq_length = 256
ori_tgt_length = 16

def sel_batch_out(y):
  y = y.reshape((-1, cfg_num, ori_tgt_length))
  #y = y[:, :, 0:tgt_length].reshape((-1, cfg_num * tgt_length))
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  y = y[:, :cfg_num-1, 2]
  return y
