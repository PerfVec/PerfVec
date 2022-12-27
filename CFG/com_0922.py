import numpy as np

# Data set configuration.
data_set_dir = 'Data4/'
data_set_idx = 8
#data_set_idx = 1
datasets = [
  (data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 1035397081, 16511420),
  (data_set_dir + '508.namd_r.in.mmap.norm', 1105247403, 1105245194),
  (data_set_dir + '500.perlbench_r.in.mmap.norm', 1184866587, 1184864557),
  (data_set_dir + '502.gcc_r.in.mmap.norm', 16220981, 69094),
  (data_set_dir + '519.lbm_r.in.mmap.norm', 1104042304, 1103901340),
  (data_set_dir + '521.wrf_r.in.mmap.norm', 1084282512, 329386),
  (data_set_dir + '505.mcf_r.in.mmap.norm', 1045214900, 1044819469),
  (data_set_dir + '523.xalancbmk_r.in.mmap.norm', 385894391, 1000130)
]

sim_datasets = [
  datasets[2],
  datasets[3],
  datasets[0],
  datasets[1],
  datasets[6],
  datasets[4],
  datasets[5],
  datasets[7]
]

def get_out_name(name):
  return name.replace("in.mmap.norm", "out0922.mmap")

feature_format = np.float32
target_format = np.uint16
if data_set_idx == 8:
  # total batch number is 4,456,740,590 / 4096 = 1,088,071.43
  # training set size = 921600 * 4096 = 3774873600
  testbatchnum = 972800
  testbatchsize = 1024
  validbatchnum = 921600
  validbatchsize = 51200
elif data_set_idx == 4:
  # total batch number is 2,306,690,265 / 4096 = 563,156.80
  # training set size = 524288 * 4096 = 2147483648
  testbatchnum = 549888
  testbatchsize = 1024
  validbatchnum = 524288
  validbatchsize = 25600
elif data_set_idx == 1:
  datasets = datasets[4:5]
  # total batch number is 1103901340 / 4096 = 269,507.16
  # training set size = 243000 * 4096 = 995328000
  testbatchnum = 256500
  testbatchsize = 1024
  validbatchnum = 243000
  validbatchsize = 13500

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 51
#tgt_length = 16
tgt_length = 1
cfg_num = 77
seq_length = 256
ori_tgt_length = 16

def sel_batch_out(y):
  y = y.reshape((-1, cfg_num, ori_tgt_length))
  #y = y[:, :, 0:tgt_length].reshape((-1, cfg_num * tgt_length))
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  y = y[:, :, 2]
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  return y
