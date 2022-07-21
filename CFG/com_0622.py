import numpy as np

# Data set configuration.
data_set_dir = 'Data4/'
data_set_idx = 8
datasets = [
  (data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 1035397081, 24818298),
  (data_set_dir + '508.namd_r.in.mmap.norm', 1105247403, 1105245197),
  (data_set_dir + '500.perlbench_r.in.mmap.norm', 1184866587, 1184864561),
  (data_set_dir + '502.gcc_r.in.mmap.norm', 16220981, 8606664),
  (data_set_dir + '519.lbm_r.in.mmap.norm', 1104042304, 1103901432),
  (data_set_dir + '521.wrf_r.in.mmap.norm', 1084282512, 423028515),
  (data_set_dir + '505.mcf_r.in.mmap.norm', 1045214900, 1044819469),
  (data_set_dir + '523.xalancbmk_r.in.mmap.norm', 385894391, 1000130)
]

sim_datasets = [
  #(data_set_dir + '508.namd_r.in.mmap.norm', 1105247403, 1105245197),
  #(data_set_dir + '500.perlbench_r.in.mmap.norm', 1184866587, 1184864561)
  #(data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 1035397081, 24818298),
  #(data_set_dir + '502.gcc_r.in.mmap.norm', 16220981, 8606664)
  (data_set_dir + '519.lbm_r.in.mmap.norm', 1104042304, 1103901432),
  (data_set_dir + '521.wrf_r.in.mmap.norm', 1084282512, 423028515),
  (data_set_dir + '505.mcf_r.in.mmap.norm', 1045214900, 1044819469)
  #(data_set_dir + '523.xalancbmk_r.in.mmap.norm', 385894391, 1000130)
]

def get_out_name(name):
  return name.replace("in.mmap.norm", "out0622.mmap")

feature_format = np.float32
target_format = np.int16
if data_set_idx == 8:
  # total batch number is 4,896,284,266 / 4096 = 1,195,381.90
  # training set size = 1048576 * 4096 = 4294967296
  testbatchnum = 1114112
  testbatchsize = 1024
  validbatchnum = 1048576
  validbatchsize = 65536
elif data_set_idx == 4:
  # total batch number is 2,323,534,720 / 4096 = 567,269.22
  # training set size = 524288 * 4096 = 2147483648
  testbatchnum = 557056
  testbatchsize = 1024
  validbatchnum = 524288
  validbatchsize = 32768

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 51
tgt_length = 16
cfg_num = 7
seq_length = 256
#sel_tgt_length = 12

#def sel_batch_out(y):
#  y = y.view(-1, cfg_num, tgt_length)
#  #y = y[:, :, 0:sel_tgt_length].view(-1, cfg_num * sel_tgt_length)
#  y = np.concatenate((y[:, :, 0:2], y[:, :, 6:tgt_length]), axis=2).view(-1, cfg_num * sel_tgt_length)
#  return y
