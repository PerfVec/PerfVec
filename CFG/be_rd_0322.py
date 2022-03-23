import numpy as np

# Data set configuration.
data_set_dir = 'Data3/'
data_set_idx = 4
datasets = [
  (data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 1035389426, 26450231),
  (data_set_dir + '508.namd_r.in.mmap.norm', 1105244743, 1105244724),
  (data_set_dir + '500.perlbench_r.in.mmap.norm', 1186655322, 1186655322),
  (data_set_dir + '502.gcc_r.in.mmap.norm', 16220268, 16220268)
]

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.mmap")

feature_format = np.float32
target_format = np.int16
# total batch number is 569,963.5
testbatchnum = 540000
testbatchsize = 128
validbatchnum = 500000
validbatchsize = 8192

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 49
tgt_length = 16
cfg_num = 2
seq_length = 256
