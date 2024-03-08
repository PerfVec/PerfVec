import numpy as np

# Data set configuration.
data_set_dir = 'Data4/'
data_set_idx = 2
datasets = [
  (data_set_dir + '502.gcc_r.in.mmap.norm', 16220981, 69094),
  (data_set_dir + '523.xalancbmk_r.in.mmap.norm', 385894391, 1000130)
]

sim_datasets = datasets

def get_out_name(name):
  return name.replace("in.mmap.norm", "out0922.mmap")

feature_format = np.float32
target_format = np.uint16
# total batch number is 1,069,224 / 4096 = 261.04
# training set size = 200 * 4096 = 819200
testbatchnum = 232
testbatchsize = 16
validbatchnum = 200
validbatchsize = 32

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
  y = y[:, :, 2]
  return y
