import numpy as np

# Data set configuration.
data_set_dir = 'Data_mm/'
data_set_idx = 8
datasets = [
  (data_set_dir + 'mmt1.in.mmap.norm', 1228059161, 1228059161),
  (data_set_dir + 'mmt2.in.mmap.norm', 334848793, 334848793),
  (data_set_dir + 'mmt4.in.mmap.norm', 146934681, 146934681),
  (data_set_dir + 'mmt8.in.mmap.norm', 85111769, 85111769),
  (data_set_dir + 'mmt16.in.mmap.norm', 63606060, 63606060),
  (data_set_dir + 'mmt32.in.mmap.norm', 54720764, 54720764),
  (data_set_dir + 'mmt64.in.mmap.norm', 50684532, 50684532),
  (data_set_dir + 'mmt128.in.mmap.norm', 48760507, 48760507)
]

sim_datasets = datasets

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.mmap")

feature_format = np.float32
target_format = np.uint16
if data_set_idx == 8:
  # total batch number is 2,012,726,267 / 4096 = 491,388.25
  testbatchnum = 475000
  testbatchsize = 1024
  validbatchnum = 450000
  validbatchsize = 25000

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 51
#tgt_length = 16
tgt_length = 1
cfg_num = 1
seq_length = 256
ori_tgt_length = 16

def sel_batch_out(y):
  y = y.reshape((-1, cfg_num, ori_tgt_length))
  #y = y[:, :, 0:tgt_length].reshape((-1, cfg_num * tgt_length))
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  y = y[:, :, 2]
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  return y

def sel_output(y):
  y = y.reshape((-1, 77, tgt_length))
  y = y[:, 70, :]
  return y
