import numpy as np

# Data set configuration.
data_set_dir = 'Data_npb/'
data_set_idx = 9
datasets = [
  (data_set_dir + 'bt.S.in.mmap.norm', 393218190, 393218190),
  (data_set_dir + 'cg.S.in.mmap.norm', 324132434, 324132434),
  (data_set_dir + 'ep.S.in.mmap.norm', 1090735168, 1090735168),
  (data_set_dir + 'ft.S.in.mmap.norm', 609379711, 609379711),
  (data_set_dir + 'is.S.in.mmap.norm', 31739966, 31739966),
  (data_set_dir + 'lu.S.in.mmap.norm', 133483557, 133483557),
  (data_set_dir + 'mg.S.in.mmap.norm', 24079669, 24079669),
  (data_set_dir + 'sp.S.in.mmap.norm', 180274273, 180274273),
  (data_set_dir + 'ua.S.in.mmap.norm', 1154586606, 1154586606)
]

sim_datasets = datasets

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.mmap")

feature_format = np.float32
target_format = np.uint16
if data_set_idx == 9:
  # total batch number is 3,941,629,574 / 4096 = 962,311.91
  testbatchnum = 950000
  testbatchsize = 1024
  validbatchnum = 900000
  validbatchsize = 50000

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
