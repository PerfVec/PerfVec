import numpy as np

# Data set configuration.
data_set_dir = 'Data_npb/'
#data_set_idx = 9
data_set_idx = 4
datasets = [
  (data_set_dir + 'bt.S.in.mmap.norm', 393218190, 118488715),
  (data_set_dir + 'cg.S.in.mmap.norm', 324132434, 63506089),
  (data_set_dir + 'ep.S.in.mmap.norm', 1090735168, 109187326),
  (data_set_dir + 'ft.S.in.mmap.norm', 609379711, 105854300),
  (data_set_dir + 'is.S.in.mmap.norm', 31739966, 31719304),
  (data_set_dir + 'lu.S.in.mmap.norm', 133483557, 117122425),
  (data_set_dir + 'mg.S.in.mmap.norm', 24079669, 7363287),
  (data_set_dir + 'sp.S.in.mmap.norm', 180274273, 116625259),
  (data_set_dir + 'ua.S.in.mmap.norm', 1154586606, 115464669)
]

sim_datasets = datasets

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.mmap").replace(data_set_dir, data_set_dir + "uarch/")

feature_format = np.float32
target_format = np.uint16
if data_set_idx == 9:
  # total batch number is 785,331,374 / 4096 = 191,731.29
  # training set size = 180000 * 4096 = 737280000
  testbatchnum = 190000
  testbatchsize = 1024
  validbatchnum = 180000
  validbatchsize = 10000
elif data_set_idx == 4:
  # total batch number is 397,036,430 / 4096 = 96,932.72
  # training set size = 90000 * 4096 = 368640000
  testbatchnum = 95000
  testbatchsize = 1024
  validbatchnum = 90000
  validbatchsize = 5000

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 51
#tgt_length = 16
tgt_length = 1
cfg_num = 11
seq_length = 256
ori_tgt_length = 16

def sel_batch_out(y):
  y = y.reshape((-1, cfg_num, ori_tgt_length))
  #y = y[:, :, 0:tgt_length].reshape((-1, cfg_num * tgt_length))
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  y = y[:, :, 2]
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  return y
