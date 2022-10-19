import torch
import numpy as np

# Data set configuration.
data_set_dir = 'Data4/'
#data_set_idx = 10
data_set_idx = 4
datasets = [
  (data_set_dir + '525.x264_r.in.mmap.norm', 1066019761, 1065961385),
  (data_set_dir + '531.deepsjeng_r.in.mmap.norm', 1051476976, 1049812795),
  (data_set_dir + '527.cam4_r.in.mmap.norm', 1060183121, 1048566868),
  (data_set_dir + '538.imagick_r.in.mmap.norm', 115459544, 115459544),
  (data_set_dir + '548.exchange2_r.in.mmap.norm', 1042957954, 1041812670),
  (data_set_dir + '557.xz_r.in.mmap.norm', 1043200017, 1042392179),
  (data_set_dir + '999.specrand_ir.in.mmap.norm', 59087074, 59087074),
  (data_set_dir + '544.nab_r.in.mmap.norm', 1050786112, 1050747889),
  (data_set_dir + '549.fotonik3d_r.in.mmap.norm', 1151356733, 1151355900),
  (data_set_dir + '997.specrand_fr.in.mmap.norm', 59087074, 59087074)
]

sim_datasets = [
  datasets[0],
  datasets[1],
  datasets[4],
  datasets[5],
  datasets[2],
  datasets[3],
  datasets[7],
  datasets[8]
]

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.mmap").replace(data_set_dir, data_set_dir + "prog/")

feature_format = np.float32
target_format = np.uint16
if data_set_idx == 10:
  # total batch number is 7,684,283,378 / 4096 = 1,876,045.75
  testbatchnum = 1710000
  testbatchsize = 1024
  validbatchnum = 1620000
  validbatchsize = 90000
elif data_set_idx == 4:
  # total batch number is 3,279,800,592 / 4096 = 800,732.57
  # training set size = 720000 * 4096 = 2949120000
  testbatchnum = 760000
  testbatchsize = 1024
  validbatchnum = 720000
  validbatchsize = 40000

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 51
tgt_length = 1
cfg_num = 6
seq_length = 256
ori_tgt_length = 16

def sel_batch_out(y):
  y = y.reshape((-1, cfg_num, ori_tgt_length))
  #y = y[:, :, 0:tgt_length].reshape((-1, cfg_num * tgt_length))
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  y = y[:, :, 2]
  return y

def sel_output(y):
  y = y.reshape((-1, 77, tgt_length))
  y = torch.cat((y[:, 70:72, :], y[:, 73:77, :]), 1)
  #y = y.reshape((-1, 32, tgt_length))
  #y = torch.cat((y[:, 25:27, :], y[:, 28:32, :]), 1)
  #y = y.reshape((-1, 7, tgt_length))
  #y = torch.cat((y[:, 1:3, :], y[:, 4:5, :], y[:, 0:1, :], y[:, 5:7, :]), 1) * 5
  y = y.reshape((-1, 6 * tgt_length))
  return y
