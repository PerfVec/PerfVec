import numpy as np

# Data set configuration.
data_set_dir = 'Data4/'
data_set_idx = 10
datasets = [
  (data_set_dir + '525.x264_r.in.mmap', 1066019761, 1065961385),
  (data_set_dir + '531.deepsjeng_r.in.mmap', 1051476976, 1049812795),
  (data_set_dir + '548.exchange2_r.in.mmap', 1042957954, 1041812670),
  (data_set_dir + '557.xz_r.in.mmap', 1043200017, 1042392179),
  (data_set_dir + '999.specrand_ir.in.mmap', 59087074, 59087074),
  (data_set_dir + '527.cam4_r.in.mmap', 1060183121, 1048566868),
  (data_set_dir + '538.imagick_r.in.mmap', 115459544, 115459544),
  (data_set_dir + '544.nab_r.in.mmap', 1050786112, 1050747889),
  (data_set_dir + '549.fotonik3d_r.in.mmap', 1151356733, 1151355900),
  (data_set_dir + '997.specrand_fr.in.mmap', 59087074, 59087074)
]

sim_datasets = [
]

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.prog0922.mmap")

feature_format = np.float32
target_format = np.uint16
# total batch number is  / 4096 = 
# training set size = * 4096 = 
testbatchnum = 
testbatchsize = 1024
validbatchnum = 
validbatchsize = 

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 51
tgt_length = 16
cfg_num = 6
seq_length = 256
ori_tgt_length = 16

def sel_batch_out(y):
  y = y.reshape((-1, cfg_num, ori_tgt_length))
  y = y[:, :, 0:tgt_length].reshape((-1, cfg_num * tgt_length))
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  return y
