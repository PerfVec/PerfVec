import numpy as np

# Data set configuration.
data_set_dir = 'Data4/'
data_set_idx = 17
#data_set_idx = 7
datasets = [
  (data_set_dir + '508.namd_r.in.mmap.norm', 1105247403, 1105247398),
  (data_set_dir + '500.perlbench_r.in.mmap.norm', 1184866587, 1184866587),
  (data_set_dir + '502.gcc_r.in.mmap.norm', 16220981, 8606664),
  (data_set_dir + '519.lbm_r.in.mmap.norm', 1104042304, 1104042238),
  (data_set_dir + '521.wrf_r.in.mmap.norm', 1084282512, 329410),
  (data_set_dir + '505.mcf_r.in.mmap.norm', 1045214900, 1045214855),
  (data_set_dir + '523.xalancbmk_r.in.mmap.norm', 385894391, 1000130),
  (data_set_dir + '525.x264_r.in.mmap', 1066019761, 1066019761),
  (data_set_dir + '531.deepsjeng_r.in.mmap', 1051476976, 1051476976),
  (data_set_dir + '548.exchange2_r.in.mmap', 1042957954, 1042957940),
  (data_set_dir + '557.xz_r.in.mmap', 1043200017, 1043200017),
  (data_set_dir + '999.specrand_ir.in.mmap', 59087074, 59087074),
  (data_set_dir + '527.cam4_r.in.mmap', 1060183121, 692705),
  (data_set_dir + '538.imagick_r.in.mmap', 115459544, 115459544),
  (data_set_dir + '544.nab_r.in.mmap', 1050786112, 1050786112),
  (data_set_dir + '549.fotonik3d_r.in.mmap', 1151356733, 1151356733),
  (data_set_dir + '997.specrand_fr.in.mmap', 59087074, 59087074)
]

sim_datasets = [
]

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.uarch0922.mmap")

feature_format = np.float32
target_format = np.uint16
if data_set_idx == 17:
  # total batch number is 11,089,431,218 / 4096 = 2,707,380.67
  # training set size = 2304000 * 4096 = 9437184000
  testbatchnum = 2432000
  testbatchsize = 1024
  validbatchnum = 2304000
  validbatchsize = 128000
elif data_set_idx == 7:
  # total batch number is 4,449,307,282 / 4096 = 1,086,256.66
  # training set size = 921600 * 4096 = 3774873600
  testbatchnum = 972800
  testbatchsize = 1024
  validbatchnum = 921600
  validbatchsize = 51200

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 51
tgt_length = 16
cfg_num = 11
seq_length = 256
ori_tgt_length = 16

def sel_batch_out(y):
  y = y.reshape((-1, cfg_num, ori_tgt_length))
  y = y[:, :, 0:tgt_length].reshape((-1, cfg_num * tgt_length))
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  return y
