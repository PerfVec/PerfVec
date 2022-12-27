import numpy as np

# Data set configuration.
data_set_dir = 'Data4/'
#data_set_idx = 4
data_set_idx = 10
datasets = [
  #(data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 1035397081, 16511420),
  #(data_set_dir + '508.namd_r.in.mmap.norm', 1105247403, 1105245194),
  #(data_set_dir + '500.perlbench_r.in.mmap.norm', 1184866587, 1184864557),
  #(data_set_dir + '502.gcc_r.in.mmap.norm', 16220981, 69094),
  #(data_set_dir + '519.lbm_r.in.mmap.norm', 1104042304, 1103901340),
  #(data_set_dir + '521.wrf_r.in.mmap.norm', 1084282512, 329386),
  #(data_set_dir + '505.mcf_r.in.mmap.norm', 1045214900, 1044819469),
  #(data_set_dir + '523.xalancbmk_r.in.mmap.norm', 385894391, 1000130)
  (data_set_dir + '525.x264_r.in.mmap.norm', 1066019761, 105798583),
  (data_set_dir + '531.deepsjeng_r.in.mmap.norm', 1051476976, 100002498),
  (data_set_dir + '538.imagick_r.in.mmap.norm', 115459544, 108954312),
  (data_set_dir + '544.nab_r.in.mmap.norm', 1050786112, 105377114),
  (data_set_dir + '548.exchange2_r.in.mmap.norm', 1042957954, 104453276),
  (data_set_dir + '557.xz_r.in.mmap.norm', 1043200017, 101079728),
  (data_set_dir + '999.specrand_ir.in.mmap.norm', 59087074, 59087074),
  (data_set_dir + '527.cam4_r.in.mmap.norm', 1060183121, 692705),
  (data_set_dir + '549.fotonik3d_r.in.mmap.norm', 1151356733, 115771900),
  (data_set_dir + '997.specrand_fr.in.mmap.norm', 59087074, 59087074)
]

sim_datasets = [
  datasets[0],
  datasets[1],
  datasets[4],
  datasets[5],
  datasets[6],
  datasets[7],
  datasets[2],
  datasets[3],
  datasets[8],
  datasets[9]
]

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.mmap").replace(data_set_dir, data_set_dir + "1022/")

feature_format = np.float32
target_format = np.uint16
if data_set_idx == 4:
  # total batch number is 420,132,507 / 4096 = 102,571.41
  # training set size = 90000 * 4096 = 368640000
  testbatchnum = 95000
  testbatchsize = 1024
  validbatchnum = 90000
  validbatchsize = 5000
elif data_set_idx == 10:
  # total batch number is 860,304,264 / 4096 = 210,035.22
  # training set size = 180000 * 4096 = 737280000
  testbatchnum = 190000
  testbatchsize = 1024
  validbatchnum = 180000
  validbatchsize = 10000

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
  return y
