import numpy as np

# Data set configuration.
data_set_dir = 'Data4/'
data_set_idx = 8
datasets = [
  (data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 1035397081, 15326847),
  (data_set_dir + '508.namd_r.in.mmap.norm', 1105247403, 111803697),
  (data_set_dir + '500.perlbench_r.in.mmap.norm', 1184866587, 794193),
  (data_set_dir + '502.gcc_r.in.mmap.norm', 16220981, 8606664),
  (data_set_dir + '519.lbm_r.in.mmap.norm', 1104042304, 131263455),
  (data_set_dir + '521.wrf_r.in.mmap.norm', 1084282512, 329410),
  (data_set_dir + '505.mcf_r.in.mmap.norm', 1045214900, 109655167),
  (data_set_dir + '523.xalancbmk_r.in.mmap.norm', 385894391, 1000130)
]

sim_datasets = [
  (data_set_dir + '508.namd_r.in.mmap.norm', 1105247403),
  (data_set_dir + '500.perlbench_r.in.mmap.norm', 1184866587),
  (data_set_dir + '502.gcc_r.in.mmap.norm', 16220981),
  (data_set_dir + '519.lbm_r.in.mmap.norm', 1104042304),
  (data_set_dir + '521.wrf_r.in.mmap.norm', 1084282512),
  (data_set_dir + '505.mcf_r.in.mmap.norm', 1045214900),
  (data_set_dir + '523.xalancbmk_r.in.mmap.norm', 385894391),
  (data_set_dir + '525.x264_r.in.mmap.norm', 1066019761),
  (data_set_dir + '531.deepsjeng_r.in.mmap.norm', 1051476976),
  (data_set_dir + '548.exchange2_r.in.mmap.norm', 1042957954),
  (data_set_dir + '557.xz_r.in.mmap.norm', 1043200017),
  (data_set_dir + '999.specrand_ir.in.mmap.norm', 59087074),
  (data_set_dir + '527.cam4_r.in.mmap.norm', 1060183121),
  (data_set_dir + '538.imagick_r.in.mmap.norm', 115459544),
  (data_set_dir + '544.nab_r.in.mmap.norm', 1050786112),
  (data_set_dir + '549.fotonik3d_r.in.mmap.norm', 1151356733),
  (data_set_dir + '997.specrand_fr.in.mmap.norm', 59087074)
]

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.mmap").replace(data_set_dir, data_set_dir + "cache/")

feature_format = np.float32
target_format = np.uint16
# total batch number is 378,779,563 / 4096 = 92,475.48
# training set size = 81000 * 4096 = 331776000
testbatchnum = 85500
testbatchsize = 1024
validbatchnum = 81000
validbatchsize = 4500

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 51
#tgt_length = 16
tgt_length = 1
cfg_num = 19
seq_length = 256
ori_tgt_length = 16

def sel_batch_out(y):
  y = y.reshape((-1, cfg_num, ori_tgt_length))
  #y = y[:, :, 0:tgt_length].reshape((-1, cfg_num * tgt_length))
  #y = np.concatenate((y[:, :, 0:2], y[:, :, 6:ori_tgt_length]), axis=2).reshape(-1, cfg_num * tgt_length)
  y = y[:, :cfg_num-1, 2]
  return y
