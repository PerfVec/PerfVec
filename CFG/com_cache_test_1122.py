import numpy as np

# Data set configuration.
data_set_dir = 'Data4/'
sim_datasets = [
  (data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 1035397081),
  (data_set_dir + '508.namd_r.in.mmap.norm', 1105247403),
  (data_set_dir + '519.lbm_r.in.mmap.norm', 1104042304),
  (data_set_dir + '521.wrf_r.in.mmap.norm', 1084282512),
  (data_set_dir + '500.perlbench_r.in.mmap.norm', 1184866587),
  (data_set_dir + '502.gcc_r.in.mmap.norm', 16220981),
  (data_set_dir + '505.mcf_r.in.mmap.norm', 1045214900),
  (data_set_dir + '523.xalancbmk_r.in.mmap.norm', 385894391),
  (data_set_dir + '527.cam4_r.in.mmap.norm', 1060183121),
  (data_set_dir + '538.imagick_r.in.mmap.norm', 115459544),
  (data_set_dir + '544.nab_r.in.mmap.norm', 1050786112),
  (data_set_dir + '549.fotonik3d_r.in.mmap.norm', 1151356733),
  #(data_set_dir + '997.specrand_fr.in.mmap.norm', 59087074),
  (data_set_dir + '525.x264_r.in.mmap.norm', 1066019761),
  (data_set_dir + '531.deepsjeng_r.in.mmap.norm', 1051476976),
  (data_set_dir + '548.exchange2_r.in.mmap.norm', 1042957954),
  (data_set_dir + '557.xz_r.in.mmap.norm', 1043200017),
  (data_set_dir + '999.specrand_ir.in.mmap.norm', 59087074)
]

feature_format = np.float32
target_format = np.uint16

input_length = 51
#tgt_length = 16
tgt_length = 1
cfg_num = 36
seq_length = 256
