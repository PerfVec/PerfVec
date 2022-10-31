import numpy as np

# Data set configuration.
data_set_dir = 'Data4/o3/'
data_set_idx = 17
#data_set_idx = 7
datasets = [
  (data_set_dir + '507.cactuBSSN_r.in.mmap', 112401557),
  (data_set_dir + '508.namd_r.in.mmap', 111803691),
  (data_set_dir + '502.gcc_r.in.mmap', 16221074),
  (data_set_dir + '519.lbm_r.in.mmap', 131263450),
  (data_set_dir + '521.wrf_r.in.mmap', 102924645),
  (data_set_dir + '505.mcf_r.in.mmap', 109655156),
  (data_set_dir + '523.xalancbmk_r.in.mmap', 115505219),
  (data_set_dir + '525.x264_r.in.mmap', 105801790),
  (data_set_dir + '531.deepsjeng_r.in.mmap', 100143464),
  (data_set_dir + '548.exchange2_r.in.mmap', 104533715),
  (data_set_dir + '557.xz_r.in.mmap', 101217453),
  (data_set_dir + '999.specrand_ir.in.mmap', 59087084),
  (data_set_dir + '527.cam4_r.in.mmap', 105527641),
  (data_set_dir + '538.imagick_r.in.mmap', 108956910),
  (data_set_dir + '544.nab_r.in.mmap', 105380723),
  (data_set_dir + '549.fotonik3d_r.in.mmap', 115772447),
  (data_set_dir + '997.specrand_fr.in.mmap', 59087084)
]

feature_format = np.float32
target_format = np.uint16

input_length = 51
cfg_num = 1
seq_length = 256
