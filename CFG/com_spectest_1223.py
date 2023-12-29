import numpy as np

# Data set configuration.
data_set_dir = 'Data_spectest/'
sim_datasets = [
  (data_set_dir + '507.cactuBSSN_r.in.nmmap', 21875660070),
  (data_set_dir + '508.namd_r.in.nmmap', 31536511325),
  (data_set_dir + '519.lbm_r.in.nmmap', 6978135244),
  (data_set_dir + '521.wrf_r.in.nmmap', 67963162997),
  (data_set_dir + '500.perlbench_r.in.nmmap', 2823272380),
  (data_set_dir + '502.gcc_r.in.nmmap', 16221155),
  (data_set_dir + '505.mcf_r.in.nmmap', 29408772632),
  (data_set_dir + '523.xalancbmk_r.in.nmmap', 385896258),
  #(data_set_dir + '527.cam4_r.in.nmmap', ),
  (data_set_dir + '538.imagick_r.in.nmmap', 115459424),
  (data_set_dir + '544.nab_r.in.nmmap', 9578303434),
  (data_set_dir + '549.fotonik3d_r.in.nmmap', 39311448867),
  #(data_set_dir + '997.specrand_fr.in.nmmap', 59087074),
  #(data_set_dir + '525.x264_r.in.nmmap', ),
  (data_set_dir + '531.deepsjeng_r.in.nmmap', 39473417512),
  #(data_set_dir + '548.exchange2_r.in.nmmap', ),
  (data_set_dir + '557.xz_r.in.nmmap', 2251516730),
  (data_set_dir + '999.specrand_ir.in.nmmap', 59087074)
]

feature_format = np.float32
target_format = np.uint16

input_length = 51
#tgt_length = 16
tgt_length = 1
cfg_num = 1
seq_length = 256
