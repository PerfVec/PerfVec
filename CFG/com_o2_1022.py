import numpy as np

# Data set configuration.
data_set_dir = 'Data4/o2/'
sim_datasets = [
  (data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 112536013),
  (data_set_dir + '508.namd_r.in.mmap.norm', 111803761),
  (data_set_dir + '502.gcc_r.in.mmap.norm', 17534494),
  (data_set_dir + '519.lbm_r.in.mmap.norm', 155383446),
  (data_set_dir + '521.wrf_r.in.mmap.norm', 102902846),
  (data_set_dir + '505.mcf_r.in.mmap.norm', 109655154),
  (data_set_dir + '523.xalancbmk_r.in.mmap.norm', 115310856),
  (data_set_dir + '525.x264_r.in.mmap.norm', 109494995),
  (data_set_dir + '531.deepsjeng_r.in.mmap.norm', 100144309),
  (data_set_dir + '548.exchange2_r.in.mmap.norm', 106811228),
  (data_set_dir + '557.xz_r.in.mmap.norm', 101142821),
  (data_set_dir + '999.specrand_ir.in.mmap.norm', 59280851),
  (data_set_dir + '527.cam4_r.in.mmap.norm', 105180804),
  (data_set_dir + '538.imagick_r.in.mmap.norm', 110100057),
  (data_set_dir + '544.nab_r.in.mmap.norm', 105420759),
  (data_set_dir + '549.fotonik3d_r.in.mmap.norm', 115818920),
  (data_set_dir + '997.specrand_fr.in.mmap.norm', 59280851)
]

feature_format = np.float32
target_format = np.uint16

input_length = 51
seq_length = 256
tgt_length = 1
