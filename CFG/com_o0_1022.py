import numpy as np

# Data set configuration.
data_set_dir = 'Data4/o0/'
sim_datasets = [
  (data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 116513919),
  (data_set_dir + '508.namd_r.in.mmap.norm', 111390471),
  (data_set_dir + '502.gcc_r.in.mmap.norm', 35570647),
  (data_set_dir + '519.lbm_r.in.mmap.norm', 100022078),
  (data_set_dir + '521.wrf_r.in.mmap.norm', 148596083),
  (data_set_dir + '505.mcf_r.in.mmap.norm', 109064890),
  (data_set_dir + '523.xalancbmk_r.in.mmap.norm', 111793321),
  (data_set_dir + '525.x264_r.in.mmap.norm', 101926647),
  (data_set_dir + '531.deepsjeng_r.in.mmap.norm', 100142179),
  (data_set_dir + '548.exchange2_r.in.mmap.norm', 101939356),
  (data_set_dir + '557.xz_r.in.mmap.norm', 100676458),
  (data_set_dir + '999.specrand_ir.in.mmap.norm', 73656320),
  (data_set_dir + '527.cam4_r.in.mmap.norm', 109768370),
  (data_set_dir + '538.imagick_r.in.mmap.norm', 106101730),
  (data_set_dir + '544.nab_r.in.mmap.norm', 102979307),
  (data_set_dir + '549.fotonik3d_r.in.mmap.norm', 115580739),
  (data_set_dir + '997.specrand_fr.in.mmap.norm', 73656320)
]

feature_format = np.float32
target_format = np.uint16

input_length = 51
seq_length = 256
tgt_length = 1
