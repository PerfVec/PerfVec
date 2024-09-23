import numpy as np

# Data set configuration.
data_set_dir = 'tracefiles/'
data_set_idx = 1
datasets = [
  (data_set_dir + 'trace.in.nmmap', 5908, 5908),
]

# sim_datasets = [
#   (data_set_dir + '508.namd_r.in.mmap.norm', 1105247403, 1105245197),
#   (data_set_dir + '500.perlbench_r.in.mmap.norm', 1184866587, 1184864563)
#   #(data_set_dir + '507.cactuBSSN_r.in.mmap.norm', 1035397081, 24818299),
#   #(data_set_dir + '502.gcc_r.in.mmap.norm', 16220981, 13602728)
# ]

def get_out_name(name):
  return name.replace("in.mmap.norm", "out.mmap")

feature_format = np.float32
target_format = np.int16
# total batch number is 2328530787 / 4096 = 568488.96
# training set size = 524288 * 4096 = 2147483648
# testbatchnum = 1
# testbatchsize = 1000
# validbatchnum = 1
# validbatchsize = 256

ori_batch_size = 256
ori_tgt_length = 16
# test_start = testbatchnum * ori_batch_size
# test_end = (testbatchnum + testbatchsize) * ori_batch_size
# valid_start = validbatchnum * ori_batch_size
# valid_end = (validbatchnum + validbatchsize) * ori_batch_size
test_start = 4000
test_end = 5000
valid_start = 5001
valid_end = 5907
print('Valid Start',valid_start)
print('Valid End', valid_end)

input_length = 50
tgt_length = 16 
cfg_num = 1
seq_length = 256
