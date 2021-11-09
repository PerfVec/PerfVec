import numpy as np

# Data set configuration.
data_set_dir = 'Data/'
datasets = [
  (data_set_dir + '507.cactuBSSN_r.seq.mmap', 1011073),
  (data_set_dir + '508.namd_r.seq.mmap', 1079195),
  (data_set_dir + '500.perlbench_r.seq.mmap', 1158521),
  (data_set_dir + '502.gcc_r.seq.mmap', 15836),
  (data_set_dir + '999.specrand_ir.seq.mmap', 57694),
  (data_set_dir + 'all.mmap', 3264625)
]

data_item_format = np.uint16
# total batch number is 797.03
testbatchnum = 750
validbatchnum = 700
validbatchsize = 4

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 1) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

seq_length = 1024
input_length = 47
tgt_length = 3
input_start = 5
inst_length = input_start + input_length
