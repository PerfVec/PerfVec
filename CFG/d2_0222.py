import numpy as np

# Data set configuration.
data_set_dir = 'Data2/'
data_set_idx = 4
datasets = [
  (data_set_dir + '507.cactuBSSN_r.seq.mmap', 1035389426),
  (data_set_dir + '508.namd_r.seq.mmap', 1105244743),
  (data_set_dir + '500.perlbench_r.seq.mmap', 1186655322),
  (data_set_dir + '502.gcc_r.seq.mmap', 16220268)
]

data_item_format = np.float32
# total batch number is 1,659.84
testbatchnum = 1580
validbatchnum = 1500
validbatchsize = 16

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 1) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 48
tgt_length = 15
input_start = 15
inst_length = input_start + input_length
