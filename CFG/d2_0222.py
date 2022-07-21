import numpy as np

# Data set configuration.
data_set_dir = 'Data2/'
data_set_idx = 4
datasets = [
  (data_set_dir + '507.cactuBSSN_r.inst.mmap.norm', 1035389426),
  (data_set_dir + '508.namd_r.inst.mmap.norm', 1105244743),
  (data_set_dir + '500.perlbench_r.inst.mmap.norm', 1186655322),
  (data_set_dir + '502.gcc_r.inst.mmap.norm', 16220268)
]

sim_datasets = [
  (data_set_dir + '507.cactuBSSN_r.inst.mmap.norm', 1035389426),
  (data_set_dir + '508.namd_r.inst.mmap.norm', 1105244743),
  (data_set_dir + '500.perlbench_r.inst.mmap.norm', 1186655322),
  (data_set_dir + '502.gcc_r.inst.mmap.norm', 16220268)
]

data_item_format = np.float32
# total batch number is 816,286.56
testbatchnum = 760000
validbatchnum = 720000
validbatchsize = 8192
testbatchsize = 128

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

input_length = 48
tgt_length = 15
input_start = 15
inst_length = input_start + input_length
seq_length = 256
