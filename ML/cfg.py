import numpy as np

# Data set configuration.
data_set_name = "Data"

data_file_name = data_set_name + "/999.specrand_ir.seq.mmap"
data_item_format = np.uint16
total_size = 57694
# total batch number is 14.09
testbatchnum = 13
validbatchnum = 12
validbatchsize = 1

ori_batch_size = 4096
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + 1) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

seq_length = 1024
inst_length = 52
tgt_length = 5
