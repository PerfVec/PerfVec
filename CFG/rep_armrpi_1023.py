# Data set configuration.
dataset = 'Data_rep/armrpi.pt' # 8563 samples

# total batch number is 8563 / 1024 = 8.36
testbatchnum = 7
testbatchsize = 1
validbatchnum = 6
validbatchsize = 1

ori_batch_size = 1024
test_start = testbatchnum * ori_batch_size
test_end = (testbatchnum + testbatchsize) * ori_batch_size
valid_start = validbatchnum * ori_batch_size
valid_end = (validbatchnum + validbatchsize) * ori_batch_size

cfg_num = 1
seq_length = 1
input_length = 256
tgt_length = 1
