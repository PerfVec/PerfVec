import numpy as np

# Fill in the gem5 path below.
data_set_dir = '<path to gem5>' + '/'
sim_datasets = [
  # Fill in the number of instructions in the trace below.
  (data_set_dir + 'trace.in.nmmap', <instruction number>)
]

feature_format = np.float32
target_format = np.uint16

input_length = 51
tgt_length = 1
cfg_num = 1
seq_length = 256

def sel_output(y):
  y = y.reshape((-1, 77, tgt_length))
  y = y[:, 70, :]
  return y
