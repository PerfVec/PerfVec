import numpy as np

# Data set configuration.
data_set_dir = '<path to gem5>'
sim_datasets = [
  (data_set_dir + 'trace.in.nmmap', <instruction number>)
]

feature_format = np.float32
target_format = np.uint16

input_length = 51
tgt_length = 1
cfg_num = 1
seq_length = 256
