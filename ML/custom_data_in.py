import os
import torch
import numpy as np
from torch.utils.data import Dataset
from CFG import *


assert 'tgt_length' not in globals()
mm_batch_size = 512

class MemMappedBatchDataset(Dataset):

    def __init__(self, file_name, in_size, start, end):
        self.in_arr = np.memmap(file_name, dtype=feature_format, mode='r',
                                shape=(in_size, input_length))
        self.batchsize = mm_batch_size
        if end < start or end * self.batchsize > in_size:
            raise AttributeError("End is illegal.")
        self.start = start
        self.size = end - start

    def __init__(self, cfgs, start, end):
        assert len(cfgs) == 2
        self.__init__(cfgs[0], cfgs[1], start, end)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx += self.start
        idx *= self.batchsize
        if idx < seq_length:
            x = np.zeros((seq_length+self.batchsize-1, input_length))
            x[seq_length-(idx+1):seq_length+self.batchsize-1, :] = np.copy(self.in_arr[0:idx+self.batchsize, :])
        else:
            x = np.copy(self.in_arr[idx+1-seq_length:idx+self.batchsize, :])
        x = torch.from_numpy(x.astype('f'))
        y = torch.zeros(self.batchsiz, 1)
        return x, y


class CombinedMMBDataset(Dataset):

    def __init__(self, file_num, start, end):
        if file_num > len(datasets):
            raise AttributeError("Require more files than that exist.")
        total_size = 0
        for i in range(file_num):
            total_size += datasets[i][1]
        if end < start or end > total_size:
            raise AttributeError("End is illegal.")
        if start % mm_batch_size != 0 or end % mm_batch_size != 0:
            raise AttributeError("Start or end is not aligned.")
        start = int(start / mm_batch_size)
        end = int(end / mm_batch_size)
        total_size /= mm_batch_size
        # Calculate start and end for each dataset.
        self.file_num = file_num
        self.size = end - start
        frac = self.size / total_size
        self.mm_sets = []
        self.starts = []
        self.mm_sizes = []
        self.bounds = [0]
        cum_start = 0
        cum_size = 0
        for i in range(file_num-1):
            self.starts.append(int(datasets[i][1] * (start / total_size) / mm_batch_size))
            self.mm_sizes.append(int(datasets[i][1] * frac / mm_batch_size))
            print('Open', datasets[i][0], '(%d %d)' % (self.starts[i], self.mm_sizes[i]))
            self.mm_sets.append(MemMappedBatchDataset(datasets[i][0], datasets[i][1],
                                self.starts[i], self.starts[i] + self.mm_sizes[i]))
            cum_start += self.starts[i]
            cum_size += self.mm_sizes[i]
            self.bounds.append(cum_size)
        self.starts.append(start - cum_start)
        self.mm_sizes.append(self.size - cum_size)
        print('Open', datasets[file_num-1][0], '(%d %d)' % (self.starts[file_num-1], self.mm_sizes[file_num-1]))
        self.mm_sets.append(MemMappedBatchDataset(datasets[file_num-1][0], datasets[file_num-1][1],
                            self.starts[file_num-1], self.starts[file_num-1] + self.mm_sizes[file_num-1]))
        self.bounds.append(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Find which set the idx falls in.
        for i in range(self.file_num):
            if idx < self.bounds[i+1]:
                #print('Index:', i, idx - self.bounds[i])
                return self.mm_sets[i].__getitem__(idx - self.bounds[i])
        raise RuntimeError("Idx is too large.")
