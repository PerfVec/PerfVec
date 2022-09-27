import os
import torch
import numpy as np
from torch.utils.data import Dataset
from CFG import *


class MemMappedDataset(Dataset):

    def __init__(self, file_name, in_size, out_size, start, end):
        self.in_arr = np.memmap(file_name, dtype=feature_format, mode='r',
                                shape=(in_size, input_length))
        self.out_arr = np.memmap(get_out_name(file_name), dtype=target_format, mode='r',
                                 shape=(out_size, ori_tgt_length * cfg_num))
        assert in_size >= out_size
        if end < start or end > out_size:
            raise AttributeError("End is illegal.")
        self.start = start
        self.size = end - start

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx += self.start
        if idx < seq_length:
            x = np.zeros((seq_length, input_length))
            x[seq_length-(idx+1):seq_length, :] = np.copy(self.in_arr[0:idx+1, :])
        else:
            x = np.copy(self.in_arr[idx+1-seq_length:idx+1, :])
        y = np.copy(self.out_arr[idx, :])
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        return x, y


class CombinedMMDataset(Dataset):

    def __init__(self, file_num, start, end):
        if file_num > len(datasets):
            raise AttributeError("Require more files than that exist.")
        total_size = 0
        for i in range(file_num):
            total_size += datasets[i][2]
        if end < start or end > total_size:
            raise AttributeError("End is illegal.")
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
            self.starts.append(int(datasets[i][2] * (start / total_size)))
            self.mm_sizes.append(int(datasets[i][2] * frac))
            print('Open', datasets[i][0], '(%d %d)' % (self.starts[i], self.mm_sizes[i]))
            self.mm_sets.append(MemMappedDataset(datasets[i][0], datasets[i][1], datasets[i][2],
                                self.starts[i], self.starts[i] + self.mm_sizes[i]))
            cum_start += self.starts[i]
            cum_size += self.mm_sizes[i]
            self.bounds.append(cum_size)
        self.starts.append(start - cum_start)
        self.mm_sizes.append(self.size - cum_size)
        print('Open', datasets[file_num-1][0], '(%d %d)' % (self.starts[file_num-1], self.mm_sizes[file_num-1]))
        self.mm_sets.append(MemMappedDataset(datasets[file_num-1][0], datasets[file_num-1][1], datasets[file_num-1][2],
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


mm_batch_size = 512

class MemMappedBatchDataset(Dataset):

    def __init__(self, file_name, in_size, out_size, start, end):
        self.in_arr = np.memmap(file_name, dtype=feature_format, mode='r',
                                shape=(in_size, input_length))
        self.out_arr = np.memmap(get_out_name(file_name), dtype=target_format, mode='r',
                                 shape=(out_size, ori_tgt_length * cfg_num))
        self.batchsize = mm_batch_size
        assert in_size >= out_size
        if end < start or end * self.batchsize > out_size:
            raise AttributeError("End is illegal.")
        self.start = start
        self.size = end - start

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
        y = np.copy(self.out_arr[idx:idx+self.batchsize, :])
        if tgt_length != ori_tgt_length:
            y = sel_batch_out(y)
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        return x, y


class CombinedMMBDataset(Dataset):

    def __init__(self, file_num, start, end):
        if file_num > len(datasets):
            raise AttributeError("Require more files than that exist.")
        total_size = 0
        for i in range(file_num):
            total_size += datasets[i][2]
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
            self.starts.append(int(datasets[i][2] * (start / total_size) / mm_batch_size))
            self.mm_sizes.append(int(datasets[i][2] * frac / mm_batch_size))
            print('Open', datasets[i][0], '(%d %d)' % (self.starts[i], self.mm_sizes[i]))
            self.mm_sets.append(MemMappedBatchDataset(datasets[i][0], datasets[i][1], datasets[i][2],
                                self.starts[i], self.starts[i] + self.mm_sizes[i]))
            cum_start += self.starts[i]
            cum_size += self.mm_sizes[i]
            self.bounds.append(cum_size)
        self.starts.append(start - cum_start)
        self.mm_sizes.append(self.size - cum_size)
        print('Open', datasets[file_num-1][0], '(%d %d)' % (self.starts[file_num-1], self.mm_sizes[file_num-1]))
        self.mm_sets.append(MemMappedBatchDataset(datasets[file_num-1][0], datasets[file_num-1][1], datasets[file_num-1][2],
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
