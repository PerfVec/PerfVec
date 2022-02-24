import os
import torch
import numpy as np
from torch.utils.data import Dataset
from CFG import data_item_format, seq_length, inst_length, input_start, datasets


class MemMappedDataset(Dataset):

    def __init__(self, file_name, seqs, start, end):
        self.arr = np.memmap(file_name, dtype=data_item_format, mode='r',
                             shape=(seqs, inst_length))
        if end <= start or end > seqs:
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
            x = np.zeros((seq_length, inst_length - input_start))
            x[seq_length-(idx+1):seq_length, :] = np.copy(self.arr[0:idx+1, input_start:inst_length])
        else:
            x = np.copy(self.arr[idx+1-seq_length:idx+1, input_start:inst_length])
        y = np.copy(self.arr[idx, 0:input_start])
        #y = np.concatenate((self.arr[idx, :, 0:1], self.arr[idx, :, 2:3], self.arr[idx, :, 4:5]), axis=1)
        #y_diff = y[1:seq_length, :] - y[0:seq_length-1, :]
        #y[1:seq_length, :] = y_diff
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        return x, y


class CombinedMMDataset(Dataset):

    def __init__(self, file_num, start, end):
        if file_num > len(datasets):
            raise AttributeError("Require more files than that exist.")
        total_size = 0
        for i in range(file_num):
            total_size += datasets[i][1]
        if end <= start or end > total_size:
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
            self.starts.append(int(datasets[i][1] * (start / total_size)))
            self.mm_sizes.append(int(datasets[i][1] * frac))
            print('Open', datasets[i][0], '(%d %d)' % (self.starts[i], self.mm_sizes[i]))
            self.mm_sets.append(MemMappedDataset(datasets[i][0], datasets[i][1],
                                self.starts[i], self.starts[i] + self.mm_sizes[i]))
            cum_start += self.starts[i]
            cum_size += self.mm_sizes[i]
            self.bounds.append(cum_size)
        self.starts.append(start - cum_start)
        self.mm_sizes.append(self.size - cum_size)
        print('Open', datasets[file_num-1][0], '(%d %d)' % (self.starts[file_num-1], self.mm_sizes[file_num-1]))
        self.mm_sets.append(MemMappedDataset(datasets[file_num-1][0], datasets[file_num-1][1],
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
