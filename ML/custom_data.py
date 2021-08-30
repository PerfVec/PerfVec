import torch
import numpy as np
from torch.utils.data import Dataset
from cfg import data_item_format, seq_length, inst_length, tgt_length


class MemMappedDataset(Dataset):

    def __init__(self, file_name, seqs, start, end, stride=1, batch_size=1):
        self.arr = np.memmap(file_name, dtype=data_item_format, mode='r',
                             shape=(seqs, seq_length, inst_length))
        if (end - start) % (batch_size * stride) != 0:
            raise AttributeError("Size is not aligned.")
        self.start = start
        self.stride = stride
        self.batch_size = batch_size
        self.size = (end - start) // stride

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Find the batch index.
        batch_idx = idx // self.batch_size
        batch_offset = idx % self.batch_size
        batch_idx *= self.stride
        idx = self.start + batch_idx * self.batch_size + batch_offset

        x = np.copy(self.arr[idx, :, tgt_length:inst_length])
        y = np.copy(self.arr[idx, :, 0:tgt_length])
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        return x, y
