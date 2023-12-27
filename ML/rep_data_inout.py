import os
import torch
import numpy as np
from torch.utils.data import Dataset


class RepDataset(Dataset):

    def __init__(self, file_name, start, end):
        self.data = torch.load(file_name)
        if end < start or end > self.data.shape[0]:
            raise AttributeError("End is illegal.")
        self.start = start
        self.size = end - start
        self.rep_dim = self.data.shape[1] - 1
        print("Open %s (%d %d %d)" % (file_name, start, end, self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx += self.start
        x = self.data[idx, 0:self.rep_dim]
        y = self.data[idx, self.rep_dim:self.rep_dim+1] * 10000000000
        return x, y
