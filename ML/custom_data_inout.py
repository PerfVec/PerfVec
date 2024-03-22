import os
import torch
import numpy as np
from torch.utils.data import Dataset


class MemMappedDataset(Dataset):

    def __init__(self, cfg, paras, start, end):
        assert len(paras) == 3
        file_name = paras[0]
        in_size = paras[1]
        out_size = paras[2]
        self.in_arr = np.memmap(file_name, dtype=cfg.feature_format, mode='r',
                                shape=(in_size, cfg.input_length))
        self.out_arr = np.memmap(cfg.get_out_name(file_name), dtype=cfg.target_format, mode='r',
                                 shape=(out_size, cfg.ori_tgt_length * cfg.cfg_num))
        assert in_size >= out_size
        if end < start or end > out_size:
            raise AttributeError("End is illegal.")
        self.start = start
        self.size = end - start
        print("Open %s (%d %d %d)" % (file_name, start, end, self.size))

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

    def __init__(self, cfg, file_num, start, end):
        if file_num > len(cfg.datasets):
            raise AttributeError("Require more files than that exist.")
        total_size = 0
        for i in range(file_num):
            total_size += cfg.datasets[i][2]
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
            self.starts.append(int(cfg.datasets[i][2] * (start / total_size)))
            self.mm_sizes.append(int(cfg.datasets[i][2] * frac))
            self.mm_sets.append(MemMappedDataset(cfg, cfg.datasets[i],
                                self.starts[i], self.starts[i] + self.mm_sizes[i]))
            cum_start += self.starts[i]
            cum_size += self.mm_sizes[i]
            self.bounds.append(cum_size)
        self.starts.append(start - cum_start)
        self.mm_sizes.append(self.size - cum_size)
        self.mm_sets.append(MemMappedDataset(cfg, cfg.datasets[file_num-1],
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

    def __init__(self, cfg, paras, start, end, rank):
      file_name = paras[0]
      in_size = paras[1]
      if len(paras) == 4:
        out_size = paras[3]
        out_use_size = paras[2]
        assert out_use_size <= out_size
      else:
        assert len(paras) == 3
        out_size = paras[2]
        out_use_size = out_size
      assert in_size >= out_size
      self.in_arr = np.memmap(file_name, dtype=cfg.feature_format, mode='r',
                              shape=(in_size, cfg.input_length))
      self.out_arr = np.memmap(cfg.get_out_name(file_name), dtype=cfg.target_format, mode='r',
                               shape=(out_size, cfg.ori_tgt_length * cfg.cfg_num))
      self.batchsize = mm_batch_size
      if end < start:
        raise AttributeError("End is illegal.")
      elif end * self.batchsize > out_use_size:
        if rank == 0:
          print("Use the maximum size of", out_use_size // self.batchsize,
                "instead of", end, flush=True)
        end = out_use_size // self.batchsize
      self.seq_length = cfg.seq_length
      self.input_length = cfg.input_length
      if hasattr(cfg, 'sel_batch_out'):
        self.sel_out = cfg.sel_batch_out
      else:
        self.sel_out = None
      if hasattr(cfg, 'sel_in'):
        self.sel_in = cfg.sel_in
      else:
        self.sel_in = None
      self.start = start
      self.size = end - start
      if rank == 0:
        print("Open %s (%d %d %d) (sel_in %s) (sel_out %s)" % (cfg.get_out_name(file_name), start, end, self.size,
                                                               "yes" if self.sel_in is not None else "no",
                                                               "yes" if self.sel_out is not None else "no"),
                                                               flush=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx += self.start
        idx *= self.batchsize
        if idx < self.seq_length:
            x = np.zeros((self.seq_length+self.batchsize-1, self.input_length))
            x[self.seq_length-(idx+1):self.seq_length+self.batchsize-1, :] = np.copy(self.in_arr[0:idx+self.batchsize, :])
        else:
            x = np.copy(self.in_arr[idx+1-self.seq_length:idx+self.batchsize, :])
        y = np.copy(self.out_arr[idx:idx+self.batchsize, :])
        if self.sel_out is not None:
            y = self.sel_out(y)
        if self.sel_in is not None:
            x = self.sel_in(x)
        x = torch.from_numpy(x.astype('f'))
        y = torch.from_numpy(y.astype('f'))
        return x, y


class CombinedMMBDataset(Dataset):

    def __init__(self, cfg, file_num, start, end, rank):
        if file_num > len(cfg.datasets):
            raise AttributeError("Require more files than that exist.")
        total_size = 0
        for i in range(file_num):
            total_size += cfg.datasets[i][2]
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
            self.starts.append(int(cfg.datasets[i][2] * (start / total_size) / mm_batch_size))
            self.mm_sizes.append(int(cfg.datasets[i][2] * frac / mm_batch_size))
            self.mm_sets.append(MemMappedBatchDataset(cfg, cfg.datasets[i],
                                self.starts[i], self.starts[i] + self.mm_sizes[i], rank))
            cum_start += self.starts[i]
            cum_size += self.mm_sizes[i]
            self.bounds.append(cum_size)
        self.starts.append(start - cum_start)
        self.mm_sizes.append(self.size - cum_size)
        self.mm_sets.append(MemMappedBatchDataset(cfg, cfg.datasets[file_num-1],
                            self.starts[file_num-1], self.starts[file_num-1] + self.mm_sizes[file_num-1], rank))
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
