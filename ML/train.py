from __future__ import print_function
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR

from custom_data import *
from utils import profile_model, generate_model_name
from models import *
from cfg import *


loss_fn = nn.MSELoss()


class ModelSet:
    def __init__(self, idx, name, model, optimizer):
        self.idx = idx
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.min_loss = float("inf")
        self.cur_loss = 0
        self.total_loss = 0


def train_mul(args, models, device, train_loader, epoch, rank):
    for ms in models:
        ms.model.train()
        ms.total_loss = 0
    start_t = time.time()
    print_threshold = max(len(train_loader) // 100, 1)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for ms in models:
            ms.optimizer.zero_grad()
            output = ms.model(data)
            loss = loss_fn(output, target)
            ms.total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(ms.model.parameters(), args.clip)
            ms.optimizer.step()
        if batch_idx % print_threshold == print_threshold - 1 and rank == 0:
            print('.', flush=True, end='')
        if args.dry_run:
            break
    if rank == 0:
        print('', flush=True)
    end_t = time.time()
    if args.distributed:
        dist.barrier()
    for ms in models:
        ms.total_loss /= len(train_loader)
        print('Train Epoch {} {}: {} \tLoss: {:.6f} \tTime: {:.1f}'.format(
            epoch, ms.idx, rank, ms.total_loss, end_t - start_t), flush=True)


def test_mul(args, models, device, test_loader, rank):
    for ms in models:
        ms.model.eval()
        ms.total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for ms in models:
                output = ms.model(data)
                ms.total_loss += loss_fn(output, target).item()
    for ms in models:
        ms.total_loss /= len(test_loader)
        ms.cur_loss = ms.total_loss
        print('Test set {} {}: Loss: {:.6f}'.format(
            ms.idx, rank, ms.total_loss), flush=True)


def save_checkpoint(name, model, optimizer, epoch, best_loss, lr, best=False):
    if lr != 0:
        lr_name = '_lr' + str(lr)
    else:
        lr_name = ''
    if best:
        name = 'checkpoints/' + generate_model_name(name) + lr_name + '_best.pt'
    else:
        name = 'checkpoints/' + generate_model_name(name, epoch) + lr_name + '.pt'
    saved_dict = {'epoch': epoch,
                  'best_loss': best_loss,
                  'optimizer_state_dict': optimizer.state_dict()}
    if torch.cuda.device_count() > 1:
        model_dict = {'model_state_dict': model.module.state_dict()}
    else:
        model_dict = {'model_state_dict': model.state_dict()}
    saved_dict.update(model_dict)
    torch.save(saved_dict, name)
    print("Saved checkpoint", name)


def load_checkpoint(rank, name, model, optimizer, device):
    assert 'checkpoints/' in name
    cp = torch.load(name, map_location='cpu')
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(cp['model_state_dict'])
    else:
        model.load_state_dict(cp['model_state_dict'])
    optimizer.load_state_dict(cp['optimizer_state_dict'])
    #for state in optimizer.state.values():
    #    for k, v in state.items():
    #        if torch.is_tensor(v):
    #            state[k] = v.to(device)
    start_epoch = cp['epoch']
    if rank == 0:
        print("Loaded checkpoint", name)
    return start_epoch + 1, cp['best_loss']


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main_rank(rank, args):
    if args.distributed:
        # create default process group
        global_rank = args.node_rank * args.gpus + rank
        dist.init_process_group("nccl", rank=global_rank, world_size=args.world_size)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    #dataset1 = MemMappedDataset(data_file_name, total_size, 0, args.train_size)
    #dataset2 = MemMappedDataset(data_file_name, total_size, valid_start, valid_end)
    dataset1 = CombinedMMDataset(4, 0, args.train_size)
    dataset2 = CombinedMMDataset(4, valid_start, valid_end)
    #print(dataset1[0][0].size())
    #print(dataset1[0])
    #print(dataset1[12686])
    #print(dataset2[0])
    #print(dataset1[1])
    #exit()
    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': False}
        kwargs.update(cuda_kwargs)
    if args.distributed:
        #train_sampler = torch.utils.data.distributed.DistributedSampler(
        #    dataset1, num_replicas=args.world_size, rank=global_rank, shuffle=False)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset1, shuffle=False)
        train_loader = torch.utils.data.DataLoader(dataset1, sampler=train_sampler, **kwargs)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset2, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset2, sampler=test_sampler, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    models = []
    i = 0
    for name in args.models:
        model = eval(name)
        if rank == 0:
            profile_model(model)
        device = torch.device("cuda" if use_cuda else "cpu")
        if args.distributed:
            device = rank
            model = DDP(model.to(device), device_ids=[device])
        elif torch.cuda.device_count() > 1:
            print ('Available devices ', torch.cuda.device_count())
            print ('Current cuda device ', torch.cuda.current_device())
            model = nn.DataParallel(model).to(device)
        else:
            model.to(device)
        if args.lr != 0:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = optim.Adam(model.parameters())
        models.append(ModelSet(i, name, model, optimizer))
        i += 1
        #ori_lr = optimizer.defaults['lr']
    start_epoch = 1

    #scheduler = StepLR(optimizer, step_size=1)
    for epoch in range(start_epoch, args.epochs + 1):
        if args.distributed:
            dist.barrier()
            train_sampler.set_epoch(epoch - 1)
        #lr = adjust_learning_rate(optimizer, epoch - 1, ori_lr)
        #if rank == 0:
        #    print("Epoch", epoch, "with lr", lr, flush=True)
        train_mul(args, models, device, train_loader, epoch, rank)
        if args.distributed:
            test_sampler.set_epoch(epoch - 1)
        test_mul(args, models, device, test_loader, rank)
        for ms in models:
            if args.distributed:
                cur_loss = torch.tensor(ms.cur_loss).to(device)
                dist.all_reduce(cur_loss, op=dist.ReduceOp.SUM)
                ms.cur_loss = cur_loss.item() / args.world_size
            if rank == 0:
                if ms.cur_loss < ms.min_loss:
                    print("Find new minimal loss", ms.cur_loss, "to replace", ms.min_loss, "of model", ms.idx)
                    ms.min_loss = ms.cur_loss
                    if not args.no_save_model:
                        save_checkpoint(ms.name, ms.model, ms.optimizer, epoch, ms.min_loss, args.lr, True)
                if (not args.no_save_model) and epoch % args.save_interval == 0:
                    save_checkpoint(ms.name, ms.model, ms.optimizer, epoch, ms.min_loss, args.lr)
        #scheduler.step()

    if args.distributed:
        # Clean up.
        dist.destroy_process_group()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Trace2Vec Training')
    parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                        help='input batch size (default: 4096)')
    parser.add_argument('--train-size', type=int, default=4096, metavar='N',
                        help='input size for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0, metavar='N',
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.5, metavar='N',
                        help='gradient normalization value (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                        help='how many epochs to save models')
    parser.add_argument('--no-save-model', action='store_true', default=False,
                        help='disable model saving')
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='whether to use distributed training')
    parser.add_argument('--nodes', type=int, default=1, metavar='N',
                        help='number of nodes (default: 1)')
    parser.add_argument('--node-rank', type=int, default=0, metavar='N',
                        help='rank of this node (default: 0)')
    parser.add_argument('--gpus', type=int, default=1, metavar='N',
                        help='number of gpus per node (default: 1)')
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()
    if args.distributed:
        args.world_size = args.gpus * args.nodes
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        mp.spawn(main_rank, args=(args,), nprocs=args.gpus, join=True)
    else:
        main_rank(0, args)


if __name__ == '__main__':
    main()
