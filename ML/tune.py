from __future__ import print_function
import argparse
import os
import sys
import time
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR

from ML.custom_data import *
from ML.utils import profile_model, generate_model_name, get_representation_dim
from ML.models import *


loss_fn = nn.MSELoss()


class ModelSet:
    def __init__(self, idx, name, model, optimizer, scheduler):
        self.idx = idx
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.min_loss = float("inf")
        self.cur_loss = 0
        self.total_loss = 0
        self.scheduler = scheduler


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
            if args.clip > 0:
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
        ms.total_loss /= len(train_loader.dataset)
        print('Train Epoch {} {}: {} \tLoss: {:.6f} \tTime: {:.1f}'.format(
            epoch, ms.idx, rank, ms.total_loss, end_t - start_t), flush=True)


def train_sbatch_mul(args, cfg, models, device, train_loader, epoch, rank):
    for ms in models:
        ms.model.train()
        ms.total_loss = 0
    start_t = time.time()
    print_threshold = max(len(train_loader) // 100, 1)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for i in range(args.sbatch_size):
            cur_data = data[:,i:i+cfg.seq_length,:]
            cur_target = target[:,i,:]
            for ms in models:
                ms.optimizer.zero_grad()
                output = ms.model(cur_data)
                loss = loss_fn(output, cur_target)
                ms.total_loss += loss.item()
                loss.backward()
                if args.clip > 0:
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
        ms.total_loss /= len(train_loader.dataset)
        ms.total_loss /= args.sbatch_size
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
        ms.total_loss /= len(test_loader.dataset)
        ms.cur_loss = ms.total_loss
        print('Test set {} {}: Loss: {:.6f}'.format(
            ms.idx, rank, ms.total_loss), flush=True)


def test_sbatch_mul(args, cfg, models, device, test_loader, rank):
    for ms in models:
        ms.model.eval()
        ms.total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for i in range(args.sbatch_size):
                cur_data = data[:,i:i+cfg.seq_length,:]
                cur_target = target[:,i,:]
                for ms in models:
                    output = ms.model(cur_data)
                    ms.total_loss += loss_fn(output, cur_target).item()
    for ms in models:
        ms.total_loss /= len(test_loader.dataset)
        ms.total_loss /= args.sbatch_size
        ms.cur_loss = ms.total_loss
        print('Test set {} {}: Loss: {:.6f}'.format(
            ms.idx, rank, ms.total_loss), flush=True)


def save_checkpoint(name, model, optimizer, epoch, best_loss, cfg, lr, best=False):
    if lr != 0:
        lr_name = '_lr' + str(lr)
    else:
        lr_name = ''
    if best:
        name = 'checkpoints/' + generate_model_name(name) + '_' + cfg + lr_name + '_best.pt'
    else:
        name = 'checkpoints/' + generate_model_name(name, epoch) + '_' + cfg + lr_name + '.pt'
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


def load_checkpoint(name, model):
    assert 'checkpoints/' in name
    cp = torch.load(name, map_location=torch.device('cpu'))
    model.load_state_dict(cp['model_state_dict'])
    print("Loaded checkpoint", name)


def main_rank(rank, args):
    if rank == 0:
        print("Load config", args.cfg, flush=True)
    cfg = importlib.import_module("CFG.%s" % args.cfg)

    if args.distributed:
        # create default process group
        global_rank = args.node_rank * args.gpus + rank
        dist.init_process_group("nccl", rank=global_rank, world_size=args.world_size)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.sbatch:
        dataset1 = CombinedMMBDataset(cfg, cfg.data_set_idx, 0, args.train_size, rank)
        dataset2 = CombinedMMBDataset(cfg, cfg.data_set_idx, cfg.valid_start, cfg.valid_end, rank)
    else:
        dataset1 = CombinedMMDataset(cfg, cfg.data_set_idx, 0, args.train_size)
        dataset2 = CombinedMMDataset(cfg, cfg.data_set_idx, cfg.valid_start, cfg.valid_end)
    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        if args.distributed:
            num_workers = 0
        else:
            num_workers = 2
        cuda_kwargs = {'num_workers': num_workers,
                       'pin_memory': True}
        kwargs.update(cuda_kwargs)
    if args.distributed:
        #train_sampler = torch.utils.data.distributed.DistributedSampler(
        #    dataset1, num_replicas=args.world_size, rank=global_rank, shuffle=False)
        shuffle_kwargs = {'shuffle': True}
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset1, **shuffle_kwargs)
        train_loader = torch.utils.data.DataLoader(dataset1, sampler=train_sampler, **kwargs)
        shuffle_kwargs = {'shuffle': False}
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset2, **shuffle_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, sampler=test_sampler, **kwargs)
    else:
        shuffle_kwargs = {'shuffle': True}
        train_loader = torch.utils.data.DataLoader(dataset1, **kwargs, **shuffle_kwargs)
        shuffle_kwargs = {'shuffle': False}
        test_loader = torch.utils.data.DataLoader(dataset2, **kwargs, **shuffle_kwargs)

    assert len(args.models) == 1
    models = []
    model = eval(args.models[0])
    rep_dim = get_representation_dim(cfg, model)
    load_checkpoint(args.checkpoints, model)
    if args.uarch_net:
        for param in model.parameters():
            param.requires_grad = False
        model.init_paras()
        for param in model.uarch_net.parameters():
            param.requires_grad = True
    else:
        if args.uarch:
            for param in model.parameters():
                param.requires_grad = False
        # Replace the linear layer.
        model.linear = nn.Linear(rep_dim, cfg.cfg_num * cfg.tgt_length, bias=args.bias)
    if rank == 0:
        profile_model(cfg, model)
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
    opt_args = {}
    if args.lr != 0:
        lr_arg = {'lr': args.lr}
        opt_args.update(lr_arg)
    if args.wd != 0:
        wd_arg = {'weight_decay': args.wd}
        opt_args.update(wd_arg)
    if args.uarch_net:
        if args.distributed or torch.cuda.device_count() > 1:
            optimizer = optim.Adam(model.module.uarch_net.parameters(), **opt_args)
        else:
            optimizer = optim.Adam(model.uarch_net.parameters(), **opt_args)
    elif args.uarch:
        if args.distributed or torch.cuda.device_count() > 1:
            optimizer = optim.Adam(model.module.linear.parameters(), **opt_args)
        else:
            optimizer = optim.Adam(model.linear.parameters(), **opt_args)
    else:
        optimizer = optim.Adam(model.parameters(), **opt_args)
    scheduler = None
    if args.lr_step > 0:
        scheduler = StepLR(optimizer, step_size=args.lr_step, verbose=True)
    if args.uarch_net:
        name = "unet_" + args.models[0]
    elif args.uarch:
        name = "uarch_" + args.models[0]
    else:
        name = "ft_" + args.models[0]
    models.append(ModelSet(0, name, model, optimizer, scheduler))
    start_epoch = 1

    for epoch in range(start_epoch, args.epochs + 1):
        if args.distributed:
            dist.barrier()
            train_sampler.set_epoch(epoch - 1)
        if args.sbatch:
            train_sbatch_mul(args, cfg, models, device, train_loader, epoch, rank)
        else:
            train_mul(args, models, device, train_loader, epoch, rank)
        if args.distributed:
            test_sampler.set_epoch(epoch - 1)
        if args.sbatch:
            test_sbatch_mul(args, cfg, models, device, test_loader, rank)
        else:
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
                        save_checkpoint(ms.name, ms.model, ms.optimizer, epoch, ms.min_loss, args.cfg, args.lr, True)
                if (not args.no_save_model) and epoch % args.save_interval == 0:
                    save_checkpoint(ms.name, ms.model, ms.optimizer, epoch, ms.min_loss, args.cfg, args.lr)
            if args.lr_step > 0:
                ms.scheduler.step()

    if args.distributed:
        # Clean up.
        dist.destroy_process_group()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PerfVec Tuning')
    parser.add_argument('--cfg', required=True, help='config file')
    parser.add_argument('--uarch', action='store_true', default=False,
                        help='learn micro-architecture representations only')
    parser.add_argument('--uarch-net', action='store_true', default=False,
                        help='learn the micro-architecture representation model only')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='use bias for the linear layer')
    parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                        help='input batch size (default: 4096)')
    parser.add_argument('--train-size', type=int, default=4096, metavar='N',
                        help='input size for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0, metavar='N',
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0, metavar='N',
                        help='weight decay rate')
    parser.add_argument('--lr-step', type=int, default=0,
                        help='lr scheduler step size')
    parser.add_argument('--clip', type=float, default=0, metavar='N',
                        help='gradient normalization value (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--sbatch', action='store_true', default=False,
                        help='uses small batch training')
    parser.add_argument('--sbatch-size', type=int, default=512, metavar='N',
                        help='small batch size (default: 512)')
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
    parser.add_argument('--checkpoints', required=True)
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
