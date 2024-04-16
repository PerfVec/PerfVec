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
try:
  import torch._dynamo
except:
  pass

from ML.custom_data import *
from ML.utils import *
from ML.models import *


class ModelSet:
    def __init__(self, idx, name, model, optimizer, scheduler, min_loss):
        self.idx = idx
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.min_loss = min_loss
        self.train_loss = 0
        self.test_loss = 0


def train_mul(args, models, device, train_loader, loss_fn, epoch, rank):
    for ms in models:
        ms.model.train()
        ms.train_loss = 0
    start_t = time.time()
    print_threshold = max(len(train_loader) // 100, 1)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for ms in models:
            ms.optimizer.zero_grad()
            output = ms.model(data)
            loss = loss_fn(output, target)
            ms.train_loss += loss.item()
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
    train_print(args, models, device, epoch, rank, len(train_loader), end_t - start_t)


def train_sbatch_mul(args, cfg, models, device, train_loader, loss_fn, epoch, rank):
    for ms in models:
        ms.model.train()
        ms.train_loss = 0
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
                ms.train_loss += loss.item()
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
    train_print(args, models, device, epoch, rank,
        len(train_loader) * args.sbatch_size, end_t - start_t)


def train_print(args, models, device, epoch, rank, size, time):
  for ms in models:
    ms.train_loss /= size
    if args.distributed:
      train_loss = torch.tensor(ms.train_loss).to(device)
      if rank == 0:
        gather_list = [torch.zeros_like(train_loss) for _ in range(args.world_size)]
        dist.gather(train_loss, gather_list, dst=0)
        avg_loss = torch.mean(torch.stack(gather_list))
        print('Train {} at Epoch {}: \tLoss: {:.6f} ({}) \tTime: {:.1f} s'.format(
            ms.idx, epoch, avg_loss.item(), tensorlist2str(gather_list), time), flush=True)
      else:
        dist.gather(train_loss, dst=0)
    else:
      print('Train {} at Epoch {}: \tLoss: {:.6f} \tTime: {:.1f} s'.format(
          ms.idx, epoch, ms.train_loss, time), flush=True)


def test_mul(args, models, device, test_loader, loss_fn, epoch, rank):
    for ms in models:
        ms.model.eval()
        ms.test_loss = 0
    start_t = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for ms in models:
                output = ms.model(data)
                ms.test_loss += loss_fn(output, target).item()
    end_t = time.time()
    test_postprocess(args, models, device, epoch, rank, len(test_loader), end_t - start_t)


def test_sbatch_mul(args, cfg, models, device, test_loader, loss_fn, epoch, rank):
    for ms in models:
        ms.model.eval()
        ms.test_loss = 0
    start_t = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for i in range(args.sbatch_size):
                cur_data = data[:,i:i+cfg.seq_length,:]
                cur_target = target[:,i,:]
                for ms in models:
                    output = ms.model(cur_data)
                    ms.test_loss += loss_fn(output, cur_target).item()
    end_t = time.time()
    test_postprocess(args, models, device, epoch, rank,
        len(test_loader) * args.sbatch_size, end_t - start_t)


def test_postprocess(args, models, device, epoch, rank, size, time):
  for ms in models:
    ms.test_loss /= size
    if args.distributed:
      test_loss = torch.tensor(ms.test_loss).to(device)
      if rank == 0:
        gather_list = [torch.zeros_like(test_loss) for _ in range(args.world_size)]
        dist.gather(test_loss, gather_list, dst=0)
        avg_loss = torch.mean(torch.stack(gather_list))
        ms.test_loss = avg_loss.item()
        print('Test {}: \tLoss: {:.6f} ({}) \tTime: {:.1f} s'.format(
            ms.idx, ms.test_loss, tensorlist2str(gather_list), time), flush=True)
      else:
        dist.gather(test_loss, dst=0)
    else:
      print('Test {}: \tLoss: {:.6f} \tTime: {:.1f} s'.format(
          ms.idx, ms.test_loss, time), flush=True)

    # Update minimal loss.
    if rank == 0:
      if ms.test_loss < ms.min_loss:
        print("Find new minimal loss", ms.test_loss, "to replace", ms.min_loss, "of model", ms.idx)
        ms.min_loss = ms.test_loss
        if not args.no_save_model:
          save_checkpoint(ms, epoch, args, True)
      if (not args.no_save_model) and epoch % args.save_interval == 0:
        save_checkpoint(ms, epoch, args)


def save_checkpoint(ms, epoch, args, best=False):
  extra_name = ''
  if args.lr != 0:
    extra_name = '_lr' + str(args.lr)
  if args.loss != "MSE":
    extra_name += '_lo' + args.loss
  if best:
    file_name = generate_model_name(ms.name) + '_' + args.cfg + extra_name + '_best.pt'
  else:
    file_name = generate_model_name(ms.name, epoch) + '_' + args.cfg + extra_name + '.pt'
  saved_dict = {'name': ms.name,
                'epoch': epoch,
                'best_loss': ms.min_loss,
                'optimizer_state_dict': ms.optimizer.state_dict()}
  #model = getattr(ms.model, '_orig_mod', ms.model)
  if args.distributed or torch.cuda.device_count() > 1:
    model_dict = {'model_state_dict': ms.model.module.state_dict()}
  else:
    model_dict = {'model_state_dict': ms.model.state_dict()}
  saved_dict.update(model_dict)
  if ms.scheduler is not None:
    sch_dict = {'scheduler_state_dict': ms.scheduler.state_dict()}
    saved_dict.update(sch_dict)
  torch.save(saved_dict, 'checkpoints/' + file_name)
  print("Saved checkpoint at", 'checkpoints/' + file_name)


def load_checkpoint(rank, name, model):
  assert 'checkpoints/' in name
  cp = torch.load(name, map_location='cpu')
  # Address compiled model loading.
  keys_list = list(cp['model_state_dict'].keys())
  for key in keys_list:
    if key.startswith('_orig_mod.'):
      new_key = key[len('_orig_mod.'):]
      if new_key.startswith('module.'):
        new_key = new_key[len('module.'):]
      cp['model_state_dict'][new_key] = cp['model_state_dict'][key]
      del cp['model_state_dict'][key]
  model.load_state_dict(cp['model_state_dict'])
  start_epoch = cp['epoch']
  if rank == 0:
    print("Loaded checkpoint", name, "at epoch", start_epoch)
  return start_epoch + 1, cp['best_loss']


def load_optimizer_scheduler(rank, name, optimizer, scheduler):
  assert 'checkpoints/' in name
  cp = torch.load(name, map_location='cpu')
  optimizer.load_state_dict(cp['optimizer_state_dict'])
  if 'scheduler_state_dict' in cp:
    assert scheduler is not None
    scheduler.load_state_dict(cp['scheduler_state_dict'])
    if rank == 0:
      print("Loaded scheduler.")


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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

    #dataset1 = MemMappedDataset(data_file_name, total_size, 0, args.train_size)
    #dataset2 = MemMappedDataset(data_file_name, total_size, valid_start, valid_end)
    if args.learn_rep:
        dataset1 = RepDataset(cfg.dataset, 0, args.train_size)
        dataset2 = RepDataset(cfg.dataset, cfg.valid_start, cfg.valid_end)
    elif args.sbatch:
        dataset1 = CombinedMMBDataset(cfg, cfg.data_set_idx, 0, args.train_size, rank)
        dataset2 = CombinedMMBDataset(cfg, cfg.data_set_idx, cfg.valid_start, cfg.valid_end, rank)
    else:
        dataset1 = CombinedMMDataset(cfg, cfg.data_set_idx, 0, args.train_size)
        dataset2 = CombinedMMDataset(cfg, cfg.data_set_idx, cfg.valid_start, cfg.valid_end)
    #dataset1 = MemMappedDataset(datasets[data_set_idx][0], datasets[data_set_idx][1], 0, args.train_size)
    #dataset2 = MemMappedDataset(datasets[data_set_idx][0], datasets[data_set_idx][1], valid_start, valid_end)
    #dataset1 = NormMemMappedDataset(datasets[data_set_idx][0], datasets[data_set_idx][1], 0, args.train_size)
    #dataset2 = NormMemMappedDataset(datasets[data_set_idx][0], datasets[data_set_idx][1], valid_start, valid_end)
    #print(dataset1[0][0].size())
    #print(dataset1[0])
    #print(dataset1[12686])
    #print(dataset2[0])
    #exit()
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

    models = []
    i = 0
    start_epoch = 1
    min_loss = float("inf")
    for name in args.models:
        model = eval(name)
        if rank == 0:
            profile_model(cfg, model)
        if args.checkpoints is not None:
            assert len(args.models) == 1
            start_epoch, min_loss = load_checkpoint(rank, args.checkpoints, model)
        device = torch.device("cuda" if use_cuda else "cpu")
        if args.distributed:
            device = rank
            model = DDP(model.to(device), device_ids=[device])
            if int(torch.__version__[0]) >= 2:
                if rank == 0:
                    print ('Enable PyTorch 2.0 compile.')
                model = torch.compile(model)
                torch.set_float32_matmul_precision('high')
                torch._dynamo.config.suppress_errors = True
        elif torch.cuda.device_count() > 1:
            print ('Warning: data parallel will be deprecated.')
            print ('Available devices', torch.cuda.device_count())
            print ('Current cuda device', torch.cuda.current_device())
            model = nn.DataParallel(model).to(device)
        else:
            model.to(device)
            if int(torch.__version__[0]) >= 2:
                print ('Enable PyTorch 2.0 compile.')
                model = torch.compile(model)
                torch.set_float32_matmul_precision('high')
                torch._dynamo.config.suppress_errors = True
        opt_args = {}
        if args.lr != 0:
            lr_arg = {'lr': args.lr}
            opt_args.update(lr_arg)
        if args.wd != 0:
            wd_arg = {'weight_decay': args.wd}
            opt_args.update(wd_arg)
        optimizer = optim.Adam(model.parameters(), **opt_args)
        scheduler = None
        if args.lr_step > 0:
            scheduler = StepLR(optimizer, step_size=args.lr_step)
            if rank == 0:
                print ('Use a scheduler with a step of %s.' % args.lr_step)
        if args.checkpoints is not None:
            load_optimizer_scheduler(rank, args.checkpoints, optimizer, scheduler)
        models.append(ModelSet(i, name, model, optimizer, scheduler, min_loss))
        i += 1
        #ori_lr = optimizer.defaults['lr']
    if args.loss == "MSE":
        loss_fn = nn.MSELoss()
    elif args.loss == "L1":
        if rank == 0:
            print("Use L1Loss.")
        loss_fn = nn.L1Loss()
    elif args.loss == "NMSE":
        if rank == 0:
            print("Use normalized MSELoss.")
        loss_fn = NormMSELoss()
    elif args.loss == "NL1":
        if rank == 0:
            print("Use normalized L1Loss.")
        loss_fn = NormL1Loss()
    elif args.loss == "RMSE":
        if rank == 0:
            print("Use root MSELoss.")
        loss_fn = RMSELoss()
    else:
        raise AttributeError("%s is an invalid loss function." % args.loss)

    for epoch in range(start_epoch, args.epochs + 1):
        if args.distributed:
            dist.barrier()
            train_sampler.set_epoch(epoch - 1)
        if args.sbatch:
            train_sbatch_mul(args, cfg, models, device, train_loader, loss_fn, epoch, rank)
        else:
            train_mul(args, models, device, train_loader, loss_fn, epoch, rank)
        if args.distributed:
            test_sampler.set_epoch(epoch - 1)
        if args.sbatch:
            test_sbatch_mul(args, cfg, models, device, test_loader, loss_fn, epoch, rank)
        else:
            test_mul(args, models, device, test_loader, loss_fn, epoch, rank)
        for ms in models:
            if ms.scheduler is not None:
                ms.scheduler.step()
            #lr = adjust_learning_rate(ms.optimizer, epoch - 1, ori_lr)
            #if rank == 0:
            #    print("Epoch", epoch, "with lr", lr, flush=True)

    if args.distributed:
        # Clean up.
        dist.destroy_process_group()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PerfVec Training')
    parser.add_argument('--cfg', required=True, help='config file')
    parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                        help='input batch size (default: 4096)')
    parser.add_argument('--train-size', type=int, default=4096, metavar='N',
                        help='input size for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--loss', default="MSE", help='loss function')
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
    parser.add_argument('--learn-rep', action='store_true', default=False,
                        help='learns representations')
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
    parser.add_argument('--checkpoints', default=None)
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()

    if args.distributed:
        args.world_size = args.gpus * args.nodes
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        mp.spawn(main_rank, args=(args,), nprocs=args.gpus, join=True)
    else:
        args.world_size = 1
        main_rank(0, args)


if __name__ == '__main__':
    main()
