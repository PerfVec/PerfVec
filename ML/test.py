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
try:
  import torch._dynamo
except:
  pass

from .models import *
from .custom_data import *
from .utils import profile_model, get_representation_dim, tensorlist2str


loss_fn = nn.MSELoss()


def analyze(args, cfg, output, target, cfg_num, seq=False):
    target = target.view(-1, cfg_num, cfg.tgt_length)
    output = output.view(-1, cfg_num, cfg.tgt_length)
    target = target.detach().numpy()
    output = output.detach().numpy()
    np.set_printoptions(suppress=True)
    print(output.shape)
    for c in range(cfg_num + 1):
        print("Config", c)
        for i in range(cfg.tgt_length):
            print(i, ":")
            if c == cfg_num:
                if seq:
                    cur_output = output[:,:,:,i].reshape(-1)
                    cur_target = target[:,:,:,i].reshape(-1)
                else:
                    cur_output = output[:,:,i].reshape(-1)
                    cur_target = target[:,:,i].reshape(-1)
            else:
                if seq:
                    cur_output = output[:,:,c,i]
                    cur_target = target[:,:,c,i]
                else:
                    cur_output = output[:,c,i]
                    cur_target = target[:,c,i]
            print("\ttarget:", cur_target)
            if not args.pred:
                cur_output = np.rint(cur_output)
                cur_target = np.rint(cur_target)
            errs = cur_target - cur_output
            print("\tnorm output:", cur_output)
            print("\terrors:", errs)
            errs = errs.ravel()

            flat_target = cur_target.ravel()
            norm_errs = errs / (flat_target + 1)
            print("\tError abs avg, norm abs avg, norm avg, and norm std:", np.average(np.abs(errs)), "\t", np.average(np.abs(norm_errs)), "\t", np.average(norm_errs), "\t", np.std(norm_errs))
            if seq:
                output_sum = np.sum(cur_output, axis=1)
                target_sum = np.sum(cur_target, axis=1)
                sum_errs = target_sum - output_sum
                sum_errs[sum_errs < 0] = -sum_errs[sum_errs < 0]
                print("Sum err avg, persentage, and std:", np.average(sum_errs), "\t", np.average(sum_errs / target_sum), "\t", np.std(sum_errs))
            his = np.histogram(errs, bins=range(-10, 10))
            print(his[0] / errs.size)


def test(args, cfg, model, device, test_loader, cfg_num, rank):
  model.eval()
  start_t = time.time()
  total_loss = 0
  total_output = torch.zeros(0, cfg_num * cfg.tgt_length, device=device)
  total_target = torch.zeros(0, cfg_num * cfg.tgt_length, device=device)
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      if args.sbatch:
        for i in range(args.sbatch_size):
          cur_data = data[:,i:i+cfg.seq_length,:]
          cur_target = target[:,i,:]
          output = model(cur_data)
          if args.select:
            output = cfg.sel_output(output)
          total_loss += loss_fn(output, cur_target).item()
          total_output = torch.cat((total_output, output), 0)
          total_target = torch.cat((total_target, cur_target), 0)
      else:
        output = model(data)
        if args.select:
          output = cfg.sel_output(output)
        total_loss += loss_fn(output, target).item()
        total_output = torch.cat((total_output, output), 0)
        total_target = torch.cat((total_target, target), 0)
  end_t = time.time()
  total_loss /= len(test_loader)
  if args.sbatch:
    total_loss /= args.sbatch_size
  if args.distributed:
    test_loss = torch.tensor(total_loss).to(device)
    if rank == 0:
      gather_list = [torch.zeros_like(test_loss) for _ in range(args.world_size)]
      dist.gather(test_loss, gather_list, dst=0)
      avg_loss = torch.mean(torch.stack(gather_list))
      total_loss = avg_loss.item()
      print('Test loss: {:.6f} ({}) \tTime: {:.1f} s'.format(
          total_loss, tensorlist2str(gather_list), end_t - start_t), flush=True)
    else:
      dist.gather(test_loss, dst=0)
  else:
    print('Test loss: {:.6f} \tTime: {:.1f} s'.format(total_loss, end_t - start_t), flush=True)
    analyze(args, cfg, total_output.cpu(), total_target.cpu(), cfg_num)


def infer(args, cfg, model, device, test_loader, rank, out_dim, with_target=True):
  model.eval()
  start_t = time.time()
  output_sum = torch.zeros(out_dim, device=device)
  batch_output_sum = torch.zeros(out_dim, device=device)
  if with_target:
    total_loss = 0
    target_sum = torch.zeros(out_dim, device=device)
    batch_target_sum = torch.zeros(out_dim, device=device)
  if args.phase:
    ph_num = len(test_loader)
    if args.distributed:
      ph_num_t = torch.tensor(ph_num).to(device)
      dist.all_reduce(ph_num_t, op=dist.ReduceOp.MAX)
      ph_num = ph_num_t.item()
    if rank == 0:
      print(ph_num, "phases in total.")
    ph_res = torch.zeros(ph_num, 2 if with_target else 1, out_dim, device=device)
    ph_idx = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      batch_output_sum = 0
      if with_target:
        batch_target_sum = 0
      if args.sbatch:
        for i in range(args.sbatch_size):
          cur_data = data[:,i:i+cfg.seq_length,:]
          output = model(cur_data)
          if args.select:
            output = cfg.sel_output(output)
          batch_output_sum += torch.sum(output, dim=0)
          if with_target:
            cur_target = target[:,i,:]
            batch_target_sum += torch.sum(cur_target, dim=0)
            total_loss += loss_fn(output, cur_target).item()
      else:
        output = model(data)
        if args.select:
          output = cfg.sel_output(output)
        batch_output_sum += torch.sum(output, dim=0)
        if with_target:
          batch_target_sum += torch.sum(target, dim=0)
          total_loss += loss_fn(output, target).item()
      output_sum += batch_output_sum
      if with_target:
        target_sum += batch_target_sum
      if args.phase:
        ph_res[ph_idx, 0] = batch_output_sum
        if with_target:
          ph_res[ph_idx, 1] = batch_target_sum
        ph_idx += 1

  if args.distributed:
    if rank == 0:
      gather_list = [torch.zeros_like(output_sum) for _ in range(args.world_size)]
      dist.gather(output_sum, gather_list, dst=0)
      output_sum = torch.sum(torch.stack(gather_list), dim=0)
    else:
      dist.gather(output_sum, dst=0)
    if with_target:
      if rank == 0:
        gather_list = [torch.zeros_like(target_sum) for _ in range(args.world_size)]
        dist.gather(target_sum, gather_list, dst=0)
        target_sum = torch.sum(torch.stack(gather_list), dim=0)
      else:
        dist.gather(target_sum, dst=0)
    if args.phase:
      if rank == 0:
        gather_list = [torch.zeros_like(ph_res) for _ in range(args.world_size)]
        dist.gather(ph_res, gather_list, dst=0)
        ph_res = torch.sum(torch.stack(gather_list), dim=0)
      else:
        dist.gather(ph_res, dst=0)
  elif args.phase:
    assert ph_idx == ph_num

  end_t = time.time()
  res = {'time': end_t - start_t,
         'output_sum': output_sum.cpu()}
  if with_target:
    total_loss /= len(test_loader.dataset)
    if args.sbatch:
      total_loss /= args.sbatch_size
    # FIXME: total_loss needs to be gathered.
    extra_res = {'target_sum': target_sum.cpu(),
                 'loss': total_loss}
    res.update(extra_res)
  if args.phase:
    extra_res = {'ph_res': ph_res.cpu()}
    res.update(extra_res)
  return res


def simulate(args, cfg, model, device, test_loader, name, cfg_num, rank):
  res = infer(args, cfg, model, device, test_loader, rank, cfg_num * cfg.tgt_length, with_target=True)
  if rank == 0:
    target_sum = res['target_sum'].view(cfg_num, cfg.tgt_length)
    output_sum = res['output_sum'].view(cfg_num, cfg.tgt_length)
    error = (output_sum - target_sum) / target_sum
    max_sum = torch.max(target_sum, output_sum)
    norm_error = (output_sum - target_sum) / max_sum
    print("Target:", target_sum)
    print("Output:", output_sum)
    print("Error:", error)
    print("Mean error:", torch.mean(torch.abs(error), dim=0))
    print("Mean normalized error:", torch.mean(torch.abs(norm_error), dim=0))
    if args.uarch:
      print("Mean unseen error:", torch.mean(torch.abs(error[1:]), dim=0))
      print("Mean normalized unseen error:", torch.mean(torch.abs(norm_error[1:]), dim=0))
    if cfg.tgt_length >= 3:
      averaged_sum = torch.mean(output_sum[:, 0:3], dim=1)
      averaged_error = (averaged_sum  - target_sum[:, 2]) / target_sum[:, 2]
      norm_averaged_error = (averaged_sum  - target_sum[:, 2]) / max_sum[:, 2]
      print("Averaged time:", averaged_sum)
      print("Averaged error:", averaged_error)
      print("Mean averaged error:", torch.mean(torch.abs(averaged_error), dim=0).item())
      print("Mean normalized averaged error:", torch.mean(torch.abs(norm_averaged_error), dim=0).item())
      if args.uarch:
        print("Mean averaged unseen error:", torch.mean(torch.abs(averaged_error[1:]), dim=0).item())
        print("Mean normalized averaged unseen error:", torch.mean(torch.abs(norm_averaged_error[1:]), dim=0).item())
    if args.uarch_net_unseen:
      file_name = args.checkpoints.replace("checkpoints/", "res/sim_%s_%s_" % (args.cfg, name))
      print("Save simulation results to", file_name)
      torch.save(output_sum.cpu(), file_name)
    if args.phase:
      file_name = args.checkpoints.replace("checkpoints/", "res/ph_%s_%s_" % (args.cfg, name))
      torch.save(res['ph_res'], file_name)
      print("Saved phase results to", file_name)
    print('Loss: {:.6f} \tTime: {:.1f}'.format(res['loss'], res['time']), flush=True)
    if args.save_sim:
      target_sum = target_sum.view(1, cfg_num, cfg.tgt_length)
      output_sum = output_sum.view(1, cfg_num, cfg.tgt_length)
      error = error.view(1, cfg_num, cfg.tgt_length)
      res = torch.cat((target_sum, output_sum, error), 0).cpu()
      return res
  return 0


def get_program_representation(args, cfg, model, device, test_loader, rep_dim, name, rank):
  res = infer(args, cfg, model, device, test_loader, rank, rep_dim, with_target=False)
  if rank == 0:
    rep_sum = res['output_sum']
    print("Representation:", rep_sum)
    if args.phase:
      file_name = args.checkpoints.replace("checkpoints/", "res/phrep_%s_%s_" % (args.cfg, name))
      ph_res = res['ph_res'].view(-1, rep_dim)
      torch.save(ph_res, file_name)
      print("Saved phase representations to", file_name)
    if args.save_rep_sep:
      file_name = args.checkpoints.replace("checkpoints/", "res/prep_%s_%s_" % (args.cfg, name))
      torch.save(rep_sum, file_name)
      print("Saved", name, "representation to", file_name)
    print('Time: {:.1f}'.format(res['time']), flush=True)
    return rep_sum
  else:
    return 0


def load_checkpoint(cp_name, cfg, init, model_name=None, training=False, optimizer=None):
  assert 'checkpoints/' in cp_name
  cp = torch.load(cp_name, map_location=torch.device('cpu'))
  if model_name is None:
    model = eval(cp['name'])
  else:
    model = eval(model_name)
  if init:
    model.init_paras()
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
  if training:
    assert optimizer is not None
    optimizer.load_state_dict(cp['optimizer_state_dict'])
  return model


def save_ts_model(cfg, name, model, device):
    assert 'checkpoints/' in name
    name = name.replace('checkpoints/', 'models/')
    model.eval()
    traced_script_module = torch.jit.trace(model, torch.rand(1, cfg.seq_length, cfg.input_length).to(device))
    traced_script_module.save(name)
    print("Saved model", name)


def test_main(rank, args):
  if rank == 0:
    print("Load config", args.cfg)
  cfg = importlib.import_module("CFG.%s" % args.cfg)

  if args.distributed:
    # create default process group
    global_rank = args.node_rank * args.gpus + rank
    dist.init_process_group("nccl", rank=global_rank, world_size=args.world_size)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  torch.manual_seed(args.seed)

  model_init = args.uarch_net or args.uarch_net_unseen
  if len(args.models) == 0:
    model = load_checkpoint(args.checkpoints, cfg, model_init)
  else:
    assert len(args.models) == 1
    model = load_checkpoint(args.checkpoints, cfg, model_init, args.models[0])
  if rank == 0:
    print("Loaded checkpoint", args.checkpoints)

  if args.rep:
    rep_dim = get_representation_dim(cfg, model)
  elif hasattr(cfg, 'sel_cfg_num'):
    cfg_num = cfg.sel_cfg_num
  else:
    cfg_num = cfg.cfg_num
  if args.uarch_net:
    cfg_num -= 1
  if args.uarch_net_unseen or args.pred:
    model.setup_test()
  #profile_model(cfg, model)
  if args.rep:
    model = RepExtractor(model)
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
  elif int(torch.__version__[0]) >= 2:
    print ('Enable PyTorch 2.0 compile.')
    model = torch.compile(model)
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.suppress_errors = True
    model.to(device)

  kwargs = {'batch_size': args.batch_size}
  if use_cuda:
    if args.distributed:
      num_workers = 0
    else:
      num_workers = 2
    cuda_kwargs = {'num_workers': num_workers,
                   'pin_memory': True}
    kwargs.update(cuda_kwargs)

  do_test = not args.no_test and not args.rep and not args.uarch_net_unseen
  if do_test:
    if args.pred:
      assert not args.sbatch
      dataset = RepDataset(cfg.dataset, cfg.test_start, cfg.test_end)
    elif args.sbatch:
      dataset = CombinedMMBDataset(cfg, cfg.data_set_idx, cfg.test_start, cfg.test_end, rank)
    else:
      dataset = CombinedMMDataset(cfg, cfg.data_set_idx, cfg.test_start, cfg.test_end)
    if args.distributed:
      shuffle_kwargs = {'shuffle': False}
      test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, **shuffle_kwargs)
      test_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, **kwargs)
      if rank == 0:
        print("Warning: standard test does not work correctly with DDP.")
    else:
      test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, **kwargs)
    if args.select:
      if rank == 0:
        print("Test with different micro-architecture arrangement.")
      assert hasattr(cfg, 'sel_output')
      assert not args.rep
    else:
      assert not hasattr(cfg, 'sel_output')
    test(args, cfg, model, device, test_loader, cfg_num, rank)
    #if not args.no_save and torch.cuda.device_count() <= 1:
    if rank == 0 and not args.no_save:
      save_ts_model(cfg, args.checkpoints, model, device)

  if args.sim or args.rep:
    if rank == 0:
      print("Run", args.sim_length, "instructions.", flush=True)
    if args.rep:
      all_rep = torch.zeros(len(cfg.sim_datasets), rep_dim)
      torch.set_printoptions(threshold=1000)
    else:
      res = torch.zeros(len(cfg.sim_datasets), 3, cfg_num, cfg.tgt_length)
    for i in range(len(cfg.sim_datasets)):
      name = cfg.sim_datasets[i][0].replace(cfg.data_set_dir, '').replace(".in.mmap.norm", '').replace(".in.nmmap", '')
      if rank == 0:
        print(name, flush=True)
      if args.sbatch:
        cur_dataset = MemMappedBatchDataset(cfg, cfg.sim_datasets[i], 0, args.sim_length // args.sbatch_size + 1, rank)
      else:
        cur_dataset = MemMappedDataset(cfg, cfg.sim_datasets[i][0], cfg.sim_datasets[i][1], 0, args.sim_length)
      if args.distributed:
        shuffle_kwargs = {'shuffle': False}
        test_sampler = torch.utils.data.distributed.DistributedSampler(cur_dataset, **shuffle_kwargs)
        test_loader = torch.utils.data.DataLoader(cur_dataset, sampler=test_sampler, **kwargs)
        if args.phase and rank == 0:
          print("Warning: phase level data may not be accurate with DDP.")
      else:
        test_loader = torch.utils.data.DataLoader(cur_dataset, **kwargs)
      if args.rep:
        all_rep[i] = get_program_representation(args, cfg, model, device, test_loader, rep_dim, name, rank)
      else:
        res[i] = simulate(args, cfg, model, device, test_loader, name, cfg_num, rank)
      if rank == 0:
        print('', flush=True)
    if args.rep and rank == 0:
      name = args.checkpoints.replace("checkpoints/", "res/prep_%s_" % args.cfg)
      print("Save program representations to", name, flush=True)
      torch.save(all_rep, name)
    if args.sim and args.save_sim and rank == 0:
      name = args.checkpoints.replace("checkpoints/", "res/simres_%s_" % args.cfg)
      print("Save simulation results to", name, flush=True)
      torch.save(res, name)


def main():
  # Test settings
  parser = argparse.ArgumentParser(description='PerfVec Testing')
  parser.add_argument('--cfg', required=True, help='config file')
  parser.add_argument('--sim', action='store_true', default=False,
                      help='simulates traces')
  parser.add_argument('--rep', action='store_true', default=False,
                      help='extracts program representations')
  parser.add_argument('--no-test', action='store_true', default=False,
                      help='does not perform standard testing')
  parser.add_argument('--uarch', action='store_true', default=False,
                      help='tests unseen micro-architectures')
  parser.add_argument('--uarch-net', action='store_true', default=False,
                      help='tests micro-architecture nets')
  parser.add_argument('--uarch-net-unseen', action='store_true', default=False,
                      help='tests micro-architecture nets on unseen programs')
  parser.add_argument('--pred', action='store_true', default=False,
                      help='tests predictors')
  parser.add_argument('--phase', action='store_true', default=False,
                      help='phase simulation')
  parser.add_argument('--sim-length', type=int, default=100000000, metavar='N',
                      help='simulation length (default: 100000000)')
  parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                      help='input batch size (default: 4096)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='disables CUDA')
  parser.add_argument('--no-save', action='store_true', default=False,
                      help='do not save model')
  parser.add_argument('--distributed', action='store_true', default=False,
                      help='whether to use DDP')
  parser.add_argument('--nodes', type=int, default=1, metavar='N',
                      help='number of nodes (default: 1)')
  parser.add_argument('--node-rank', type=int, default=0, metavar='N',
                      help='rank of this node (default: 0)')
  parser.add_argument('--gpus', type=int, default=1, metavar='N',
                      help='number of gpus per node (default: 1)')
  parser.add_argument('--save-sim', action='store_true', default=False,
                      help='save simulation results')
  parser.add_argument('--save-rep-sep', action='store_true', default=False,
                      help='save representations separately')
  parser.add_argument('--sbatch', action='store_true', default=False,
                      help='uses small batch training')
  parser.add_argument('--sbatch-size', type=int, default=512, metavar='N',
                      help='small batch size (default: 512)')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--select', action='store_true', default=False,
                      help='test set is a subset of training set')
  parser.add_argument('--checkpoints', required=True)
  parser.add_argument('models', nargs='*')
  args = parser.parse_args()

  if args.distributed:
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'
    mp.spawn(test_main, args=(args,), nprocs=args.gpus, join=True)
  else:
    args.world_size = 1
    test_main(0, args)


if __name__ == '__main__':
  main()
