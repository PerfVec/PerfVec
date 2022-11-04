import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .custom_data import *
from .utils import profile_model, get_representation_dim
from .models import *
from CFG import *


loss_fn = nn.MSELoss()


def analyze(args, output, target, seq=False):
    target = target.view(-1, cfg_num, tgt_length)
    output = output.view(-1, cfg_num, tgt_length)
    target = target.detach().numpy()
    output = output.detach().numpy()
    np.set_printoptions(suppress=True)
    print(output.shape)
    for c in range(cfg_num + 1):
        print("Config", c)
        for i in range(tgt_length):
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
            #print("output:", cur_output)
            print("\ttarget:", cur_target)
            cur_output = np.rint(cur_output)
            cur_target = np.rint(cur_target)
            print("\tnorm output:", cur_output)
            errs = cur_target - cur_output
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


def test(args, model, device, test_loader):
    model.eval()
    total_loss = 0
    total_output = torch.zeros(0, cfg_num * tgt_length)
    total_target = torch.zeros(0, cfg_num * tgt_length)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.sbatch:
                for i in range(args.sbatch_size):
                    cur_data = data[:,i:i+seq_length,:]
                    cur_target = target[:,i,:]
                    output = model(cur_data)
                    if args.select:
                        output = sel_output(output)
                    total_loss += loss_fn(output, cur_target).item()
                    if not args.no_cuda:
                        output = output.cpu()
                        cur_target = cur_target.cpu()
                    total_output = torch.cat((total_output, output), 0)
                    total_target = torch.cat((total_target, cur_target), 0)
            else:
                output = model(data)
                if args.select:
                    output = sel_output(output)
                total_loss += loss_fn(output, target).item()
                if not args.no_cuda:
                    output = output.cpu()
                    target = target.cpu()
                total_output = torch.cat((total_output, output), 0)
                total_target = torch.cat((total_target, target), 0)
    total_loss /= len(test_loader.dataset)
    if args.sbatch:
        total_loss /= args.sbatch_size
    print('Test set: Loss: {:.6f}'.format(total_loss), flush=True)
    analyze(args, total_output, total_target)


def simulate(args, model, device, test_loader, name):
    model.eval()
    start_t = time.time()
    total_loss = 0
    target_sum = torch.zeros(cfg_num * tgt_length, device=device)
    output_sum = torch.zeros(cfg_num * tgt_length, device=device)
    batch_target_sum = torch.zeros(cfg_num * tgt_length, device=device)
    batch_output_sum = torch.zeros(cfg_num * tgt_length, device=device)
    if args.phase:
        ph_num = len(test_loader)
        print(ph_num, "phases in total.")
        ph_res = torch.zeros(ph_num, 2, cfg_num * tgt_length, device=device)
        ph_idx = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_target_sum = 0
            batch_output_sum = 0
            if args.sbatch:
                for i in range(args.sbatch_size):
                    cur_data = data[:,i:i+seq_length,:]
                    cur_target = target[:,i,:]
                    output = model(cur_data)
                    if args.select:
                        output = sel_output(output)
                    batch_target_sum += torch.sum(cur_target, dim=0)
                    batch_output_sum += torch.sum(output, dim=0)
                    total_loss += loss_fn(output, cur_target).item()
            else:
                output = model(data)
                if args.select:
                    output = sel_output(output)
                batch_target_sum += torch.sum(target, dim=0)
                batch_output_sum += torch.sum(output, dim=0)
                total_loss += loss_fn(output, target).item()
            target_sum += batch_target_sum
            output_sum += batch_output_sum
            if args.phase:
                ph_res[ph_idx, 0] = batch_target_sum
                ph_res[ph_idx, 1] = batch_output_sum
                ph_idx += 1
    end_t = time.time()
    target_sum = target_sum.view(cfg_num, tgt_length)
    output_sum = output_sum.view(cfg_num, tgt_length)
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
    if tgt_length >= 3:
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
    if args.phase:
        assert ph_idx == ph_num
        file_name = args.checkpoints.replace("checkpoints/", "res/ph_" + name + '_')
        print("Save phase results to", file_name)
        torch.save(ph_res.cpu(), file_name)
    total_loss /= len(test_loader.dataset)
    if args.sbatch:
        total_loss /= args.sbatch_size
    print('Loss: {:.6f} \tTime: {:.1f}'.format(total_loss, end_t - start_t), flush=True)


def get_program_representation(args, model, device, test_loader, rep_dim, name):
    model.eval()
    start_t = time.time()
    rep_sum = torch.zeros(rep_dim, device=device)
    batch_rep_sum = torch.zeros(rep_dim, device=device)
    if args.phase:
        ph_num = len(test_loader)
        print(ph_num, "phases in total.")
        ph_res = torch.zeros(ph_num, rep_dim, device=device)
        ph_idx = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            batch_rep_sum = 0
            if args.sbatch:
                for i in range(args.sbatch_size):
                    cur_data = data[:,i:i+seq_length,:]
                    #rep = model.extract_representation(cur_data)
                    _, rep = model(cur_data)
                    batch_rep_sum += torch.sum(rep, dim=0)
            else:
                rep = model.extract_representation(data)
                batch_rep_sum += torch.sum(rep, dim=0)
            rep_sum += batch_rep_sum
            if args.phase:
                ph_res[ph_idx] = batch_rep_sum
                ph_idx += 1
    end_t = time.time()
    rep_sum = rep_sum.cpu()
    print("Representation:", rep_sum)
    if args.phase:
        assert ph_idx == ph_num
        file_name = args.checkpoints.replace("checkpoints/", "res/phrep_" + name + '_')
        print("Save phase representations to", file_name)
        torch.save(ph_res.cpu(), file_name)
    print('Time: {:.1f}'.format(end_t - start_t), flush=True)
    return rep_sum


def load_checkpoint(name, model, training=False, optimizer=None):
    assert 'checkpoints/' in name
    cp = torch.load(name, map_location=torch.device('cpu'))
    model.load_state_dict(cp['model_state_dict'])
    if training:
        assert optimizer is not None
        optimizer.load_state_dict(cp['optimizer_state_dict'])
    print("Loaded checkpoint", name)


def save_ts_model(name, model, device):
    assert 'checkpoints/' in name
    name = name.replace('checkpoints/', 'models/')
    model.eval()
    traced_script_module = torch.jit.trace(model, torch.rand(1, seq_length, input_length).to(device))
    traced_script_module.save(name)
    print("Saved model", name)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Trace2Vec Testing')
    parser.add_argument('--sim', action='store_true', default=False,
                        help='simulates traces')
    parser.add_argument('--rep', action='store_true', default=False,
                        help='extracts program representations')
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
    parser.add_argument('--sbatch', action='store_true', default=False,
                        help='uses small batch training')
    parser.add_argument('--sbatch-size', type=int, default=512, metavar='N',
                        help='small batch size (default: 512)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--select', action='store_true', default=False,
                        help='test set is a subset of training set')
    parser.add_argument('--uarch', action='store_true', default=False,
                        help='tests unseen micro-architectures')
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    assert len(args.models) == 1
    model = eval(args.models[0])
    rep_dim = get_representation_dim(model)
    load_checkpoint(args.checkpoints, model)
    #profile_model(model)
    device = torch.device("cuda" if use_cuda else "cpu")
    if torch.cuda.device_count() > 1:
        print ('Available devices', torch.cuda.device_count())
        print ('Current cuda device', torch.cuda.current_device())
        model = nn.DataParallel(model)
    model.to(device)

    kwargs = {'batch_size': args.batch_size,
              'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        kwargs.update(cuda_kwargs)

    if not args.rep:
        if args.sbatch:
            dataset = CombinedMMBDataset(data_set_idx, test_start, test_end)
        else:
            dataset = CombinedMMDataset(data_set_idx, test_start, test_end)
            #dataset = MemMappedDataset(datasets[data_set_idx][0], datasets[data_set_idx][1], test_start, test_end)
            #dataset = NormMemMappedDataset(datasets[data_set_idx][0], datasets[data_set_idx][1], test_start, test_end)
        test_loader = torch.utils.data.DataLoader(dataset, **kwargs)
        if args.select:
            print("Test with different micro-architecture arrangement.")
            assert "sel_output" in globals()
        test(args, model, device, test_loader)
        if not args.no_save and torch.cuda.device_count() <= 1:
            save_ts_model(args.checkpoints, model, device)

    if args.sim or args.rep:
        print("Run", args.sim_length, "instructions.")
        if args.rep:
            all_rep = torch.zeros(len(sim_datasets), rep_dim)
            torch.set_printoptions(threshold=1000)
        for i in range(len(sim_datasets)):
            name = sim_datasets[i][0].replace(data_set_dir, '').replace(".in.mmap.norm", '')
            print(sim_datasets[i][0], flush=True)
            if args.sbatch:
                cur_dataset = MemMappedBatchDataset(sim_datasets[i], 0, args.sim_length // args.sbatch_size + 1)
            else:
                cur_dataset = MemMappedDataset(sim_datasets[i][0], sim_datasets[i][1], 0, args.sim_length)
            test_loader = torch.utils.data.DataLoader(cur_dataset, **kwargs)
            if args.rep:
                all_rep[i] = get_program_representation(args, model, device, test_loader, rep_dim, name)
            else:
                simulate(args, model, device, test_loader, name)
            print('', flush=True)
        if args.rep:
            name = args.checkpoints.replace("checkpoints/", "res/prep_")
            print("Save program representations to", name)
            torch.save(all_rep, name)


if __name__ == '__main__':
    main()
