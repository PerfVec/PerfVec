import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .custom_data import *
from .utils import profile_model
from .models import *
from CFG import *


loss_fn = nn.MSELoss()
inst_type = -2


def analyze(args, output, target, data):
    target = target.detach().numpy()
    data = data.detach().numpy()
    output = output.detach().numpy()
    np.set_printoptions(suppress=True)
    print(output.shape)
    for i in range(tgt_length):
        cur_output = output[:,:,i]
        cur_target = target[:,:,i]
        #print("output:", cur_output)
        print("target:", cur_target)
        cur_output = np.rint(cur_output)
        cur_target = np.rint(cur_target)
        print("norm output:", cur_output)
        errs = cur_target - cur_output
        print("errors:", errs)
        errs = errs.ravel()
        errs[errs < 0] = -errs[errs < 0]
        #errs[cur_cla_target == num_classes - 1] = -1

        #if inst_type >= -1:
        #    for i in range(errs.size):
        #        cur_inst_type = get_inst_type(data[i], 0, fs) - 1
        #        #print(cur_inst_type)
        #        assert cur_inst_type >= 0 and cur_inst_type < 37
        #        if inst_type >= 0 and cur_inst_type != inst_type:
        #            errs[i] = -1
        #        elif inst_type == -1 and (cur_inst_type == 25 or cur_inst_type == 26):
        #            errs[i] = -1
        #    print(errs)

        flat_target = cur_target.ravel()
        print("Err avg, persentage, and std:", np.average(errs[errs != -1]), "\t", np.sum(errs[errs != -1]) / np.sum(flat_target[errs != -1]), "\t", np.std(errs[errs != -1]))
        print("data percentage:", errs[errs != -1].size / errs.size)
        output_sum = np.sum(cur_output, axis=1)
        target_sum = np.sum(cur_target, axis=1)
        sum_errs = target_sum - output_sum
        sum_errs[sum_errs < 0] = -sum_errs[sum_errs < 0]
        print("Sum err avg, persentage, and std:", np.average(sum_errs), "\t", np.average(sum_errs / target_sum), "\t", np.std(sum_errs))
        his = np.histogram(errs, bins=range(-1, 100))
        print(his[0] / errs[errs != -1].size)


def test(args, model, device, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_fn(output, target).item()
            if not args.no_cuda:
                data, target, output = data.cpu(), target.cpu(), output.cpu()
            analyze(args, output, target, data)
    total_loss /= len(test_loader)
    print('Test set: Loss: {:.6f}'.format(total_loss), flush=True)


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
    parser = argparse.ArgumentParser(description='SIMNET Testing')
    parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                        help='input batch size (default: 4096)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA')
    parser.add_argument('--no-save', action='store_true', default=False,
                        help='do not save model')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    #dataset = MemMappedDataset(data_file_name, total_size, test_start, test_end)
    #dataset = CombinedMMDataset(4, test_start, test_end)
    dataset = MemMappedDataset(datasets[data_set_idx][0], datasets[data_set_idx][1], test_start, test_end)
    #dataset = NormMemMappedDataset(datasets[data_set_idx][0], datasets[data_set_idx][1], test_start, test_end)
    kwargs = {'batch_size': args.batch_size,
              'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        kwargs.update(cuda_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    assert len(args.models) == 1
    model = eval(args.models[0])
    load_checkpoint(args.checkpoints, model)
    #profile_model(model)
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    test(args, model, device, test_loader)
    if not args.no_save:
        save_ts_model(args.checkpoints, model, device)


if __name__ == '__main__':
    main()
