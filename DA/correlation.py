import argparse
import importlib
import torch
#from scipy import stats


def main():
    # Settings
    parser = argparse.ArgumentParser(description='PerfVec Pearson Correlation')
    parser.add_argument('--cfg', required=True, help='config file')
    parser.add_argument('--res', required=True)
    parser.add_argument('--skip', action='store_true', default=False,
                        help='skip first uarch')
    args = parser.parse_args()

    print("Load config", args.cfg)
    cfg = importlib.import_module("CFG.%s" % args.cfg)

    res = torch.load(args.res, map_location=torch.device('cpu'))
    print(res.shape)
    if args.skip:
        res = res[:, :, 1:, :]
    target = res[:, 0].reshape(1, -1)
    output = res[:, 1].reshape(1, -1)
    pc = torch.corrcoef(torch.cat((target, output), 0))
    print("Pearson correlation coefficient matrix:", pc)

    # IPC
    target = res[:, 0]
    output = res[:, 1]
    assert target.shape[0] == len(cfg.sim_datasets)
    for i in range(len(cfg.sim_datasets)):
        inst_num = cfg.sim_datasets[i][2]
        target[i] = inst_num / res[i, 0]
        output[i] = inst_num / res[i, 1]
    target = target.reshape(1, -1)
    output = output.reshape(1, -1)
    pc = torch.corrcoef(torch.cat((target, output), 0))
    print("IPC Pearson correlation coefficient matrix:", pc)
    #target = target.reshape(-1)
    #output = output.reshape(-1)
    #pc = stats.pearsonr(target.detach().numpy(), output.detach().numpy())
    #print("IPC Pearson correlation coefficient matrix:", pc)


if __name__ == '__main__':
    main()
