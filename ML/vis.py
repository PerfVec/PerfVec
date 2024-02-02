import argparse
import os
import sys
import time
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .custom_data import *
from .utils import generate_model_name
from .models import *
from ML.test import load_checkpoint


def norm(rep, opt_lvl):
    # Normalize based on the number of instructions.
    cfg = importlib.import_module("CFG.com_o%d_1022" % opt_lvl)
    for i in range(len(cfg.sim_datasets)):
        rep[i] /= cfg.sim_datasets[i][1]
    return rep


def vis(args):
    if args.prog:
        assert "res/" in args.checkpoints
        rep = torch.load(args.checkpoints, map_location=torch.device('cpu'))
        rep = rep.detach().numpy()
        print("Program representations' shape is", rep.shape)
    elif args.opt:
        assert "res/" in args.checkpoints
        rep = torch.load(args.checkpoints, map_location=torch.device('cpu'))
        #rep = norm(rep, 0)
        rep = rep.view(rep.shape[0], 1, rep.shape[1])
        for i in range(1, 4):
            file_name = args.checkpoints.replace("_o0_", "_o%d_" % i)
            print("Open", file_name)
            cur_rep = torch.load(file_name, map_location=torch.device('cpu'))
            #cur_rep = norm(cur_rep, i)
            cur_rep = cur_rep.view(cur_rep.shape[0], 1, cur_rep.shape[1])
            rep = torch.cat((rep, cur_rep), dim=1)
        rep = rep.detach().numpy()
        print("Input shape is", rep.shape)
        rep = rep.reshape((-1, rep.shape[2]))
    else:
        assert len(args.models) == 1
        model = eval(args.models[0])
        load_checkpoint(args.checkpoints, model)
        rep = model.linear.weight.detach().numpy()
        print("Micro-architecture representations' shape is", rep.shape)
        print(rep)

    # Project representations.
    num = rep.shape[0]
    if args.tsne:
        tsne = TSNE(args.dim, verbose=1)
        proj = tsne.fit_transform(rep)
    else:
        pca = PCA(n_components=args.dim)
        pca.fit(rep)
        proj = pca.transform(rep)
        print("PCA:", pca.explained_variance_ratio_)

    # Visualize representations.
    mpl.rcParams['text.usetex'] = True
    font = {'size' : 24}
    plt.rc('font', **font)
    ms = 100
    fig_size = 8
    if args.dim == 3:
        fig = plt.figure(figsize=(fig_size, fig_size))
        ax = fig.add_subplot(111, projection='3d')
    elif args.dim == 2:
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    else:
        print("Do not support the dimension of", args.dim)
        sys.exit()

    if args.prog:
        from CFG import sim_datasets, data_set_dir
        assert num == len(sim_datasets)
        cmap = cm.get_cmap('tab20')
        for idx in range(num):
            name = sim_datasets[idx][0]
            name = name.replace(data_set_dir, "").replace("_r.in.mmap.norm", "")
            if args.dim == 2:
                ax.scatter(proj[idx,0], proj[idx,1], c=np.array(cmap(idx)).reshape(1,4), label=name, alpha=0.5)
            elif args.dim == 3:
                ax.scatter(proj[idx,0], proj[idx,1], proj[idx,2], c=np.array(cmap(idx)).reshape(1,4), label=name, alpha=0.5)
        ax.legend(markerscale=2)
        file_name = args.checkpoints.replace("res/", "fig/")
    elif args.opt:
        from CFG.com_o0_1022 import sim_datasets, data_set_dir
        assert num == 4 * len(sim_datasets)
        proj = proj.reshape((len(sim_datasets), 4, args.dim))
        cmap = cm.get_cmap('tab10')
        markers = ["o", "^", "s", "d"]
        #mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        mask = [2, 3, 4, 7, 8, 9, 11]
        #mask = [0, 1, 4, 5, 6, 7, 9, 10, 12, 13, 14]
        #for idx in range(len(sim_datasets)):
        cur_idx = 0
        for idx in mask:
            name = sim_datasets[idx][0]
            name = name.replace(data_set_dir, "").replace(".in.mmap.norm", "").replace("_r", "").replace("_ir", "")
            for i in range(4):
                cidx = 4 * idx + i
                if args.dim == 2:
                    if cur_idx == 0 or i == 0:
                        if cur_idx == 0:
                            label = name + "_O%d" % i
                        else:
                            label = name
                        ax.scatter(proj[idx,i,0], proj[idx,i,1], c=np.array(cmap(cur_idx)).reshape(1,4), label=label, alpha=0.5, marker=markers[i], s=ms)
                    else:
                        ax.scatter(proj[idx,i,0], proj[idx,i,1], c=np.array(cmap(cur_idx)).reshape(1,4), alpha=0.5, marker=markers[i], s=ms)
                    if i < 3:
                        x = [proj[idx,i,0], proj[idx,i+1,0]]
                        y = [proj[idx,i,1], proj[idx,i+1,1]]
                        plt.plot(x, y, c=np.array(cmap(cur_idx)).reshape(1,4), alpha=0.5)
                elif args.dim == 3:
                    label = name + "_O%d" % i
                    ax.scatter(proj[idx,i,0], proj[idx,i,1], proj[idx,i,2], c=np.array(cmap(cur_idx)).reshape(1,4), label=label, alpha=0.5, marker=markers[i])
            cur_idx += 1
        ax.legend(markerscale=1.5, shadow=False, borderpad=0.1, borderaxespad=0.1, framealpha=0.3)
        file_name = args.checkpoints.replace("res/", "fig/opt_")
    else:
        cmap = cm.get_cmap('tab10')
        all_idx = np.arange(0, num, dtype=int)
        for label in range(2):
            if label == 0:
                idx = all_idx < 60
                idx |= (all_idx >= 70) & (all_idx < 74) 
                name = "out-of-order"
            elif label == 1:
                idx = (all_idx < 70) & (all_idx >= 60)
                idx |= (all_idx >= 74) & (all_idx < 77) 
                name = "in-order"
            if args.dim == 2:
                ax.scatter(proj[idx,0], proj[idx,1], c=np.array(cmap(label)).reshape(1,4), label=name, s=ms, alpha=0.5)
            elif args.dim == 3:
                ax.scatter(proj[idx,0], proj[idx,1], proj[idx,2], c=np.array(cmap(label)).reshape(1,4), label=name, s=ms, alpha=0.5)
        #for idx in range(num):
        #    label = idx
        #    if args.dim == 2:
        #        ax.scatter(proj[idx,0], proj[idx,1], c=np.array(cmap(label)).reshape(1,4), label=label, alpha=0.5)
        #    elif args.dim == 3:
        #        ax.scatter(proj[idx,0], proj[idx,1], proj[idx,2], c=np.array(cmap(label)).reshape(1,4), label=label, alpha=0.5)
        ax.legend(markerscale=1.5, shadow=False, borderpad=0.1, borderaxespad=0.1, framealpha=0.5)
        file_name = args.checkpoints.replace("checkpoints/", "fig/urep_")

    file_name = file_name.replace(".pt", "")
    if args.tsne:
        file_name += "_tsne"
    else:
        file_name += "_pca"
    file_name += "_dim" + str(args.dim) + ".pdf"
    print("Save to", file_name)
    fig.tight_layout()
    fig.savefig(file_name)
    plt.show()


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PerfVec Visualization')
    parser.add_argument('--prog', action='store_true', default=False,
                        help='visualizes program representations')
    parser.add_argument('--opt', action='store_true', default=False,
                        help='visualizes program representations under optimization levels')
    parser.add_argument('--tsne', action='store_true', default=False,
                        help='uses t-SNE')
    parser.add_argument('--dim', type=int, default=2, metavar='N',
                        help='projection dimension')
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()
    assert not (args.prog and args.opt)

    vis(args)
