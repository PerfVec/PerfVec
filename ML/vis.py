import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .custom_data import *
from .utils import generate_model_name
from .models import *
from CFG import *
from ML.test import load_checkpoint


def vis():
    # Training settings
    parser = argparse.ArgumentParser(description='SIMNET Testing')
    parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                        help='input batch size (default: 4096)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA')
    parser.add_argument('--sbatch', action='store_true', default=False,
                        help='uses small batch training')
    parser.add_argument('--sbatch-size', type=int, default=512, metavar='N',
                        help='small batch size (default: 512)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--tsne', action='store_true', default=False,
                        help='uses t-SNE')
    parser.add_argument('--dim', type=int, default=2, metavar='N',
                        help='projection dimension')
    parser.add_argument('--prog', action='store_true', default=False,
                        help='visualizes program representations')
    parser.add_argument('--checkpoints', required=True)
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()

    if args.prog:
        assert "res/" in args.checkpoints
        rep = torch.load(args.checkpoints, map_location=torch.device('cpu'))
        rep = rep.detach().numpy()
        print("Program representations' shape is", rep.shape)
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
    if args.dim == 3:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
    elif args.dim == 2:
        fig, ax = plt.subplots(figsize=(8,8))
    else:
        print("Do not support the dimension of", args.dim)
        sys.exit()
    all_idx = np.arange(0, num, dtype=int)

    if args.prog:
        assert num == len(sim_datasets)
        cmap = cm.get_cmap('tab20')
        for idx in range(num):
            name = sim_datasets[idx][0]
            name = name.replace(data_set_dir, "").replace("_r.in.mmap.norm", "")
            if args.dim == 2:
                ax.scatter(proj[idx,0], proj[idx,1], c=np.array(cmap(idx)).reshape(1,4), label=name, alpha=0.5)
            elif args.dim == 3:
                ax.scatter(proj[idx,0], proj[idx,1], proj[idx,2], c=np.array(cmap(idx)).reshape(1,4), label=name, alpha=0.5)
        ax.legend(fontsize='small', markerscale=2)
        file_name = args.checkpoints.replace("res/", "fig/")
    else:
        cmap = cm.get_cmap('tab10')
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
                ax.scatter(proj[idx,0], proj[idx,1], c=np.array(cmap(label)).reshape(1,4), label=name, alpha=0.5)
            elif args.dim == 3:
                ax.scatter(proj[idx,0], proj[idx,1], proj[idx,2], c=np.array(cmap(label)).reshape(1,4), label=name, alpha=0.5)
        #for idx in range(num):
        #    label = idx
        #    if args.dim == 2:
        #        ax.scatter(proj[idx,0], proj[idx,1], c=np.array(cmap(label)).reshape(1,4), label=label, alpha=0.5)
        #    elif args.dim == 3:
        #        ax.scatter(proj[idx,0], proj[idx,1], proj[idx,2], c=np.array(cmap(label)).reshape(1,4), label=label, alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
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
    vis()
