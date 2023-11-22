from GCN import GCN

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch
import torch_geometric

from random import randint
import math
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax
from sklearn.utils import murmurhash3_32

import numpy as np

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, Sequential
from torch_geometric.utils import add_self_loops, k_hop_subgraph, coalesce
import torch_geometric.transforms as T

from torch_geometric.nn import aggr

from typing import Union

from sys import maxsize

import code
import networkx as nx

from utils import load_dataset, add_k_hop_edges

from tqdm import tqdm

from torch_sparse import SparseStorage, SparseTensor

from main import MinHashClustering

import argparse

def parse_args():
    """
    Parse the arguments.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument('--dataset', default='Cora',
                        help='Dataset. Default is Cora. ')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of layers in mlp. Default is 3. ')
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help='Number of hidden channels in mlp. Default is 128. ')
    parser.add_argument("-m", "--method", type=str, choices=["metis","minhash"], 
                        help="The minibatching method to use.")
    # parser.add_argument('--eta', type=float, default=0.5,
    #                     help='Proportion of added edges. Default is 0.0. ')
    # parser.add_argument('--alpha', type=float, default=0.5,
    #                     help='Topological weight. Default is 0.0. ')
    # parser.add_argument('--beta', type=float, default=0.25,
    #                     help='Trained weight. Default is 1.0. ')
    # parser.add_argument('--add-self-loop', type=literal_eval, default=True,
    #                     help='Whether to add self-loops to all nodes. Default is False. ')
    # parser.add_argument('--trained-edge-weight-batch-size', type=int, default=50000,
    #                     help='Batch size for computing the trained edge weights. Default is 20000. ')
    # parser.add_argument('--graph-learning-type', default='mlp',
    #                     help='Type of the graph learning component. Default is mlp. ')
    # parser.add_argument('--num-layers', type=int, default=3,
    #                     help='Number of layers in mlp. Default is 3. ')
    # parser.add_argument('--hidden-channels', type=int, default=128,
    #                     help='Number of hidden channels in mlp. Default is 128. ')
    # parser.add_argument('--dropout', type=float, default=0.5,
    #                     help='Dropout rate. Default is 0.5. ')
    # parser.add_argument('--topological-heuristic-type', default='ac',
    #                     help='Type of the topological heuristic component. Default is ac. ')
    # parser.add_argument('--scaling-parameter', type=int, default=3,
    #                     help='Scaling parameter of ac. Default is 3. ')
    # parser.add_argument('--lr', type=float, default=0.001,
    #                     help='Learning rate. Default is 0.001. ')
    # parser.add_argument('--epochs', type=int, default=250,
    #                     help='Number of epochs. Default is 250. ')
    # parser.add_argument('--runs', type=int, default=3,
    #                     help='Number of runs. Default is 3. ')
    # parser.add_argument('--train-batch-ratio', type=float, default=0.01,
    #                     help='Ratio of training edges per train batch. Default is 0.01. ')
    # parser.add_argument('--val-batch-ratio', type=float, default=0.01,
    #                     help='Ratio of val edges per train batch. Default is 0.01. ')
    # parser.add_argument('--sample-ratio', type=float, default=0.1,
    #                     help='Ratio of training edges per train batch. Default is 0.01. ')
    # parser.add_argument('--random-seed', type=int, default=0,
    #                     help='Random seed for training. Default is 1. ')
    # parser.add_argument('--cuda', type=int, default=0,
    #                     help='Index of cuda device to use. Default is 0. ')
    # parser.add_argument('--num-partitions', type=int, default=1)
    # parser.add_argument('--partition-eval', action='store_true')
    # parser.add_argument('--sparse-s', type=literal_eval, default=False)
    # parser.add_argument('--full-training', action='store_true')
    # parser.add_argument('--no-wandb', action='store_true')
    # parser.add_argument('--intracluster-only', action='store_true')
    # parser.add_argument('--no-batch', action='store_true')
    # parser.add_argument('--heart-path', type=str, default='')
    
    return parser.parse_args()



if __name__ == '__main__':
    
    args = argparse()
    
    data, pyg_data = load_dataset("Karate")
    
    x = torch.arange(data.num_nodes)[:, None]
    
    transform = T.Compose([T.ToUndirected(), T.LargestConnectedComponents(connection='strong')])
    
    data = transform(data)
    
    
    
    if args.method == "minhash":
        # We need this to get connected subgraphs.
        # Otherwise the node that has the minhash will not
        # be able to "link" two subgraphs.
        data.edge_index = add_self_loops(data.edge_index)[0]
        new_edge_index = add_k_hop_edges(data, k=2)

        min_hash_clustering_hashes = [MinHashClustering(d=maxsize, seed=1, num_layers=2), 
                                  MinHashClustering(d=maxsize, seed=2, num_layers=2)]
        
        