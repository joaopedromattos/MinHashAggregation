import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torch_geometric.datasets import Planetoid, Amazon, KarateClub
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges, k_hop_subgraph
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm
import argparse 
from ast import literal_eval
import code
import torch_geometric
import os
from torch_sparse.tensor import SparseTensor
from itertools import combinations
from tqdm import tqdm
import scipy.sparse as ssp
import datetime



def load_dataset(dataset):
    """
    Load the dataset from PyG.
    :param dataset: name of the dataset. Options: 'Karate'
    :return: PyG dataset data.
    """
    data_folder = f'data/{dataset}/'
    if dataset in ('Karate'):
        pyg_dataset = KarateClub(data_folder)
    elif dataset in ('Cora', 'CiteSeer', 'PubMed'):
        pyg_dataset = Planetoid(data_folder, dataset)
    elif dataset in ('Photo', 'Computers'):
        pyg_dataset = Amazon(data_folder, dataset)
    elif dataset in ('ogbl-ppa'):
        pyg_dataset = PygLinkPropPredDataset('ogbl-ppa', root=data_folder)
    elif dataset in ('ogbl-ddi', 'ogbl-collab', 'ogbl-ppa', 'ogbl-wikikg2', 'ogbl-vessel', 'ogbl-biokg'):
        pyg_dataset = PygLinkPropPredDataset(dataset, root=data_folder)
    else:
        raise NotImplementedError(f'{dataset} not supported. ')
    data = pyg_dataset.data
    
    return data, pyg_dataset
