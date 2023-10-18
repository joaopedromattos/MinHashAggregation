import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch
import torch_geometric


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
from torch_geometric.utils import add_self_loops, spmm
import torch_geometric.transforms as T

from torch_geometric.nn import aggr

from typing import Union

from sys import maxsize

import code
import networkx as nx

from utils import load_dataset

from tqdm import tqdm


class MinHashClustering:
    def __init__(self, d: Union[list, int], seed: Union[list, int]) -> None:
        self.d = d
        self.seed = seed
        
        # We will use PyG MinAggregation to
        # aggregate the minimum hash in each node's neighborhood
        self.min_aggr = aggr.MinAggregation()
    
    def __call__(self, x: Tensor, edge_index: Tensor, *args: Any, **kwds: Any) -> Tensor:
        
        x = np.array(x, dtype=np.int32)
        
        hashes = torch.tensor((murmurhash3_32(x, positive=True, seed=self.seed) % self.d), dtype=torch.int64)
        
        # This aggregates following "source to target"
        hashes = self.min_aggr(hashes[edge_index[0]], edge_index[1])
        
        return hashes
    
    
if __name__ == '__main__':

    data, pyg_data = load_dataset("Karate")
    
    x = torch.arange(data.num_nodes)[:, None]
    
    transform = T.Compose([T.ToUndirected(), T.LargestConnectedComponents(connection='strong')])
    
    data = transform(data)
    
    # We need this to get connected subgraphs.
    # Otherwise the node that has the minhash will not
    # be able to "link" two subgraphs.
    data.edge_index = add_self_loops(data.edge_index)[0]
    
    num_layers = 1 # Running with a single layer for now
    min_hash_clustering_hashes = [MinHashClustering(d=maxsize, seed=i) for i in range(num_layers)]
    
    output = x
    
    for min_hash_clustering in tqdm(min_hash_clustering_hashes):
        print(output)
        output = min_hash_clustering(output, data.edge_index)
    
        # Sanity checks
        test_nx = torch_geometric.utils.to_networkx(data)
        print("Input graph is connected...", nx.is_connected(test_nx.to_undirected()))
        
        for cur_cluster in output.unique():
            test = data.subgraph(output.squeeze(1) == cur_cluster)
            test_nx = torch_geometric.utils.to_networkx(test)
            print(f"Cluster {cur_cluster} ({test.num_nodes}) is connected...", nx.is_connected(test_nx.to_undirected()))
            
    # Opens interactive mode to inspection
    print("Interactive mode - Ctrl + D to leave...")
    code.interact(local=locals())

    