import torch
from torch_geometric.utils import add_self_loops
import torch
import torch_geometric
from typing import Any
import torch
from torch import Tensor
from sklearn.utils import murmurhash3_32
import numpy as np
import torch
from torch_geometric.utils import add_self_loops
import torch_geometric.transforms as T
from torch_geometric.nn import aggr
from sys import maxsize
import code
import networkx as nx
from utils import load_dataset, add_k_hop_edges, edge_cut, convert_to_single_hash
from tqdm import tqdm
from random import Random


class MinHashClustering:
    def __init__(self, d: int, seed: int, num_layers:int) -> None:
        self.d = d
        self.random = Random(seed)
        self.num_layers = num_layers
        
        self.random_seeds = [self.random.randint(0, 2**32 - 1) for i in range(num_layers)]

        # We will use PyG MinAggregation to
        # aggregate the minimum hash in each node's neighborhood
        self.min_aggr = aggr.MinAggregation()
               
    def __call__(self, x: Tensor, edge_index: Tensor, *args: Any, **kwds: Any) -> Tensor:
        
        # x: (N, 1)
        hashes = x
        
        for i in range(self.num_layers):            
            
            hashes = np.array(hashes.squeeze(1), dtype=np.int32)
            
            hashes = torch.tensor((murmurhash3_32(hashes, positive=True, seed=self.random_seeds[i]) % self.d), dtype=torch.int64)
        
            hashes = hashes[:, None]

            # This aggregates following "source to target"
            hashes = self.min_aggr(hashes[edge_index[0]], edge_index[1])
        
        return hashes
    
    
if __name__ == '__main__':
    
    data, pyg_data = load_dataset("CiteSeer")
    
    x = torch.arange(data.num_nodes)[:, None]
    
    transform = T.Compose([T.ToUndirected(), T.LargestConnectedComponents(connection='strong')])
    
    data = transform(data)
    
    # We need this to get connected subgraphs.
    # Otherwise the node that has the minhash will not
    # be able to "link" two subgraphs.
    data.edge_index = add_self_loops(data.edge_index)[0]
        
    
    min_hash_clustering_hashes = [MinHashClustering(d=maxsize, seed=1, num_layers=30), 
                                  MinHashClustering(d=maxsize, seed=2, num_layers=30)]
    
    
    output = torch.zeros((data.num_nodes, len(min_hash_clustering_hashes)), dtype=torch.int64)
    
    for i, min_hash_clustering in enumerate(min_hash_clustering_hashes):
        
        cur_output = min_hash_clustering(x, data.edge_index)
        
        output[:, i] = cur_output.squeeze(1)
    
    
    output = convert_to_single_hash(output)            
    
    edge_cut_percentage = edge_cut(output, data.edge_index)
    print(f"[Minhash] - Edge cut: {edge_cut_percentage}")
    
    for cur_cluster in output.unique():
        test = data.subgraph(output.squeeze(1) == cur_cluster)
        test_nx = torch_geometric.utils.to_networkx(test)
        
        classes, counts = test.y.unique(return_counts=True)
        
        print(f"Cluster {cur_cluster} ({test.num_nodes}) is connected...", nx.is_connected(test_nx.to_undirected()), " - Classes:", [f"{cur_class}: {cur_count}" for cur_class, cur_count in zip(classes, counts)])
    