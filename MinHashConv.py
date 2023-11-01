import torch
import torch_geometric


import math
from typing import Optional

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
from torch_geometric.utils import add_self_loops, degree

from typing import Union




# class MinHashAggregation(Aggregation):
#     r"""An aggregation operator that sums up features across a set of elements

#     .. math::
#         \mathrm{sum}(\mathcal{X}) = \sum_{\mathbf{x}_i \in \mathcal{X}}
#         \mathbf{x}_i.
#     """
#     def __init__(self, d=int, seed=Union[list, int]):
#         super().__init__()        
#         self.d = d
#         self.seed = seed
    
#     def forward(self, x: Tensor, index: Optional[Tensor] = None,
#                 ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
#                 dim: int = -2) -> Tensor:
        
#         if type(self.seed) is int:
#             hashes = (murmurhash3_32(np.array(x, dtype=np.int32), positive=True, seed=self.seed) % self.d).astype(np.int32)
#             hashes = torch.tensor(hashes)
#         else:
#             hashes = None
#             for cur_seed in self.seed:
#                 cur_hash = (murmurhash3_32(np.array(x, dtype=np.int32), positive=True, seed=cur_seed) % self.d).astype(np.int32)
#                 if hashes is not None:
#                     hashes = np.hstack((hashes, cur_hash))
#                 else:
#                     hashes = cur_hash
            
#             hashes = torch.tensor(hashes)
#             print(hashes.shape)
        
#         return self.reduce(hashes, index, ptr, dim_size, dim, reduce='min')


class MinHashConv(MessagePassing):
    def __init__(self, d=int, seed=Union[list, int]):
        '''
        This is just a simple layer used to run message passing
        using the MinHashAggregation.
        '''
        self.d = d
        self.seed = seed
        super().__init__(aggr=min)
        
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)
        
        # When we have more than a single hash [h1, h2, h3]
        # reduce only computes min(h1), min(h2), min(h3).
        # So to get min(min(h1), min(h2), min(h3)) we need this 
        # additional step
        out = out.min(dim=1)[0][:, None]
        return out

    def message(self, x_j):
        
        if type(self.seed) is int:
            hashes = (murmurhash3_32(np.array(x_j, dtype=np.int32), positive=True, seed=self.seed) % self.d).astype(np.int32)
            hashes = torch.tensor(hashes)
        else:
            hashes = None
            for cur_seed in self.seed:
                cur_hash = (murmurhash3_32(np.array(x_j, dtype=np.int32), positive=True, seed=cur_seed) % self.d).astype(np.int32)
                if hashes is not None:
                    hashes = np.hstack((hashes, cur_hash))
                else:
                    hashes = cur_hash
            hashes = torch.tensor(hashes)
            
        return hashes
    