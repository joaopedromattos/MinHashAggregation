import torch
from torch_geometric.datasets import Planetoid, Amazon, KarateClub
from ogb.linkproppred import PygLinkPropPredDataset

from torch_sparse.tensor import SparseTensor



def load_dataset(dataset):
    """
    Load dataset.
    :param dataset: name of the dataset.
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



def add_k_hop_edges(data, k=2):
    """
    Add k-hop edges to the graph.
    :param data: PyG dataset data.
    :param k: number of hops.
    :return: (2, E) torch.Tensor.
    """
   
    edge_index = SparseTensor.from_edge_index(data.edge_index, edge_attr=torch.ones(size=(data.edge_index.shape[1],)), sparse_sizes=(data.num_nodes, data.num_nodes))
    
    final_edge_index = edge_index
    for _ in range(k):
        final_edge_index = edge_index.spspmm(final_edge_index.t())
 
    return torch.vstack((final_edge_index.storage.row(), final_edge_index.storage.col()))


convert_to_single_hash = lambda x: (x @ torch.tensor([2**32 - 1, 1]).unsqueeze(1)).to(torch.int64)


def edge_cut(clusters, edges):
    # num_clusters = clusters.unique()
    # intra_cluster_edges = clusters[edges[0]] == clusters[edges[1]]
    # for cluster in range(num_clusters):
    #     intra_cluster_edges == cluster
    return torch.sum(clusters[edges[0]] != clusters[edges[1]]) / edges.shape[-1]