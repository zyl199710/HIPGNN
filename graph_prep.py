import sys
import os
import math
import time
import pickle as pkl
import argparse
import scipy as sp
from scipy import io
import numpy as np
import pandas as pd
import networkx as nx
import dgl
import torch
from sklearn.preprocessing import label_binarize
from numpy.linalg import eig, eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
from scipy.sparse import diags
import torch
import scipy.sparse as sp

parser = argparse.ArgumentParser(description='Process some graphs.')
parser.add_argument('--dataset', default='STRINGdb', help='Name of the dataset (default: STRINGdb)')
args = parser.parse_args()

if args.dataset == "STRINGdb":
    save_path = 'data/graph_STRINGdb.bin'
elif args.dataset == "CPDB":
    save_path = 'data/graph_CPDB.bin'
loaded_graphs, labels = dgl.load_graphs(save_path)
ori_graph = loaded_graphs[0]
confidences = ori_graph.edata['confidence']
print(confidences)

non_zero_confidence_edges = (confidences != 0).nonzero(as_tuple=False).squeeze()

src, dst = ori_graph.edges()
zero_confidence_edges = (confidences == 0).nonzero(as_tuple=False).squeeze()
zero_src = src[zero_confidence_edges]
zero_dst = dst[zero_confidence_edges]

zero_confidence_edge_indices = np.stack((zero_src.numpy(), zero_dst.numpy()), axis=0)


graph = dgl.edge_subgraph(ori_graph, non_zero_confidence_edges, relabel_nodes = False)


confidence_values = graph.edata['confidence']

src, dst = graph.edges()
non_zero_confidence_edge_indices = np.stack((src.numpy(), dst.numpy()), axis=0)
weights = graph.edata['confidence'].numpy()

non_zero_edges_with_weights = np.vstack((non_zero_confidence_edge_indices, weights))


from sklearn.model_selection import train_test_split
pos_train_edge, pos_test_edge = train_test_split(non_zero_edges_with_weights.T, test_size=0.2, random_state=42)
train_edge_index = pos_train_edge[:, :2].T
test_edge_index = pos_test_edge[:, :2].T
test_edge_index_tensor = torch.tensor(test_edge_index, dtype=torch.int64)
src_indices = test_edge_index_tensor[0, :]
dst_indices = test_edge_index_tensor[1, :]
edge_ids = graph.edge_ids(src_indices, dst_indices)
graph.remove_edges(edge_ids, store_ids=True)


new_graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)
src, dst = new_graph.edges()
weights = new_graph.edata['confidence'].numpy()


N = new_graph.number_of_nodes()
adj_weighted = coo_matrix((weights, (src.numpy(), dst.numpy())), shape=(N, N))
degrees = np.array(adj_weighted.sum(axis=1)).flatten()
D = diags(degrees)
D_inv_sqrt = diags(1.0 / np.sqrt(degrees + 1e-10))
L_sym = sp.eye(N) - D_inv_sqrt @ adj_weighted @ D_inv_sqrt
L_dense = L_sym.toarray()
L_torch = torch.from_numpy(L_dense).to(device='cuda')
e, u = torch.linalg.eigh(L_torch)


x = graph.ndata['feat']
y = graph.ndata['label']
train_mask = graph.ndata['train_mask']
test_mask = graph.ndata['test_mask']
val_mask = graph.ndata['val_mask']


zero_weights = np.zeros(zero_confidence_edge_indices.shape[1])


zero_edges_with_weights = np.vstack((zero_confidence_edge_indices, zero_weights))

data_to_save = {
    'e': e,  # Tensor
    'u': u,  # Tensor
    'x': x,  # Tensor
    'y': y,  # Tensor
    'train_mask': train_mask.numpy(),
    'test_mask': test_mask.numpy(),
    'val_mask': val_mask.numpy(),
    'non_zero_edges_with_weights': non_zero_edges_with_weights,
    'zero_edges_with_weights': zero_edges_with_weights,
}

if args.dataset == "STRINGdb":
    torch.save(data_to_save, 'data/STRINGdb_weighted.pt')
elif args.dataset == "CPDB":
    torch.save(data_to_save, 'data/CPDB_weighted.pt')