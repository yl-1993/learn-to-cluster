#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # if rowsum <= 0, keep its previous value
    rowsum[rowsum <= 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def build_symmetric_adj(adj, self_loop=True):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    return adj


def sparse_mx_to_indices_values(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    values = sparse_mx.data
    shape = np.array(sparse_mx.shape)
    return indices, values, shape


def indices_values_to_sparse_tensor(indices, values, shape):
    import torch
    indices = torch.from_numpy(indices)
    values = torch.from_numpy(values)
    shape = torch.Size(shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices, values, shape = sparse_mx_to_indices_values(sparse_mx)
    return indices_values_to_sparse_tensor(indices, values, shape)
