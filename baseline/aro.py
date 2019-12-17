#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from functools import partial
from tqdm import tqdm

from utils import build_knns, knns2ordered_nbrs, Timer
"""
paper: https://arxiv.org/pdf/1604.00989.pdf
original code https://github.com/varun-suresh/Clustering

To run `aro`:
1. pip install pyflann
2. 2to3 -w path/site-packages/pyflann/
Refer [No module named 'index'](https://github.com/primetang/pyflann/issues/1) for more details.

For `knn_aro`, we replace the pyflann with more advanced knn searching methods.
"""

__all__ = ['aro', 'knn_aro']


def build_index(dataset, n_neighbors):
    """
    Takes a dataset, returns the "n" nearest neighbors
    """
    # Initialize FLANN
    import pyflann
    pyflann.set_distance_type(distance_type='euclidean')
    flann = pyflann.FLANN()
    params = flann.build_index(dataset, algorithm='kdtree', trees=4)
    #print params
    nbrs, dists = flann.nn_index(dataset, n_neighbors, checks=params['checks'])

    return nbrs, dists


def create_neighbor_lookup(nbrs):
    """
    Key is the reference face, values are the neighbors.
    """
    nn_lookup = {}
    for i in range(nbrs.shape[0]):
        nn_lookup[i] = nbrs[i, :]
    return nn_lookup


def calculate_symmetric_dist_row(nbrs, nn_lookup, row_no):
    """
    This function calculates the symmetric distances for one row in the
    matrix.
    """
    dist_row = np.zeros([1, nbrs.shape[1]])
    f1 = nn_lookup[row_no]
    for idx, neighbor in enumerate(f1[1:]):
        Oi = idx + 1
        co_neighbor = True
        try:
            row = nn_lookup[neighbor]
            Oj = np.where(row == row_no)[0][0] + 1
        except IndexError:
            Oj = nbrs.shape[1] + 1
            co_neighbor = False
        # dij
        f11 = set(f1[0:Oi])
        f21 = set(nn_lookup[neighbor])
        dij = len(f11.difference(f21))
        # dji
        f12 = set(f1)
        f22 = set(nn_lookup[neighbor][0:Oj])
        dji = len(f22.difference(f12))

        if not co_neighbor:
            dist_row[0, Oi] = 9999.0
        else:
            dist_row[0, Oi] = float(dij + dji) / min(Oi, Oj)

    return dist_row


def calculate_symmetric_dist(nbrs, num_process):
    """
    This function calculates the symmetric distance matrix.
    """
    d = np.zeros(nbrs.shape)
    if num_process > 1:
        from multiprocessing import Pool
        p = Pool(processes=num_process)
        num = nbrs.shape[0]
        batch_size = 2000000
        batch_num = int(num / batch_size) + 1
        results = []
        for i in range(batch_num):
            start = i * batch_size
            end = min(num, (i + 1) * batch_size)
            sub_nbrs = nbrs[start:end]
            nn_lookup = create_neighbor_lookup(sub_nbrs)
            func = partial(calculate_symmetric_dist_row, sub_nbrs, nn_lookup)
            results += p.map(func, range(sub_nbrs.shape[0]))
        for row_no, row_val in enumerate(results):
            d[row_no, :] = row_val
    else:
        nn_lookup = create_neighbor_lookup(nbrs)
        for row_no in tqdm(range(nbrs.shape[0])):
            row_val = calculate_symmetric_dist_row(nbrs, nn_lookup, row_no)
            d[row_no, :] = row_val
    return d


def aro_clustering(nbrs, dists, thresh):
    '''
    Approximate rank-order clustering. Takes in the nearest neighbors matrix
    and outputs clusters - list of lists.
    '''
    # Clustering
    clusters = []
    # Start with the first face
    nodes = set(list(np.arange(0, dists.shape[0])))
    plausible_neighbors = create_plausible_neighbor_lookup(nbrs, dists, thresh)
    while nodes:
        # Get a node
        n = nodes.pop()
        # This contains the set of connected nodes
        group = {n}
        # Build a queue with this node in it
        queue = [n]
        # Iterate over the queue
        while queue:
            n = queue.pop(0)
            neighbors = plausible_neighbors[n]
            # Remove neighbors we've already visited
            neighbors = nodes.intersection(neighbors)
            neighbors.difference_update(group)

            # Remove nodes from the global set
            nodes.difference_update(neighbors)

            # Add the connected neighbors
            group.update(neighbors)

            # Add the neighbors to the queue to visit them next
            queue.extend(neighbors)
        # Add the group to the list of groups
        clusters.append(group)
    return clusters


def create_plausible_neighbor_lookup(nbrs, dists, thresh):
    """
    Create a dictionary where the keys are the row numbers(face numbers) and
    the values are the plausible neighbors.
    """
    n_vectors = nbrs.shape[0]
    plausible_neighbors = {}
    for i in range(n_vectors):
        plausible_neighbors[i] = set(
            list(nbrs[i, np.where(dists[i, :] <= thresh)][0]))
    return plausible_neighbors


def clusters2labels(clusters, num):
    labels_ = -1 * np.ones((num), dtype=np.int)
    for lb, c in enumerate(clusters):
        idxs = np.array([int(x) for x in list(c)])
        labels_[idxs] = lb
    return labels_


def aro(feats, knn, th_sim, num_process, **kwargs):
    """
    Master function. Takes the descriptor matrix and returns clusters.
    n_neighbors are the number of nearest neighbors considered and thresh
    is the clustering distance threshold
    """
    with Timer('[aro] search knn with pyflann'):
        nbrs, _ = build_index(feats, n_neighbors=knn)
    dists = calculate_symmetric_dist(nbrs, num_process)
    print('symmetric dist:', dists.max(), dists.min(), dists.mean())
    clusters = aro_clustering(nbrs, dists, 1. - th_sim)
    labels_ = clusters2labels(clusters, feats.shape[0])
    return labels_


def knn_aro(feats, prefix, name, knn_method, knn, th_sim, num_process,
            **kwargs):
    knn_prefix = os.path.join(prefix, 'knns', name)
    knns = build_knns(knn_prefix, feats, knn_method, knn)
    _, nbrs = knns2ordered_nbrs(knns, sort=False)
    dists = calculate_symmetric_dist(nbrs, num_process)
    clusters = aro_clustering(nbrs, dists, 1. - th_sim)
    labels_ = clusters2labels(clusters, feats.shape[0])
    return labels_
