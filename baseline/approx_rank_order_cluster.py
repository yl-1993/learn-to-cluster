#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyflann
import numpy as np
from time import time
from multiprocessing import Pool
from functools import partial
"""
paper: https://arxiv.org/pdf/1604.00989.pdf
original code https://github.com/varun-suresh/Clustering
To run approx_rank_order:
    1. pip install pyflann
    2. 2to3 -w path/site-packages/pyflann/
Refer [No module named 'index'](https://github.com/primetang/pyflann/issues/1) for more details.
"""

__all__ = ['approx_rank_order']


def build_index(dataset, n_neighbors):
    """
    Takes a dataset, returns the "n" nearest neighbors
    """
    # Initialize FLANN
    pyflann.set_distance_type(distance_type='euclidean')
    flann = pyflann.FLANN()
    params = flann.build_index(dataset, algorithm='kdtree', trees=4)
    #print params
    nearest_neighbors, dists = flann.nn_index(dataset,
                                              n_neighbors,
                                              checks=params['checks'])

    return nearest_neighbors, dists


def create_neighbor_lookup(nearest_neighbors):
    """
    Key is the reference face, values are the neighbors.
    """
    nn_lookup = {}
    for i in range(nearest_neighbors.shape[0]):
        nn_lookup[i] = nearest_neighbors[i, :]
    return nn_lookup


def calculate_symmetric_dist_row(nearest_neighbors, nn_lookup, row_no):
    """
    This function calculates the symmetric distances for one row in the
    matrix.
    """
    dist_row = np.zeros([1, nearest_neighbors.shape[1]])
    f1 = nn_lookup[row_no]
    for idx, neighbor in enumerate(f1[1:]):
        Oi = idx + 1
        co_neighbor = True
        try:
            row = nn_lookup[neighbor]
            Oj = np.where(row == row_no)[0][0] + 1
        except IndexError:
            Oj = nearest_neighbors.shape[1] + 1
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


def calculate_symmetric_dist(app_nearest_neighbors):
    """
    This function calculates the symmetric distance matrix.
    """
    # dist_calc_time = time()
    nn_lookup = create_neighbor_lookup(app_nearest_neighbors)
    d = np.zeros(app_nearest_neighbors.shape)
    p = Pool(processes=4)
    func = partial(calculate_symmetric_dist_row, app_nearest_neighbors,
                   nn_lookup)
    results = p.map(func, range(app_nearest_neighbors.shape[0]))
    for row_no, row_val in enumerate(results):
        d[row_no, :] = row_val
    # d_time = time()-dist_calc_time
    # print('Distance calculation time : {}'.format(d_time))
    return d


def aro_clustering(app_nearest_neighbors, distance_matrix, thresh):
    '''
    Approximate rank-order clustering. Takes in the nearest neighbors matrix
    and outputs clusters - list of lists.
    '''
    # Clustering
    clusters = []
    # Start with the first face
    nodes = set(list(np.arange(0, distance_matrix.shape[0])))
    # print 'Nodes initial : {}'.format(nodes)
    # tc = time()
    plausible_neighbors = create_plausible_neighbor_lookup(
        app_nearest_neighbors, distance_matrix, thresh)
    # print('Time to create plausible_neighbors lookup : {}'.format(time() - tc))
    # ctime = time()
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
    # print('Clustering Time : {}'.format(time() - ctime))
    return clusters


def create_plausible_neighbor_lookup(app_nearest_neighbors, distance_matrix,
                                     thresh):
    """
    Create a dictionary where the keys are the row numbers(face numbers) and
    the values are the plausible neighbors.
    """
    n_vectors = app_nearest_neighbors.shape[0]
    plausible_neighbors = {}
    for i in range(n_vectors):
        plausible_neighbors[i] = set(
            list(app_nearest_neighbors[
                i, np.where(distance_matrix[i, :] <= thresh)][0]))
    return plausible_neighbors


def approx_rank_order(feat, knn, th_sim, **kwargs):
    """
    Master function. Takes the descriptor matrix and returns clusters.
    n_neighbors are the number of nearest neighbors considered and thresh
    is the clustering distance threshold
    """
    app_nearest_neighbors, _ = build_index(feat, n_neighbors=knn)
    distance_matrix = calculate_symmetric_dist(app_nearest_neighbors)
    clusters = aro_clustering(app_nearest_neighbors, distance_matrix,
                              1. - th_sim)
    labels_ = -1 * np.ones((feat.shape[0]), dtype=np.int)
    for lb, c in enumerate(clusters):
        idxs = np.array([int(x) for x in list(c)])
        labels_[idxs] = lb
    return labels_
