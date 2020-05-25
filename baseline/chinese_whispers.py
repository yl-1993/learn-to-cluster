import os
import random
import numpy as np
from scipy.sparse import identity, csr_matrix

from utils import (build_knns, knns2ordered_nbrs, fast_knns2spmat,
                   build_symmetric_adj, clusters2labels, Timer)


def chinese_whispers(feats, prefix, name, knn_method, knn, th_sim, iters,
                     **kwargs):
    """ Chinese Whispers Clustering Algorithm

    Paper: Chinese whispers: an efficient graph clustering algorithm
            and its application to natural language processing problems.
    Reference code:
        - http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/
        - https://github.com/zhly0/facenet-face-cluster-chinese-whispers-
    """
    import networkx as nx

    assert len(feats) > 1

    with Timer('create graph'):
        knn_prefix = os.path.join(prefix, 'knns', name)
        knns = build_knns(knn_prefix, feats, knn_method, knn)
        spmat = fast_knns2spmat(knns, knn, th_sim, use_sim=True)

        size = len(feats)
        nodes = [(n_i, {'cluster': n_i}) for n_i in range(size)]
        c = spmat.tocoo()
        edges = [(n_i, n_j, {
            'weight': s
        }) for n_i, n_j, s in zip(c.row, c.col, c.data)]

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        node_num = G.number_of_nodes()
        edge_num = G.number_of_edges()
        assert size == node_num
        print('#nodes: {}, #edges: {}'.format(node_num, edge_num))

    with Timer('whisper iteratively (iters={})'.format(iters)):
        cluster_nodes = list(G.nodes())
        for _ in range(iters):
            idxs = [i for i in range(node_num)]
            random.shuffle(idxs)
            for idx in idxs:
                node = cluster_nodes[idx]
                nbrs = G[node]
                if len(nbrs) == 0:
                    continue
                cluster2weight = {}
                for nbr in nbrs:
                    assigned_cluster = G.nodes[nbr]['cluster']
                    edge_weight = G[node][nbr]['weight']
                    if assigned_cluster not in cluster2weight:
                        cluster2weight[assigned_cluster] = 0
                    cluster2weight[assigned_cluster] += edge_weight

                # set the class of node to its neighbor with largest weight
                cluster2weight = sorted(cluster2weight.items(),
                                        key=lambda kv: kv[1],
                                        reverse=True)
                G.nodes[node]['cluster'] = cluster2weight[0][0]

    clusters = {}
    for (node, data) in G.nodes.items():
        assigned_cluster = data['cluster']

        if assigned_cluster not in clusters:
            clusters[assigned_cluster] = []
        clusters[assigned_cluster].append(node)

    print('#cluster: {}'.format(len(clusters)))
    labels = clusters2labels(clusters.values())
    labels = list(labels.values())

    return labels


def _matrix2array(m):
    return np.asarray(m).reshape(-1)


def _maxrow(D, n):
    row = np.arange(n)
    col = _matrix2array(D.argmax(axis=1))
    data = _matrix2array(D[row, col])
    D = csr_matrix((data, (row, col)), shape=(n, n))
    return D


def chinese_whispers_fast(feats, prefix, name, knn_method, knn, th_sim, iters,
                          **kwargs):
    """ Chinese Whispers Clustering Algorithm

    Paper: Chinese whispers: an efficient graph clustering algorithm
            and its application to natural language processing problems.
    This implementation follows the matrix operation as described in Figure.4
    int the paper. We switch the `maxrow` and `D^{t-1} * A_G` to make it
    easier for post-processing.
    The current result is inferior to `chinese_whispers` as it lacks of the
    random mechanism as the iterative algorithm. The paper introduce two
    operations to tackle this issue, namely `random mutation` and `keep class`.
    However, it is not very clear how to set this two hyper-parameters.
    """
    assert len(feats) > 1

    with Timer('create graph'):
        knn_prefix = os.path.join(prefix, 'knns', name)
        knns = build_knns(knn_prefix, feats, knn_method, knn)
        spmat = fast_knns2spmat(knns, knn, th_sim, use_sim=True)
        A = build_symmetric_adj(spmat, self_loop=False)

        node_num = len(feats)
        edge_num = A.nnz
        print('#nodes: {}, #edges: {}'.format(node_num, edge_num))

    with Timer('whisper iteratively (iters={})'.format(iters)):
        D = identity(node_num)
        for _ in range(iters):
            D = D * A  # it is equal to D.dot(A)
            D = _maxrow(D, node_num)

        assert D.nnz == node_num

    clusters = {}
    assigned_clusters = D.tocoo().col
    for (node, assigned_cluster) in enumerate(assigned_clusters):
        if assigned_cluster not in clusters:
            clusters[assigned_cluster] = []
        clusters[assigned_cluster].append(node)

    print('#cluster: {}'.format(len(clusters)))
    labels = clusters2labels(clusters.values())
    labels = list(labels.values())

    return labels
