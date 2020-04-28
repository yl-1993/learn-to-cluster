import os
import random
import numpy as np
import networkx as nx

from utils import (build_knns, knns2ordered_nbrs, fast_knns2spmat,
                   clusters2labels, Timer)


def chinese_whispers(feats, prefix, name, knn_method, knn, th_sim, iters,
                     **kwargs):
    """ Chinese Whispers Clustering Algorithm

    Paper: Chinese whispers: an efficient graph clustering algorithm
            and its application to natural language processing problems.
    Reference code:
        - http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/
        - https://github.com/zhly0/facenet-face-cluster-chinese-whispers-
    """

    assert len(feats) > 1

    with Timer('create graph'):
        nodes = []
        edges = []

        knn_prefix = os.path.join(prefix, 'knns', name)
        knns = build_knns(knn_prefix, feats, knn_method, knn)
        dists, nbrs = knns2ordered_nbrs(knns, sort=True)
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
        node_num = G.number_of_nodes()
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
                    assigned_cluster = G.node[nbr]['cluster']
                    edge_weight = G[node][nbr]['weight']
                    if assigned_cluster not in cluster2weight:
                        cluster2weight[assigned_cluster] = 0
                    cluster2weight[assigned_cluster] += edge_weight

                # set the class of node to its neighbor with largest weight
                cluster2weight = sorted(cluster2weight.items(),
                                        key=lambda kv: kv[1],
                                        reverse=True)
                G.node[node]['cluster'] = cluster2weight[0][0]

    clusters = {}
    for (node, data) in G.node.items():
        assigned_cluster = data['cluster']

        if assigned_cluster not in clusters:
            clusters[assigned_cluster] = []
        clusters[assigned_cluster].append(node)

    print('#cluster: {}'.format(len(clusters)))
    labels = clusters2labels(clusters.values())
    labels = list(labels.values())

    return labels
