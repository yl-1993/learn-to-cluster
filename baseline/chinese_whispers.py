import os
import random
import numpy as np
import networkx as nx

from utils import (build_knns, knns2ordered_nbrs, clusters2labels, Timer)


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
        sims = 1 - dists

        for node_i, (sim, nbr) in enumerate(zip(sims, nbrs)):
            # initialize 'cluster' to unique value (cluster of itself)
            node = (node_i, {'cluster': node_i})
            nodes.append(node)

            edges_i = []
            for _sim, node_j in zip(sim, nbr):
                # remove self-loop and prune edge with small similarity
                if (node_i == node_j) or (_sim <= th_sim):
                    continue
                edges_i.append((node_i, node_j, {'weight': _sim}))

            edges = edges + edges_i

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        node_num = G.number_of_nodes()
        edge_num = G.number_of_edges()
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
                cluster2weight = {}
                for nbr in nbrs:
                    assigned_cluster = G.node[nbr]['cluster']
                    edge_weight = G[node][nbr]['weight']
                    if assigned_cluster not in cluster2weight:
                        cluster2weight[assigned_cluster] = 0
                    cluster2weight[assigned_cluster] += edge_weight

                # set the class of node to its neighbor with largest weight
                if len(cluster2weight) > 0:
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
