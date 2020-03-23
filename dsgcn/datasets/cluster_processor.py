import numpy as np


class ClusterProcessor(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dtype = np.float32

    def __len__(self):
        return self.dataset.size

    def build_adj(self, node, edge):
        node = list(node)
        abs2rel = {}
        rel2abs = {}
        for i, n in enumerate(node):
            abs2rel[n] = i
            rel2abs[i] = n
        size = len(node)
        adj = np.eye(size)
        for e in edge:
            w = 1.
            if len(e) == 2:
                e1, e2 = e
            elif len(e) == 3:
                e1, e2, dist = e
                if not self.dataset.wo_weight:
                    w = 1. - dist
            else:
                raise ValueError('Unknown length of e: {}'.format(e))
            v1 = abs2rel[e1]
            v2 = abs2rel[e2]
            adj[v1][v2] = w
            adj[v2][v1] = w
        if self.dataset.is_norm_adj:
            adj /= adj.sum(axis=1, keepdims=True)
        return adj, abs2rel, rel2abs

    def build_features(self, node):
        if self.dataset.featureless:
            features = np.ones(len(node)).reshape(-1, 1)
        else:
            features = self.dataset.features[node, :]
        return features

    def __getitem__(self, idx):
        raise NotImplementedError
