import numpy as np

from utils import (read_meta, read_probs, l2norm, knns2ordered_nbrs,
                   intdict2ndarray, Timer)


class ClusterDataset(object):
    def __init__(self, cfg):
        feat_path = cfg['feat_path']
        label_path = cfg.get('label_path', None)
        knn_graph_path = cfg['knn_graph_path']

        self.k_at_hop = cfg['k_at_hop']
        self.depth = len(self.k_at_hop)
        self.active_connection = cfg['active_connection']
        self.feature_dim = cfg['feature_dim']
        self.is_norm_feat = cfg.get('is_norm_feat', True)
        self.is_sort_knns = cfg.get('is_sort_knns', True)
        self.is_test = cfg.get('is_test', False)

        with Timer('read meta and feature'):
            if label_path is not None:
                _, idx2lb = read_meta(label_path)
                self.inst_num = len(idx2lb)
                self.labels = intdict2ndarray(idx2lb)
                self.ignore_label = False
            else:
                self.labels = None
                self.inst_num = -1
                self.ignore_label = True
            self.features = read_probs(feat_path, self.inst_num,
                                       self.feature_dim)
            if self.is_norm_feat:
                self.features = l2norm(self.features)
            if self.inst_num == -1:
                self.inst_num = self.features.shape[0]
            self.size = self.inst_num

        with Timer('read knn graph'):
            knns = np.load(knn_graph_path)['data']
            _, self.knn_graph = knns2ordered_nbrs(knns, sort=self.is_sort_knns)
        assert np.mean(self.k_at_hop) >= self.active_connection

        print('feature shape: {}, norm_feat: {}, sort_knns: {} '
              'k_at_hop: {}, active_connection: {}'.format(
                  self.features.shape, self.is_norm_feat, self.is_sort_knns,
                  self.k_at_hop, self.active_connection))

    def __getitem__(self, index):
        '''
        return the vertex feature and the adjacent matrix A, together
        with the indices of the center node and its 1-hop nodes
        '''
        if index is None or index > self.size:
            raise ValueError('index({}) is not in the range of {}'.format(
                index, self.size))

        center_node = index

        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        hops = []
        hops.append(set(self.knn_graph[center_node][1:]))

        # Actually we dont need the loop since the depth is fixed here,
        # But we still remain the code for further revision
        for d in range(1, self.depth):
            hops.append(set())
            for h in hops[-2]:
                hops[-1].update(set(self.knn_graph[h][1:self.k_at_hop[d] + 1]))

        hops_set = set([h for hop in hops for h in hop])
        hops_set.update([
            center_node,
        ])
        uniq_nodes = np.array(list(hops_set), dtype=np.int64)
        uniq_nodes_map = {j: i for i, j in enumerate(uniq_nodes)}

        center_idx = np.array([uniq_nodes_map[center_node]], dtype=np.int64)
        one_hop_idxs = np.array([uniq_nodes_map[i] for i in hops[0]],
                                dtype=np.int64)
        center_feat = self.features[center_node]
        feat = self.features[uniq_nodes]
        feat = feat - center_feat

        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        num_nodes = len(uniq_nodes)
        A = np.zeros([num_nodes, num_nodes], dtype=feat.dtype)

        res_num_nodes = max_num_nodes - num_nodes
        if res_num_nodes > 0:
            pad_feat = np.zeros([res_num_nodes, self.feature_dim],
                                dtype=feat.dtype)
            feat = np.concatenate([feat, pad_feat], axis=0)

        for node in uniq_nodes:
            neighbors = self.knn_graph[node, 1:self.active_connection + 1]
            for n in neighbors:
                if n in uniq_nodes:
                    i, j = uniq_nodes_map[node], uniq_nodes_map[n]
                    A[i, j] = 1
                    A[j, i] = 1

        D = A.sum(1, keepdims=True)
        A = A / D
        A_ = np.zeros([max_num_nodes, max_num_nodes], dtype=A.dtype)
        A_[:num_nodes, :num_nodes] = A

        if self.ignore_label:
            return (feat, A_, center_idx, one_hop_idxs)

        labels = self.labels[uniq_nodes]
        one_hop_labels = labels[one_hop_idxs]
        center_label = labels[center_idx]
        edge_labels = (center_label == one_hop_labels).astype(np.int64)

        if self.is_test:
            if res_num_nodes > 0:
                pad_nodes = np.zeros(res_num_nodes, dtype=uniq_nodes.dtype)
                uniq_nodes = np.concatenate([uniq_nodes, pad_nodes], axis=0)
            return (feat, A_, one_hop_idxs,
                    edge_labels), center_idx, uniq_nodes
        else:
            return (feat, A_, one_hop_idxs, edge_labels)

    def __len__(self):
        return self.size
