import os
import numpy as np

from utils import (read_meta, read_probs, l2norm, build_knns,
                   knns2ordered_nbrs, fast_knns2spmat, row_normalize,
                   build_symmetric_adj, sparse_mx_to_indices_values,
                   intdict2ndarray, Timer)
from vegcn.confidence import (confidence, confidence_to_peaks)


class GCNVDataset(object):
    def __init__(self, cfg):
        feat_path = cfg['feat_path']
        label_path = cfg.get('label_path', None)
        knn_graph_path = cfg.get('knn_graph_path', None)

        self.k = cfg['k']
        self.feature_dim = cfg['feature_dim']
        self.is_norm_feat = cfg.get('is_norm_feat', True)
        self.save_decomposed_adj = cfg.get('save_decomposed_adj', False)

        self.th_sim = cfg.get('th_sim', 0.)
        self.max_conn = cfg.get('max_conn', 1)
        self.conf_metric = cfg.get('conf_metric')

        with Timer('read meta and feature'):
            if label_path is not None:
                self.lb2idxs, self.idx2lb = read_meta(label_path)
                self.inst_num = len(self.idx2lb)
                self.gt_labels = intdict2ndarray(self.idx2lb)
                self.ignore_label = False
            else:
                self.inst_num = -1
                self.ignore_label = True
            self.features = read_probs(feat_path, self.inst_num,
                                       self.feature_dim)
            if self.is_norm_feat:
                self.features = l2norm(self.features)
            if self.inst_num == -1:
                self.inst_num = self.features.shape[0]
            self.size = 1 # take the entire graph as input

        with Timer('read knn graph'):
            if os.path.isfile(knn_graph_path):
                knns = np.load(knn_graph_path)['data']
            else:
                if knn_graph_path is not None:
                    print('knn_graph_path does not exist: {}'.format(
                        knn_graph_path))
                knn_prefix = os.path.join(cfg.prefix, 'knns', cfg.name)
                knns = build_knns(knn_prefix, self.features, cfg.knn_method,
                                  cfg.knn)

            adj = fast_knns2spmat(knns, self.k, self.th_sim, use_sim=True)

            # build symmetric adjacency matrix
            adj = build_symmetric_adj(adj, self_loop=True)
            adj = row_normalize(adj)
            if self.save_decomposed_adj:
                adj = sparse_mx_to_indices_values(adj)
                self.adj_indices, self.adj_values, self.adj_shape = adj
            else:
                self.adj = adj

            # convert knns to (dists, nbrs)
            self.dists, self.nbrs = knns2ordered_nbrs(knns)

        print('feature shape: {}, k: {}, norm_feat: {}'.format(
            self.features.shape, self.k, self.is_norm_feat))

        if not self.ignore_label:
            with Timer('Prepare ground-truth label'):
                self.labels = confidence(feats=self.features,
                                         dists=self.dists,
                                         nbrs=self.nbrs,
                                         metric=self.conf_metric,
                                         idx2lb=self.idx2lb,
                                         lb2idxs=self.lb2idxs)
                if cfg.eval_interim:
                    _, self.peaks = confidence_to_peaks(
                        self.dists, self.nbrs, self.labels, self.max_conn)

    def __getitem__(self, index):
        ''' return the entire graph for training.
        To accelerate training or cope with larger graph,
        we can sample sub-graphs in this function.
        '''

        assert index == 0
        return (self.features, self.adj_indices, self.adj_values,
                self.adj_shape, self.labels)

    def __len__(self):
        return self.size
