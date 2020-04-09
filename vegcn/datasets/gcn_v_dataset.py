import os
import numpy as np

from utils import (read_meta, read_probs, l2norm, build_knns,
                   knns2ordered_nbrs, fast_knns2spmat, row_normalize,
                   build_symmetric_adj, intdict2ndarray, Timer)
from vegcn.confidence import (confidence, confidence_to_peaks)


class GCNVDataset(object):
    def __init__(self, cfg):
        feat_path = cfg['feat_path']
        label_path = cfg.get('label_path', None)
        knn_graph_path = cfg.get('knn_graph_path', None)

        self.k = cfg['k']
        self.feature_dim = cfg['feature_dim']
        self.is_norm_feat = cfg.get('is_norm_feat', True)
        self.is_sort_knns = cfg.get('is_sort_knns', True)

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
            self.size = self.inst_num
            assert self.size == self.features.shape[0]

        with Timer('read knn graph'):
            if knn_graph_path is not None:
                knns = np.load(knn_graph_path)['data']
            else:
                knn_prefix = os.path.join(cfg.prefix, 'knns', cfg.name)
                knns = build_knns(knn_prefix, self.features, cfg.knn_method,
                                  cfg.knn)

            adj = fast_knns2spmat(knns, self.k, self.th_sim, use_sim=True)

            # build symmetric adjacency matrix
            adj = build_symmetric_adj(adj, self_loop=True)
            self.adj = row_normalize(adj)

            # convert knns to (dists, nbrs)
            self.dists, self.nbrs = knns2ordered_nbrs(knns)

        print('feature shape: {}, k: {}, norm_feat: {}, sort_knns: {}'.format(
            self.features.shape, self.k, self.is_norm_feat, self.is_sort_knns))

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
        '''
        return the entire graph for training.
        To accelerate training or cope with larger graph,
        we can sample sub-graphs in this function.
        '''

        assert index == 0
        return self.features, self.adj, self.labels

    def __len__(self):
        return self.size
