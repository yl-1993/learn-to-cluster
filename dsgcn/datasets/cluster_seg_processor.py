import random
import numpy as np

from utils import load_data
from proposals import compute_iou, get_majority
from .cluster_processor import ClusterProcessor


class ClusterSegProcessor(ClusterProcessor):
    def __init__(self, dataset):
        super().__init__(dataset)

    @classmethod
    def get_node_lb(cls, pred, label):
        gt_set = set(label)
        lbs = []
        for node in pred:
            if node in gt_set:
                lbs.append(1)
            else:
                lbs.append(0)
        return np.array(lbs)

    def build_graph(self, fn_node, fn_edge):
        ''' build graph from graph file
            - nodes: NxD,
                     each row represents the feature of a node
            - adj:   NxN,
                     a symmetric similarity matrix with self-connection
        '''
        node = load_data(fn_node)
        edge = load_data(fn_edge)
        assert len(node) > 1, '#node of {}: {}'.format(fn_node, len(node))

        adj, abs2rel, rel2abs = self.build_adj(node, edge)
        # compute label & mask
        if self.dataset.use_random_seed:
            ''' except using node with max degree as seed,
                you can explore more creative designs.
                e.g., applying random seed for multiple times,
                and take the best results.
            '''
            if self.dataset.use_max_degree_seed:
                s = adj.sum(axis=1, keepdims=True)
                rel_center_idx = np.argmax(s)
                center_idx = rel2abs[rel_center_idx]
            else:
                center_idx = random.choice(node)
                rel_center_idx = abs2rel[center_idx]
            mask = np.zeros(len(node))
            mask[rel_center_idx] = 1
            mask = mask.reshape(-1, 1)
            if not self.dataset.ignore_label:
                lb = self.dataset.idx2lb[center_idx]
                gt_node = self.dataset.lb2idxs[lb]
        else:
            # do not use mask
            if not self.dataset.ignore_label:
                lb2cnt = {}
                for idx in node:
                    if idx not in self.dataset.idx2lb:
                        continue
                    lb = self.dataset.idx2lb[idx]
                    if lb not in lb2cnt:
                        lb2cnt[lb] = 0
                    lb2cnt[lb] += 1
                gt_lb, _ = get_majority(lb2cnt)
                gt_node = self.dataset.lb2idxs[gt_lb]

        if not self.dataset.ignore_label:
            g_label = self.get_node_lb(node, gt_node)
        else:
            g_label = np.zeros_like(node)

        features = self.build_features(node)
        if self.dataset.use_random_seed:
            features = np.concatenate((features, mask), axis=1)
        return features, adj, g_label

    def __getitem__(self, idx):
        ''' each features is a NxD matrix,
            each adj is a NxN matrix,
            each label is a Nx2 matrix,
            which indicates the quality of the proposal.
        '''
        if idx is None or idx > self.dataset.size:
            raise ValueError('idx({}) is not in the range of {}'.format(
                idx, self.dataset.size))
        fn_node, fn_edge = self.dataset.lst[idx]
        ret = self.build_graph(fn_node, fn_edge)
        assert ret is not None
        features, adj, label = ret
        return features.astype(self.dtype), adj.astype(self.dtype), label
