import numpy as np

from utils import load_data
from proposals import compute_iou, compute_iop, get_majority
from .cluster_processor import ClusterProcessor


class ClusterDetProcessor(ClusterProcessor):
    def __init__(self, dataset):
        super().__init__(dataset)

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
        # take majority as label of the graph
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
            if self.dataset.det_label == 'iou':
                label = compute_iou(node, gt_node)
            elif self.dataset.det_label == 'iop':
                label = compute_iop(node, gt_node)
            else:
                raise KeyError('Unknown det_label type: {}'.format(
                    self.dataset.det_label))
        else:
            label = -1.

        adj, _, _ = self.build_adj(node, edge)
        features = self.build_features(node)
        return features, adj, label

    def __getitem__(self, idx):
        ''' each features is a NxD matrix,
            each adj is a NxN matrix,
            each label is a floating point number,
            which indicates the quality of the proposal.
        '''
        if idx is None or idx > self.dataset.size:
            raise ValueError('idx({}) is not in the range of {}'.format(
                idx, self.dataset.size))
        fn_node, fn_edge = self.dataset.lst[idx]
        ret = self.build_graph(fn_node, fn_edge)
        assert ret is not None
        features, adj, label = ret
        return features.astype(self.dtype), adj.astype(self.dtype), np.array(
            label, dtype=self.dtype)
