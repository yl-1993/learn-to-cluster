import numpy as np

from utils import load_data
from proposals import compute_iou, get_majority


class ClusterDetProcessor(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.dtype = np.float32

    def __len__(self):
        return self.dataset.size

    def build_graph(self, fn_node, fn_edge):
        """ build graph from graph file
            - nodes: NxD,
                     each row represents the feature of a node
            - adj:   NxN,
                     a symmetric similarity matrix with self-connection
        """
        node = load_data(fn_node)
        edge = load_data(fn_edge)
        assert len(node) > 1, '#node of {}: {}'.format(fn_node, len(node))
        # take majority as label of the graph
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
        iou = compute_iou(node, gt_node)
        # compute adj
        node = list(node)
        abs2rel = {}
        for i, n in enumerate(node):
            abs2rel[n] = i
        size = len(node)
        adj = np.eye(size)
        for e in edge:
            if len(e) == 2:
                e1, e2 = e
                w = 1
            elif len(e) == 3:
                e1, e2, dist = e
                if self.dataset.wo_weight:
                    w = 1
                else:
                    w = 1 - dist
            else:
                raise ValueError('Unknown length of e: {}'.format(e))
            v1 = abs2rel[e1]
            v2 = abs2rel[e2]
            adj[v1][v2] = w
            adj[v2][v1] = w
        if self.dataset.featureless:
            vertices = adj.sum(axis=1, keepdims=True)
            vertices /= vertices.sum(axis=1, keepdims=True)
        else:
            vertices = self.dataset.features[node, :]
        if self.dataset.is_norm_adj:
            adj /= adj.sum(axis=1, keepdims=True)
        return vertices, adj, iou

    def __getitem__(self, idx):
        """ each vertices is a NxD matrix,
            each adj is a NxN matrix,
            each label is a Nx1 matrix,
            which is a 0 or 1 representing the foreground and background
        """
        if idx is None or idx > self.dataset.size:
            raise ValueError('idx({}) is not in the range of {}'.format(idx, self.dataset.size))
        fn_node, fn_edge = self.dataset.lst[idx]
        ret = self.build_graph(fn_node, fn_edge)
        assert ret is not None
        vertices, adj, label = ret
        return vertices.astype(self.dtype), \
               adj.astype(self.dtype), \
               np.array(label, dtype=self.dtype)
