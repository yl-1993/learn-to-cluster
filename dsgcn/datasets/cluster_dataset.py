import glob
import os.path as osp
import numpy as np

from utils import (read_meta, read_probs, l2norm, load_data, intdict2ndarray,
                   Timer)
from proposals import compute_iop, get_majority


class ClusterDataset(object):
    def __init__(self, cfg):
        self.fn_node_pattern = '*_node.npz'
        self.fn_edge_pattern = '*_edge.npz'

        feat_path = cfg['feat_path']
        label_path = cfg.get('label_path', None)
        proposal_folders = cfg['proposal_folders']

        self.feature_dim = cfg['feature_dim']
        self.featureless = cfg.get('featureless', False)
        self.is_norm_adj = cfg.get('is_norm_adj', True)
        self.num_class = cfg.get('num_class', 1)
        self.th_iop_min = cfg.get('th_iop_min', None)
        self.th_iop_max = cfg.get('th_iop_max', None)
        self.wo_weight = cfg.get('wo_weight', False)
        self.det_label = cfg.get('det_label', 'iou')
        self.use_random_seed = cfg.get('use_random_seed', True)
        self.use_max_degree_seed = cfg.get('use_max_degree_seed', False)
        self.pred_iop_score = cfg.get('pred_iop_score', '')

        if self.th_iop_min is not None and self.th_iop_max is not None:
            assert 0 <= self.th_iop_min < self.th_iop_max <= 1
            self.do_iop_check = True
        else:
            assert self.th_iop_min is None and self.th_iop_max is None
            self.do_iop_check = False

        self.fn2iop = None
        if self.pred_iop_score != '' and self.pred_iop_score is not None:
            assert osp.isfile(self.pred_iop_score), '{} is not a file'.format(
                self.pred_iop_score)
            print('read predicted iop from {}'.format(self.pred_iop_score))
            d = np.load(self.pred_iop_score, allow_pickle=True)
            pred_scores = d['data']
            meta = d['meta'].item()
            _proposals = []
            _proposal_folders = meta['proposal_folders']
            if callable(_proposal_folders):
                _proposal_folders = _proposal_folders()
            for _proposal_folder in _proposal_folders:
                fn_clusters = sorted(
                    glob.glob(osp.join(_proposal_folder,
                                       self.fn_node_pattern)))
                _proposals.extend([fn_node for fn_node in fn_clusters])
            self.fn2iop = {}
            for fn, iop in zip(_proposals, pred_scores):
                self.fn2iop[fn] = iop

        self._read(feat_path, label_path, proposal_folders)

        print('#cluster: {}, #num_class: {}, feature shape: {}, '
              'norm_adj: {}, wo_weight: {}'.format(self.size, self.num_class,
                                                   self.features.shape,
                                                   self.is_norm_adj,
                                                   self.wo_weight))

    def _read(self, feat_path, label_path, proposal_folders):
        with Timer('read meta and feature'):
            if label_path is not None:
                self.lb2idxs, self.idx2lb = read_meta(label_path)
                self.labels = intdict2ndarray(self.idx2lb)
                self.inst_num = len(self.idx2lb)
                self.ignore_label = False
            else:
                self.lb2idxs, self.idx2lb = None, None
                self.labels = None
                self.inst_num = -1
                self.ignore_label = True
            if not self.featureless:
                features = read_probs(feat_path, self.inst_num,
                                      self.feature_dim)
                self.features = l2norm(features)
                if self.inst_num == -1:
                    self.inst_num = features.shape[0]
            else:
                assert self.inst_num > 0
                self.feature_dim = 1
                self.features = np.ones(self.inst_num).reshape(-1, 1)

        with Timer('read proposal list'):
            self.lst = []
            self.tot_lst = []
            if callable(proposal_folders):
                proposal_folders = proposal_folders()
            for proposal_folder in proposal_folders:
                print('read proposals from folder: ', proposal_folder)
                fn_nodes = sorted(
                    glob.glob(osp.join(proposal_folder, self.fn_node_pattern)))
                fn_edges = sorted(
                    glob.glob(osp.join(proposal_folder, self.fn_edge_pattern)))
                assert len(fn_nodes) == len(
                    fn_edges), "node files({}) vs edge files({})".format(
                        len(fn_nodes), len(fn_edges))
                assert len(fn_nodes) > 0, 'files under {} is 0'.format(
                    proposal_folder)
                for fn_node, fn_edge in zip(fn_nodes, fn_edges):
                    # sanity check
                    assert fn_node[:fn_node.rfind(
                        '_')] == fn_edge[:fn_edge.rfind('_'
                                                        )], "{} vs {}".format(
                                                            fn_node, fn_edge)
                    if self._check_iop(fn_node):
                        self.lst.append([fn_node, fn_edge])
                    self.tot_lst.append([fn_node, fn_edge])

            self.size = len(self.lst)
            self.tot_size = len(self.tot_lst)
            assert self.size <= self.tot_size

            if self.size < self.tot_size:
                print('select {} / {} = {:.2f} proposals '
                      'with iop between ({:.2f}, {:.2f})'.format(
                          self.size, self.tot_size,
                          1. * self.size / self.tot_size, self.th_iop_min,
                          self.th_iop_max))

    def _check_iop(self, fn_node):
        if not self.do_iop_check:
            return True
        node = load_data(fn_node)
        if not self.ignore_label and not self.fn2iop:
            lb2cnt = {}
            for idx in node:
                if idx not in self.idx2lb:
                    continue
                lb = self.idx2lb[idx]
                if lb not in lb2cnt:
                    lb2cnt[lb] = 0
                lb2cnt[lb] += 1
            gt_lb, _ = get_majority(lb2cnt)
            gt_node = self.lb2idxs[gt_lb]
            iop = compute_iop(node, gt_node)
        else:
            iop = self.fn2iop[fn_node]
        return (iop >= self.th_iop_min) and (iop <= self.th_iop_max)

    def __len__(self):
        return self.size
