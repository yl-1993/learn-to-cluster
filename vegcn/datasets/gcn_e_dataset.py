import os
import numpy as np
from tqdm import tqdm

from utils import (read_meta, read_probs, l2norm, build_knns,
                   knns2ordered_nbrs, fast_knns2spmat, row_normalize,
                   build_symmetric_adj, intdict2ndarray, Timer)


class GCNEDataset(object):
    def __init__(self, cfg):
        feat_path = cfg['feat_path']
        label_path = cfg.get('label_path', None)
        knn_graph_path = cfg.get('knn_graph_path', None)

        self.k = cfg['k']
        self.feature_dim = cfg['feature_dim']
        self.is_norm_feat = cfg.get('is_norm_feat', True)

        self.th_sim = cfg.get('th_sim', 0.)
        self.max_conn = cfg.get('max_conn', 1)

        self.ignore_ratio = cfg.get('ignore_ratio', 0.8)
        self.ignore_small_confs = cfg.get('ignore_small_confs', True)
        self.use_candidate_set = cfg.get('use_candidate_set', True)

        self.nproc = cfg.get('nproc', 1)
        self.max_qsize = cfg.get('max_qsize', int(1e5))

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

        print('feature shape: {}, k: {}, norm_feat: {}'.format(
            self.features.shape, self.k, self.is_norm_feat))

        with Timer('read knn graph'):
            if knn_graph_path is not None:
                knns = np.load(knn_graph_path)['data']
            else:
                knn_prefix = os.path.join(cfg.prefix, 'knns', cfg.name)
                knns = build_knns(knn_prefix, self.features, cfg.knn_method,
                                  cfg.knn)
            assert self.inst_num == len(knns), "{} vs {}".format(
                self.inst_num, len(knns))

            adj = fast_knns2spmat(knns, self.k, self.th_sim, use_sim=True)

            # build symmetric adjacency matrix
            adj = build_symmetric_adj(adj, self_loop=True)
            self.adj = row_normalize(adj)

            # convert knns to (dists, nbrs)
            self.dists, self.nbrs = knns2ordered_nbrs(knns, sort=True)

            if cfg.pred_confs != '':
                print('read estimated confidence from {}'.format(
                    cfg.pred_confs))
                self.confs = np.load(cfg.pred_confs)['pred_confs']
            else:
                print('use unsupervised density as confidence')
                assert self.radius
                from vegcn.confidence import density
                self.confs = density(self.dists, radius=self.radius)

            assert 0 <= self.ignore_ratio <= 1
            if self.ignore_ratio == 1:
                self.ignore_set = set(np.arange(len(self.confs)))
            else:
                num = int(len(self.confs) * self.ignore_ratio)
                confs = self.confs
                if not self.ignore_small_confs:
                    confs = -confs
                self.ignore_set = set(np.argpartition(confs, num)[:num])

        print(
            'ignore_ratio: {}, ignore_small_confs: {}, use_candidate_set: {}'.
            format(self.ignore_ratio, self.ignore_small_confs,
                   self.use_candidate_set))
        print('#ignore_set: {} / {} = {:.3f}'.format(
            len(self.ignore_set), self.inst_num,
            1. * len(self.ignore_set) / self.inst_num))

        with Timer('Prepare sub-graphs'):
            # construct subgraphs with larger confidence
            self.peaks = {i: [] for i in range(self.inst_num)}
            self.dist2peak = {i: [] for i in range(self.inst_num)}

            if self.nproc > 1:
                # multi-process
                import multiprocessing as mp
                pool = mp.Pool(self.nproc)
                results = []
                num = int(self.inst_num / self.max_qsize) + 1
                for i in tqdm(range(num)):
                    beg = int(i * self.max_qsize)
                    end = min(beg + self.max_qsize, self.inst_num)
                    lst = [j for j in range(beg, end)]
                    results.extend(
                        list(
                            tqdm(pool.map(self.get_subgraph, lst),
                                 total=len(lst))))
                pool.close()
                pool.join()
            else:
                results = [
                    self.get_subgraph(i) for i in tqdm(range(self.inst_num))
                ]

            self.adj_lst = []
            self.feat_lst = []
            self.lb_lst = []
            self.subset_gt_labels = []
            self.subset_idxs = []
            self.subset_nbrs = []
            self.subset_dists = []
            for result in results:
                if result is None:
                    continue
                elif len(result) == 3:
                    i, nbr, dist = result
                    self.peaks[i].extend(nbr)
                    self.dist2peak[i].extend(dist)
                    continue
                i, nbr, dist, feat, adj, lb = result
                self.subset_idxs.append(i)
                self.subset_nbrs.append(nbr)
                self.subset_dists.append(dist)
                self.feat_lst.append(feat)
                self.adj_lst.append(adj)
                if not self.ignore_label:
                    self.subset_gt_labels.append(self.idx2lb[i])
                    self.lb_lst.append(lb)
            self.subset_gt_labels = np.array(self.subset_gt_labels)

            self.size = len(self.feat_lst)
            assert self.size == len(self.adj_lst)
            if not self.ignore_label:
                assert self.size == len(self.lb_lst)

    def get_subgraph(self, i):
        nbr = self.nbrs[i]
        dist = self.dists[i]
        idxs = np.where(self.confs[nbr] > self.confs[i])[0]

        if len(idxs) == 0:
            return None
        elif len(idxs) == 1 or i in self.ignore_set:
            nbr_lst = []
            dist_lst = []
            for j in idxs[:self.max_conn]:
                nbr_lst.append(nbr[j])
                dist_lst.append(self.dists[i, j])
            return i, nbr_lst, dist_lst

        if self.use_candidate_set:
            nbr = nbr[idxs]
            dist = dist[idxs]

        # present `direction`
        feat = self.features[nbr] - self.features[i]
        adj = self.adj[nbr, :][:, nbr]
        adj = row_normalize(adj).toarray().astype(np.float32)

        if not self.ignore_label:
            lb = [int(self.idx2lb[i] == self.idx2lb[n]) for n in nbr]
        else:
            lb = [0 for _ in nbr]  # dummy labels
        lb = np.array(lb)

        return i, nbr, dist, feat, adj, lb

    def __getitem__(self, index):
        features = self.feat_lst[index]
        adj = self.adj_lst[index]
        if not self.ignore_label:
            labels = self.lb_lst[index]
        else:
            labels = -1
        return features, adj, labels

    def __len__(self):
        return self.size
