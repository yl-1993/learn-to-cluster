#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import argparse
import numpy as np
import multiprocessing as mp
from utils import load_data, dump_data, mkdir_if_no_exists, Timer

__all__ = [
    'knn_brute_force', 'knn_hnsw', 'knn_faiss', 'knns2spmat', 'build_knns',
    'filter_knns'
]


def knns_recall(ners, idx2lb, lb2idxs):
    with Timer('compute recall'):
        recs = []
        cnt = 0
        for idx, (n, _) in enumerate(ners):
            lb = idx2lb[idx]
            idxs = lb2idxs[lb]
            n = list(n)
            if len(n) == 1:
                cnt += 1
            s = set(idxs) & set(n)
            recs += [1. * len(s) / len(idxs)]
        print('there are {} / {} = {:.3f} isolated anchors.'\
                .format(cnt, len(ners), 1. * cnt / len(ners)))
    recall = np.mean(recs)
    return recall


def filter_knns(knns, k, th):
    pairs = []
    scores = []
    n = len(knns)
    ners = np.zeros([n, k], dtype=np.int32) - 1
    simi = np.zeros([n, k]) - 1
    for i, (ner, dist) in enumerate(knns):
        assert len(ner) == len(dist)
        ners[i, :len(ner)] = ner
        simi[i, :len(ner)] = 1. - dist
    anchor = np.tile(np.arange(n).reshape(n, 1), (1, k))

    # filter
    selidx = np.where((simi >= th) & (ners != -1) & (ners != anchor))
    pairs = np.hstack((anchor[selidx].reshape(-1,
                                              1), ners[selidx].reshape(-1, 1)))
    scores = simi[selidx]

    # keep uniq pairs
    pairs = np.sort(pairs, axis=1)
    pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
    scores = scores[unique_idx]
    return pairs, scores


def knns2spmat(knns, k, th_sim=0.7):
    # convert knns to  symmetric sparse matrix
    from scipy.sparse import csr_matrix
    n = len(knns)
    row, col, data = [], [], []
    for row_i, knn in enumerate(knns):
        nbrs, dists = knn
        for nbr, dist in zip(nbrs, dists):
            if 1 - dist < th_sim or nbr == -1:
                continue
            row.append(row_i)
            col.append(nbr)
            data.append(dist)
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def build_knns(knn_prefix, feats, knn_method, k, is_rebuild=False):
    knn_prefix = os.path.join(knn_prefix, '{}_k_{}'.format(knn_method, k))
    mkdir_if_no_exists(knn_prefix)
    knn_path = knn_prefix + '.npz'
    if not os.path.isfile(knn_path) or is_rebuild:
        index_path = knn_prefix + '.index'
        with Timer('build index'):
            if knn_method == 'hnsw':
                index = knn_hnsw(feats, k, index_path)
            elif knn_method == 'faiss':
                index = knn_faiss(feats, k, index_path)
            else:
                raise KeyError('Unsupported method({}). \
                        Only support hnsw and faiss currently'.format(
                    knn_method))
            knns = index.get_knns()
        with Timer('dump knns to {}'.format(knn_path)):
            dump_data(knn_path, knns, force=True)
    else:
        print('read knn from {}'.format(knn_path))
        knns = load_data(knn_path)
    return knns


class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_ners = []
        th_dists = []
        ners, dists = self.knns[i]
        for n, dist in zip(ners, dists):
            if 1 - dist < self.th:
                continue
            th_ners.append(n)
            th_dists.append(dist)
        th_ners = np.array(th_ners)
        th_dists = np.array(th_dists)
        return (th_ners, th_dists)

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.\
                format(th, nproc), self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_brute_force(knn):
    def __init__(self, feats, k, index_path='', verbose=True):
        self.verbose = verbose
        with Timer('[brute force] build index', verbose):
            feats = feats.astype('float32')
            sim = feats.dot(feats.T)
        with Timer('[brute force] query topk {}'.format(k), verbose):
            ners = np.argpartition(-sim, kth=k)[:, :k]
            idxs = np.array([i for i in range(ners.shape[0])])
            dists = 1 - sim[idxs.reshape(-1, 1), ners]
            self.knns = [(np.array(ner, dtype=np.int32), np.array(dist, dtype=np.float32)) \
                            for ner, dist in zip(ners, dists)]


class knn_hnsw(knn):
    def __init__(self, feats, k, index_path='', verbose=True):
        import nmslib
        self.verbose = verbose
        with Timer('[hnsw] build index', verbose):
            """ higher ef leads to better accuracy, but slower search
                higher M leads to higher accuracy/run_time at fixed ef, but consumes more memory
            """
            # space_params = {
            #     'ef': 100,
            #     'M': 16,
            # }
            # index = nmslib.init(method='hnsw', space='cosinesimil', space_params=space_params)
            index = nmslib.init(method='hnsw', space='cosinesimil')
            if index_path != '' and os.path.isfile(index_path):
                index.loadIndex(index_path)
            else:
                index.addDataPointBatch(feats)
                index.createIndex({
                    'post': 2,
                    'indexThreadQty': 1
                },
                                  print_progress=verbose)
                if index_path:
                    print('[hnsw] save index to {}'.format(index_path))
                    mkdir_if_no_exists(index_path)
                    index.saveIndex(index_path)
        with Timer('[hnsw] query topk {}'.format(k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                print('[hnsw] read knns from {}'.format(knn_ofn))
                self.knns = [(knn[0, :].astype(np.int32), knn[1, :].astype(np.float32)) \
                                for knn in np.load(knn_ofn)['data']]
            else:
                self.knns = index.knnQueryBatch(feats, k=k)


class knn_faiss(knn):
    def __init__(self,
                 feats,
                 k,
                 index_path='',
                 index_key='',
                 nprobe=128,
                 verbose=True):
        import faiss
        self.verbose = verbose
        with Timer('[faiss] build index', verbose):
            if index_path != '' and os.path.exists(index_path):
                print('[faiss] read index from {}'.format(index_path))
                index = faiss.read_index(index_path)
            else:
                feats = feats.astype('float32')
                size, dim = feats.shape
                index = faiss.IndexFlatIP(dim)
                if index_key != '':
                    assert index_key.find(
                        'HNSW') < 0, 'HNSW returns distances insted of sims'
                    metric = faiss.METRIC_INNER_PRODUCT
                    nlist = min(4096, 8 * round(math.sqrt(size)))
                    if index_key == 'IVF':
                        quantizer = index
                        index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                                   metric)
                    else:
                        index = faiss.index_factory(dim, index_key, metric)
                    if index_key.find('Flat') < 0:
                        assert not index.is_trained
                    index.train(feats)
                    index.nprobe = min(nprobe, nlist)
                    assert index.is_trained
                    print('nlist: {}, nprobe: {}'.format(nlist, nprobe))
                index.add(feats)
                if index_path != '':
                    print('[faiss] save index to {}'.format(index_path))
                    mkdir_if_no_exists(index_path)
                    faiss.write_index(index, index_path)
        with Timer('[faiss] query topk {}'.format(k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                print('[faiss] read knns from {}'.format(knn_ofn))
                self.knns = [(knn[0, :].astype(np.int32), knn[1, :].astype(np.float32)) \
                                for knn in np.load(knn_ofn)['data']]
            else:
                sims, ners = index.search(feats, k=k)
                self.knns = [(np.array(ner, dtype=np.int32), 1 - np.array(sim, dtype=np.float32)) \
                                for ner, sim in zip(ners, sims)]


if __name__ == '__main__':
    from utils import l2norm

    k = 30
    d = 256
    nfeat = 10000
    np.random.seed(42)

    feats = np.random.random((nfeat, d)).astype('float32')
    feats = l2norm(feats)

    index1 = knn_hnsw(feats, k)
    index2 = knn_faiss(feats, k)
    index3 = knn_faiss(feats, k, index_key='Flat')
    index4 = knn_faiss(feats, k, index_key='IVF')
    index5 = knn_faiss(feats, k, index_key='IVF100,PQ32')

    print(index1.knns[0])
    print(index2.knns[0])
    print(index3.knns[0])
    print(index4.knns[0])
    print(index5.knns[0])
