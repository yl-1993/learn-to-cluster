#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import argparse
import numpy as np
import multiprocessing as mp
from utils import Timer, mkdir_if_no_exists


def knns_recall(ners, idx2lb, lb2idxs):
    import time
    start = time.time()
    rs = []
    cnt = 0
    for idx, (n, _) in enumerate(ners):
        lb = idx2lb[idx]
        idxs = lb2idxs[lb]
        n = list(n)
        if len(n) == 1:
            cnt += 1
        s = set(idxs) & set(n)
        rs += [1. * len(s) / len(idxs)]
    print('there are {} / {} = {:.3f} isolated anchors.'\
            .format(cnt, len(ners), 1. * cnt / len(ners)))
    print('compute recall consumes: {} s'.format(time.time() - start))
    recall = np.mean(rs)
    return recall


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
                th_knns = list(tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
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
                index.createIndex({'post': 2, 'indexThreadQty': 1}, print_progress=verbose)
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

    def __init__(self, feats, k, index_path='', index_key='', nprobe=128, verbose=True):
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
                    assert index_key.find('HNSW') < 0, 'HNSW returns distances insted of sims'
                    metric = faiss.METRIC_INNER_PRODUCT
                    nlist = min(4096, 8 * round(math.sqrt(size)))
                    if index_key == 'IVF':
                        quantizer = index
                        index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric)
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
