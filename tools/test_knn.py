#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

from utils import (BasicDataset, build_knns, knns2spmat, fast_knns2spmat,
                   is_spmat_eq, Timer)


def parse_args():
    parser = argparse.ArgumentParser(description='Test KNN')
    parser.add_argument("--name",
                        type=str,
                        default='part1_test',
                        help="image features")
    parser.add_argument("--prefix",
                        type=str,
                        default='./data',
                        help="prefix of dataset")
    parser.add_argument("--dim",
                        type=int,
                        default=256,
                        help="dimension of feature")
    parser.add_argument('--knn', default=80, type=int)
    parser.add_argument('--th_sim', default=0.6, type=float)
    parser.add_argument('--knn_method',
                        default='faiss',
                        choices=['faiss', 'faiss_gpu', 'hnsw'])
    parser.add_argument('--num_process', default=None, type=int)
    parser.add_argument("--no_normalize",
                        action='store_true',
                        help="normalize feature by default")
    parser.add_argument("--test_all", action='store_true')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    ds = BasicDataset(name=args.name,
                      prefix=args.prefix,
                      dim=args.dim,
                      normalize=args.no_normalize)
    ds.info()

    with Timer('[{}] build_knns'.format(args.knn_method)):
        if args.num_process is None:
            import multiprocessing as mp
            args.num_process = mp.cpu_count()
        print('use {} CPU for computation'.format(args.num_process))
        knn_prefix = os.path.join(args.prefix, 'knns', args.name)
        knns = build_knns(knn_prefix,
                          ds.features,
                          args.knn_method,
                          args.knn,
                          num_process=args.num_process)

    if args.test_all:
        with Timer('knns2spmat'):
            adj1 = knns2spmat(knns, args.knn, args.th_sim, use_sim=True)

        with Timer('fast_knns2spmat'):
            adj2 = fast_knns2spmat(knns, args.knn, args.th_sim, use_sim=True)

        print('#adj: {}, #adj2: {}, #non-eq: {}'.format(
            adj1.nnz, adj2.nnz, (adj1 != adj2).nnz))

        assert is_spmat_eq(adj1, adj2), "adj1 and adj2 are not equal"
        print('Output of knns2spmat and fast_knns2spmat are equal')
