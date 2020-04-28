#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import inspect
import argparse

import baseline
from utils import (write_meta, set_random_seed, mkdir_if_no_exists,
                   BasicDataset, Timer)

funcs = inspect.getmembers(baseline, inspect.isfunction)
method_names = [n for n, _ in funcs]


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Clustering')
    parser.add_argument("--name",
                        type=str,
                        default='part1_test',
                        help="image features")
    parser.add_argument("--prefix",
                        type=str,
                        default='./data',
                        help="prefix of dataset")
    parser.add_argument("--oprefix",
                        type=str,
                        default='./data/baseline_results',
                        help="prefix of saving clustering results")
    parser.add_argument("--dim",
                        type=int,
                        default=256,
                        help="dimension of feature")
    parser.add_argument("--no_normalize",
                        action='store_true',
                        help="whether to normalize feature")
    parser.add_argument('--method', choices=method_names, required=True)
    # args for different methods
    parser.add_argument('--n_clusters',
                        default=2,
                        type=int,
                        help="KMeans, HAC")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--eps', default=0.7, type=float)
    parser.add_argument('--distance', default=0.7, type=float)
    parser.add_argument('--min_samples', default=10, type=int)
    parser.add_argument('--hmethod', default='single', type=str)
    parser.add_argument('--radius', default=0.1, type=float)
    parser.add_argument('--min_conn', default=1, type=int)
    parser.add_argument('--bw', default=1, type=float)
    parser.add_argument('--min_bin_freq', default=1, type=int)
    parser.add_argument('--knn', default=80, type=int)
    parser.add_argument('--th_sim', default=0.7, type=float)
    parser.add_argument('--iters', default=20, type=int)
    parser.add_argument('--knn_method',
                        default='faiss',
                        choices=['faiss', 'faiss_gpu', 'hnsw'])
    parser.add_argument('--num_process', default=1, type=int)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    return args


def get_output_path(args, ofn='pred_labels.txt'):
    method2name = {
        'aro':
        'k_{}_th_{}'.format(args.knn, args.th_sim),
        'knn_aro':
        'k_{}_th_{}'.format(args.knn, args.th_sim),
        'dbscan':
        'eps_{}_min_{}'.format(args.eps, args.min_samples),
        'knn_dbscan':
        'eps_{}_min_{}_{}_k_{}_th_{}'.format(args.eps, args.min_samples,
                                             args.knn_method, args.knn,
                                             args.th_sim),
        'our_dbscan':
        'min_{}_k_{}_th_{}'.format(args.min_samples, args.knn, args.th_sim),
        'hdbscan':
        'min_{}'.format(args.min_samples),
        'fast_hierarchy':
        'dist_{}_hmethod_{}'.format(args.distance, args.hmethod),
        'hierarchy':
        'n_{}_k_{}'.format(args.n_clusters, args.knn),
        'knn_hierarchy':
        'n_{}_k_{}_th_{}'.format(args.n_clusters, args.knn, args.th_sim),
        'mini_batch_kmeans':
        'n_{}_bs_{}'.format(args.n_clusters, args.batch_size),
        'kmeans':
        'n_{}'.format(args.n_clusters),
        'spectral':
        'n_{}'.format(args.n_clusters),
        'dask_spectral':
        'n_{}'.format(args.n_clusters),
        'knn_spectral':
        'n_{}_k_{}_th_{}'.format(args.n_clusters, args.knn, args.th_sim),
        'densepeak':
        'k_{}_th_{}_r_{}_m_{}'.format(args.knn, args.th_sim, args.radius,
                                      args.min_conn),
        'meanshift':
        'bw_{}_bin_{}'.format(args.bw, args.min_bin_freq),
        'chinese_whispers':
        '{}_k_{}_th_{}_iters_{}'.format(args.knn_method, args.knn, args.th_sim,
                                        args.iters),
        'chinese_whispers_fast':
        '{}_k_{}_th_{}_iters_{}'.format(args.knn_method, args.knn, args.th_sim,
                                        args.iters),
    }

    if args.method in method2name:
        name = '{}_{}_{}'.format(args.name, args.method,
                                 method2name[args.method])
    else:
        name = '{}_{}'.format(args.name, args.method)

    opath = os.path.join(args.oprefix, name, ofn)
    if os.path.exists(opath) and not args.force:
        raise FileExistsError(
            '{} has already existed. Please set force=True to overwrite.'.
            format(opath))
    mkdir_if_no_exists(opath)

    return opath


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)

    cluster_func = baseline.__dict__[args.method]

    ds = BasicDataset(name=args.name,
                      prefix=args.prefix,
                      dim=args.dim,
                      normalize=not args.no_normalize)
    ds.info()
    feats = ds.features

    opath = get_output_path(args)

    with Timer('{}'.format(args.method)):
        pred_labels = cluster_func(feats, **args.__dict__)

    # save clustering results
    idx2lb = {}
    for idx, lb in enumerate(pred_labels):
        if lb == -1:
            continue
        idx2lb[idx] = lb
    inst_num = len(pred_labels)
    print('coverage: {} / {} = {:.4f}'.format(len(idx2lb), inst_num,
                                              1. * len(idx2lb) / inst_num))
    write_meta(opath, idx2lb, inst_num=inst_num)
