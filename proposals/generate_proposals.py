#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from tqdm import tqdm

from utils import (dump_data, read_meta, write_meta, build_knns,
                   labels2clusters, clusters2labels, BasicDataset, Timer)
from proposals import super_vertex


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Proposals')
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
                        default='./data/cluster_proposals',
                        help="prefix of saving super vertx")
    parser.add_argument("--dim",
                        type=int,
                        default=256,
                        help="dimension of feature")
    parser.add_argument("--no_normalize",
                        action='store_true',
                        help="normalize feature by default")
    parser.add_argument('--knn', default=80, type=int)
    parser.add_argument('--th_knn', default=0.7, type=float)
    parser.add_argument('--th_step', default=0.05, type=float)
    parser.add_argument('--knn_method',
                        default='faiss',
                        choices=['faiss', 'hnsw'])
    parser.add_argument('--max_size', default=300, type=int)
    parser.add_argument('--min_size', default=3, type=int)
    parser.add_argument('--is_rebuild', action='store_true')
    parser.add_argument('--is_save_proposals', action='store_true')
    args = parser.parse_args()

    return args


def filter_clusters(clusters, min_size=None, max_size=None):
    if min_size is not None:
        clusters = [c for c in clusters if len(c) >= min_size]
    if max_size is not None:
        clusters = [c for c in clusters if len(c) <= max_size]
    return clusters


def save_proposals(clusters, knns, ofolder, force=False):
    for lb, nodes in enumerate(tqdm(clusters)):
        nodes = set(nodes)
        edges = []
        visited = set()
        # get edges from knn
        for idx in nodes:
            ners, dists = knns[idx]
            for n, dist in zip(ners, dists):
                if n == idx or n not in nodes:
                    continue
                idx1, idx2 = (idx, n) if idx < n else (n, idx)
                key = '{}-{}'.format(idx1, idx2)
                if key not in visited:
                    visited.add(key)
                    edges.append([idx1, idx2, dist])
        # save to npz file
        opath_node = os.path.join(ofolder, '{}_node.npz'.format(lb))
        opath_edge = os.path.join(ofolder, '{}_edge.npz'.format(lb))
        nodes = list(nodes)
        dump_data(opath_node, data=nodes, force=force)
        dump_data(opath_edge, data=edges, force=force)


def generate_proposals(oprefix,
                       knn_prefix,
                       feats,
                       feat_dim=256,
                       knn_method='faiss',
                       k=80,
                       th_knn=0.6,
                       th_step=0.05,
                       min_size=3,
                       max_size=300,
                       is_rebuild=False,
                       is_save_proposals=False):
    print('k={}, th_knn={}, th_step={}, max_size={}, is_rebuild={}'.\
            format(k, th_knn, th_step, max_size, is_rebuild))

    # build knns
    knns = build_knns(knn_prefix, feats, knn_method, k, is_rebuild)

    # obtain cluster proposals
    ofolder = os.path.join(oprefix,
            '{}_k_{}_th_{}_step_{}_minsz_{}_maxsz_{}_iter_0'.\
            format(knn_method, k, th_knn, th_step, min_size, max_size))
    ofn_pred_labels = os.path.join(ofolder, 'pred_labels.txt')
    if not os.path.exists(ofolder):
        os.makedirs(ofolder)
    if not os.path.isfile(ofn_pred_labels) or is_rebuild:
        with Timer('build super vertices'):
            clusters = super_vertex(knns, k, th_knn, th_step, max_size)
        with Timer('dump clustering to {}'.format(ofn_pred_labels)):
            labels = clusters2labels(clusters)
            write_meta(ofn_pred_labels, labels)
    else:
        print('read clusters from {}'.format(ofn_pred_labels))
        lb2idxs, _ = read_meta(ofn_pred_labels)
        clusters = labels2clusters(lb2idxs)
    clusters = filter_clusters(clusters, min_size)

    # output cluster proposals
    if is_save_proposals:
        ofolder = os.path.join(ofolder, 'proposals')
        print('saving cluster proposals to {}'.format(ofolder))
        if not os.path.exists(ofolder):
            os.makedirs(ofolder)
        save_proposals(clusters, knns, ofolder=ofolder, force=True)


if __name__ == '__main__':
    args = parse_args()

    ds = BasicDataset(name=args.name,
                      prefix=args.prefix,
                      dim=args.dim,
                      normalize=not args.no_normalize)
    ds.info()

    generate_proposals(os.path.join(args.oprefix, args.name),
                       os.path.join(args.prefix, 'knns', args.name),
                       ds.features,
                       args.dim,
                       args.knn_method,
                       args.knn,
                       args.th_knn,
                       args.th_step,
                       args.min_size,
                       args.max_size,
                       is_rebuild=args.is_rebuild,
                       is_save_proposals=args.is_save_proposals)
