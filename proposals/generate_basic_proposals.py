#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import os.path as osp
from tqdm import tqdm

from utils import (dump_data, read_meta, write_meta, build_knns,
                   filter_clusters, labels2clusters, clusters2labels,
                   BasicDataset, Timer)
from proposals import super_vertex


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Basic Proposals')
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
    parser.add_argument('--k', default=80, type=int)
    parser.add_argument('--th_knn', default=0.7, type=float)
    parser.add_argument('--th_step', default=0.05, type=float)
    parser.add_argument('--knn_method',
                        default='faiss',
                        choices=['faiss', 'hnsw'])
    parser.add_argument('--maxsz', default=300, type=int)
    parser.add_argument('--minsz', default=3, type=int)
    parser.add_argument('--is_rebuild', action='store_true')
    parser.add_argument('--is_save_proposals', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    return args


def save_proposals(clusters, knns, ofolder, force=False):
    for lb, nodes in enumerate(tqdm(clusters)):
        opath_node = osp.join(ofolder, '{}_node.npz'.format(lb))
        opath_edge = osp.join(ofolder, '{}_edge.npz'.format(lb))
        if not force and osp.exists(opath_node) and osp.exists(opath_edge):
            continue
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
        nodes = list(nodes)
        dump_data(opath_node, data=nodes, force=force)
        dump_data(opath_edge, data=edges, force=force)


def generate_basic_proposals(oprefix,
                             knn_prefix,
                             feats,
                             feat_dim=256,
                             knn_method='faiss',
                             k=80,
                             th_knn=0.6,
                             th_step=0.05,
                             minsz=3,
                             maxsz=300,
                             is_rebuild=False,
                             is_save_proposals=True,
                             force=False,
                             **kwargs):
    print('k={}, th_knn={}, th_step={}, maxsz={}, is_rebuild={}'.format(
        k, th_knn, th_step, maxsz, is_rebuild))

    # build knns
    knns = build_knns(knn_prefix, feats, knn_method, k, is_rebuild)

    # obtain cluster proposals
    ofolder = osp.join(
        oprefix, '{}_k_{}_th_{}_step_{}_minsz_{}_maxsz_{}_iter_0'.format(
            knn_method, k, th_knn, th_step, minsz, maxsz))
    ofn_pred_labels = osp.join(ofolder, 'pred_labels.txt')
    if not osp.exists(ofolder):
        os.makedirs(ofolder)
    if not osp.isfile(ofn_pred_labels) or is_rebuild:
        with Timer('build super vertices'):
            clusters = super_vertex(knns, k, th_knn, th_step, maxsz)
        with Timer('dump clustering to {}'.format(ofn_pred_labels)):
            labels = clusters2labels(clusters)
            write_meta(ofn_pred_labels, labels)
    else:
        print('read clusters from {}'.format(ofn_pred_labels))
        lb2idxs, _ = read_meta(ofn_pred_labels)
        clusters = labels2clusters(lb2idxs)
    clusters = filter_clusters(clusters, minsz)

    # output cluster proposals
    ofolder_proposals = osp.join(ofolder, 'proposals')
    if is_save_proposals:
        print('saving cluster proposals to {}'.format(ofolder_proposals))
        if not osp.exists(ofolder_proposals):
            os.makedirs(ofolder_proposals)
        save_proposals(clusters, knns, ofolder=ofolder_proposals, force=force)

    return ofolder_proposals, ofn_pred_labels


if __name__ == '__main__':
    args = parse_args()

    ds = BasicDataset(name=args.name,
                      prefix=args.prefix,
                      dim=args.dim,
                      normalize=not args.no_normalize)
    ds.info()

    generate_basic_proposals(osp.join(args.oprefix, args.name),
                             osp.join(args.prefix, 'knns', args.name),
                             ds.features,
                             args.dim,
                             args.knn_method,
                             args.k,
                             args.th_knn,
                             args.th_step,
                             args.minsz,
                             args.maxsz,
                             is_rebuild=args.is_rebuild,
                             is_save_proposals=args.is_save_proposals,
                             force=args.force)
