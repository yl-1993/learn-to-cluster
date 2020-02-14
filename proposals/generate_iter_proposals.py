#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from utils import (read_meta, write_meta, build_knns, labels2clusters,
                   clusters2labels, BasicDataset, Timer)
from proposals import super_vertex, filter_clusters, save_proposals


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Iterative Proposals')
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
    parser.add_argument('--k', default=3, type=int)
    parser.add_argument('--th_knn', default=0.6, type=float)
    parser.add_argument('--th_step', default=0.05, type=float)
    parser.add_argument('--knn_method',
                        default='faiss',
                        choices=['faiss', 'hnsw'])
    parser.add_argument('--minsz', default=3, type=int)
    parser.add_argument('--maxsz', default=500, type=int)
    parser.add_argument('--sv_minsz', default=2, type=int)
    parser.add_argument('--sv_maxsz', default=5, type=int)
    parser.add_argument("--sv_labels",
                        type=str,
                        default=None,
                        help="super vertex labels")
    parser.add_argument("--sv_knn_prefix",
                        type=str,
                        default=None,
                        help="super vertex precomputed knn")
    parser.add_argument('--is_rebuild', action='store_true')
    parser.add_argument('--is_save_proposals', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    return args


def parse_path(s):
    s = os.path.dirname(s)
    s = s.split('/')[-1]
    lst = s.split('_')
    lst.insert(0, 'knn_method')
    dic1 = {}
    for i in range(0, len(lst), 2):
        dic1[lst[i]] = lst[i + 1]
    dic = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    assert dic == dic1
    return dic


def get_iter_from_path(s):
    return int(parse_path(s)['iter'])


def get_knns_from_path(s, knn_prefix, feats):
    dic = parse_path(s)
    k = int(dic['k'])
    knn_method = dic['knn_method']
    knns = build_knns(knn_prefix, feats, knn_method, k, is_rebuild=False)
    return knns


def generate_iter_proposals(oprefix,
                            knn_prefix,
                            feats,
                            feat_dim=256,
                            knn_method='faiss',
                            k=80,
                            th_knn=0.6,
                            th_step=0.05,
                            minsz=3,
                            maxsz=300,
                            sv_minsz=2,
                            sv_maxsz=5,
                            sv_labels=None,
                            sv_knn_prefix=None,
                            is_rebuild=False,
                            is_save_proposals=True,
                            force=False,
                            **kwargs):

    assert sv_minsz >= 2, "sv_minsz >= 2 to avoid duplicated proposals"
    print('k={}, th_knn={}, th_step={}, minsz={}, maxsz={}, '
          'sv_minsz={}, sv_maxsz={}, is_rebuild={}'.format(
              k, th_knn, th_step, minsz, maxsz, sv_minsz, sv_maxsz,
              is_rebuild))

    if not os.path.exists(sv_labels):
        raise FileNotFoundError('{} not found.'.format(sv_labels))

    if sv_knn_prefix is None:
        sv_knn_prefix = knn_prefix

    # get iter and knns from super vertex path
    _iter = get_iter_from_path(sv_labels) + 1
    knns_inst = get_knns_from_path(sv_labels, sv_knn_prefix, feats)
    print('read sv_clusters from {}'.format(sv_labels))
    sv_lb2idxs, sv_idx2lb = read_meta(sv_labels)
    inst_num = len(sv_idx2lb)
    sv_clusters = labels2clusters(sv_lb2idxs)
    # sv_clusters = filter_clusters(sv_clusters, minsz)
    feats = np.array([feats[c, :].mean(axis=0) for c in sv_clusters])
    print('average feature of super vertices:', feats.shape)

    # build knns
    knns = build_knns(knn_prefix, feats, knn_method, k, is_rebuild)

    # obtain cluster proposals
    ofolder = os.path.join(
        oprefix,
        '{}_k_{}_th_{}_step_{}_minsz_{}_maxsz_{}_sv_minsz_{}_maxsz_{}_iter_{}'.
        format(knn_method, k, th_knn, th_step, minsz, maxsz, sv_minsz,
               sv_maxsz, _iter))
    ofn_pred_labels = os.path.join(ofolder, 'pred_labels.txt')
    if not os.path.exists(ofolder):
        os.makedirs(ofolder)
    if not os.path.isfile(ofn_pred_labels) or is_rebuild:
        with Timer('build super vertices (iter={})'.format(_iter)):
            clusters = super_vertex(knns, k, th_knn, th_step, sv_maxsz)
            clusters = filter_clusters(clusters, sv_minsz)
            clusters = [[x for c in cluster for x in sv_clusters[c]]
                        for cluster in clusters]
        with Timer('dump clustering to {}'.format(ofn_pred_labels)):
            labels = clusters2labels(clusters)
            write_meta(ofn_pred_labels, labels, inst_num=inst_num)
    else:
        print('read clusters from {}'.format(ofn_pred_labels))
        lb2idxs, _ = read_meta(ofn_pred_labels)
        clusters = labels2clusters(lb2idxs)
    clusters = filter_clusters(clusters, minsz, maxsz)

    # output cluster proposals
    ofolder_proposals = os.path.join(ofolder, 'proposals')
    if is_save_proposals:
        print('saving cluster proposals to {}'.format(ofolder_proposals))
        if not os.path.exists(ofolder_proposals):
            os.makedirs(ofolder_proposals)
        save_proposals(clusters,
                       knns_inst,
                       ofolder=ofolder_proposals,
                       force=force)

    return ofolder_proposals, ofn_pred_labels


if __name__ == '__main__':
    args = parse_args()

    ds = BasicDataset(name=args.name,
                      prefix=args.prefix,
                      dim=args.dim,
                      normalize=not args.no_normalize)
    ds.info()

    sv_folder = os.path.dirname(args.sv_labels)
    generate_iter_proposals(sv_folder,
                            os.path.join(sv_folder, 'knns'),
                            ds.features,
                            args.dim,
                            args.knn_method,
                            args.k,
                            args.th_knn,
                            args.th_step,
                            args.minsz,
                            args.maxsz,
                            args.sv_minsz,
                            args.sv_maxsz,
                            sv_labels=args.sv_labels,
                            sv_knn_prefix=args.sv_knn_prefix,
                            is_rebuild=args.is_rebuild,
                            is_save_proposals=args.is_save_proposals,
                            force=args.force)
