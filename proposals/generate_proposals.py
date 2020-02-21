#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp
import argparse

from utils import BasicDataset
from proposals import generate_basic_proposals, generate_iter_proposals


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
    args = parser.parse_args()

    return args


def generate_proposals(params, prefix, oprefix, name, dim, no_normalize=False):
    ds = BasicDataset(name=name,
                      prefix=prefix,
                      dim=dim,
                      normalize=not no_normalize)
    ds.info()

    folders = []
    for param in params:
        oprefix_i0 = osp.join(oprefix, name)
        knn_prefix_i0 = osp.join(prefix, 'knns', name)
        folder_i0, pred_labels_i0 = generate_basic_proposals(
            oprefix=oprefix_i0,
            knn_prefix=knn_prefix_i0,
            feats=ds.features,
            feat_dim=dim,
            **param)

        iter0 = param.get('iter0', True)
        if iter0:
            folders.append(folder_i0)

        iter1_params = param.get('iter1_params', [])
        for param_i1 in iter1_params:
            oprefix_i1 = osp.dirname(folder_i0)
            knn_prefix_i1 = osp.join(oprefix_i1, 'knns')
            folder_i1, _ = generate_iter_proposals(oprefix=oprefix_i1,
                                                   knn_prefix=knn_prefix_i1,
                                                   feats=ds.features,
                                                   feat_dim=dim,
                                                   sv_labels=pred_labels_i0,
                                                   sv_knn_prefix=knn_prefix_i0,
                                                   **param_i1)
            folders.append(folder_i1)

    return folders


if __name__ == '__main__':
    args = parse_args()
    k = 80
    knn_method = 'faiss'

    step_i0 = 0.05
    minsz_i0 = 3
    maxsz_i0 = 300

    th_i1 = 0.4
    step_i1 = 0.05
    minsz_i1 = 3
    maxsz_i1 = 500
    sv_minsz_i1 = 2

    params = [
        dict(k=k,
             knn_method=knn_method,
             th_knn=0.6,
             th_step=step_i0,
             minsz=minsz_i0,
             maxsz=maxsz_i0,
             iter1_params=[
                 dict(k=2,
                      knn_method=knn_method,
                      th_knn=th_i1,
                      th_step=step_i1,
                      minsz=minsz_i1,
                      maxsz=maxsz_i1,
                      sv_minsz=sv_minsz_i1,
                      sv_maxsz=8),
                 dict(k=3,
                      knn_method=knn_method,
                      th_knn=th_i1,
                      th_step=step_i1,
                      minsz=minsz_i1,
                      maxsz=maxsz_i1,
                      sv_minsz=sv_minsz_i1,
                      sv_maxsz=5)
             ]),
        dict(
            k=k,
            knn_method=knn_method,
            th_knn=0.7,
            th_step=step_i0,
            minsz=minsz_i0,
            maxsz=maxsz_i0,
            iter0=False,  # do not include the iter0 proposals
            iter1_params=[
                dict(k=2,
                     knn_method=knn_method,
                     th_knn=th_i1,
                     th_step=step_i1,
                     minsz=minsz_i1,
                     maxsz=maxsz_i1,
                     sv_minsz=sv_minsz_i1,
                     sv_maxsz=8),
                dict(k=3,
                     knn_method=knn_method,
                     th_knn=th_i1,
                     th_step=step_i1,
                     minsz=minsz_i1,
                     maxsz=maxsz_i1,
                     sv_minsz=sv_minsz_i1,
                     sv_maxsz=5)
            ]),
    ]
    folders = generate_proposals(params,
                                 args.prefix,
                                 args.oprefix,
                                 args.name,
                                 args.dim,
                                 no_normalize=args.no_normalize)
    print('generate proposals in the following {} folders: {}'.format(
        len(folders), folders))
