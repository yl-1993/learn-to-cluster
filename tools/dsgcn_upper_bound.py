#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from utils import load_data, read_meta, write_meta, labels2clusters
from proposals import get_majority, compute_iou
from post_process import nms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN Upper Bound')
    parser.add_argument('--cluster_path', nargs='+')
    parser.add_argument('--th_pos', default=-1, type=float)
    parser.add_argument('--th_iou', default=1, type=float)
    parser.add_argument('--gt_labels', type=str, required=True)
    parser.add_argument('--output_name', default='', type=str)
    parser.add_argument('--output_folder',
                        default='./data/results/gcn_ub/',
                        type=str)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    assert args.th_iou >= 0

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    cluster_name = args.output_name + '_' if args.output_name != '' else ''
    pred_label_fn = os.path.join(
        args.output_folder,
        '{}th_iou_{}_pos_{}_pred_labels.txt'.format(cluster_name, args.th_iou,
                                                    args.th_pos))

    if os.path.exists(pred_label_fn) and not args.force:
        print('{} has already existed. Please set force=True to overwrite.'.
              format(pred_label_fn))
        exit()

    # read label
    lb2idxs, idx2lb = read_meta(args.gt_labels)
    tot_inst_num = len(idx2lb)

    clusters = []
    for path in args.cluster_path:
        path = path.replace('\\', '')
        if path.endswith('.npz'):
            clusters.extend(load_data(path))
        elif path.endswith('.txt'):
            lb2idxs_, _ = read_meta(path)
            clusters.extend(labels2clusters(lb2idxs_))
        else:
            raise ValueError('Unkown suffix', path)

    # get ground-truth iou
    ious = []
    for cluster in clusters:
        lb2cnt = {}
        cluster = set(cluster)
        # take majority as label of the graph
        for idx in cluster:
            if idx not in idx2lb:
                print('[warn] {} is not found'.format(idx))
                continue
            lb = idx2lb[idx]
            if lb not in lb2cnt:
                lb2cnt[lb] = 0
            lb2cnt[lb] += 1
        lb, _ = get_majority(lb2cnt)
        if lb is None:
            iou = -1e6
        else:
            idxs = lb2idxs[lb]
            iou = compute_iou(cluster, idxs)
        ious.append(iou)
    ious = np.array(ious)

    # rank by iou
    pos_g_labels = np.where(ious > args.th_pos)[0]
    clusters = [[clusters[i], ious[i]] for i in pos_g_labels]
    clusters = sorted(clusters, key=lambda x: x[1], reverse=True)
    clusters = [n for n, _ in clusters]

    inst_num = len(idx2lb)
    pos_idx_set = set()
    for c in clusters:
        pos_idx_set |= set(c)
    print('inst-coverage before nms: {}'.format(1. * len(pos_idx_set) /
                                                inst_num))

    # nms
    idx2lb, _ = nms(clusters, args.th_iou)

    # output stats
    inst_num = len(idx2lb)
    cls_num = len(idx2lb.values())

    print('#inst: {}, #class: {}'.format(inst_num, cls_num))
    print('#inst-coverage: {:.2f}'.format(1. * inst_num / tot_inst_num))

    # save to file
    write_meta(pred_label_fn, idx2lb, inst_num=tot_inst_num)
