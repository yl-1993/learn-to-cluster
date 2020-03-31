#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from utils import read_meta, load_data
from proposals.metrics import compute_iou, compute_iop, compute_iog


def get_majority(lb2cnt):
    max_cnt = 0
    max_lb = None
    for lb, cnt in lb2cnt.items():
        if cnt > max_cnt:
            max_cnt = cnt
            max_lb = lb
    return max_lb, max_cnt


def compute_avg_size(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)


def coverage(s, size):
    if not isinstance(s, set):
        raise TypeError('s should be set type')
    return 1. * len(s) / size


def inst2cls(inst_sets, idx2lb):
    cls_sets = []
    for inst_set in inst_sets:
        cls_set = set()
        for idx in inst_set:
            cls_set.add(idx2lb[idx])
        cls_sets.append(cls_set)
    return cls_sets


def analyze_clusters(clusters, idx2lb, lb2idxs, th_pos, th_neg):
    pos_set, neg_set = set(), set()
    pos_idx_set, neg_idx_set = set(), set()
    num_nodes = []
    ious = []
    iops = []
    iogs = []
    for nodes in clusters:
        lb2cnt = {}
        nodes = set(nodes)
        # take majority as label of the graph
        for idx in nodes:
            lb = idx2lb[idx]
            if lb not in lb2cnt:
                lb2cnt[lb] = 0
            lb2cnt[lb] += 1
        lb, _ = get_majority(lb2cnt)
        idxs = lb2idxs[lb]
        # compute stat
        iou = compute_iou(nodes, idxs)
        iop = compute_iop(nodes, idxs)
        iog = compute_iog(nodes, idxs)
        ious.append(iou)
        iops.append(iop)
        iogs.append(iog)
        num_nodes.append(len(nodes))
        if iou > th_pos:
            pos_set.add(lb)
            pos_idx_set |= nodes
        elif iou < th_neg:
            neg_set.add(lb)
            neg_idx_set |= nodes
        else:
            pass

    ious = np.array(ious)
    iops = np.array(iops)
    iogs = np.array(iogs)
    num_nodes = np.array(num_nodes)
    return num_nodes, ious, iops, iogs, pos_set, neg_set, pos_idx_set, neg_idx_set


def mse_error(arr, n):
    return np.dot(n - arr, n - arr) / arr.size


def stat_cluster(clusters, idx2lb, lb2idxs, inst_num, cls_num, th_pos, th_neg):
    print('#clusters:', len(clusters))
    num_nodes, ious, iops, iogs, pos_set, neg_set, pos_idx_set, neg_idx_set = \
            analyze_clusters(clusters, idx2lb, lb2idxs, th_pos, th_neg)

    ## compute statistics
    print('isolated anchor: ', len(np.where(num_nodes == 1)[0]))
    avg_node_size = compute_avg_size(num_nodes)
    print('#all_avg_node: {}'.format(int(avg_node_size)))
    pos_num_nodes = num_nodes[np.where(ious > th_pos)]
    if len(pos_num_nodes) > 0:
        avg_node_size = compute_avg_size(pos_num_nodes)
        print('#pos_avg_node: {}, #max_node: {}, #min_node: {}'.format(
            int(avg_node_size), pos_num_nodes.max(), pos_num_nodes.min()))
    neg_num_nodes = num_nodes[np.where(ious < th_pos)]
    if len(neg_num_nodes) > 0:
        avg_node_size = compute_avg_size(neg_num_nodes)
        print('#neg_avg_node: {}, #max_node: {}, #min_node: {}'.format(
            int(avg_node_size), neg_num_nodes.max(), neg_num_nodes.min()))

    pos_g_labels = np.where(ious > th_pos)[0]
    neg_g_labels = np.where(ious < th_neg)[0]
    print('#tot: {}, #pos: {}, #neg: {}'.format(len(ious), pos_g_labels.size,
                                                neg_g_labels.size))

    err0 = mse_error(ious, 0)
    err1 = mse_error(ious, 1)
    print('random guess error: 0({}), 1({})'.format(err0, err1))

    pos_c = coverage(pos_idx_set, inst_num)
    neg_c = coverage(neg_idx_set, inst_num)
    all_c = coverage(pos_idx_set | neg_idx_set, inst_num)
    print(
        '[instance-level] pos coverage: {:.2f}, neg coverage: {:.2f}, all coverage: {:.2f}'
        .format(pos_c, neg_c, all_c))

    pos_c = coverage(pos_set, cls_num)
    neg_c = coverage(neg_set, cls_num)
    all_c = coverage(pos_set | neg_set, cls_num)
    print(
        '[class-level] pos coverage: {:.2f}, neg coverage: {:.2f}, all coverage: {:.2f}'
        .format(pos_c, neg_c, all_c))
