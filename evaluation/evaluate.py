#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from utils import Timer, TextColors
from evaluation import bcubed, pairwise


def _read_meta(fn):
    labels = list()
    lb_set = set()
    with open(fn) as f:
        for lb in f.readlines():
            lb = int(lb.strip())
            labels.append(lb)
            lb_set.add(lb)
    return np.array(labels), lb_set


def evaluate(gt_labels, pred_labels, metric='pairwise'):
    print('[gt_labels] {}'.format(gt_labels))
    print('[pred_labels] {}'.format(pred_labels))
    gt_labels, gt_lb_set = _read_meta(gt_labels)
    pred_labels, pred_lb_set = _read_meta(pred_labels)

    print('#inst: gt({}) vs pred({})'.format(len(gt_labels), len(pred_labels)))
    print('#cls: gt({}) vs pred({})'.format(len(gt_lb_set), len(pred_lb_set)))

    if metric == 'bcubed':
        func = bcubed
    elif metric == 'pairwise':
        func = pairwise
    else:
        raise KeyError('Unsupported evaluation metric', args.metric)

    with Timer('evaluate with {}{}{}'.format(TextColors.FATAL, metric, TextColors.ENDC)):
        ave_pre, ave_rec, fscore = func(gt_labels, pred_labels)
    print('{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}{}'.\
            format(TextColors.OKGREEN, ave_pre, ave_rec, fscore, TextColors.ENDC))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Cluster')
    parser.add_argument('--gt_labels', type=str, required=True)
    parser.add_argument('--pred_labels', type=str, required=True)
    parser.add_argument('--metric',
                        default='pairwise',
                        choices=['pairwise', 'bcubed'])
    args = parser.parse_args()

    evaluate(args.gt_labels, args.pred_labels, args.metric)
