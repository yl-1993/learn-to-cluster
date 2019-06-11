#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

from utils import load_data, write_meta
from post_process import nms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Super-vertex Deoverlap')
    parser.add_argument('--pred_score', type=str)
    parser.add_argument('--th_pos', default=-1, type=float)
    parser.add_argument('--th_iou', default=1, type=float)
    parser.add_argument('--output_name', default='', type=str)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    assert args.th_iou >= 0

    assert args.pred_score.endswith('.npz')
    if args.output_name == '':
        pos = args.pred_score.rfind('.npz')
        pred_label_fn = args.pred_score[:pos] + '_th_iou_{}_pos_{}_pred_label.txt'.format(args.th_iou, args.th_pos)
    else:
        pred_label_fn = args.output_name

    print('th_pos={}, th_iou={}, pred_score={}, pred_label_fn={}'.\
            format(args.th_pos, args.th_iou, args.pred_score, pred_label_fn))

    d = np.load(args.pred_score)
    probs = d['data']
    meta = d['meta'].item()
    proposal_folders = meta['proposal_folders']
    tot_inst_num = meta['tot_inst_num']

    print('avg_score(mean: {:.2f}, max: {:.2f}, min: {:.2f})'.format(probs.mean(), probs.max(), probs.min()))

    proposals = []
    fn_node_pattern = '*_node.npz'
    for proposal_folder in proposal_folders:
        fn_clusters = sorted(glob.glob(os.path.join(proposal_folder, fn_node_pattern)))
        proposals.extend([fn_node for fn_node in fn_clusters])
    assert len(proposals) == len(probs)

    pos_lst = []
    for idx, prob in enumerate(probs):
        if prob < args.th_pos:
            continue
        pos_lst.append([idx, prob])
    pos_lst = sorted(pos_lst, key=lambda x:x[1], reverse=True)

    # get all clusters
    clusters = []
    for idx, _ in tqdm(pos_lst):
        cluster = load_data(proposals[idx])
        clusters.append(cluster)

    idx2lb, idx2lbs = nms(clusters, args.th_iou)

    # output stats
    multi_lb_num = 0
    for _, lbs in idx2lbs.items():
        if len(lbs) > 1:
            multi_lb_num += 1
    inst_num = len(idx2lb)
    cls_num = len(set(idx2lb.values()))

    print('#inst: {}, #class: {}, #multi-label: {}'.format(inst_num, cls_num, multi_lb_num))
    print('#inst-coverage: {:.2f}'.format(1. * inst_num / tot_inst_num))

    # save to file
    write_meta(pred_label_fn, idx2lb, inst_num=tot_inst_num)
