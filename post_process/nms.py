#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from proposals.iou import compute_iou


def nms(clusters, th=1.):
    # nms
    t0 = time.time()
    suppressed = set()
    if th < 1:
        start_idx = 0
        tot_size = len(clusters)
        while start_idx < tot_size:
            if start_idx in suppressed:
                start_idx += 1
                continue
            cluster = clusters[start_idx]
            for j in range(start_idx+1, tot_size):
                if j in suppressed:
                    continue
                if compute_iou(cluster, clusters[j]) > th:
                    suppressed.add(j)
            start_idx += 1
    else:
        print('th={} >= 1, skip the nms'.format(th))
    print('nms consumes {} s'.format(time.time() - t0))

    # assign label
    lb = 0
    idx2lbs = {}
    for i, cluster in enumerate(clusters):
        if i in suppressed:
            continue
        for v in cluster:
            if v not in idx2lbs:
                idx2lbs[v] = []
            idx2lbs[v].append(lb)
        lb += 1

    # deoverlap (choose the one belongs to the highest predicted iou)
    idx2lb = {}
    for idx, lbs in idx2lbs.items():
        idx2lb[idx] = lbs[0]

    return idx2lb, idx2lbs
