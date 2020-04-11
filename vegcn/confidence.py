#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from itertools import groupby

__all__ = ['confidence', 'confidence_to_peaks']


def density(dists, radius=0.3, use_weight=True):
    row, col = (dists < radius).nonzero()

    num, _ = dists.shape
    if use_weight:
        density = np.zeros((num, ), dtype=np.float32)
        for r, c in zip(row, col):
            density[r] += 1 - dists[r, c]
    else:
        density = np.zeros((num, ), dtype=np.int32)
        for k, g in groupby(row):
            density[k] = len(list(g))
    return density


def s_nbr(dists, nbrs, idx2lb, **kwargs):
    ''' use supervised confidence defined on neigborhood
    '''
    num, _ = dists.shape
    conf = np.zeros((num, ), dtype=np.float32)
    contain_neg = 0
    for i, (nbr, dist) in enumerate(zip(nbrs, dists)):
        lb = idx2lb[i]
        pos, neg = 0, 0
        for j, n in enumerate(nbr):
            if idx2lb[n] == lb:
                pos += 1 - dist[j]
            else:
                neg += 1 - dist[j]
        conf[i] = pos - neg
        if neg > 0:
            contain_neg += 1
    print('#contain_neg:', contain_neg)
    conf /= np.abs(conf).max()
    return conf


def s_nbr_size_norm(dists, nbrs, idx2lb, **kwargs):
    ''' use supervised confidence defined on neigborhood (norm by size)
    '''
    num, _ = dists.shape
    conf = np.zeros((num, ), dtype=np.float32)
    contain_neg = 0
    max_size = 0
    for i, (nbr, dist) in enumerate(zip(nbrs, dists)):
        size = 0
        pos, neg = 0, 0
        lb = idx2lb[i]
        for j, n in enumerate(nbr):
            if idx2lb[n] == lb:
                pos += 1 - dist[j]
            else:
                neg += 1 - dist[j]
            size += 1
        conf[i] = pos - neg
        max_size = max(max_size, size)
        if neg > 0:
            contain_neg += 1
    print('#contain_neg:', contain_neg)
    print('max_size: {}'.format(max_size))
    conf /= max_size
    return conf


def s_avg(feats, idx2lb, lb2idxs, **kwargs):
    ''' use average similarity of intra-nodes
    '''
    num = len(idx2lb)
    conf = np.zeros((num, ), dtype=np.float32)
    for i in range(num):
        lb = idx2lb[i]
        idxs = lb2idxs[lb]
        idxs.remove(i)
        if len(idxs) == 0:
            continue
        feat = feats[i, :]
        conf[i] = feat.dot(feats[idxs, :].T).mean()
    eps = 1e-6
    assert -1 - eps <= conf.min() <= conf.max(
    ) <= 1 + eps, "min: {}, max: {}".format(conf.min(), conf.max())
    return conf


def s_center(feats, idx2lb, lb2idxs, **kwargs):
    ''' use average similarity of intra-nodes
    '''
    num = len(idx2lb)
    conf = np.zeros((num, ), dtype=np.float32)
    for i in range(num):
        lb = idx2lb[i]
        idxs = lb2idxs[lb]
        if len(idxs) == 0:
            continue
        feat = feats[i, :]
        feat_center = feats[idxs, :].mean(axis=0)
        conf[i] = feat.dot(feat_center.T)
    eps = 1e-6
    assert -1 - eps <= conf.min() <= conf.max(
    ) <= 1 + eps, "min: {}, max: {}".format(conf.min(), conf.max())
    return conf


def confidence(metric='s_nbr', **kwargs):
    metric2func = {
        's_nbr': s_nbr,
        's_nbr_size_norm': s_nbr_size_norm,
        's_avg': s_avg,
        's_center': s_center,
    }
    if metric in metric2func:
        func = metric2func[metric]
    else:
        raise KeyError('Only support confidence metircs: {}'.format(
            metric2func.keys()))

    conf = func(**kwargs)
    return conf


def confidence_to_peaks(dists, nbrs, confidence, max_conn=1):
    # Note that dists has been sorted in ascending order
    assert dists.shape[0] == confidence.shape[0]
    assert dists.shape == nbrs.shape

    num, _ = dists.shape
    dist2peak = {i: [] for i in range(num)}
    peaks = {i: [] for i in range(num)}

    for i, nbr in tqdm(enumerate(nbrs)):
        nbr_conf = confidence[nbr]
        for j, c in enumerate(nbr_conf):
            nbr_idx = nbr[j]
            if i == nbr_idx or c <= confidence[i]:
                continue
            dist2peak[i].append(dists[i, j])
            peaks[i].append(nbr_idx)
            if len(dist2peak[i]) >= max_conn:
                break
    return dist2peak, peaks
