#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import pickle
import random
import numpy as np
import scipy.sparse as sp
import torch


class TextColors:
    HEADER = '\033[35m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[33m'
    FATAL = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None


def set_random_seed(seed, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def is_l2norm(features, size):
    rand_i = random.choice(range(size))
    norm_ = np.dot(features[rand_i, :], features[rand_i, :])
    return abs(norm_ - 1) < 1e-6


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # if rowsum <= 0, keep its previous value
    rowsum[rowsum <= 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def is_spmat_eq(a, b):
    return (a != b).nnz == 0


def aggregate(features, adj, times):
    dtype = features.dtype
    for i in range(times):
        features = adj * features
    return features.astype(dtype)


def read_probs(path, inst_num, feat_dim, dtype=np.float32, verbose=False):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
    probs = np.fromfile(path, dtype=dtype, count=count)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs


def read_meta(fn_meta, start_pos=0, verbose=True):
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


def write_meta(ofn, idx2lb, inst_num=None):
    if inst_num is None:
        inst_num = max(idx2lb.keys()) + 1
    cls_num = len(set(idx2lb.values()))

    idx2newlb = {}
    current_lb = 0
    discard_lb = 0
    map2newlb = {}
    for idx in range(inst_num):
        if idx in idx2lb:
            lb = idx2lb[idx]
            if lb in map2newlb:
                newlb = map2newlb[lb]
            else:
                newlb = current_lb
                map2newlb[lb] = newlb
                current_lb += 1
        else:
            newlb = cls_num + discard_lb
            discard_lb += 1
        idx2newlb[idx] = newlb
    assert current_lb == cls_num, '{} vs {}'.format(current_lb, cls_num)

    print('#discard: {}, #lbs: {}'.format(discard_lb, current_lb))
    print('#inst: {}, #class: {}'.format(inst_num, cls_num))
    if ofn is not None:
        print('save label to', ofn)
        with open(ofn, 'w') as of:
            for idx in range(inst_num):
                of.write(str(idx2newlb[idx]) + '\n')

    pred_labels = intdict2ndarray(idx2newlb)
    return pred_labels


def write_feat(ofn, features):
    print('save features to', ofn)
    features.tofile(ofn)


def dump2npz(ofn, data, force=False):
    if os.path.exists(ofn) and not force:
        return
    np.savez_compressed(ofn, data=data)


def dump2json(ofn, data, force=False):
    if os.path.exists(ofn) and not force:
        return

    def default(obj):
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, set) or isinstance(obj, np.ndarray):
            return list(obj)
        else:
            raise TypeError("Unserializable object {} of type {}".format(
                obj, type(obj)))

    with open(ofn, 'w') as of:
        json.dump(data, of, default=default)


def dump2pkl(ofn, data, force=False):
    if os.path.exists(ofn) and not force:
        return
    with open(ofn, 'wb') as of:
        pickle.dump(data, of)


def dump_data(ofn, data, force=False, verbose=False):
    if os.path.exists(ofn) and not force:
        if verbose:
            print(
                '{} already exists. Set force=True to overwrite.'.format(ofn))
        return
    mkdir_if_no_exists(ofn)
    if ofn.endswith('.json'):
        dump2json(ofn, data, force=force)
    elif ofn.endswith('.pkl'):
        dump2pkl(ofn, data, force=force)
    else:
        dump2npz(ofn, data, force=force)


def load_npz(fn):
    return np.load(fn, allow_pickle=True)['data']


def load_pkl(fn):
    return pickle.load(open(fn, 'rb'))


def load_json(fn):
    return json.load(open(fn, 'r'))


def load_data(ofn):
    if ofn.endswith('.json'):
        return load_json(ofn)
    elif ofn.endswith('.pkl'):
        return load_pkl(ofn)
    else:
        return load_npz(ofn)


def labels2clusters(lb2idxs):
    clusters = [idxs for _, idxs in lb2idxs.items()]
    return clusters


def clusters2labels(clusters):
    idx2lb = {}
    for lb, cluster in enumerate(clusters):
        for v in cluster:
            idx2lb[v] = lb
    return idx2lb


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def mkdir_if_no_exists(path, subdirs=[''], is_folder=False):
    if path == '':
        return
    for sd in subdirs:
        if sd != '' or is_folder:
            d = os.path.dirname(os.path.join(path, sd))
        else:
            d = os.path.dirname(path)
        if not os.path.exists(d):
            os.makedirs(d)


def rm_suffix(s):
    return s[:s.rfind(".")]


def rand_argmax(v):
    assert len(v.squeeze().shape) == 1
    return np.random.choice(np.flatnonzero(v == v.max()))


def create_temp_file_if_exist(path, suffix=''):
    path_with_suffix = path + suffix
    if not os.path.exists(path_with_suffix):
        return path_with_suffix
    else:
        i = 0
        while i < 1000:
            temp_path = '{}_{}'.format(path, i) + suffix
            i += 1
            if not os.path.exists(temp_path):
                return temp_path
