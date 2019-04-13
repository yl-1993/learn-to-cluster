#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from proposals.graph import graph_clustering_dynamic_th


def filter_knn(knn, k, th):
    pairs = []
    scores = []
    n = len(knn)
    ners = np.zeros([n, k], dtype=np.int32) - 1
    simi = np.zeros([n, k]) - 1
    for i, (ner, dist) in enumerate(knn):
        assert len(ner) == len(dist)
        ners[i, :len(ner)] = ner
        simi[i, :len(ner)] = 1. - dist
    anchor = np.tile(np.arange(n).reshape(n, 1), (1, k))
    selidx = np.where((simi > th) & (ners != -1) & (ners != anchor))

    pairs = np.hstack((anchor[selidx].reshape(-1, 1), ners[selidx].reshape(-1, 1)))
    scores = simi[selidx]

    pairs = np.sort(pairs, axis=1)
    pairs, unique_idx = np.unique(pairs, return_index=True, axis=0)
    scores = scores[unique_idx]
    return pairs, scores


def super_vertex(knn, k, th, th_step, max_sz):
    pairs, scores = filter_knn(knn, k, th)
    comps = graph_clustering_dynamic_th(pairs, scores, max_sz, th_step)
    clusters = [sorted([n.name for n in c]) for c in comps]
    return clusters
