#!/usr/bin/env python
# -*- coding: utf-8 -*-

from proposals.graph import graph_clustering_dynamic_th
from utils import filter_knns


def super_vertex(knns, k, th, th_step, max_sz):
    pairs, scores = filter_knns(knns, k, th)
    assert len(pairs) == len(scores)
    if len(pairs) == 0:
        return []
    components = graph_clustering_dynamic_th(pairs, scores, max_sz, th_step)
    return components
