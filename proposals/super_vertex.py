#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from proposals.graph import graph_clustering_dynamic_th
from utils import filter_knns


def super_vertex(knns, k, th, th_step, max_sz):
    pairs, scores = filter_knns(knns, k, th)
    comps = graph_clustering_dynamic_th(pairs, scores, max_sz, th_step)
    clusters = [sorted([n.name for n in c]) for c in comps]
    return clusters
