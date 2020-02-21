#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from igraph import Graph, plot

from utils import load_data


def draw_graph(ofolder, idx2lb, g_label, idx, prob):
    fpath = os.path.join(ofolder, '{}.npz'.format(idx))
    ograph_folder = 'graph/' + ofolder.split('/')[-1]
    if not os.path.exists(ograph_folder):
        os.makedirs(ograph_folder)
    color_dict = {1: "red", 0: "lightblue"}
    vertices, raw_edges = load_data(fpath)
    vertices = list(vertices)
    lb = idx2lb[idx]
    abs2rel = {}
    for i, v in enumerate(vertices):
        abs2rel[v] = i
    edges = [(abs2rel[p1], abs2rel[p2]) for p1, p2, _ in raw_edges]
    g = Graph(vertex_attrs={"label": vertices}, edges=edges, directed=False)
    edge_weights = [1 - d for _, _, d in raw_edges]
    if len(edge_weights) > 0:
        w_mean = sum(edge_weights) / len(edge_weights)
        w_max = max(edge_weights)
        w_min = min(edge_weights)
    else:
        w_mean, w_max, w_min = 1, 1, 1

    visual_style = {}
    visual_style["vertex_color"] = [
        color_dict[lb == idx2lb[v]] for v in vertices
    ]
    visual_style['edge_width'] = [5 * w for w in edge_weights]

    plot(g,
         **visual_style,
         target="{}/{}_{}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.png".format(
             ograph_folder, g_label, idx, prob, w_mean, w_min, w_max))


def draw_graphs(err, idx2lb, gt_folder, draw_err_num=10):
    for lb in err:
        lst = err[lb]
        random.shuffle(lst)
        for idx, prob in lst[:draw_err_num]:
            print(idx, prob[lb])
            draw_graph(gt_folder, idx2lb, lb, idx, prob[lb])
