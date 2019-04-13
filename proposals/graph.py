# Graph Toolbox.
# By Xiaohang Zhan (xiaohangzhan@outlook.com)

import numpy as np


class Data(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other, score):
        self.__links.add(other)
        other.__links.add(self)


def connected_components(nodes, edges):
    result = []
    nodes = set(nodes)
    node2ners = {n:set() for n in nodes}
    for n1, n2, _ in edges:
        node2ners[n1].add(n2)
        node2ners[n2].add(n1)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        while queue:
            n = queue.pop(0)
            neighbors = node2ners[n]
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
        result.append(group)
    return result


def connected_components_constraint(nodes, max_sz, score_dict=None, th=None):
    '''
    only use edges whose scores are above `th`
    if a component is larger than `max_sz`, all the nodes in this component are added into `remain` and returned for next iteration.
    '''
    result = []
    remain = set()
    nodes = set(nodes)
    while nodes:
        n = nodes.pop()
        group = {n}
        queue = [n]
        valid = True
        while queue:
            n = queue.pop(0)
            if th is not None:
                neighbors = {l for l in n.links if score_dict[tuple(sorted([n.name, l.name]))] >= th}
            else:
                neighbors = n.links
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)
            if len(group) > max_sz or len(remain.intersection(neighbors)) > 0:
                # if this group is larger than `max_sz`, add the nodes into `remain`
                valid = False
                remain.update(group)
                break
        if valid: # if this group is smaller than or equal to `max_sz`, finalize it.
            result.append(group)
    return result, remain


def graph_clustering_dynamic_th(edges, score, max_sz, step=0.05, max_iter=100):
    edges = np.sort(edges, axis=1)
    th = score.min()

    # construct graph
    score_dict = {} # score lookup table
    for i,e in enumerate(edges):
        score_dict[e[0], e[1]] = score[i]

    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max()+1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_idx = mapping[edges]
    vertex = [Data(n) for n in nodes]
    for l, s in zip(link_idx, score):
        vertex[l[0]].add_link(vertex[l[1]], s)

    # first iteration
    comps, remain = connected_components_constraint(vertex, max_sz)

    # iteration
    components = comps[:]
    Iter = 0
    while remain:
        th = th + (1 - th) * step
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
        Iter += 1
        if Iter >= max_iter:
            print("\t Force stopping at: th {}, remain {}".format(th, len(remain)))
            components.append(remain)
            remain = {}

    return components
