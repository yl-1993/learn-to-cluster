#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sklearn.cluster as cluster
import multiprocessing as mp

from proposals import knns2spmat, build_knns


def kmeans(feat, n_clusters, **kwargs):
    kmeans = cluster.KMeans(n_clusters=n_clusters,
                            n_jobs=mp.cpu_count(),
                            random_state=0).fit(feat)
    return kmeans.labels_


def mini_batch_kmeans(feat, n_clusters, batch_size, **kwargs):
    kmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters,
                                    batch_size=batch_size,
                                    random_state=0).fit(feat)
    return kmeans.labels_


def spectral(feat, n_clusters, **kwargs):
    spectral = cluster.SpectralClustering(n_clusters=n_clusters,
                                        assign_labels="discretize",
                                        affinity="nearest_neighbors",
                                        random_state=0).fit(feat)
    return spectral.labels_


def hierarchy(feat, n_clusters, knn, **kwargs):
    from sklearn.neighbors import kneighbors_graph
    knn_graph = kneighbors_graph(feat, knn, include_self=False)
    hierarchy = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                                connectivity=knn_graph,
                                                linkage='ward').fit(feat)
    return hierarchy.labels_


def fast_hierarchy(feat, distance, hmethod='single', **kwargs):
    import fastcluster
    import scipy.cluster
    links = fastcluster.linkage_vector(feat,
                                       method=hmethod)
    labels_ = scipy.cluster.hierarchy.fcluster(links,
                                               distance,
                                               criterion='distance')
    return labels_


def dbscan(feat, eps, min_samples, **kwargs):
    db = cluster.DBSCAN(eps=eps, min_samples=min_samples,
                        n_jobs=mp.cpu_count()).fit(feat)
    return db.labels_


def knn_dbscan(feats, eps, min_samples, prefix, knn_method, knn, th_sim, **kwargs):
    knn_prefix = os.path.join(prefix, 'knns')
    knns = build_knns(knn_prefix, feats, knn_method, knn)
    sparse_affinity = knns2spmat(knns, knn, th_sim)
    db = cluster.DBSCAN(eps=eps, min_samples=min_samples,
                        n_jobs=mp.cpu_count(),
                        metric='precomputed').fit(sparse_affinity)
    return db.labels_


def hdbscan(feat, min_samples, **kwargs):
    import hdbscan
    db = hdbscan.HDBSCAN(min_cluster_size=min_samples)
    labels_ = db.fit_predict(feat)
    return labels_
