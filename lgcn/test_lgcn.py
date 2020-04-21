from __future__ import division

import os
import torch
import numpy as np
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from lgcn.datasets import build_dataset, build_dataloader
from lgcn.online_evaluation import online_evaluate

from utils import (clusters2labels, intdict2ndarray, get_cluster_idxs,
                   write_meta)
from proposals.graph import graph_clustering_dynamic_th
from evaluation import evaluate


def test(model, dataset, cfg, logger):
    if cfg.load_from:
        print('load from {}'.format(cfg.load_from))
        load_checkpoint(model, cfg.load_from, strict=True, logger=logger)

    losses = []
    edges = []
    scores = []

    if cfg.gpus == 1:
        data_loader = build_dataloader(dataset,
                                       cfg.batch_size_per_gpu,
                                       cfg.workers_per_gpu,
                                       train=False)

        model = MMDataParallel(model, device_ids=range(cfg.gpus))
        if cfg.cuda:
            model.cuda()

        model.eval()
        for i, (data, cid, node_list) in enumerate(data_loader):
            with torch.no_grad():
                _, _, h1id, gtmat = data
                pred, loss = model(data, return_loss=True)
                losses += [loss.item()]
                pred = F.softmax(pred, dim=1)
                if i % cfg.log_config.interval == 0:
                    if dataset.ignore_label:
                        logger.info('[Test] Iter {}/{}'.format(
                            i, len(data_loader)))
                    else:
                        acc, p, r = online_evaluate(gtmat, pred)
                        logger.info(
                            '[Test] Iter {}/{}: Loss {:.4f}, '
                            'Accuracy {:.4f}, Precision {:.4f}, Recall {:.4f}'.
                            format(i, len(data_loader), loss, acc, p, r))

                node_list = node_list.numpy()
                bs = len(cid)
                h1id_num = len(h1id[0])
                for b in range(bs):
                    cidb = cid[b].int().item()
                    nlst = node_list[b]
                    center_idx = nlst[cidb]
                    for j, n in enumerate(h1id[b]):
                        edges.append([center_idx, nlst[n.item()]])
                        scores.append(pred[b * h1id_num + j, 1].item())
    else:
        raise NotImplementedError

    if not dataset.ignore_label:
        avg_loss = sum(losses) / len(losses)
        logger.info('[Test] Overall Loss {:.4f}'.format(avg_loss))

    return np.array(edges), np.array(scores), len(dataset)


def test_lgcn(model, cfg, logger):
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)
    dataset = build_dataset(cfg.test_data)

    ofn_pred = os.path.join(cfg.work_dir, 'pred_edges_scores.npz')
    if os.path.isfile(ofn_pred) and not cfg.force:
        data = np.load(ofn_pred)
        edges = data['edges']
        scores = data['scores']
        inst_num = data['inst_num']
        if inst_num != len(dataset):
            logger.warn(
                'instance number in {} is different from dataset: {} vs {}'.
                format(ofn_pred, inst_num, len(dataset)))
    else:
        edges, scores, inst_num = test(model, dataset, cfg, logger)

    # produce predicted labels
    clusters = graph_clustering_dynamic_th(edges,
                                           scores,
                                           max_sz=cfg.max_sz,
                                           step=cfg.step,
                                           pool=cfg.pool)
    pred_idx2lb = clusters2labels(clusters)
    pred_labels = intdict2ndarray(pred_idx2lb)

    if cfg.save_output:
        print('save predicted edges and scores to {}'.format(ofn_pred))
        np.savez_compressed(ofn_pred,
                            edges=edges,
                            scores=scores,
                            inst_num=inst_num)
        ofn_meta = os.path.join(cfg.work_dir, 'pred_labels.txt')
        write_meta(ofn_meta, pred_idx2lb, inst_num=inst_num)

    # evaluation
    if not dataset.ignore_label:
        print('==> evaluation')
        gt_labels = dataset.labels
        for metric in cfg.metrics:
            evaluate(gt_labels, pred_labels, metric)

        single_cluster_idxs = get_cluster_idxs(clusters, size=1)
        print('==> evaluation (removing {} single clusters)'.format(
            len(single_cluster_idxs)))
        remain_idxs = np.setdiff1d(np.arange(len(dataset)),
                                   np.array(single_cluster_idxs))
        remain_idxs = np.array(remain_idxs)
        for metric in cfg.metrics:
            evaluate(gt_labels[remain_idxs], pred_labels[remain_idxs], metric)
