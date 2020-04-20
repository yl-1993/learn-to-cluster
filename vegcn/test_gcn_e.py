from __future__ import division

import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from vegcn.datasets import build_dataset
from vegcn.deduce import peaks_to_labels
from lgcn.datasets import build_dataloader

from utils import (list2dict, write_meta, mkdir_if_no_exists, Timer)
from evaluation import evaluate, accuracy


def output_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def test(model, dataset, cfg, logger):
    if cfg.load_from:
        print('load from {}'.format(cfg.load_from))
        load_checkpoint(model, cfg.load_from)

    losses = []
    accs = []
    pred_conns = []

    max_lst = []
    multi_max = []

    if cfg.gpus == 1:
        data_loader = build_dataloader(dataset,
                                       cfg.batch_size_per_gpu,
                                       cfg.workers_per_gpu,
                                       train=False)
        size = len(data_loader)

        model = MMDataParallel(model, device_ids=range(cfg.gpus))
        if cfg.cuda:
            model.cuda()

        model.eval()
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                output, loss = model(data, return_loss=True)
                output = F.log_softmax(output, dim=-1)
                if not dataset.ignore_label:
                    labels = data[2].view(-1)
                    if not cfg.regressor:
                        acc = output_accuracy(output, labels)
                        accs += [acc.item()]
                    losses += [loss.item()]
                if not cfg.regressor:
                    output = output[:, 1]
                if cfg.max_conn == 1:
                    output_max = output.max()
                    pred = (output == output_max).nonzero().view(-1)
                    pred_size = len(pred)
                    if pred_size > 1:
                        multi_max.append(pred_size)
                        pred_i = np.random.choice(np.arange(pred_size))
                    else:
                        pred_i = 0
                    pred = [int(pred[pred_i].detach().cpu().numpy())]
                    max_lst.append(output_max.detach().cpu().numpy())
                elif cfg.max_conn > 1:
                    output = output.detach().cpu().numpy()
                    pred = output.argpartition(cfg.max_conn)[:cfg.max_conn]
                pred_conns.append(pred)
                if i % cfg.log_config.interval == 0:
                    if dataset.ignore_label:
                        logger.info('[Test] Iter {}/{}'.format(i, size))
                    else:
                        logger.info('[Test] Iter {}/{}: Loss {:.4f}'.format(
                            i, size, loss))
    else:
        raise NotImplementedError

    if not dataset.ignore_label:
        avg_loss = sum(losses) / len(losses)
        logger.info('[Test] Overall Loss {:.4f}'.format(avg_loss))
        if not cfg.regressor:
            avg_acc = sum(accs) / len(accs)
            logger.info('[Test] Overall Accuracy {:.4f}'.format(avg_acc))
    if size > 0:
        logger.info('max val: mean({:.2f}), max({:.2f}), min({:.2f})'.format(
            sum(max_lst) / size, max(max_lst), min(max_lst)))
    multi_max_size = len(multi_max)
    if multi_max_size > 0:
        logger.info('multi-max({:.2f}): mean({:.1f}), max({}), min({})'.format(
            1. * multi_max_size / size,
            sum(multi_max) / multi_max_size, max(multi_max), min(multi_max)))

    return np.array(pred_conns)


def test_gcn_e(model, cfg, logger):
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)
    dataset = build_dataset(cfg.model['type'], cfg.test_data)

    pred_peaks = dataset.peaks
    pred_dist2peak = dataset.dist2peak

    ofn_pred = osp.join(cfg.work_dir, 'pred_conns.npz')
    if osp.isfile(ofn_pred) and not cfg.force:
        data = np.load(ofn_pred)
        pred_conns = data['pred_conns']
        inst_num = data['inst_num']
        if inst_num != dataset.inst_num:
            logger.warn(
                'instance number in {} is different from dataset: {} vs {}'.
                format(ofn_pred, inst_num, len(dataset)))
    else:
        if cfg.random_conns:
            pred_conns = []
            for nbr, dist, idx in zip(dataset.subset_nbrs,
                                      dataset.subset_dists,
                                      dataset.subset_idxs):
                for _ in range(cfg.max_conn):
                    pred_rel_nbr = np.random.choice(np.arange(len(nbr)))
                    pred_abs_nbr = nbr[pred_rel_nbr]
                    pred_peaks[idx].append(pred_abs_nbr)
                    pred_dist2peak[idx].append(dist[pred_rel_nbr])
                    pred_conns.append(pred_rel_nbr)
            pred_conns = np.array(pred_conns)
        else:
            pred_conns = test(model, dataset, cfg, logger)
            for pred_rel_nbr, nbr, dist, idx in zip(pred_conns,
                                                    dataset.subset_nbrs,
                                                    dataset.subset_dists,
                                                    dataset.subset_idxs):
                pred_abs_nbr = nbr[pred_rel_nbr]
                pred_peaks[idx].extend(pred_abs_nbr)
                pred_dist2peak[idx].extend(dist[pred_rel_nbr])
        inst_num = dataset.inst_num

    if len(pred_conns) > 0:
        logger.info(
            'pred_conns (nbr order): mean({:.1f}), max({}), min({})'.format(
                pred_conns.mean(), pred_conns.max(), pred_conns.min()))

    if not dataset.ignore_label and cfg.eval_interim:
        subset_gt_labels = dataset.subset_gt_labels
        for i in range(cfg.max_conn):
            pred_peaks_labels = np.array([
                dataset.idx2lb[pred_peaks[idx][i]]
                for idx in dataset.subset_idxs
            ])

            acc = accuracy(pred_peaks_labels, subset_gt_labels)
            logger.info(
                '[{}-th] accuracy of pred_peaks labels ({}): {:.4f}'.format(
                    i, len(pred_peaks_labels), acc))

            # the rule for nearest nbr is only appropriate when nbrs is sorted
            nearest_idxs = np.where(pred_conns[:, i] == 0)[0]
            acc = accuracy(pred_peaks_labels[nearest_idxs],
                           subset_gt_labels[nearest_idxs])
            logger.info(
                '[{}-th] accuracy of pred labels (nearest: {}): {:.4f}'.format(
                    i, len(nearest_idxs), acc))

            not_nearest_idxs = np.where(pred_conns[:, i] > 0)[0]
            acc = accuracy(pred_peaks_labels[not_nearest_idxs],
                           subset_gt_labels[not_nearest_idxs])
            logger.info(
                '[{}-th] accuracy of pred labels (not nearest: {}): {:.4f}'.
                format(i, len(not_nearest_idxs), acc))

    with Timer('Peaks to clusters (th_cut={})'.format(cfg.tau)):
        pred_labels = peaks_to_labels(pred_peaks, pred_dist2peak, cfg.tau,
                                      inst_num)

    if cfg.save_output:
        logger.info(
            'save predicted connectivity and labels to {}'.format(ofn_pred))
        if not osp.isfile(ofn_pred) or cfg.force:
            np.savez_compressed(ofn_pred,
                                pred_conns=pred_conns,
                                inst_num=inst_num)

        # save clustering results
        idx2lb = list2dict(pred_labels, ignore_value=-1)

        folder = '{}_gcne_k_{}_th_{}_ig_{}'.format(cfg.test_name, cfg.knn,
                                                   cfg.th_sim,
                                                   cfg.test_data.ignore_ratio)
        opath_pred_labels = osp.join(cfg.work_dir, folder,
                                     'tau_{}_pred_labels.txt'.format(cfg.tau))
        mkdir_if_no_exists(opath_pred_labels)
        write_meta(opath_pred_labels, idx2lb, inst_num=inst_num)

    # evaluation
    if not dataset.ignore_label:
        print('==> evaluation')
        for metric in cfg.metrics:
            evaluate(dataset.gt_labels, pred_labels, metric)
