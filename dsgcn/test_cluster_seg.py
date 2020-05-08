from __future__ import division

import glob
import torch
import numpy as np
import os.path as osp
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from utils import list2dict, write_meta
from dsgcn.datasets import (build_dataset, build_processor, build_dataloader)
from post_process import deoverlap
from evaluation import evaluate


def test_cluster_seg(model, cfg, logger):
    assert osp.isfile(cfg.pred_iou_score)

    if cfg.load_from:
        logger.info('load pretrained model from: {}'.format(cfg.load_from))
        load_checkpoint(model, cfg.load_from, strict=True, logger=logger)

    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)

    setattr(cfg.test_data, 'pred_iop_score', cfg.pred_iop_score)

    dataset = build_dataset(cfg.test_data)
    processor = build_processor(cfg.stage)

    inst_num = dataset.inst_num

    # read pred_scores from file and do sanity check
    d = np.load(cfg.pred_iou_score, allow_pickle=True)
    pred_scores = d['data']
    meta = d['meta'].item()
    assert inst_num == meta['tot_inst_num'], '{} vs {}'.format(
        inst_num, meta['tot_inst_num'])

    proposals = [fn_node for fn_node, _ in dataset.tot_lst]
    _proposals = []
    fn_node_pattern = '*_node.npz'
    for proposal_folder in meta['proposal_folders']:
        fn_clusters = sorted(
            glob.glob(osp.join(proposal_folder, fn_node_pattern)))
        _proposals.extend([fn_node for fn_node in fn_clusters])
    assert proposals == _proposals, '{} vs {}'.format(len(proposals),
                                                      len(_proposals))

    losses = []
    pred_outlier_scores = []
    stats = {'mean': []}

    if cfg.gpus == 1:
        data_loader = build_dataloader(dataset,
                                       processor,
                                       cfg.test_batch_size_per_gpu,
                                       cfg.workers_per_gpu,
                                       train=False)

        model = MMDataParallel(model, device_ids=range(cfg.gpus))
        if cfg.cuda:
            model.cuda()

        model.eval()
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                output, loss = model(data, return_loss=True)
                losses += [loss.item()]
                if i % cfg.log_config.interval == 0:
                    if dataset.ignore_label:
                        logger.info('[Test] Iter {}/{}'.format(
                            i, len(data_loader)))
                    else:
                        logger.info('[Test] Iter {}/{}: Loss {:.4f}'.format(
                            i, len(data_loader), loss))
                if cfg.save_output:
                    output = F.softmax(output, dim=1)
                    output = output[:, 1, :]
                    scores = output.data.cpu().numpy()
                    pred_outlier_scores.extend(list(scores))
                    stats['mean'] += [scores.mean()]
    else:
        raise NotImplementedError

    if not dataset.ignore_label:
        avg_loss = sum(losses) / len(losses)
        logger.info('[Test] Overall Loss {:.4f}'.format(avg_loss))

    scores_mean = 1. * sum(stats['mean']) / len(stats['mean'])
    logger.info('mean of pred_outlier_scores: {:.4f}'.format(scores_mean))

    # save predicted scores
    if cfg.save_output:
        if cfg.load_from:
            fn = osp.basename(cfg.load_from)
        else:
            fn = 'random'
        opath = osp.join(cfg.work_dir, fn[:fn.rfind('.pth')] + '.npz')
        meta = {
            'tot_inst_num': inst_num,
            'proposal_folders': cfg.test_data.proposal_folders,
        }
        logger.info('dump pred_outlier_scores ({}) to {}'.format(
            len(pred_outlier_scores), opath))
        np.savez_compressed(opath, data=pred_outlier_scores, meta=meta)

    # post-process
    outlier_scores = {
        fn_node: outlier_score
        for (fn_node,
             _), outlier_score in zip(dataset.lst, pred_outlier_scores)
    }

    # de-overlap (w gcn-s)
    pred_labels_w_seg = deoverlap(pred_scores,
                                  proposals,
                                  inst_num,
                                  cfg.th_pos,
                                  cfg.th_iou,
                                  outlier_scores=outlier_scores,
                                  th_outlier=cfg.th_outlier,
                                  keep_outlier=cfg.keep_outlier)

    # de-overlap (wo gcn-s)
    pred_labels_wo_seg = deoverlap(pred_scores, proposals, inst_num,
                                   cfg.th_pos, cfg.th_iou)

    # save predicted labels
    if cfg.save_output:
        ofn_meta_w_seg = osp.join(cfg.work_dir, 'pred_labels_w_seg.txt')
        ofn_meta_wo_seg = osp.join(cfg.work_dir, 'pred_labels_wo_seg.txt')
        print('save predicted labels to {} and {}'.format(
            ofn_meta_w_seg, ofn_meta_wo_seg))
        pred_idx2lb_w_seg = list2dict(pred_labels_w_seg, ignore_value=-1)
        pred_idx2lb_wo_seg = list2dict(pred_labels_wo_seg, ignore_value=-1)
        write_meta(ofn_meta_w_seg, pred_idx2lb_w_seg, inst_num=inst_num)
        write_meta(ofn_meta_wo_seg, pred_idx2lb_wo_seg, inst_num=inst_num)

    # evaluation
    if not dataset.ignore_label:
        gt_labels = dataset.labels
        print('==> evaluation (with gcn-s)')
        for metric in cfg.metrics:
            evaluate(gt_labels, pred_labels_w_seg, metric)
        print('==> evaluation (without gcn-s)')
        for metric in cfg.metrics:
            evaluate(gt_labels, pred_labels_wo_seg, metric)
