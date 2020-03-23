from __future__ import division

import os
import torch
import numpy as np

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from dsgcn.datasets import build_dataset, build_processor, build_dataloader
from post_process import deoverlap
from evaluation import evaluate


def test_cluster_det(model, cfg, logger):
    if cfg.load_from:
        logger.info('load pretrained model from: {}'.format(cfg.load_from))
        load_checkpoint(model, cfg.load_from, strict=True, logger=logger)

    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)
    dataset = build_dataset(cfg.test_data)
    processor = build_processor(cfg.stage)

    losses = []
    pred_scores = []

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
                    output = output.view(-1)
                    prob = output.data.cpu().numpy()
                    pred_scores.append(prob)
    else:
        raise NotImplementedError

    if not dataset.ignore_label:
        avg_loss = sum(losses) / len(losses)
        logger.info('[Test] Overall Loss {:.4f}'.format(avg_loss))

    # save predicted scores
    if cfg.save_output:
        if cfg.load_from:
            fn = os.path.basename(cfg.load_from)
        else:
            fn = 'random'
        opath = os.path.join(cfg.work_dir, fn[:fn.rfind('.pth')] + '.npz')
        meta = {
            'tot_inst_num': dataset.inst_num,
            'proposal_folders': cfg.test_data.proposal_folders,
        }
        print('dump pred_score to {}'.format(opath))
        pred_scores = np.concatenate(pred_scores).ravel()
        np.savez_compressed(opath, data=pred_scores, meta=meta)

    # de-overlap
    proposals = [fn_node for fn_node, _ in dataset.lst]
    pred_labels = deoverlap(pred_scores, proposals, dataset.inst_num,
                            cfg.th_pos, cfg.th_iou)

    # evaluation
    if not dataset.ignore_label:
        print('==> evaluation')
        gt_labels = dataset.labels
        for metric in cfg.metrics:
            evaluate(gt_labels, pred_labels, metric)
