from __future__ import division

import os
import torch
import numpy as np

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from dsgcn.datasets import build_dataset, build_processor, build_dataloader


def test_cluster_det(model, cfg, logger):
    if cfg.load_from:
        load_checkpoint(model, cfg.load_from)

    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)
    dataset = build_dataset(cfg.test_data)
    processor = build_processor(cfg.stage)

    mseloss = torch.nn.MSELoss()

    losses = []
    output_probs = []

    if cfg.gpus == 1:
        data_loader = build_dataloader(
                dataset,
                processor,
                cfg.batch_size_per_gpu,
                cfg.workers_per_gpu,
                cfg.gpus,
                train=False)

        model = MMDataParallel(model, device_ids=range(cfg.gpus))
        if cfg.cuda:
            model.cuda()

        model.eval()
        for i, (x, adj, label) in enumerate(data_loader):
            with torch.no_grad():
                if cfg.cuda:
                    label = label.cuda(non_blocking=True)
                output = model(x, adj).view(-1)
                loss = mseloss(output, label).item()
                losses += [loss]
                if i % cfg.print_freq == 0:
                    logger.info('[Test] Iter {}/{}: Loss {:.4f}'.format(i, len(data_loader), loss))
                if cfg.save_output:
                    prob = output.data.cpu().numpy()
                    output_probs.append(prob)
    else:
        raise NotImplementedError

    avg_loss = sum(losses) / len(losses)
    logger.info('[Test] Overall Loss {:.4f}'.format(avg_loss))

    if cfg.save_output:
        fn = os.path.basename(cfg.load_from)
        opath = os.path.join(cfg.work_dir, fn[:fn.rfind('.pth.tar')] + '.npz')
        meta = {
            'tot_inst_num': len(dataset.idx2lb),
            'proposal_folders': cfg.test_data.proposal_folders,
        }
        print('dump output to {}'.format(opath))
        np.savez_compressed(opath, data=output_probs, meta=meta)
