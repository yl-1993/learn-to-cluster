from __future__ import division

import os
import torch
import argparse
import numpy as np

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from utils import (create_logger, set_random_seed,
                    rm_suffix, mkdir_if_no_exists)

from dsgcn.datasets import build_dataset, build_dataloader, build_processor
from dsgcn.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Test Cluster Detection')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--processor_type', choices=['det', 'seg'], default='det')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--load_from', help='the checkpoint file to load from')
    parser.add_argument('--gpus', type=int, default=1,
            help='number of gpus(only applicable to non-distributed training)')
    parser.add_argument('--save_output', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    args = parser.parse_args()

    return args


def test_cluster_det(model, dataset, processor, cfg, logger=None):
    if cfg.load_from:
        load_checkpoint(model, cfg.load_from)

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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set cuda
    cfg.cuda = not args.no_cuda and torch.cuda.is_available()

    # set cudnn_benchmark & cudnn_deterministic
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('cudnn_deterministic', False):
        torch.backends.cudnn.deterministic = True

    # update configs according to args
    if not hasattr(cfg, 'work_dir'):
        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        else:
            cfg_name = rm_suffix(os.path.basename(args.config))
            cfg.work_dir = os.path.join('./data/work_dir', cfg_name)
    mkdir_if_no_exists(cfg.work_dir, is_folder=True)
    if not hasattr(cfg, 'processor_type'):
        cfg.processor_type = args.processor_type
    if args.load_from is not None:
        cfg.load_from = args.load_from

    cfg.gpus = args.gpus
    cfg.save_output = args.save_output

    logger = create_logger()

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_model(cfg.model['type'], **cfg.model['kwargs'])

    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)
    test_dataset = build_dataset(cfg.test_data)
    test_processor = build_processor(cfg.processor_type)

    test_cluster_det(model, test_dataset, test_processor, cfg, logger)


if __name__ == '__main__':
    main()
