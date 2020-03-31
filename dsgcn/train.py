from __future__ import division

import torch
from mmcv.runner import DistSamplerSeedHook, obj_from_dict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from dsgcn.datasets import build_dataset, build_processor, build_dataloader
from dsgcn.runner import Runner


def train_cluster(model, cfg, logger, batch_processor):
    # prepare data loaders
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.train_data, k, v)

    dataset = build_dataset(cfg.train_data)
    assert not dataset.ignore_label, 'Please specify label_path for training'

    processor = build_processor(cfg.stage)
    data_loaders = [
        build_dataloader(dataset,
                         processor,
                         cfg.batch_size_per_gpu,
                         cfg.workers_per_gpu,
                         train=True,
                         shuffle=True)
    ]

    # train
    if cfg.distributed:
        _dist_train(model, data_loaders, batch_processor, cfg)
    else:
        _single_train(model, data_loaders, batch_processor, cfg)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    assert paramwise_options is None
    return obj_from_dict(optimizer_cfg, torch.optim,
                         dict(params=model.parameters()))


def _dist_train(model, data_loaders, batch_processor, cfg):
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _single_train(model, data_loaders, batch_processor, cfg):
    if cfg.gpus > 1:
        raise NotImplemented
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model,
                    batch_processor,
                    optimizer,
                    cfg.work_dir,
                    cfg.log_level,
                    iter_size=cfg.iter_size)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
