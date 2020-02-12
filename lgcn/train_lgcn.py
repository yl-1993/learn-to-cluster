from __future__ import division

from collections import OrderedDict

import torch
from mmcv.runner import Runner, obj_from_dict
from mmcv.parallel import MMDataParallel
from lgcn.datasets import build_dataset, build_dataloader
from lgcn.online_evaluation import online_evaluate


def batch_processor(model, data, train_mode):
    assert train_mode

    pred, loss = model(data, return_loss=True)

    log_vars = OrderedDict()
    _, _, _, gtmat = data
    acc, p, r = online_evaluate(gtmat, pred)
    log_vars['loss'] = loss.item()
    log_vars['accuracy'] = acc
    log_vars['precision'] = p
    log_vars['recall'] = r

    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(gtmat))

    return outputs


def train_lgcn(model, cfg, logger):
    # prepare data loaders
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.train_data, k, v)
    dataset = build_dataset(cfg.train_data)
    data_loaders = [
        build_dataloader(dataset,
                         cfg.batch_size_per_gpu,
                         cfg.workers_per_gpu,
                         train=True,
                         shuffle=True)
    ]

    # train
    if cfg.distributed:
        raise NotImplementedError
    else:
        _single_train(model, data_loaders, cfg)


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


def _single_train(model, data_loaders, cfg):
    if cfg.gpus > 1:
        raise NotImplemented
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
