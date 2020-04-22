from __future__ import division

from collections import OrderedDict

from dsgcn.runner import Runner
from mmcv.parallel import MMDataParallel

from dsgcn.train import build_optimizer
from vegcn.datasets import build_dataset
from lgcn.datasets import build_dataloader


def batch_processor(model, data, train_mode):
    assert train_mode

    _, loss = model(data, return_loss=True)

    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()

    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data[2]))

    return outputs


def train_gcn_e(model, cfg, logger):
    # prepare data loaders
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.train_data, k, v)

    dataset = build_dataset(cfg.model['type'], cfg.train_data)
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


def _single_train(model, data_loaders, cfg):
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
