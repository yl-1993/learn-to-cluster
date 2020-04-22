from __future__ import division

from collections import OrderedDict

import torch
from vegcn.runner import Runner
from dsgcn.train import build_optimizer
from vegcn.datasets import build_dataset
from utils import sparse_mx_to_torch_sparse_tensor


def batch_processor(model, data, train_mode):
    assert train_mode

    _, loss = model(data, return_loss=True)

    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()

    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data[2]))

    return outputs


def train_gcn_v(model, cfg, logger):
    # prepare dataset
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.train_data, k, v)
    dataset = build_dataset(cfg.model['type'], cfg.train_data)

    # train
    if cfg.distributed:
        raise NotImplementedError
    else:
        _single_train(model, dataset, cfg)


def _single_train(model, dataset, cfg):
    if cfg.gpus > 1:
        raise NotImplemented

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

    features = torch.FloatTensor(dataset.features)
    adj = sparse_mx_to_torch_sparse_tensor(dataset.adj)
    labels = torch.FloatTensor(dataset.labels)

    if cfg.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

    train_data = [[features, adj, labels]]
    runner.run(train_data, cfg.workflow, cfg.total_epochs)
