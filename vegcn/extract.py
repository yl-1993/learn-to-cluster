from __future__ import division

import torch
import os.path as osp
import numpy as np

from vegcn.datasets import build_dataset
from vegcn.models import build_model
from vegcn.test_gcn_v import test

from utils import create_logger, write_feat, mkdir_if_no_exists


def extract_gcn_v(opath_feat, opath_pred_confs, data_name, cfg):
    if osp.isfile(opath_feat) and osp.isfile(opath_pred_confs):
        print('{} and {} already exist.'.format(opath_feat, opath_pred_confs))
        return
    cfg.cuda = torch.cuda.is_available()

    logger = create_logger()

    model = build_model(cfg.model['type'], **cfg.model['kwargs'])

    for k, v in cfg.model['kwargs'].items():
        setattr(cfg[data_name], k, v)
    cfg[data_name].eval_interim = False

    dataset = build_dataset(cfg.model['type'], cfg[data_name])

    pred_confs, gcn_feat = test(model, dataset, cfg, logger)

    logger.info('save predicted confs to {}'.format(opath_pred_confs))
    mkdir_if_no_exists(opath_pred_confs)
    np.savez_compressed(opath_pred_confs,
                        pred_confs=pred_confs,
                        inst_num=dataset.inst_num)

    logger.info('save gcn features to {}'.format(opath_feat))
    mkdir_if_no_exists(opath_feat)
    write_feat(opath_feat, gcn_feat)
