import torch

from collections import OrderedDict

from dsgcn.train import train_cluster
from evaluation import accuracy


def batch_processor(model, data, train_mode):
    assert train_mode
    pred, loss = model(data, return_loss=True)

    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    _, _, gt_labels = data
    # TODO: remove pad_label when computing batch accuracy
    pred_labels = torch.argmax(pred.cpu(), dim=1).long()
    gt_labels = gt_labels.cpu().numpy()
    pred_labels = pred_labels.numpy()
    log_vars['acc'] = accuracy(gt_labels, pred_labels)

    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data[-1]))

    return outputs


def train_cluster_seg(model, cfg, logger):
    train_cluster(model, cfg, logger, batch_processor)
