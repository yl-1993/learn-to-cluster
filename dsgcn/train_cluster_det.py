from collections import OrderedDict

from dsgcn.train import train_cluster


def batch_processor(model, data, train_mode):
    assert train_mode
    _, loss = model(data, return_loss=True)
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()

    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data[-1]))

    return outputs


def train_cluster_det(model, cfg, logger):
    train_cluster(model, cfg, logger, batch_processor)
