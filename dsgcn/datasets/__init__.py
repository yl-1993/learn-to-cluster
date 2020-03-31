from .cluster_dataset import ClusterDataset
from .cluster_det_processor import ClusterDetProcessor
from .cluster_seg_processor import ClusterSegProcessor
from .build_dataloader import build_dataloader

__factory__ = {
    'det': ClusterDetProcessor,
    'seg': ClusterSegProcessor,
}


def build_dataset(cfg):
    return ClusterDataset(cfg)


def build_processor(name):
    if name not in __factory__:
        raise KeyError("Unknown processor:", name)
    return __factory__[name]
