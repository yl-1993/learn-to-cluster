from .cluster_dataset import ClusterDataset
from .build_dataloader import build_dataloader


def build_dataset(cfg):
    return ClusterDataset(cfg)
