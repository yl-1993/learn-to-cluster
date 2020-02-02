from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader

from dsgcn.datasets.sampler import (DistributedSampler,
                                    DistributedSequentialSampler)


def build_dataloader(dataset,
                     batch_size_per_gpu,
                     workers_per_gpu,
                     shuffle=False,
                     train=False,
                     **kwargs):
    rank, world_size = get_dist_info()
    if train:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle)
    else:
        sampler = DistributedSequentialSampler(dataset, world_size, rank)
    batch_size = batch_size_per_gpu
    num_workers = workers_per_gpu

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             pin_memory=False,
                             **kwargs)

    return data_loader
