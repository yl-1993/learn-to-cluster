import torch
import torch.nn.functional as F
import numpy as np

from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from dsgcn.datasets.sampler import (DistributedSampler,
                                    DistributedSequentialSampler)


def collate_graphs(batch):
    bs = len(batch)
    if bs > 1:
        feat, adj, lb = zip(*batch)
        sizes = [f.shape[0] for f in feat]
        max_size = max(sizes)
        lb = torch.from_numpy(np.array(lb))
        # pad to [X, 0]
        pad_feat = [
            F.pad(torch.from_numpy(f), (0, 0, 0, max_size - s), value=0)
            for f, s in zip(feat, sizes)
        ]
        # pad to [[A, 0], [0, 0]]
        pad_adj = [
            F.pad(torch.from_numpy(a), (0, max_size - s, 0, max_size - s),
                  value=0) for a, s in zip(adj, sizes)
        ]
        # pad to [[A, 0], [0, I]]
        pad_adj = [
            a + F.pad(torch.eye(max_size - s),
                      (s, 0, s, 0), value=0) if s < max_size else a
            for a, s in zip(pad_adj, sizes)
        ]
        pad_feat = default_collate(pad_feat)
        pad_adj = default_collate(pad_adj)
        return pad_feat, pad_adj, lb
    else:
        return default_collate(batch)


def build_dataloader(dataset,
                     processor,
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

    data_loader = DataLoader(processor(dataset),
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=collate_graphs,
                             pin_memory=False,
                             **kwargs)

    return data_loader
