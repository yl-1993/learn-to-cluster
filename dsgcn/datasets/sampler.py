import math
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import (DistributedSampler as
                                          _DistributedSampler)

__all__ = ["DistributedSampler", "DistributedSequentialSampler"]


class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistributedSequentialSampler(Sampler):
    def __init__(self, dataset, world_size, rank):
        assert rank >= 0
        assert dataset.size >= world_size, '{} vs {}'.format(
            dataset.size, world_size)
        sub_num = int(math.ceil(1. * dataset.size / world_size))
        # add extra samples to make it evenly divisible
        tot_num = sub_num * world_size
        self.beg = sub_num * rank
        self.end = min(self.beg + sub_num, tot_num)

    def __iter__(self):
        indices = list(range(self.beg, self.end))
        return iter(indices)

    def __len__(self):
        return self.end - self.beg
