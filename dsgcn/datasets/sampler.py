import math
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


__all__ = ["DistGivenIterationSampler", "DistSequentialSampler"]


class DistGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size, rank):
        self.dsize = dataset.size
        self.total_size = total_iter * batch_size
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            self.indices = self.gen_all_iter_indices()
            return iter(self.indices)
        else:
            raise RuntimeError("this sampler is designed to be called only once!!")

    def gen_all_iter_indices(self, rank):
        # each process shuffle independently
        np.random.seed(rank)

        indices = np.arange(len(self.dataset))
        indices = indices[:self.total_size]
        num_repeat = (self.total_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:self.total_size]

        for beg in range(0, self.total_size, self.dsize):
            end = min(beg + self.dsize, self.total_size)
            np.random.shuffle(indices[beg:end])

        assert len(indices) == self.total_size
        return indices

    def __len__(self):
        return self.total_size


class DistSequentialSampler(Sampler):
    def __init__(self, dataset, world_size, rank):
        assert rank >= 0
        assert dataset.size >= world_size, '{} vs {}'.format(dataset.size, world_size)
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
