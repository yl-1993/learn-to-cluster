from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader

from dsgcn.datasets.sampler import DistGivenIterationSampler, DistSequentialSampler


def build_dataloader(dataset,
                     processor,
                     batch_size_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     train=False,
                     **kwargs):
    rank, world_size = get_dist_info()
    if train:
        sampler = DistGivenIterationSampler(dataset, cfg.max_iter, \
                                batch_size_per_gpu, world_size, rank)
        batch_size = batch_size_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = DistSequentialSampler(dataset, world_size, rank)
        batch_size = num_gpus * batch_size_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        processor(dataset),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        **kwargs)

    return data_loader
