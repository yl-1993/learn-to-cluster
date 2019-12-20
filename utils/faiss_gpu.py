import os
import gc
import numpy as np
from tqdm import tqdm

import faiss

__all__ = ['faiss_search_approx_knn']


class faiss_index_wrapper():
    def __init__(self,
                 target,
                 nprobe=128,
                 num_gpu=None,
                 index_factory_str=None,
                 verbose=False,
                 mode='proxy',
                 using_gpu=True):
        self._res_list = []

        found_gpu = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
        if found_gpu == 0:
            raise RuntimeError(
                "No GPU found. Please export CUDA_VISIBLE_DEVICES")
        if num_gpu is None or num_gpu > found_gpu:
            num_gpu = found_gpu
        print('[faiss gpu] #GPU: {}'.format(num_gpu))

        size, dim = target.shape
        assert size > 0, "size: {}".format(size)
        index_factory_str = "IVF{},PQ{}".format(
            min(8192, 16 * round(np.sqrt(size))),
            32) if index_factory_str is None else index_factory_str
        cpu_index = faiss.index_factory(dim, index_factory_str)
        cpu_index.nprobe = nprobe

        if mode == 'proxy':
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False

            index = faiss.IndexProxy()
            for i in range(num_gpu):
                res = faiss.StandardGpuResources()
                self._res_list.append(res)
                sub_index = faiss.index_cpu_to_gpu(
                    res, i, cpu_index, co) if using_gpu else cpu_index
                index.addIndex(sub_index)
        elif mode == 'shard':
            raise NotImplementedError
        else:
            raise KeyError("Unknown index mode")

        index = faiss.IndexIDMap(index)
        index.verbose = verbose

        # get nlist to decide how many samples used for training
        nlist = int([
            item for item in index_factory_str.split(",") if 'IVF' in item
        ][0].replace("IVF", ""))

        # training
        if not index.is_trained:
            indexes_sample_for_train = np.random.randint(
                0, size, nlist * 256)
            index.train(target[indexes_sample_for_train])

        # add with ids
        target_ids = np.arange(0, size)
        index.add_with_ids(target, target_ids)
        self.index = index

    def search(self, *args, **kargs):
        return self.index.search(*args, **kargs)

    def __del__(self):
        self.index.reset()
        del self.index
        for res in self._res_list:
            del res


def batch_search(index, query, k, bs, verbose=False):
    n = len(query)
    dists = np.zeros((n, k), dtype=np.float32)
    nbrs = np.zeros((n, k), dtype=np.int64)

    for sid in tqdm(range(0, n, bs),
                    desc="faiss searching...",
                    disable=not verbose):
        eid = min(n, sid + bs)
        dists[sid:eid], nbrs[sid:eid] = index.search(query[sid:eid], k)
    return dists, nbrs


def faiss_search_approx_knn(query,
                            target,
                            k,
                            nprobe=128,
                            bs=int(1e6),
                            num_gpu=None,
                            index_factory_str=None,
                            verbose=False):
    index = faiss_index_wrapper(target,
                                nprobe=nprobe,
                                num_gpu=num_gpu,
                                index_factory_str=index_factory_str,
                                verbose=verbose)
    dists, nbrs = batch_search(index, query, k=k, bs=bs, verbose=verbose)

    del index
    gc.collect()
    return dists, nbrs
